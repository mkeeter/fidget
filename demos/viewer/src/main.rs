use anyhow::Result;
use clap::Parser;
use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::{
    egui,
    egui_wgpu::{self, wgpu},
};
use env_logger::Env;
use log::{debug, error, info, warn};
use nalgebra::{Point2, Point3};
use notify::Watcher;
use zerocopy::{FromBytes, Immutable, IntoBytes};

use fidget::render::{
    GeometryPixel, ImageRenderConfig, RotateHandle, TranslateHandle, View2,
    View3, VoxelRenderConfig,
};

use std::{error::Error, path::Path};

/// Minimal viewer, using Fidget to render a Rhai script
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// File to watch
    target: String,
}

fn file_watcher_thread(
    path: &Path,
    rx: Receiver<()>,
    tx: Sender<String>,
) -> Result<()> {
    let read_file = || -> Result<String> {
        let out = String::from_utf8(std::fs::read(path)?).unwrap();
        Ok(out)
    };
    let mut contents = read_file()?;
    tx.send(contents.clone())?;

    loop {
        // Wait for a file change notification
        rx.recv()?;
        let new_contents = loop {
            match read_file() {
                Ok(c) => break c,
                Err(e) => {
                    warn!("file read error: {e:?}");
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
        };
        if contents != new_contents {
            contents = new_contents;
            debug!("file contents changed!");
            tx.send(contents.clone())?;
        }
    }
}

fn rhai_script_thread(
    rx: Receiver<String>,
    tx: Sender<Result<fidget::rhai::ScriptContext, String>>,
) -> Result<()> {
    let mut engine = fidget::rhai::Engine::new();
    loop {
        let script = rx.recv()?;
        debug!("rhai script thread received script");
        let r = engine.run(&script).map_err(|e| e.to_string());
        debug!("rhai script thread is sending result to render thread");
        tx.send(r)?;
    }
}

struct RenderSettings {
    image_size: fidget::render::ImageSize,
    mode: RenderMode,
}

enum ImageData {
    Rgba(Vec<[u8; 4]>),
    Geometry(Vec<GeometryPixel>),
}

struct RenderResult {
    images: Vec<ImageData>,
    render_time: std::time::Duration,
    image_size: fidget::render::ImageSize,
}

fn render_thread<F>(
    cfg: Receiver<RenderSettings>,
    rx: Receiver<Result<fidget::rhai::ScriptContext, String>>,
    tx: Sender<Result<RenderResult, String>>,
    wake: Sender<()>,
) -> Result<()>
where
    F: fidget::eval::Function
        + fidget::eval::MathFunction
        + fidget::render::RenderHints,
{
    // This is our target framerate; updates faster than this will be merged.
    const DT: std::time::Duration = std::time::Duration::from_millis(16);

    let mut config = None;
    let mut script_ctx = None;
    let mut timeout_time: Option<std::time::Instant> = None;
    loop {
        let timeout = if let Some(t) = timeout_time {
            t.checked_duration_since(std::time::Instant::now())
                .unwrap_or(std::time::Duration::ZERO)
        } else {
            std::time::Duration::from_secs(u64::MAX)
        };
        crossbeam_channel::select! {
            recv(rx) -> msg => match msg? {
                Ok(s) => {
                    debug!("render thread got a new result");
                    script_ctx = Some(s);
                    if timeout_time.is_none() {
                        timeout_time = Some(std::time::Instant::now() + DT);
                    }
                    continue;
                },
                Err(e) => {
                    error!("render thread got error {e:?}; forwarding");
                    script_ctx = None;
                    tx.send(Err(e.to_string()))?;
                    wake.send(()).unwrap();
                }
            },
            recv(cfg) -> msg => {
                debug!("render thread got a new config");
                config = Some(msg?);
                if timeout_time.is_none() {
                    timeout_time = Some(std::time::Instant::now() + DT);
                }
                continue;
            },
            default(timeout) => debug!("render thread timed out"),
        }

        // Reset our timer
        if timeout_time.take().is_none() {
            continue;
        }

        if let (Some(out), Some(render_config)) = (&script_ctx, &config) {
            debug!("Rendering...");
            let render_start = std::time::Instant::now();
            let mut images = vec![];
            for s in out.shapes.iter() {
                let tape = fidget::shape::Shape::<F>::from(s.tree.clone());
                let out = render(
                    &render_config.mode,
                    tape,
                    render_config.image_size,
                    s.color_rgb,
                );
                images.push(out);
            }
            let dt = render_start.elapsed();
            tx.send(Ok(RenderResult {
                images,
                render_time: dt,
                image_size: render_config.image_size,
            }))?;
            wake.send(()).unwrap();
        }
    }
}

fn render<F: fidget::eval::Function + fidget::render::RenderHints>(
    mode: &RenderMode,
    shape: fidget::shape::Shape<F>,
    image_size: fidget::render::ImageSize,
    color: [u8; 3],
) -> ImageData {
    match mode {
        RenderMode::TwoD { view, mode, .. } => {
            let config = ImageRenderConfig {
                image_size,
                tile_sizes: F::tile_sizes_2d(),
                view: *view,
                ..Default::default()
            };

            let out = match mode {
                Mode2D::Color => {
                    let image = config
                        .run::<_, fidget::render::BitRenderMode>(shape)
                        .unwrap();
                    let c = [color[0], color[1], color[2], u8::MAX];
                    image.map(|p| if *p { c } else { [0u8; 4] })
                }

                Mode2D::Sdf => config
                    .run::<_, fidget::render::SdfRenderMode>(shape)
                    .unwrap()
                    .map(|&[r, g, b]| [r, g, b, u8::MAX]),

                Mode2D::ExactSdf => config
                    .run::<_, fidget::render::SdfPixelRenderMode>(shape)
                    .unwrap()
                    .map(|&[r, g, b]| [r, g, b, u8::MAX]),

                Mode2D::Debug => {
                    let image = config
                        .run::<_, fidget::render::DebugRenderMode>(shape)
                        .unwrap();
                    image.map(|p| p.as_debug_color())
                }
            };
            let (data, _) = out.take();
            ImageData::Rgba(data)
        }
        RenderMode::ThreeD { view, mode, .. } => {
            // XXX allow selection of depth?
            let config = VoxelRenderConfig {
                image_size: fidget::render::VoxelSize::new(
                    image_size.width(),
                    image_size.height(),
                    image_size.width().max(image_size.height()),
                ),
                tile_sizes: F::tile_sizes_3d(),
                view: *view,
                ..Default::default()
            };

            // Get the geometry buffer from the voxel rendering process
            let geometry_buffer = config.run(shape).unwrap();

            match mode {
                Mode3D::Color | Mode3D::Heightmap => {
                    // For both rendering modes, we'll just pass the GeometryPixel data
                    // to the GPU, which will apply the appropriate rendering effect
                    let (data, _) = geometry_buffer.take();
                    ImageData::Geometry(data)
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .init();
    let args = Args::parse();

    // This is a pipelined process with separate threads for each stage.
    // Unbounded channels are used to send data through the pipeline
    //
    // - File watcher (via `notify`) produces () notifications
    // - Loading from a file produces the text of the script
    // - Script evaluation produces the Rhai result (or error)
    // - Rendering produces the image (or an error)
    // - Posting wake events to the GUI
    //
    // In addition, the GUI (main) thread will send new rendering configuration
    // to the render thread when the user changes things.
    let (file_watcher_tx, file_watcher_rx) = unbounded();
    let (rhai_script_tx, rhai_script_rx) = unbounded();
    let (rhai_result_tx, rhai_result_rx) = unbounded();
    let (render_tx, render_rx) = unbounded();
    let (config_tx, config_rx) = unbounded();
    let (wake_tx, wake_rx) = unbounded();

    let path = Path::new(&args.target).to_owned();
    std::thread::spawn(move || {
        let _ = file_watcher_thread(&path, file_watcher_rx, rhai_script_tx);
        info!("file watcher thread is done");
    });
    std::thread::spawn(move || {
        let _ = rhai_script_thread(rhai_script_rx, rhai_result_tx);
        info!("rhai script thread is done");
    });
    std::thread::spawn(move || {
        #[cfg(feature = "jit")]
        type F = fidget::jit::JitFunction;

        #[cfg(not(feature = "jit"))]
        type F = fidget::vm::VmFunction;

        let _ =
            render_thread::<F>(config_rx, rhai_result_rx, render_tx, wake_tx);
        info!("render thread is done");
    });

    // Automatically select the best implementation for your platform.
    let mut watcher = notify::recommended_watcher(move |res| match res {
        Ok(event) => {
            info!("file watcher: {event:?}");
            file_watcher_tx.send(()).unwrap();
        }
        Err(e) => panic!("watch error: {:?}", e),
    })
    .unwrap();
    watcher
        .watch(Path::new(&args.target), notify::RecursiveMode::NonRecursive)
        .unwrap();

    let mut options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    let size = egui::Vec2::new(640.0, 480.0);
    options.viewport.inner_size = Some(size);

    eframe::run_native(
        "Fidget",
        options,
        Box::new(move |cc| {
            // Run a worker thread which listens for wake events and pokes the
            // UI whenever they come in.
            let egui_ctx = cc.egui_ctx.clone();
            std::thread::spawn(move || {
                while let Ok(()) = wake_rx.recv() {
                    egui_ctx.request_repaint();
                }
                info!("wake thread is done");
            });

            Ok(Box::new(ViewerApp::new(cc, config_tx, render_rx)))
        }),
    )?;

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Eq, PartialEq)]
enum Mode2D {
    Color,
    Sdf,
    ExactSdf,
    Debug,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum Mode3D {
    Color,
    Heightmap,
}

#[derive(Copy, Clone)]
enum Drag3D {
    Pan(TranslateHandle<3>),
    Rotate(RotateHandle),
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
enum RenderMode {
    TwoD {
        view: View2,

        /// Drag start position (in model coordinates)
        drag_start: Option<TranslateHandle<2>>,
        mode: Mode2D,
    },
    ThreeD {
        view: View3,

        /// Drag start position (in model coordinates)
        drag_start: Option<Drag3D>,

        mode: Mode3D,
    },
}

impl RenderMode {
    fn set_2d_mode(&mut self, new_mode: Mode2D) -> bool {
        match self {
            RenderMode::TwoD { mode, .. } => {
                let changed = *mode != new_mode;
                *mode = new_mode;
                changed
            }
            RenderMode::ThreeD { .. } => {
                *self = RenderMode::TwoD {
                    // TODO get parameters from 3D camera here?
                    view: Default::default(),
                    drag_start: None,
                    mode: new_mode,
                };
                true
            }
        }
    }
    fn set_3d_mode(&mut self, new_mode: Mode3D) -> bool {
        match self {
            RenderMode::TwoD { .. } => {
                *self = RenderMode::ThreeD {
                    view: View3::default(),
                    drag_start: None,
                    mode: new_mode,
                };
                true
            }
            RenderMode::ThreeD { mode, .. } => {
                let changed = *mode != new_mode;
                *mode = new_mode;
                changed
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TextureType {
    Rgba,
    Geometry,
}

/// Configuration for 3D rendering with geometry data
#[repr(C)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, Immutable)]
struct RenderConfig {
    render_mode: u32, // 0 = heightmap, 1 = shaded
    light_direction: [f32; 3],
    ambient_intensity: f32,
    diffuse_intensity: f32,
    _padding: [u32; 3], // Ensure 16-byte alignment
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            render_mode: 0,                   // Default to heightmap
            light_direction: [0.5, 0.5, 0.5], // Light from top-right-front
            ambient_intensity: 0.3,
            diffuse_intensity: 0.7,
            _padding: [0; 3],
        }
    }
}

#[allow(unused)]
struct CustomTexture {
    bind_group: wgpu::BindGroup,

    // These are unused but must remain alive
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,

    // Type of texture (RGBA or Geometry)
    texture_type: TextureType,
}

struct CustomResources {
    // Pipelines for different texture types
    rgba_pipeline: wgpu::RenderPipeline,
    geometry_pipeline: wgpu::RenderPipeline,

    // Textures with their metadata
    tex: Option<(fidget::render::ImageSize, Vec<CustomTexture>)>,

    // Samplers
    rgba_sampler: wgpu::Sampler,
    geometry_sampler: wgpu::Sampler,

    // Bind group layouts
    rgba_bind_group_layout: wgpu::BindGroupLayout,
    geometry_bind_group_layout: wgpu::BindGroupLayout,

    // Uniform buffer for render configuration
    render_config: RenderConfig,
    render_config_buffer: wgpu::Buffer,
}

impl CustomResources {
    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        images: &[ImageData],
        image_size: fidget::render::ImageSize,
        mode: Mode3D,
    ) {
        let (width, height) = (image_size.width(), image_size.height());
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // Function to create a new RGBA texture
        let new_rgba_tex = || -> CustomTexture {
            // Create the texture
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("RGBA Texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Create the texture view
            let texture_view =
                texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create the bind group for this texture
            let bind_group =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RGBA Bind Group"),
                    layout: &self.rgba_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &texture_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                &self.rgba_sampler,
                            ),
                        },
                    ],
                });

            CustomTexture {
                bind_group,
                texture,
                texture_view,
                texture_type: TextureType::Rgba,
            }
        };

        // Function to create a new Geometry texture
        let new_geometry_tex = || -> CustomTexture {
            // Create the texture - we need to store depth and normals,
            // so we'll use a format that can store 4 32-bit values
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Geometry Texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Create the texture view
            let texture_view =
                texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create the bind group for this texture
            let bind_group =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Geometry Bind Group"),
                    layout: &self.geometry_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &texture_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                &self.geometry_sampler,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(
                                wgpu::BufferBinding {
                                    buffer: &self.render_config_buffer,
                                    offset: 0,
                                    size: None,
                                },
                            ),
                        },
                    ],
                });

            CustomTexture {
                bind_group,
                texture,
                texture_view,
                texture_type: TextureType::Geometry,
            }
        };

        // Check to see whether we can reuse textures
        let mut textures = Vec::with_capacity(images.len());
        match &mut self.tex {
            Some((tex_size, tex_data)) if *tex_size == image_size => {
                // We can reuse the texture size, but we need to check each texture type
                tex_data.clear();
                for image in images {
                    match image {
                        ImageData::Rgba(_) => textures.push(new_rgba_tex()),
                        ImageData::Geometry(_) => {
                            textures.push(new_geometry_tex())
                        }
                    }
                }
                *tex_data = textures;
            }
            Some(..) | None => {
                // Create new textures for each image
                for image in images {
                    match image {
                        ImageData::Rgba(_) => textures.push(new_rgba_tex()),
                        ImageData::Geometry(_) => {
                            textures.push(new_geometry_tex())
                        }
                    }
                }
                self.tex = Some((image_size, textures));
            }
        }

        // Update render config - set render mode based on current mode
        self.render_config.render_mode = match mode {
            Mode3D::Heightmap => 0,
            Mode3D::Color => 1,
        };

        // Update the render config buffer
        queue.write_buffer(
            &self.render_config_buffer,
            0,
            self.render_config.as_bytes(),
        );

        // Upload all of the images to textures
        for (image_data, tex) in
            images.iter().zip(self.tex.as_ref().unwrap().1.iter())
        {
            match (image_data, tex.texture_type) {
                (ImageData::Rgba(rgba_data), TextureType::Rgba) => {
                    // Upload RGBA image data
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &tex.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        rgba_data.as_bytes(),
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4 * width),
                            rows_per_image: Some(height),
                        },
                        texture_size,
                    );
                }
                (ImageData::Geometry(geometry_data), TextureType::Geometry) => {
                    // Upload geometry data using AsBytes trait
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &tex.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        geometry_data.as_bytes(),
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(16 * width), // 16 bytes per GeometryPixel (4 u32 values)
                            rows_per_image: Some(height),
                        },
                        texture_size,
                    );
                }
                _ => {
                    // Mismatched texture and data types, this shouldn't happen
                    panic!("Mismatched texture and data types");
                }
            }
        }
    }

    fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        // Only draw if we have a texture
        if let Some((_tex_size, tex_data)) = &self.tex {
            for tex in tex_data {
                match tex.texture_type {
                    TextureType::Rgba => {
                        render_pass.set_pipeline(&self.rgba_pipeline);
                        render_pass.set_bind_group(0, &tex.bind_group, &[]);
                    }
                    TextureType::Geometry => {
                        render_pass.set_pipeline(&self.geometry_pipeline);
                        render_pass.set_bind_group(0, &tex.bind_group, &[]);
                    }
                }
                // Draw 2 triangles (6 vertices) to form a quad
                render_pass.draw(0..6, 0..1);
            }
        }
    }
}

struct CustomCallback {
    data: Option<(Vec<ImageData>, fidget::render::ImageSize)>,
    render_mode: Mode3D,
}

impl egui_wgpu::CallbackTrait for CustomCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &mut CustomResources = resources.get_mut().unwrap();
        if let Some((image_data, image_size)) = self.data.as_ref() {
            resources.prepare(
                device,
                queue,
                image_data.as_slice(),
                *image_size,
                self.render_mode,
            )
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &CustomResources = resources.get().unwrap();
        resources.paint(render_pass);
    }
}

struct ViewerApp {
    /// Current image (or an error)
    image_data: Option<Result<RenderResult, String>>,

    /// Current render mode
    mode: RenderMode,
    image_size: fidget::render::ImageSize,

    config_tx: Sender<RenderSettings>,
    image_rx: Receiver<Result<RenderResult, String>>,
}

////////////////////////////////////////////////////////////////////////////////

impl ViewerApp {
    fn new(
        cc: &eframe::CreationContext,
        config_tx: Sender<RenderSettings>,
        image_rx: Receiver<Result<RenderResult, String>>,
    ) -> Self {
        // Initialize renderer if WGPU is available
        let wgpu_state = cc.wgpu_render_state.as_ref().unwrap();
        let device = &wgpu_state.device;
        let target_format = wgpu_state.target_format;

        // Create RGBA shader module
        let rgba_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RGBA Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/image.wgsl").into(),
                ),
            });

        // Create Geometry shader module
        let geometry_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Geometry Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/geometry.wgsl").into(),
                ),
            });

        // Create samplers
        let rgba_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("RGBA Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let geometry_sampler =
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Geometry Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest, // Use nearest for integer textures
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

        // Create bind group layout for RGBA texture and sampler
        let rgba_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RGBA Bind Group Layout"),
                entries: &[
                    // Texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: true,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
            });

        // Create bind group layout for Geometry texture and sampler
        let geometry_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Geometry Bind Group Layout"),
                entries: &[
                    // Texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::NonFiltering,
                        ),
                        count: None,
                    },
                    // Uniform buffer for render configuration
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create render pipeline layouts
        let rgba_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RGBA Render Pipeline Layout"),
                bind_group_layouts: &[&rgba_bind_group_layout],
                push_constant_ranges: &[],
            });

        let geometry_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Geometry Render Pipeline Layout"),
                bind_group_layouts: &[&geometry_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create the RGBA render pipeline
        let rgba_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("RGBA Render Pipeline"),
                layout: Some(&rgba_pipeline_layout),
                cache: None,
                vertex: wgpu::VertexState {
                    module: &rgba_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &rgba_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::OVER,
                            alpha: wgpu::BlendComponent::OVER,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        // Create the Geometry render pipeline
        let geometry_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Geometry Render Pipeline"),
                layout: Some(&geometry_pipeline_layout),
                cache: None,
                vertex: wgpu::VertexState {
                    module: &geometry_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &geometry_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::OVER,
                            alpha: wgpu::BlendComponent::OVER,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        // Create a buffer for render configuration
        let render_config = RenderConfig::default();
        let render_config_buffer =
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Render Config Buffer"),
                size: std::mem::size_of::<RenderConfig>() as u64,
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // Insert the custom resources into the renderer
        wgpu_state.renderer.write().callback_resources.insert(
            CustomResources {
                rgba_pipeline,
                geometry_pipeline,
                tex: None,
                rgba_sampler,
                geometry_sampler,
                rgba_bind_group_layout,
                geometry_bind_group_layout,
                render_config,
                render_config_buffer,
            },
        );

        Self {
            image_data: None,
            image_size: fidget::render::ImageSize::from(256),

            config_tx,
            image_rx,

            mode: RenderMode::TwoD {
                view: Default::default(),
                drag_start: None,
                mode: Mode2D::Color,
            },
        }
    }

    fn draw_menu(&mut self, ctx: &egui::Context) -> bool {
        let mut changed = false;
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("Config", |ui| {
                    let mut mode_3d = match &self.mode {
                        RenderMode::TwoD { .. } => None,
                        RenderMode::ThreeD { mode, .. } => Some(*mode),
                    };
                    ui.radio_value(
                        &mut mode_3d,
                        Some(Mode3D::Heightmap),
                        "3D heightmap",
                    );
                    ui.radio_value(
                        &mut mode_3d,
                        Some(Mode3D::Color),
                        "3D color",
                    );
                    if let Some(m) = mode_3d {
                        changed = self.mode.set_3d_mode(m);
                    }
                    ui.separator();
                    let mut mode_2d = match &self.mode {
                        RenderMode::TwoD { mode, .. } => Some(*mode),
                        RenderMode::ThreeD { .. } => None,
                    };
                    ui.radio_value(
                        &mut mode_2d,
                        Some(Mode2D::Debug),
                        "2D debug",
                    );
                    ui.radio_value(
                        &mut mode_2d,
                        Some(Mode2D::Sdf),
                        "2D SDF (approx)",
                    );
                    ui.radio_value(
                        &mut mode_2d,
                        Some(Mode2D::ExactSdf),
                        "2D SDF (exact)",
                    );
                    ui.radio_value(
                        &mut mode_2d,
                        Some(Mode2D::Color),
                        "2D Color",
                    );

                    if let Some(m) = mode_2d {
                        changed = self.mode.set_2d_mode(m);
                    }
                });
            });
        });
        changed
    }

    /// Try to receive an image from the worker thread, populating
    /// `self.texture` and `self.stats`, or `self.err`
    fn try_recv_image(&mut self) {
        if let Ok(r) = self.image_rx.try_recv() {
            self.image_data = Some(r);
        }
    }

    fn paint_image(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let pos = ui.next_widget_position();
        let size = ui.available_size();
        let painter = ui.painter_at(egui::Rect {
            min: pos,
            max: pos + size,
        });
        const PADDING: egui::Vec2 = egui::Vec2 { x: 10.0, y: 10.0 };

        let rect = ui.ctx().available_rect();

        if let Some(Ok(image_data)) = &mut self.image_data {
            // Get current 3D render mode if applicable
            let render_mode = match self.mode {
                RenderMode::ThreeD { mode, .. } => mode,
                _ => Mode3D::Heightmap, // Default mode if not in 3D
            };

            // Draw the image using WebGPU
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                CustomCallback {
                    data: if image_data.images.is_empty() {
                        None // use pre-existing data
                    } else {
                        // Pass the image buffers into the GPU renderer
                        Some((
                            std::mem::take(&mut image_data.images),
                            image_data.image_size,
                        ))
                    },
                    render_mode,
                },
            ));

            // The image has been drawn by the CustomCallback
            let layout = painter.layout(
                format!(
                    "Image size: {}×{}\nRender time: {:.2?}",
                    image_data.image_size.width(),
                    image_data.image_size.height(),
                    image_data.render_time,
                ),
                egui::FontId::proportional(14.0),
                egui::Color32::WHITE,
                f32::INFINITY,
            );
            let text_corner = rect.max - layout.size();
            painter.rect_filled(
                egui::Rect {
                    min: text_corner - 2.0 * PADDING,
                    max: rect.max,
                },
                egui::CornerRadius::default(),
                egui::Color32::from_black_alpha(128),
            );
            painter.galley(text_corner - PADDING, layout, egui::Color32::BLACK);
        }

        if let Some(Err(err)) = &self.image_data {
            let layout = painter.layout(
                err.to_string(),
                egui::FontId::proportional(14.0),
                egui::Color32::LIGHT_RED,
                f32::INFINITY,
            );

            let text_corner = rect.min + layout.size();
            painter.rect_filled(
                egui::Rect {
                    min: rect.min,
                    max: text_corner + 2.0 * PADDING,
                },
                egui::CornerRadius::default(),
                egui::Color32::from_black_alpha(128),
            );
            painter.galley(rect.min + PADDING, layout, egui::Color32::BLACK);
        }

        // Return events from the canvas in the inner response
        ui.interact(
            rect,
            egui::Id::new("canvas"),
            egui::Sense::click_and_drag(),
        )
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut render_changed = self.draw_menu(ctx);
        self.try_recv_image();

        let rect = ctx.available_rect();
        let size = rect.max - rect.min;
        let image_size = fidget::render::ImageSize::new(
            (size.x * ctx.pixels_per_point()) as u32,
            (size.y * ctx.pixels_per_point()) as u32,
        );

        if image_size != self.image_size {
            self.image_size = image_size;
            render_changed = true;
        }

        // Draw the current image and/or error
        let r = egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(egui::Color32::BLACK))
            .show(ctx, |ui| self.paint_image(ui))
            .inner;

        // Handle pan and zoom
        match &mut self.mode {
            RenderMode::TwoD {
                view, drag_start, ..
            } => {
                let image_size = fidget::render::ImageSize::new(
                    rect.width() as u32,
                    rect.height() as u32,
                );

                if let Some(pos) = r.interact_pointer_pos() {
                    let pos =
                        image_size.transform_point(Point2::new(pos.x, pos.y));
                    if let Some(prev) = drag_start {
                        render_changed |= view.translate(prev, pos);
                    } else {
                        *drag_start = Some(view.begin_translate(pos));
                    }
                } else {
                    *drag_start = None;
                }

                if r.hovered() {
                    let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                    let mouse_pos = r.hover_pos().map(|p| {
                        let p = p - rect.min;
                        image_size.transform_point(Point2::new(p.x, p.y))
                    });
                    render_changed |=
                        view.zoom((scroll / 100.0).exp2(), mouse_pos);
                }
            }
            RenderMode::ThreeD {
                view, drag_start, ..
            } => {
                let image_size = fidget::render::VoxelSize::new(
                    rect.width() as u32,
                    rect.height() as u32,
                    rect.width().max(rect.height()) as u32,
                );

                if let Some(pos) = r.interact_pointer_pos() {
                    let pos_world = image_size
                        .transform_point(Point3::new(pos.x, pos.y, 0.0));
                    match drag_start {
                        Some(Drag3D::Pan(prev)) => {
                            render_changed |= view.translate(prev, pos_world);
                        }
                        Some(Drag3D::Rotate(prev)) => {
                            render_changed |= view.rotate(prev, pos_world);
                        }
                        None => {
                            if r.dragged_by(egui::PointerButton::Primary) {
                                *drag_start = Some(Drag3D::Pan(
                                    view.begin_translate(pos_world),
                                ));
                            } else if r
                                .dragged_by(egui::PointerButton::Secondary)
                            {
                                *drag_start = Some(Drag3D::Rotate(
                                    view.begin_rotate(pos_world),
                                ));
                            }
                        }
                    }
                } else {
                    *drag_start = None;
                }

                if r.hovered() {
                    let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                    let mouse_pos =
                        ctx.input(|i| i.pointer.hover_pos()).map(|p| {
                            let p = p - rect.min;
                            image_size
                                .transform_point(Point3::new(p.x, p.y, 0.0))
                        });
                    if scroll != 0.0 {
                        view.zoom((scroll / 100.0).exp2(), mouse_pos);
                        render_changed = true;
                    }
                }
            }
        }

        // Kick off a new render if we changed any settings
        if render_changed {
            self.config_tx
                .send(RenderSettings {
                    mode: self.mode,
                    image_size: self.image_size,
                })
                .unwrap();
        }
    }
}
