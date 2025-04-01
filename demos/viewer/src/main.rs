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
    Rgba(Vec<Vec<[u8; 4]>>),
    Geometry {
        images: Vec<Vec<GeometryPixel>>,
        mode: Mode3D,
        max_depth: u32,
    },
}

struct RenderResult {
    images: ImageData,
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
            let images = match &render_config.mode {
                RenderMode::TwoD { canvas, mode } => {
                    let data = out
                        .shapes
                        .iter()
                        .map(|s| {
                            let tape =
                                fidget::shape::Shape::<F>::from(s.tree.clone());
                            render_2d(
                                *mode,
                                canvas.view(),
                                tape,
                                render_config.image_size,
                                s.color_rgb,
                            )
                        })
                        .collect();
                    ImageData::Rgba(data)
                }
                RenderMode::ThreeD { canvas, mode } => {
                    // XXX allow selection of depth?
                    let image_size = render_config.image_size;
                    let voxel_size = fidget::render::VoxelSize::new(
                        image_size.width(),
                        image_size.height(),
                        image_size.width().max(image_size.height()),
                    );
                    let data = out
                        .shapes
                        .iter()
                        .map(|s| {
                            let tape =
                                fidget::shape::Shape::<F>::from(s.tree.clone());
                            render_3d(canvas.view(), tape, voxel_size)
                        })
                        .collect();
                    ImageData::Geometry {
                        images: data,
                        mode: *mode,
                        max_depth: voxel_size.depth(),
                    }
                }
            };

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

fn render_2d<F: fidget::eval::Function + fidget::render::RenderHints>(
    mode: Mode2D,
    view: View2,
    shape: fidget::shape::Shape<F>,
    image_size: fidget::render::ImageSize,
    color: [u8; 3],
) -> Vec<[u8; 4]> {
    let config = ImageRenderConfig {
        image_size,
        tile_sizes: F::tile_sizes_2d(),
        view,
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
    data
}

fn render_3d<F: fidget::eval::Function + fidget::render::RenderHints>(
    view: View3,
    shape: fidget::shape::Shape<F>,
    image_size: fidget::render::VoxelSize,
) -> Vec<GeometryPixel> {
    let config = VoxelRenderConfig {
        image_size,
        tile_sizes: F::tile_sizes_3d(),
        view,
        ..Default::default()
    };

    // Get the geometry buffer from the voxel rendering process
    let geometry_buffer = config.run(shape).unwrap();

    // For both rendering modes, we'll just pass the GeometryPixel data
    // to the GPU, which will apply the appropriate rendering effect
    let (data, _) = geometry_buffer.take();
    data
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

impl Mode2D {
    fn description(&self) -> &'static str {
        match self {
            Self::Color => "2D color",
            Self::Sdf => "2D SDF (approx)",
            Self::ExactSdf => "2D SDF (exact)",
            Self::Debug => "2D debug",
        }
    }
}

#[derive(Copy, Clone, Default)]
struct Canvas2D {
    view: View2,
    drag_start: Option<TranslateHandle<2>>,
}

impl Canvas2D {
    fn view(&self) -> View2 {
        self.view
    }

    fn drag(&mut self, pos_world: Point2<f32>) -> bool {
        if let Some(prev) = &self.drag_start {
            self.view.translate(prev, pos_world)
        } else {
            self.drag_start = Some(self.view.begin_translate(pos_world));
            false
        }
    }

    fn end_drag(&mut self) {
        self.drag_start = None;
    }

    fn zoom(&mut self, amount: f32, pos_world: Option<Point2<f32>>) -> bool {
        self.view.zoom((amount / 100.0).exp2(), pos_world)
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum Mode3D {
    Heightmap,
    Color,
    Shaded,
}

impl Mode3D {
    fn description(&self) -> &'static str {
        match self {
            Self::Heightmap => "3D heightmap",
            Self::Color => "3D color",
            Self::Shaded => "3D shaded",
        }
    }
}

#[derive(Copy, Clone)]
enum Drag3D {
    Pan(TranslateHandle<3>),
    Rotate(RotateHandle),
}

#[derive(Copy, Clone, Default)]
struct Canvas3D {
    view: View3,
    drag_start: Option<Drag3D>,
}

#[derive(Copy, Clone)]
enum DragMode {
    Pan,
    Rotate,
}

impl Canvas3D {
    fn view(&self) -> View3 {
        self.view
    }

    fn drag(&mut self, pos_world: Point3<f32>, drag_mode: DragMode) -> bool {
        match &self.drag_start {
            Some(Drag3D::Pan(prev)) => self.view.translate(prev, pos_world),
            Some(Drag3D::Rotate(prev)) => self.view.rotate(prev, pos_world),
            None => {
                self.drag_start = Some(match drag_mode {
                    DragMode::Pan => {
                        Drag3D::Pan(self.view.begin_translate(pos_world))
                    }
                    DragMode::Rotate => {
                        Drag3D::Rotate(self.view.begin_rotate(pos_world))
                    }
                });
                false
            }
        }
    }

    fn end_drag(&mut self) {
        self.drag_start = None;
    }

    fn zoom(&mut self, amount: f32, pos: Option<Point3<f32>>) -> bool {
        self.view.zoom((amount / 100.0).exp2(), pos)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
enum RenderMode {
    TwoD { canvas: Canvas2D, mode: Mode2D },
    ThreeD { canvas: Canvas3D, mode: Mode3D },
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
                    canvas: Default::default(),
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
                    canvas: Default::default(),
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

/// Configuration for 3D rendering with geometry data
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, IntoBytes, FromBytes, Immutable)]
struct RenderConfig {
    render_mode: u32, // 0 = heightmap, 1 = shaded
    max_depth: u32,
}

#[allow(unused)]
struct CustomTexture {
    bind_group: wgpu::BindGroup,

    // These are unused but must remain alive
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
}

struct Draw2D {
    rgba_pipeline: wgpu::RenderPipeline,
    tex: Option<(fidget::render::ImageSize, Vec<CustomTexture>)>,
    rgba_sampler: wgpu::Sampler,
    rgba_bind_group_layout: wgpu::BindGroupLayout,
}

impl Draw2D {
    fn init(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        // Create RGBA shader module
        let rgba_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RGBA Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/image.wgsl").into(),
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

        // Create render pipeline layouts
        let rgba_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RGBA Render Pipeline Layout"),
                bind_group_layouts: &[&rgba_bind_group_layout],
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
        Draw2D {
            rgba_pipeline,
            tex: None,
            rgba_sampler,
            rgba_bind_group_layout,
        }
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        images: &[Vec<[u8; 4]>],
        image_size: fidget::render::ImageSize,
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
            }
        };

        // Check to see whether we can reuse textures
        match &mut self.tex {
            Some((tex_size, tex_data)) if *tex_size == image_size => {
                tex_data.resize_with(images.len(), new_rgba_tex);
            }
            Some(..) | None => {
                let textures = images.iter().map(|_i| new_rgba_tex()).collect();
                self.tex = Some((image_size, textures));
            }
        }

        // Upload all of the images to textures
        for (image_data, tex) in
            images.iter().zip(self.tex.as_ref().unwrap().1.iter())
        {
            // Upload RGBA image data
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                image_data.as_bytes(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                texture_size,
            );
        }
    }

    fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        // Only draw if we have a texture
        if let Some((_tex_size, tex_data)) = &self.tex {
            for tex in tex_data {
                render_pass.set_pipeline(&self.rgba_pipeline);
                render_pass.set_bind_group(0, &tex.bind_group, &[]);

                // Draw 2 triangles (6 vertices) to form a quad
                render_pass.draw(0..6, 0..1);
            }
        }
    }
}

struct Draw3D {
    geometry_pipeline: wgpu::RenderPipeline,
    tex: Option<(fidget::render::ImageSize, Vec<CustomTexture>)>,
    geometry_sampler: wgpu::Sampler,
    geometry_bind_group_layout: wgpu::BindGroupLayout,

    /// Local copy of render configuration
    render_config: RenderConfig,

    /// GPU buffer for render configuration
    render_config_buffer: wgpu::Buffer,
}

impl Draw3D {
    fn init(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        // Create Geometry shader module
        let geometry_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Geometry Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/geometry.wgsl").into(),
                ),
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

        let geometry_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Geometry Render Pipeline Layout"),
                bind_group_layouts: &[&geometry_bind_group_layout],
                push_constant_ranges: &[],
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
        Draw3D {
            geometry_pipeline,
            tex: None,
            geometry_sampler,
            geometry_bind_group_layout,
            render_config,
            render_config_buffer,
        }
    }
    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        images: &[Vec<GeometryPixel>],
        image_size: fidget::render::ImageSize,
        mode: Mode3D,
        max_depth: u32,
    ) {
        let (width, height) = (image_size.width(), image_size.height());
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
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
            }
        };

        // Check to see whether we can reuse textures
        match &mut self.tex {
            Some((tex_size, tex_data)) if *tex_size == image_size => {
                tex_data.resize_with(images.len(), new_geometry_tex);
            }
            Some(..) | None => {
                let textures =
                    images.iter().map(|_i| new_geometry_tex()).collect();
                self.tex = Some((image_size, textures));
            }
        }

        // Update render config locally, then copy to the GPU
        self.render_config.render_mode = match mode {
            Mode3D::Heightmap => 0,
            Mode3D::Color => 1,
            Mode3D::Shaded => 2,
        };
        self.render_config.max_depth = max_depth;
        queue.write_buffer(
            &self.render_config_buffer,
            0,
            self.render_config.as_bytes(),
        );

        // Upload all of the images to textures
        for (image_data, tex) in
            images.iter().zip(self.tex.as_ref().unwrap().1.iter())
        {
            // Upload geometry data using AsBytes trait
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                image_data.as_bytes(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16 * width), // 16 bytes per GeometryPixel (4 u32 values)
                    rows_per_image: Some(height),
                },
                texture_size,
            );
        }
    }

    fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        // Only draw if we have a texture
        if let Some((_tex_size, tex_data)) = &self.tex {
            for tex in tex_data {
                render_pass.set_pipeline(&self.geometry_pipeline);
                render_pass.set_bind_group(0, &tex.bind_group, &[]);

                // Draw 2 triangles (6 vertices) to form a quad
                render_pass.draw(0..6, 0..1);
            }
        }
    }
}

struct Draw2DCallback {
    data: Option<(Vec<Vec<[u8; 4]>>, fidget::render::ImageSize)>,
}

impl egui_wgpu::CallbackTrait for Draw2DCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &mut Draw2D = resources.get_mut().unwrap();
        if let Some((image_data, image_size)) = self.data.as_ref() {
            resources.prepare(device, queue, image_data.as_slice(), *image_size)
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &Draw2D = resources.get().unwrap();
        resources.paint(render_pass);
    }
}

struct Draw3DCallback {
    data: Option<(Vec<Vec<GeometryPixel>>, fidget::render::ImageSize)>,
    mode: Mode3D,
    max_depth: u32,
}

impl egui_wgpu::CallbackTrait for Draw3DCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &mut Draw3D = resources.get_mut().unwrap();
        if let Some((image_data, image_size)) = self.data.as_ref() {
            resources.prepare(
                device,
                queue,
                image_data.as_slice(),
                *image_size,
                self.mode,
                self.max_depth,
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
        let resources: &Draw3D = resources.get().unwrap();
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

        // Build the custom render resources for 2D and 3D rendering
        let draw2d = Draw2D::init(&wgpu_state.device, wgpu_state.target_format);
        wgpu_state
            .renderer
            .write()
            .callback_resources
            .insert(draw2d);

        let draw3d = Draw3D::init(&wgpu_state.device, wgpu_state.target_format);
        wgpu_state
            .renderer
            .write()
            .callback_resources
            .insert(draw3d);

        Self {
            image_data: None,
            image_size: fidget::render::ImageSize::from(256),

            config_tx,
            image_rx,

            mode: RenderMode::TwoD {
                canvas: Default::default(),
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
                    for m in [Mode3D::Heightmap, Mode3D::Color, Mode3D::Shaded]
                    {
                        ui.radio_value(&mut mode_3d, Some(m), m.description());
                    }
                    if let Some(m) = mode_3d {
                        changed = self.mode.set_3d_mode(m);
                    }
                    ui.separator();
                    let mut mode_2d = match &self.mode {
                        RenderMode::TwoD { mode, .. } => Some(*mode),
                        RenderMode::ThreeD { .. } => None,
                    };
                    for m in [
                        Mode2D::Debug,
                        Mode2D::Sdf,
                        Mode2D::ExactSdf,
                        Mode2D::Color,
                    ] {
                        ui.radio_value(&mut mode_2d, Some(m), m.description());
                    }

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
            match &mut image_data.images {
                ImageData::Geometry {
                    images,
                    mode,
                    max_depth,
                } => {
                    // Draw the image using WebGPU
                    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                        rect,
                        Draw3DCallback {
                            data: if images.is_empty() {
                                None
                            } else {
                                Some((
                                    std::mem::take(images),
                                    image_data.image_size,
                                ))
                            },
                            mode: *mode,
                            max_depth: *max_depth,
                        },
                    ));
                }
                ImageData::Rgba(images) => {
                    // Draw the image using WebGPU
                    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                        rect,
                        Draw2DCallback {
                            data: if images.is_empty() {
                                None // use pre-existing data
                            } else {
                                // Pass the image buffers into the GPU renderer
                                Some((
                                    std::mem::take(images),
                                    image_data.image_size,
                                ))
                            },
                        },
                    ));
                }
            };

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
            RenderMode::TwoD { canvas, .. } => {
                let image_size = fidget::render::ImageSize::new(
                    rect.width() as u32,
                    rect.height() as u32,
                );

                if let Some(pos) = r.interact_pointer_pos() {
                    let pos =
                        image_size.transform_point(Point2::new(pos.x, pos.y));
                    render_changed |= canvas.drag(pos);
                } else {
                    canvas.end_drag();
                }

                if r.hovered() {
                    let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                    let mouse_pos = r.hover_pos().map(|p| {
                        let p = p - rect.min;
                        image_size.transform_point(Point2::new(p.x, p.y))
                    });
                    render_changed |= canvas.zoom(scroll, mouse_pos);
                }
            }
            RenderMode::ThreeD { canvas, .. } => {
                let image_size = fidget::render::VoxelSize::new(
                    rect.width() as u32,
                    rect.height() as u32,
                    rect.width().max(rect.height()) as u32,
                );

                if let Some(pos) = r.interact_pointer_pos() {
                    let pos_world = image_size
                        .transform_point(Point3::new(pos.x, pos.y, 0.0));
                    let drag_mode =
                        if r.dragged_by(egui::PointerButton::Primary) {
                            Some(DragMode::Pan)
                        } else if r.dragged_by(egui::PointerButton::Secondary) {
                            Some(DragMode::Rotate)
                        } else {
                            None
                        };
                    if let Some(m) = drag_mode {
                        render_changed |= canvas.drag(pos_world, m);
                    }
                } else {
                    canvas.end_drag();
                }

                if r.hovered() {
                    let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                    let mouse_pos =
                        ctx.input(|i| i.pointer.hover_pos()).map(|p| {
                            let p = p - rect.min;
                            image_size
                                .transform_point(Point3::new(p.x, p.y, 0.0))
                        });
                    render_changed |= canvas.zoom(scroll, mouse_pos);
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
