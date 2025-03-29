use anyhow::Result;
use clap::Parser;
use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::{
    egui,
    egui_wgpu::{self, wgpu, wgpu::util::DeviceExt},
};
use env_logger::Env;
use log::{debug, error, info, warn};
use nalgebra::{Point2, Point3};
use notify::Watcher;
use zerocopy::IntoBytes;

use fidget::render::{
    ImageRenderConfig, RotateHandle, TranslateHandle, View2, View3,
    VoxelRenderConfig,
};

use std::{error::Error, num::NonZeroU64, path::Path};

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

struct RenderResult {
    dt: std::time::Duration,
    image: egui::ImageData,
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
    let mut config = None;
    let mut script_ctx = None;
    let mut changed = false;
    loop {
        let timeout_ms = if changed { 10 } else { 10_000 };
        let timeout = std::time::Duration::from_millis(timeout_ms);
        crossbeam_channel::select! {
            recv(rx) -> msg => match msg? {
                Ok(s) => {
                    debug!("render thread got a new result");
                    script_ctx = Some(s);
                    changed = true;
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
                debug!("render config got a new thread");
                config = Some(msg?);
                changed = true;
                continue;
            },
            default(timeout) => debug!("render thread timed out"),
        }

        if !changed {
            continue;
        }

        if let (Some(out), Some(render_config)) = (&script_ctx, &config) {
            debug!("Rendering...");
            let mut image = egui::ColorImage::new(
                [
                    render_config.image_size.width() as usize,
                    render_config.image_size.height() as usize,
                ],
                egui::Color32::BLACK,
            );
            let render_start = std::time::Instant::now();
            for s in out.shapes.iter() {
                let tape = fidget::shape::Shape::<F>::from(s.tree.clone());
                render(
                    &render_config.mode,
                    tape,
                    render_config.image_size,
                    s.color_rgb,
                    &mut image.pixels,
                );
            }
            let dt = render_start.elapsed();
            let image = egui::ImageData::Color(std::sync::Arc::new(image));
            tx.send(Ok(RenderResult {
                image,
                dt,
                image_size: render_config.image_size,
            }))?;
            changed = false;
            wake.send(()).unwrap();
        }
    }
}

fn render<F: fidget::eval::Function + fidget::render::RenderHints>(
    mode: &RenderMode,
    shape: fidget::shape::Shape<F>,
    image_size: fidget::render::ImageSize,
    color: [u8; 3],
    pixels: &mut [egui::Color32],
) {
    match mode {
        RenderMode::TwoD { view, mode, .. } => {
            let config = ImageRenderConfig {
                image_size,
                tile_sizes: F::tile_sizes_2d(),
                view: *view,
                ..Default::default()
            };

            match mode {
                Mode2D::Color => {
                    let image = config
                        .run::<_, fidget::render::BitRenderMode>(shape)
                        .unwrap();
                    let c = egui::Color32::from_rgba_unmultiplied(
                        color[0],
                        color[1],
                        color[2],
                        u8::MAX,
                    );
                    for (p, &i) in pixels.iter_mut().zip(&image) {
                        if i {
                            *p = c;
                        }
                    }
                }

                Mode2D::Sdf => {
                    let image = config
                        .run::<_, fidget::render::SdfRenderMode>(shape)
                        .unwrap();
                    for (p, i) in pixels.iter_mut().zip(&image) {
                        *p = egui::Color32::from_rgb(i[0], i[1], i[2]);
                    }
                }

                Mode2D::ExactSdf => {
                    let image = config
                        .run::<_, fidget::render::SdfPixelRenderMode>(shape)
                        .unwrap();
                    for (p, i) in pixels.iter_mut().zip(&image) {
                        *p = egui::Color32::from_rgb(i[0], i[1], i[2]);
                    }
                }

                Mode2D::Debug => {
                    let image = config
                        .run::<_, fidget::render::DebugRenderMode>(shape)
                        .unwrap();
                    for (p, i) in pixels.iter_mut().zip(&image) {
                        let c = i.as_debug_color();
                        *p = egui::Color32::from_rgb(c[0], c[1], c[2]);
                    }
                }
            }
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
            let image = config.run(shape).unwrap();
            match mode {
                Mode3D::Color => {
                    for (p, &i) in pixels.iter_mut().zip(image.iter()) {
                        if i.depth != 0 {
                            let c = i.to_color();
                            *p = egui::Color32::from_rgb(c[0], c[1], c[2]);
                        }
                    }
                }

                Mode3D::Heightmap => {
                    let max_depth =
                        image.iter().map(|p| p.depth).max().unwrap_or(1).max(1);
                    for (p, &d) in pixels.iter_mut().zip(&image) {
                        if d.depth != 0 {
                            let b = (d.depth * 255 / max_depth) as u8;
                            *p = egui::Color32::from_rgb(b, b, b);
                        }
                    }
                }
            }
        }
    };
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

struct CustomResources {
    render_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
}

impl CustomResources {
    fn prepare(&self, _device: &wgpu::Device, queue: &wgpu::Queue) {
        // Update our uniform buffer with the angle from the UI
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            [0.0f32, 0.0, 0.0, 0.0].as_bytes(),
        );
    }

    fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        // Draw our triangle!
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

struct CustomCallback {}

impl egui_wgpu::CallbackTrait for CustomCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &CustomResources = resources.get().unwrap();
        resources.prepare(device, queue);
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
    // Current image
    texture: Option<egui::TextureHandle>,
    stats: Option<(std::time::Duration, fidget::render::ImageSize)>,

    // Most recent result, or an error string
    // TODO: this could be combined with stats as a Result
    err: Option<String>,

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

        // Create shader module
        let shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Custom Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/image.wgsl").into(),
                ),
            });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("custom3d"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(16),
                    },
                    count: None,
                }],
            });

        // Create render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                cache: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::REPLACE,
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
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

        let uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("custom3d"),
                contents: [0.0_f32; 4].as_bytes(), // 16 bytes aligned!
                // Mapping at creation (as done by the create_buffer_init utility) doesn't require us to to add the MAP_WRITE usage
                // (this *happens* to workaround this bug )
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("custom3d"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        wgpu_state.renderer.write().callback_resources.insert(
            CustomResources {
                render_pipeline,
                bind_group,
                uniform_buffer,
            },
        );

        Self {
            texture: None,
            stats: None,

            err: None,
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
    fn try_recv_image(&mut self, ctx: &egui::Context) {
        if let Ok(r) = self.image_rx.try_recv() {
            match r {
                Ok(r) => {
                    match self.texture.as_mut() {
                        Some(t) => {
                            if t.size() == r.image.size() {
                                t.set(r.image, egui::TextureOptions::LINEAR)
                            } else {
                                *t = ctx.load_texture(
                                    "tex",
                                    r.image,
                                    egui::TextureOptions::LINEAR,
                                )
                            }
                        }
                        None => {
                            let texture = ctx.load_texture(
                                "tex",
                                r.image,
                                egui::TextureOptions::LINEAR,
                            );
                            self.texture = Some(texture);
                        }
                    }
                    self.stats = Some((r.dt, r.image_size));
                    self.err = None;
                }
                Err(e) => {
                    self.err = Some(e);
                    self.stats = None;
                }
            }
        }
    }

    fn paint_image(&self, ui: &mut egui::Ui) -> egui::Response {
        let pos = ui.next_widget_position();
        let size = ui.available_size();
        let painter = ui.painter_at(egui::Rect {
            min: pos,
            max: pos + size,
        });
        const PADDING: egui::Vec2 = egui::Vec2 { x: 10.0, y: 10.0 };

        let rect = ui.ctx().available_rect();
        let uv = egui::Rect {
            min: egui::Pos2::new(0.0, 0.0),
            max: egui::Pos2::new(1.0, 1.0),
        };

        // Draw the custom image below everything else (??)
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            CustomCallback {},
        ));

        if let Some((dt, image_size)) = self.stats {
            // Only draw the image if we have valid stats (i.e. no error)
            /*
            if let Some(t) = self.texture.as_ref() {
                let mut mesh = egui::Mesh::with_texture(t.id());

                mesh.add_rect_with_uv(rect, uv, egui::Color32::WHITE);
                painter.add(mesh);
            }
            */

            let layout = painter.layout(
                format!(
                    "Image size: {}×{}\nRender time: {dt:.2?}",
                    image_size.width(),
                    image_size.height(),
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

        if let Some(err) = &self.err {
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
        self.try_recv_image(ctx);

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
