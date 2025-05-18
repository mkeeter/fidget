use anyhow::Result;
use clap::Parser;
use crossbeam_channel::{Receiver, Sender, unbounded};
use eframe::{
    egui,
    egui_wgpu::{self, wgpu},
};
use env_logger::Env;
use log::{debug, error, info};
use nalgebra::Point2;
use notify::{Event, EventKind, Watcher};

use fidget::{
    gui::{Canvas2, Canvas3, CursorState, DragMode},
    render::{
        GeometryPixel, ImageRenderConfig, View2, View3, VoxelRenderConfig,
    },
};

use std::{error::Error, path::Path};

mod draw2d;
mod draw3d;
mod script;
mod watcher;

/// Minimal viewer, using Fidget to render a Rhai script
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// File to watch
    target: String,
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
        let _ = watcher::file_watcher_thread(
            &path,
            file_watcher_rx,
            rhai_script_tx,
        );
        info!("file watcher thread is done");
    });
    std::thread::spawn(move || {
        let _ = script::rhai_script_thread(rhai_script_rx, rhai_result_tx);
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
    let mut watcher = notify::recommended_watcher(
        move |res: Result<Event, notify::Error>| match res {
            Ok(event) => {
                info!("file watcher: {event:?}");
                if let EventKind::Modify(..) = event.kind {
                    file_watcher_tx.send(()).unwrap()
                }
            }
            Err(e) => panic!("watch error: {:?}", e),
        },
    )
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

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
enum RenderMode {
    TwoD { canvas: Canvas2, mode: Mode2D },
    ThreeD { canvas: Canvas3, mode: Mode3D },
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
                    canvas: Canvas2::new(fidget::render::ImageSize::new(0, 0)),
                    mode: new_mode,
                };
                true
            }
        }
    }
    fn set_3d_mode(&mut self, new_mode: Mode3D) -> bool {
        match self {
            RenderMode::TwoD { .. } => {
                // TODO get parameters from 2D camera here?
                *self = RenderMode::ThreeD {
                    canvas: Canvas3::new(fidget::render::VoxelSize::new(
                        0, 0, 0,
                    )),
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

struct CustomTexture {
    bind_group: wgpu::BindGroup,
    texture: wgpu::Texture,
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

        // Install custom render resources for 2D and 3D rendering
        draw2d::Draw2D::init(wgpu_state);
        draw3d::Draw3D::init(wgpu_state);

        // Pick a dummy image size; we'll fix it later
        let image_size = fidget::render::ImageSize::from(256);
        Self {
            image_data: None,
            image_size,

            config_tx,
            image_rx,

            mode: RenderMode::TwoD {
                canvas: Canvas2::new(image_size),
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
        let painter = ui.painter();
        const PADDING: egui::Vec2 = egui::Vec2 { x: 10.0, y: 10.0 };

        let rect = painter.clip_rect();

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
                        draw3d::Draw3D::new(
                            if images.is_empty() {
                                None
                            } else {
                                Some((
                                    std::mem::take(images),
                                    image_data.image_size,
                                ))
                            },
                            *mode,
                            *max_depth,
                        ),
                    ));
                }
                ImageData::Rgba(images) => {
                    // Draw the image using WebGPU
                    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                        rect,
                        draw2d::Draw2D::new(if images.is_empty() {
                            // use pre-existing data
                            None
                        } else {
                            // Pass the image buffers into the GPU renderer
                            Some((
                                std::mem::take(images),
                                image_data.image_size,
                            ))
                        }),
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
        let size = rect.size();
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

                let cursor_state =
                    match (r.interact_pointer_pos(), r.hover_pos()) {
                        (Some(p), _) => Some((p, true)),
                        (_, Some(p)) => Some((p, false)),
                        (None, None) => None,
                    }
                    .map(|(p, drag)| {
                        let p = p - rect.min;
                        CursorState {
                            screen_pos: Point2::new(
                                p.x.round() as i32,
                                p.y.round() as i32,
                            ),
                            drag,
                        }
                    });
                render_changed |= canvas.interact(
                    image_size,
                    cursor_state,
                    ctx.input(|i| i.smooth_scroll_delta.y),
                );
            }
            RenderMode::ThreeD { canvas, .. } => {
                let image_size = fidget::render::VoxelSize::new(
                    rect.width() as u32,
                    rect.height() as u32,
                    rect.width().max(rect.height()) as u32,
                );

                let cursor_state =
                    match (r.interact_pointer_pos(), r.hover_pos()) {
                        (Some(p), _) => {
                            let drag =
                                if r.dragged_by(egui::PointerButton::Primary) {
                                    Some(DragMode::Pan)
                                } else if r
                                    .dragged_by(egui::PointerButton::Secondary)
                                {
                                    Some(DragMode::Rotate)
                                } else {
                                    None
                                };

                            Some((p, drag))
                        }
                        (_, Some(p)) => Some((p, None)),
                        (None, None) => None,
                    }
                    .map(|(p, drag)| {
                        let p = p - rect.min;
                        CursorState {
                            screen_pos: Point2::new(
                                p.x.round() as i32,
                                p.y.round() as i32,
                            ),
                            drag,
                        }
                    });
                render_changed |= canvas.interact(
                    image_size,
                    cursor_state,
                    ctx.input(|i| i.smooth_scroll_delta.y),
                );
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
