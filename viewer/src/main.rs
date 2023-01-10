use anyhow::Result;
use clap::Parser;
use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;
use env_logger::Env;
use fidget::{eval::Family, render::RenderConfig};
use log::{debug, error, info};
use nalgebra::{Transform2, Transform3, Vector2, Vector3};
use notify::Watcher;

use std::path::Path;

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
    let read_file = || String::from_utf8(std::fs::read(path).unwrap()).unwrap();
    let mut contents = read_file();
    tx.send(contents.clone())?;

    loop {
        // Wait for a file change notification
        rx.recv()?;
        let new_contents = read_file();
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
    image_size: usize,
    mode: RenderMode,
}

struct RenderResult {
    dt: std::time::Duration,
    image: egui::ImageData,
    image_size: usize,
}

fn render_thread(
    cfg: Receiver<RenderSettings>,
    rx: Receiver<Result<fidget::rhai::ScriptContext, String>>,
    tx: Sender<Result<RenderResult, String>>,
    wake: Sender<()>,
) -> Result<()> {
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
            let mut image = egui::ImageData::Color(egui::ColorImage::new(
                [render_config.image_size; 2],
                egui::Color32::BLACK,
            ));
            let pixels = match &mut image {
                egui::ImageData::Color(c) => &mut c.pixels,
                _ => panic!(),
            };
            let render_start = std::time::Instant::now();
            for s in out.shapes.iter() {
                let tape: fidget::eval::Tape<fidget::jit::Eval> =
                    out.context.get_tape(s.shape).unwrap();
                render(
                    &render_config.mode,
                    tape,
                    render_config.image_size,
                    s.color_rgb,
                    pixels,
                );
            }
            let dt = render_start.elapsed();
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

fn render(
    mode: &RenderMode,
    tape: fidget::eval::Tape<fidget::jit::Eval>,
    image_size: usize,
    color: [u8; 3],
    pixels: &mut [egui::Color32],
) {
    match mode {
        RenderMode::TwoD(camera, mode) => {
            let mat = Transform2::from_matrix_unchecked(
                Transform2::identity()
                    .matrix()
                    .append_scaling(camera.scale)
                    .append_translation(&Vector2::new(
                        camera.offset.x,
                        camera.offset.y,
                    )),
            );

            let config = RenderConfig {
                image_size,
                tile_sizes: fidget::jit::Eval::tile_sizes_2d().to_vec(),
                threads: 8,

                mat,
            };
            match mode {
                TwoDMode::Color => {
                    let image = fidget::render::render2d(
                        tape,
                        &config,
                        &fidget::render::BitRenderMode,
                    );
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

                TwoDMode::Sdf => {
                    let image = fidget::render::render2d(
                        tape,
                        &config,
                        &fidget::render::SdfRenderMode,
                    );
                    for (p, i) in pixels.iter_mut().zip(&image) {
                        *p = egui::Color32::from_rgb(i[0], i[1], i[2]);
                    }
                }

                TwoDMode::Debug => {
                    let image = fidget::render::render2d(
                        tape,
                        &config,
                        &fidget::render::DebugRenderMode,
                    );
                    for (p, i) in pixels.iter_mut().zip(&image) {
                        let c = i.as_debug_color();
                        *p = egui::Color32::from_rgb(c[0], c[1], c[2]);
                    }
                }
            }
        }
        RenderMode::ThreeD(camera, mode) => {
            let mat = Transform3::from_matrix_unchecked(
                Transform3::identity()
                    .matrix()
                    .append_scaling(camera.scale)
                    .append_translation(&Vector3::new(
                        camera.offset.x,
                        camera.offset.y,
                        0.0,
                    )),
            );

            let config = RenderConfig {
                image_size,
                tile_sizes: fidget::jit::Eval::tile_sizes_2d().to_vec(),
                threads: 8,

                mat,
            };
            let (depth, color) = fidget::render::render3d(tape, &config);
            match mode {
                ThreeDMode::Color => {
                    for (p, (&d, &c)) in
                        pixels.iter_mut().zip(depth.iter().zip(&color))
                    {
                        if d != 0 {
                            *p = egui::Color32::from_rgb(c[0], c[1], c[2]);
                        }
                    }
                }

                ThreeDMode::Heightmap => {
                    let max_depth =
                        depth.iter().max().cloned().unwrap_or(1).max(1);
                    for (p, &d) in pixels.iter_mut().zip(&depth) {
                        if d != 0 {
                            let b = (d * 255 / max_depth) as u8;
                            *p = egui::Color32::from_rgb(b, b, b);
                        }
                    }
                }
            }
        }
    };
}

fn main() -> Result<()> {
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
        let _ = render_thread(config_rx, rhai_result_rx, render_tx, wake_tx);
        info!("render thread is done");
    });

    // Automatically select the best implementation for your platform.
    let mut watcher = notify::recommended_watcher(move |res| match res {
        Ok(event) => {
            println!("{event:?}");
            file_watcher_tx.send(()).unwrap();
        }
        Err(e) => panic!("watch error: {:?}", e),
    })
    .unwrap();
    watcher
        .watch(Path::new(&args.target), notify::RecursiveMode::NonRecursive)
        .unwrap();

    let options = eframe::NativeOptions::default();
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

            Box::new(ViewerApp::new(config_tx, render_rx))
        }),
    );

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
struct TwoDCamera {
    // 2D camera parameters
    scale: f32,
    offset: egui::Vec2,
    drag_start: Option<egui::Vec2>,
}

impl TwoDCamera {
    /// Converts from mouse position to a UV position within the render window
    fn mouse_to_uv(
        &self,
        rect: egui::Rect,
        uv: egui::Rect,
        p: egui::Pos2,
    ) -> egui::Vec2 {
        let r = (p - rect.min) / (rect.max - rect.min);
        const ONE: egui::Vec2 = egui::Vec2::new(1.0, 1.0);
        let pos = uv.min.to_vec2() * (ONE - r) + uv.max.to_vec2() * r;
        let out = ((pos * 2.0) - ONE) * self.scale;
        egui::Vec2::new(out.x, -out.y) + self.offset
    }
}

impl Default for TwoDCamera {
    fn default() -> Self {
        TwoDCamera {
            drag_start: None,
            scale: 1.0,
            offset: egui::Vec2::ZERO,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum TwoDMode {
    Color,
    Sdf,
    Debug,
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
struct ThreeDCamera {
    // 2D camera parameters
    scale: f32,
    offset: nalgebra::Vector3<f32>,
    drag_start: Option<egui::Vec2>,
}

impl ThreeDCamera {
    fn mouse_to_uv(
        &self,
        rect: egui::Rect,
        uv: egui::Rect,
        p: egui::Pos2,
    ) -> egui::Vec2 {
        panic!()
    }
}

impl Default for ThreeDCamera {
    fn default() -> Self {
        ThreeDCamera {
            drag_start: None,
            scale: 1.0,
            offset: nalgebra::Vector3::zeros(),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum ThreeDMode {
    Color,
    Heightmap,
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
enum RenderMode {
    TwoD(TwoDCamera, TwoDMode),
    ThreeD(ThreeDCamera, ThreeDMode),
}

impl RenderMode {
    fn set_2d_mode(&mut self, mode: TwoDMode) -> bool {
        match self {
            RenderMode::TwoD(.., m) => {
                let changed = *m != mode;
                *m = mode;
                changed
            }
            RenderMode::ThreeD(..) => {
                *self = RenderMode::TwoD(TwoDCamera::default(), mode);
                true
            }
        }
    }
    fn set_3d_mode(&mut self, mode: ThreeDMode) -> bool {
        match self {
            RenderMode::TwoD(..) => {
                *self = RenderMode::ThreeD(ThreeDCamera::default(), mode);
                true
            }
            RenderMode::ThreeD(_camera, m) => {
                let changed = *m != mode;
                *m = mode;
                changed
            }
        }
    }
}

struct ViewerApp {
    // Current image
    texture: Option<egui::TextureHandle>,
    stats: Option<(std::time::Duration, usize)>,

    // Most recent result, or an error string
    // TODO: this could be combined with stats as a Result
    err: Option<String>,

    /// Current render mode
    mode: RenderMode,
    image_size: usize,

    config_tx: Sender<RenderSettings>,
    image_rx: Receiver<Result<RenderResult, String>>,
}

////////////////////////////////////////////////////////////////////////////////

impl ViewerApp {
    fn new(
        config_tx: Sender<RenderSettings>,
        image_rx: Receiver<Result<RenderResult, String>>,
    ) -> Self {
        Self {
            texture: None,
            stats: None,

            err: None,
            image_size: 0,

            config_tx,
            image_rx,

            mode: RenderMode::TwoD(TwoDCamera::default(), TwoDMode::Color),
        }
    }

    fn draw_menu(&mut self, ctx: &egui::Context) -> bool {
        let mut changed = false;
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("Config", |ui| {
                    let mut mode_3d = match &self.mode {
                        RenderMode::TwoD(..) => None,
                        RenderMode::ThreeD(_camera, mode) => Some(*mode),
                    };
                    ui.radio_value(
                        &mut mode_3d,
                        Some(ThreeDMode::Heightmap),
                        "3D heightmap",
                    );
                    ui.radio_value(
                        &mut mode_3d,
                        Some(ThreeDMode::Color),
                        "3D color",
                    );
                    if let Some(m) = mode_3d {
                        changed = self.mode.set_3d_mode(m);
                    }
                    ui.separator();
                    let mut mode_2d = match &self.mode {
                        RenderMode::TwoD(_camera, mode) => Some(*mode),
                        RenderMode::ThreeD(..) => None,
                    };
                    ui.radio_value(
                        &mut mode_2d,
                        Some(TwoDMode::Debug),
                        "2D debug",
                    );
                    ui.radio_value(&mut mode_2d, Some(TwoDMode::Sdf), "2D SDF");
                    ui.radio_value(
                        &mut mode_2d,
                        Some(TwoDMode::Color),
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
                                t.set(r.image, egui::TextureFilter::Linear)
                            } else {
                                *t = ctx.load_texture(
                                    "tex",
                                    r.image,
                                    egui::TextureFilter::Linear,
                                )
                            }
                        }
                        None => {
                            let texture = ctx.load_texture(
                                "tex",
                                r.image,
                                egui::TextureFilter::Linear,
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

    fn paint_image(
        &self,
        rect: egui::Rect,
        uv: egui::Rect,
        ui: &mut egui::Ui,
    ) -> egui::Response {
        let pos = ui.next_widget_position();
        let size = ui.available_size();
        let painter = ui.painter_at(egui::Rect {
            min: pos,
            max: pos + size,
        });
        const PADDING: egui::Vec2 = egui::Vec2 { x: 10.0, y: 10.0 };

        if let Some((dt, image_size)) = self.stats {
            // Only draw the image if we have valid stats (i.e. no error)
            if let Some(t) = self.texture.as_ref() {
                let mut mesh = egui::Mesh::with_texture(t.id());
                mesh.add_rect_with_uv(rect, uv, egui::Color32::WHITE);
                painter.add(mesh);
            }

            let layout = painter.layout(
                format!(
                    "Image size: {0}x{0}\nRender time: {dt:.2?}",
                    image_size,
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
                egui::Rounding::none(),
                egui::Color32::from_black_alpha(128),
            );
            painter.galley(text_corner - PADDING, layout);
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
                egui::Rounding::none(),
                egui::Color32::from_black_alpha(128),
            );
            painter.galley(rect.min + PADDING, layout);
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
        let max_size = size.x.max(size.y);
        let image_size = (max_size * ctx.pixels_per_point()) as usize;

        if image_size != self.image_size {
            self.image_size = image_size;
            render_changed = true;
        }

        let uv = if size.x > size.y {
            let r = (1.0 - (size.y / size.x)) / 2.0;
            egui::Rect {
                min: egui::Pos2::new(0.0, r),
                max: egui::Pos2::new(1.0, 1.0 - r),
            }
        } else {
            let r = (1.0 - (size.x / size.y)) / 2.0;
            egui::Rect {
                min: egui::Pos2::new(r, 0.0),
                max: egui::Pos2::new(1.0 - r, 1.0),
            }
        };

        // Draw the current image and/or error
        let r = egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::BLACK))
            .show(ctx, |ui| self.paint_image(rect, uv, ui))
            .inner;

        // Handle pan and zoom
        match &mut self.mode {
            RenderMode::TwoD(camera, ..) => {
                if let Some(pos) = r.interact_pointer_pos() {
                    if let Some(start) = camera.drag_start {
                        camera.offset = egui::Vec2::ZERO;
                        let pos = camera.mouse_to_uv(rect, uv, pos);
                        camera.offset = start - pos;
                        render_changed = true;
                    } else {
                        let pos = camera.mouse_to_uv(rect, uv, pos);
                        camera.drag_start = Some(pos);
                    }
                } else {
                    camera.drag_start = None;
                }

                if r.hovered() {
                    let scroll = ctx.input().scroll_delta.y;
                    if scroll != 0.0 {
                        let mouse_pos = ctx.input().pointer.hover_pos();
                        let pos_before =
                            mouse_pos.map(|p| camera.mouse_to_uv(rect, uv, p));
                        render_changed = true;
                        camera.scale /= (scroll / 100.0).exp2();
                        if let Some(pos_before) = pos_before {
                            let pos_after = camera.mouse_to_uv(
                                rect,
                                uv,
                                mouse_pos.unwrap(),
                            );
                            camera.offset += pos_before - pos_after;
                        }
                    }
                }
            }
            RenderMode::ThreeD(camera, ..) => {
                unimplemented!()
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
