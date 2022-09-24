#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui;
use fidget::render::RenderConfig;

mod highlight;

fn main() {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Fidget",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    );
}

struct MyApp {
    first_run: bool,
    label_height: Option<f32>,
    textures: Vec<egui::TextureHandle>,

    engine: fidget::bind::Engine,

    script: String,
    out: String,
}

////////////////////////////////////////////////////////////////////////////////

impl Default for MyApp {
    fn default() -> Self {
        let engine = fidget::bind::Engine::new();

        Self {
            first_run: true,
            textures: vec![],
            engine,
            script: "// hello, world".to_owned(),
            out: "".to_string(),
            label_height: None,
        }
    }
}

impl MyApp {
    fn init(&mut self, ctx: &egui::Context) {
        let mut theme = egui::Visuals::dark();

        let f = |c: syntect::highlighting::Color| {
            egui::Color32::from_rgb(c.r, c.g, c.b)
        };
        let sol = crate::highlight::get_theme().settings;

        theme.extreme_bg_color = sol.background.map(f).unwrap();
        theme.widgets.noninteractive.bg_fill = sol.gutter.map(f).unwrap();
        theme.selection.bg_fill = sol.selection.map(f).unwrap();

        ctx.set_visuals(theme);
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.first_run {
            self.init(ctx);
            self.first_run = false;
        }

        egui::SidePanel::left("root")
            .default_width(400.0)
            .show(ctx, |ui| {
                let mut size = ui.available_size();
                if let Some(h) = self.label_height {
                    const PAD: f32 = 8.0;
                    let style = ui.style_mut();
                    style.spacing.item_spacing.y += PAD;
                    size.y -= h + 2.0 * PAD;
                }
                let r =
                    crate::highlight::code_view_ui(ui, &mut self.script, size);
                if r.changed() || self.label_height.is_none() {
                    let v = self.engine.run(&self.script);

                    self.out = format!("{:?}", v);
                }

                let label = ui.label(&self.out);
                let new_height = label.rect.height();
                if Some(new_height) != self.label_height {
                    self.label_height = Some(new_height);
                    ctx.request_repaint();
                }
            });

        {
            let script_ctx = self.engine.script_context();
            for (i, s) in script_ctx.shapes.iter().enumerate() {
                let tape = script_ctx
                    .context
                    .get_tape(*s, fidget::asm::dynasm::REGISTER_LIMIT);
                let image = fidget::render::render::<
                    fidget::asm::dynasm::JitEvalFamily,
                >(
                    tape,
                    &RenderConfig {
                        image_size: 512,
                        tile_size: 64,
                        subtile_size: 8,
                        threads: 8,
                        interval_subdiv: 3,
                    },
                );
                let pixels = image
                    .into_iter()
                    .map(|p| {
                        let [r, g, b, a] = p.as_color();
                        egui::Color32::from_rgba_unmultiplied(r, g, b, a)
                    })
                    .collect::<Vec<_>>();
                let image = egui::ImageData::Color(egui::ColorImage {
                    size: [512; 2],
                    pixels,
                });

                match self.textures.get_mut(i) {
                    Some(t) => t.set(image, egui::TextureFilter::Linear),
                    None => {
                        let texture = ctx.load_texture(
                            "tex",
                            image,
                            egui::TextureFilter::Linear,
                        );
                        self.textures.push(texture);
                    }
                }
            }
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::BLACK))
            .show(ctx, |ui| {
                let pos = ui.next_widget_position();
                let size = ui.available_size();
                let painter = ui.painter_at(egui::Rect {
                    min: pos,
                    max: pos + size,
                });
                for t in &self.textures {
                    let mut mesh = egui::Mesh::with_texture(t.id());
                    mesh.add_rect_with_uv(
                        egui::Rect {
                            min: pos,
                            max: pos + size,
                        },
                        egui::Rect {
                            min: egui::Pos2::new(0.0, 0.0),
                            max: egui::Pos2::new(1.0, 1.0),
                        },
                        egui::Color32::WHITE,
                    );
                    painter.add(mesh);
                }
            });
    }
}
