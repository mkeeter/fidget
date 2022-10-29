use eframe::egui;
use fidget::render::config::RenderConfig;
use nalgebra::{Transform2, Vector2};

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
    label_height: Option<f32>,
    texture: Option<egui::TextureHandle>,

    engine: fidget::rhai::Engine,

    scale: f32,
    offset: egui::Vec2,
    drag_start: Option<egui::Vec2>,

    script: String,
    out: Result<fidget::rhai::ScriptContext, String>,
}

////////////////////////////////////////////////////////////////////////////////

impl Default for MyApp {
    fn default() -> Self {
        let engine = fidget::rhai::Engine::new();

        Self {
            texture: None,
            engine,
            script: "draw(circle(0, 0, 0.5))".to_owned(),
            out: Err("".to_string()),
            label_height: None,

            drag_start: None,
            scale: 1.0,
            offset: egui::Vec2::ZERO,
        }
    }
}

impl MyApp {
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

    fn solarized(&mut self, ctx: &egui::Context) {
        let mut theme = egui::Visuals::dark();

        let f = |c: Option<syntect::highlighting::Color>| {
            c.map(|c| egui::Color32::from_rgb(c.r, c.g, c.b)).unwrap()
        };
        let sol = crate::highlight::get_theme().settings;

        theme.extreme_bg_color = f(sol.background);
        theme.widgets.noninteractive.bg_fill = f(sol.gutter);
        theme.widgets.hovered.bg_stroke =
            egui::Stroke::new(1.0, f(sol.selection_border));
        theme.selection.bg_fill = f(sol.selection);
        theme.selection.stroke =
            egui::Stroke::new(1.0, f(sol.selection_border));

        ctx.set_visuals(theme);
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.solarized(ctx);

        egui::SidePanel::left("root")
            .default_width(400.0)
            .frame(
                egui::Frame::none()
                    .fill(ctx.style().visuals.widgets.noninteractive.bg_fill)
                    .inner_margin(egui::style::Margin {
                        top: 7.0,
                        bottom: 5.0,
                        left: 5.0,
                        right: 5.0,
                    }),
            )
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
                    self.out = self
                        .engine
                        .run(&self.script)
                        .map_err(|e| format!("{:?}", e));
                }

                let new_height = if let Err(e) = &self.out {
                    let label = ui.label(e);
                    label.rect.height()
                } else {
                    0.0
                };
                if Some(new_height) != self.label_height {
                    self.label_height = Some(new_height);
                    ctx.request_repaint();
                }
            });

        let rect = ctx.available_rect();
        let size = rect.max - rect.min;
        let max_size = size.x.max(size.y);
        let image_size = (max_size * ctx.pixels_per_point()) as usize;
        let tile_size = 256;
        let image_size = (image_size + tile_size - 1) / tile_size * tile_size;

        let mut image = egui::ImageData::Color(egui::ColorImage::new(
            [image_size; 2],
            egui::Color32::BLACK,
        ));
        let pixels = match &mut image {
            egui::ImageData::Color(c) => &mut c.pixels,
            _ => panic!(),
        };

        // Render shapes into self.texture
        let render_start = std::time::Instant::now();

        if let Ok(script_ctx) = &self.out {
            for s in script_ctx.shapes.iter() {
                let tape = script_ctx
                    .context
                    .get_tape(s.shape, fidget::jit::REGISTER_LIMIT);
                let image = fidget::render::render2d::render::<
                    fidget::jit::JitEvalFamily,
                    fidget::render::render2d::BitRenderMode,
                >(
                    tape,
                    &RenderConfig {
                        image_size,
                        tile_sizes: vec![tile_size, tile_size / 8],
                        threads: 8,

                        mat: Transform2::from_matrix_unchecked(
                            Transform2::identity()
                                .matrix()
                                .append_scaling(self.scale)
                                .append_translation(&Vector2::new(
                                    self.offset.x,
                                    self.offset.y,
                                )),
                        ),
                    },
                );
                for i in 0..pixels.len() {
                    if image[i] {
                        pixels[i] = egui::Color32::from_rgba_unmultiplied(
                            s.color_rgb[0],
                            s.color_rgb[1],
                            s.color_rgb[2],
                            u8::MAX,
                        );
                    }
                }
            }
        }

        match self.texture.as_mut() {
            Some(t) => {
                if t.size() == [image_size; 2] {
                    t.set(image, egui::TextureFilter::Linear)
                } else {
                    *t = ctx.load_texture(
                        "tex",
                        image,
                        egui::TextureFilter::Linear,
                    )
                }
            }
            None => {
                let texture =
                    ctx.load_texture("tex", image, egui::TextureFilter::Linear);
                self.texture = Some(texture);
            }
        }

        let dt = render_start.elapsed();
        ctx.set_visuals(egui::Visuals::dark());
        egui::Window::new("debug").show(ctx, |ui| {
            ui.label(format!("Image size: {0}x{0}", image_size));
            ui.label(format!("Render time: {:.2?}", dt));
        });

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

        let r = egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::BLACK))
            .show(ctx, |ui| {
                let pos = ui.next_widget_position();
                let size = ui.available_size();
                let painter = ui.painter_at(egui::Rect {
                    min: pos,
                    max: pos + size,
                });
                if let Some(t) = self.texture.as_ref() {
                    let mut mesh = egui::Mesh::with_texture(t.id());
                    mesh.add_rect_with_uv(rect, uv, egui::Color32::WHITE);
                    painter.add(mesh);
                }
                // Return events from the canvas in the inner response
                ui.interact(
                    rect,
                    egui::Id::new("canvas"),
                    egui::Sense::click_and_drag(),
                )
            });

        // Handle pan and zoom
        if let Some(pos) = r.inner.interact_pointer_pos() {
            if let Some(start) = self.drag_start {
                self.offset = egui::Vec2::ZERO;
                let pos = self.mouse_to_uv(rect, uv, pos);
                self.offset = start - pos;
            } else {
                let pos = self.mouse_to_uv(rect, uv, pos);
                self.drag_start = Some(pos);
            }
        } else {
            self.drag_start = None;
        }

        if r.inner.hovered() {
            let mouse_pos = ctx.input().pointer.hover_pos();
            let pos_before = mouse_pos.map(|p| self.mouse_to_uv(rect, uv, p));
            self.scale /= (ctx.input().scroll_delta.y / 100.0).exp2();
            if let Some(pos_before) = pos_before {
                let pos_after = self.mouse_to_uv(rect, uv, mouse_pos.unwrap());
                self.offset += pos_before - pos_after;
            }
        }
    }
}
