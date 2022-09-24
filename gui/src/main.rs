#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::sync::{Arc, Mutex};

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

    engine: rhai::Engine,
    script_ctx: Arc<Mutex<ScriptContext>>,

    script: String,
    out: String,
}

////////////////////////////////////////////////////////////////////////////////

/// Extension trait to pull a Fidget `Context` out of a `NativeCallContext`
trait FidgetContext {
    fn with_fidget_context<F, V>(&self, f: F) -> V
    where
        F: Fn(&mut fidget::context::Context) -> V;
}

impl FidgetContext for rhai::NativeCallContext<'_> {
    fn with_fidget_context<F, V>(&self, f: F) -> V
    where
        F: Fn(&mut fidget::context::Context) -> V,
    {
        let ctx = self
            .tag()
            .unwrap()
            .clone_cast::<Arc<Mutex<ScriptContext>>>();
        let lock = &mut ctx.lock().unwrap().fidget_context;
        f(lock)
    }
}

struct ScriptContext {
    fidget_context: fidget::context::Context,
    shapes: Vec<fidget::context::Node>,
}

impl ScriptContext {
    fn new() -> Self {
        Self {
            fidget_context: fidget::context::Context::new(),
            shapes: vec![],
        }
    }
    fn clear(&mut self) {
        self.fidget_context.clear();
        self.shapes.clear();
    }
}

////////////////////////////////////////////////////////////////////////////////

fn var_x(ctx: rhai::NativeCallContext) -> fidget::context::Node {
    ctx.with_fidget_context(|c| c.x())
}
fn var_y(ctx: rhai::NativeCallContext) -> fidget::context::Node {
    ctx.with_fidget_context(|c| c.y())
}

fn draw(ctx: rhai::NativeCallContext, node: fidget::context::Node) {
    let ctx = ctx.tag().unwrap().clone_cast::<Arc<Mutex<ScriptContext>>>();
    ctx.lock().unwrap().shapes.push(node);
}

macro_rules! define_binary_fns {
    ($name:ident) => {
        mod $name {
            use super::*;
            use fidget::context::Node;
            use rhai::NativeCallContext;
            pub fn node_node(ctx: NativeCallContext, a: Node, b: Node) -> Node {
                ctx.with_fidget_context(|c| c.$name(a, b).unwrap())
            }
            pub fn node_float(ctx: NativeCallContext, a: Node, b: f64) -> Node {
                ctx.with_fidget_context(|c| {
                    let b = c.constant(b);
                    c.$name(a, b).unwrap()
                })
            }
            pub fn float_node(ctx: NativeCallContext, a: f64, b: Node) -> Node {
                ctx.with_fidget_context(|c| {
                    let a = c.constant(a);
                    c.$name(a, b).unwrap()
                })
            }
            pub fn node_int(ctx: NativeCallContext, a: Node, b: i64) -> Node {
                ctx.with_fidget_context(|c| {
                    let b = c.constant(b as f64);
                    c.$name(a, b).unwrap()
                })
            }
            pub fn int_node(ctx: NativeCallContext, a: i64, b: Node) -> Node {
                ctx.with_fidget_context(|c| {
                    let a = c.constant(a as f64);
                    c.$name(a, b).unwrap()
                })
            }
        }
    };
}

macro_rules! define_unary_fns {
    ($name:ident) => {
        mod $name {
            use super::*;
            use fidget::context::Node;
            use rhai::NativeCallContext;
            pub fn node(ctx: NativeCallContext, a: Node) -> Node {
                ctx.with_fidget_context(|c| c.$name(a).unwrap())
            }
        }
    };
}

define_binary_fns!(add);
define_binary_fns!(sub);
define_binary_fns!(mul);
define_binary_fns!(min);
define_binary_fns!(max);
define_unary_fns!(sqrt);
define_unary_fns!(square);
define_unary_fns!(neg);

macro_rules! register_binary_fns {
    ($op:literal, $name:ident, $engine:ident) => {
        $engine.register_fn($op, $name::node_node);
        $engine.register_fn($op, $name::node_float);
        $engine.register_fn($op, $name::float_node);
        $engine.register_fn($op, $name::node_int);
        $engine.register_fn($op, $name::int_node);
    };
}
macro_rules! register_unary_fns {
    ($op:literal, $name:ident, $engine:ident) => {
        $engine.register_fn($op, $name::node);
    };
}

impl Default for MyApp {
    fn default() -> Self {
        let mut engine = rhai::Engine::new();
        engine.register_type_with_name::<fidget::context::Node>("Node");
        engine.register_fn("__var_x", var_x);
        engine.register_fn("__var_y", var_y);
        engine.register_fn("__draw", draw);

        register_binary_fns!("+", add, engine);
        register_binary_fns!("-", sub, engine);
        register_binary_fns!("*", mul, engine);
        register_binary_fns!("min", min, engine);
        register_binary_fns!("max", max, engine);
        register_unary_fns!("sqrt", sqrt, engine);
        register_unary_fns!("square", square, engine);
        register_unary_fns!("-", neg, engine);

        engine.set_fast_operators(false);

        let script_ctx = Arc::new(Mutex::new(ScriptContext::new()));
        engine.set_default_tag(rhai::Dynamic::from(script_ctx.clone()));

        let ast = engine
            .compile(include_str!("../scripts/core.rhai"))
            .unwrap();
        let module =
            rhai::Module::eval_ast_as_new(rhai::Scope::new(), &ast, &engine)
                .unwrap();
        engine.register_global_module(rhai::Shared::new(module));

        Self {
            first_run: true,
            textures: vec![],
            script_ctx,
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
                    self.script_ctx.lock().unwrap().clear();
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
            let script_ctx = self.script_ctx.lock().unwrap();
            for (i, s) in script_ctx.shapes.iter().enumerate() {
                let tape = script_ctx
                    .fidget_context
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
