#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::sync::{Arc, Mutex};

use eframe::egui::{self, widgets::text_edit::TextEdit, FontId};
use fidget::context::{Context, Node};
use rhai::{Dynamic, Engine};

fn main() {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "fidget",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    );
}

struct MyApp {
    engine: Engine,
    script: String,
    out: String,
    label_height: Option<f32>,
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
        let ctx = self.tag().unwrap().clone_cast::<Arc<Mutex<Context>>>();
        let mut lock = ctx.lock().unwrap();
        f(&mut lock)
    }
}

////////////////////////////////////////////////////////////////////////////////

/*
fn circle(cx, cy, r) {
    let c = |x, y| {
        sqrt((x - cx) * (x - cx) +
             (y - cy) * (y - cy)) - r
    };
    print(c);
    c
}

let r = 1;
let f = circle(0, 0, 2);
r = 1;
print(f.call(1.0, 0.0));

*/

fn var_x(ctx: rhai::NativeCallContext) -> Node {
    ctx.with_fidget_context(|c| c.x())
}
fn var_y(ctx: rhai::NativeCallContext) -> Node {
    ctx.with_fidget_context(|c| c.y())
}

macro_rules! define_binary_fns {
    ($name:ident) => {
        mod $name {
            use super::*;
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
        let mut engine = Engine::new();
        engine.register_type_with_name::<Node>("Node");
        engine.register_fn("var_x", var_x);
        engine.register_fn("var_y", var_y);

        register_binary_fns!("+", add, engine);
        register_binary_fns!("-", sub, engine);
        register_binary_fns!("*", mul, engine);
        register_binary_fns!("min", min, engine);
        register_binary_fns!("max", max, engine);
        register_unary_fns!("sqrt", sqrt, engine);
        register_unary_fns!("square", square, engine);
        register_unary_fns!("-", neg, engine);

        engine.set_fast_operators(false);

        let fidget_ctx = Dynamic::from(Arc::new(Mutex::new(Context::new())));
        engine.set_default_tag(fidget_ctx);

        Self {
            engine,
            script: "// hello, world".to_owned(),
            out: "".to_string(),
            label_height: None,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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
                let r = ui.add_sized(
                    size,
                    TextEdit::multiline(&mut self.script)
                        .font(FontId::monospace(16.0)),
                );
                if r.changed() || self.label_height.is_none() {
                    let c = self
                        .engine
                        .default_tag()
                        .clone_cast::<Arc<Mutex<Context>>>();
                    c.lock().unwrap().clear();

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
    }
}
