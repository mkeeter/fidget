//! Rhai bindings to Fidget
use std::sync::{Arc, Mutex, MutexGuard};

use crate::context::{Context, Node};

pub struct Engine {
    engine: rhai::Engine,
    context: Arc<Mutex<ScriptContext>>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let mut engine = rhai::Engine::new();
        engine.register_type_with_name::<Node>("Node");
        engine.register_fn("__var_x", var_x);
        engine.register_fn("__var_y", var_y);
        engine.register_fn("__draw", draw);

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

        register_binary_fns!("+", add, engine);
        register_binary_fns!("-", sub, engine);
        register_binary_fns!("*", mul, engine);
        register_binary_fns!("min", min, engine);
        register_binary_fns!("max", max, engine);
        register_unary_fns!("sqrt", sqrt, engine);
        register_unary_fns!("square", square, engine);
        register_unary_fns!("-", neg, engine);

        engine.set_fast_operators(false);

        let context = Arc::new(Mutex::new(ScriptContext::new()));
        engine.set_default_tag(rhai::Dynamic::from(context.clone()));

        let ast = engine.compile(include_str!("core.rhai")).unwrap();
        let module =
            rhai::Module::eval_ast_as_new(rhai::Scope::new(), &ast, &engine)
                .unwrap();
        engine.register_global_module(rhai::Shared::new(module));

        Self { engine, context }
    }

    pub fn run(
        &mut self,
        script: &str,
    ) -> Result<(), Box<rhai::EvalAltResult>> {
        self.context.lock().unwrap().clear();
        self.engine.run(script)
    }

    pub fn script_context(&self) -> MutexGuard<ScriptContext> {
        self.context.lock().unwrap()
    }
}

////////////////////////////////////////////////////////////////////////////////

pub struct ScriptContext {
    pub context: Context,
    pub shapes: Vec<Node>,
}

impl ScriptContext {
    fn new() -> Self {
        Self {
            context: Context::new(),
            shapes: vec![],
        }
    }
    fn clear(&mut self) {
        self.context.clear();
        self.shapes.clear();
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Extension trait to pull a Fidget `Context` out of a `NativeCallContext`
trait FidgetContext {
    fn with_fidget_context<F, V>(&self, f: F) -> V
    where
        F: Fn(&mut Context) -> V;
}

impl FidgetContext for rhai::NativeCallContext<'_> {
    fn with_fidget_context<F, V>(&self, f: F) -> V
    where
        F: Fn(&mut Context) -> V,
    {
        let ctx = self
            .tag()
            .unwrap()
            .clone_cast::<Arc<Mutex<ScriptContext>>>();
        let lock = &mut ctx.lock().unwrap().context;
        f(lock)
    }
}

////////////////////////////////////////////////////////////////////////////////

fn var_x(ctx: rhai::NativeCallContext) -> Node {
    ctx.with_fidget_context(|c| c.x())
}
fn var_y(ctx: rhai::NativeCallContext) -> Node {
    ctx.with_fidget_context(|c| c.y())
}

fn draw(ctx: rhai::NativeCallContext, node: Node) {
    let ctx = ctx.tag().unwrap().clone_cast::<Arc<Mutex<ScriptContext>>>();
    ctx.lock().unwrap().shapes.push(node);
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
