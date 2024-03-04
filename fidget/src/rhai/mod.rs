//! Rhai bindings to Fidget
//!
//! There are two main ways to use these bindings.
//!
//! The simplest option is to call [`eval`], which evaluates a single expression
//! with pre-defined variables `x`, `y`, `z`.
//!
//! ```
//! use fidget::{
//!     eval::{Shape, MathShape, EzShape, TracingEvaluator},
//!     vm::VmShape,
//! };
//!
//! let (sum, ctx) = fidget::rhai::eval("x + y")?;
//! let shape = VmShape::new(&ctx, sum)?;
//! let mut eval = VmShape::new_point_eval();
//! let tape = shape.ez_point_tape();
//! assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0, &[])?.0, 3.0);
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! `eval` only evaluates a single expression.  To evaluate a full script,
//! construct an [`Engine`] then call [`Engine::run`]:
//!
//! ```
//! use fidget::{
//!     eval::{Shape, MathShape, EzShape, TracingEvaluator},
//!     vm::VmShape,
//!     rhai::Engine
//! };
//!
//! let mut engine = Engine::new();
//! let out = engine.run("draw(|x, y| x + y - 1)")?;
//!
//! assert_eq!(out.shapes.len(), 1);
//! let shape = VmShape::new(&out.context, out.shapes[0].shape)?;
//! let mut eval = VmShape::new_point_eval();
//! let tape = shape.ez_point_tape();
//! assert_eq!(eval.eval(&tape, 0.5, 2.0, 0.0, &[])?.0, 1.5);
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! Within a call to [`Engine::run`], `draw` and `draw_rgb` insert shapes into
//! [`ScriptContext::shapes`], which is returned after script evaluation is
//! complete.
//!
//! Both `draw` and `draw_rgb` expect to take a two-argument lambda function
//! `|x, y| { ... }`.  They trace evaluation of that function by calling it with
//! X and Y objects which build a math tree in a [`fidget::Context`](Context).
//! This means that the lambda function must be a **pure function** that only
//! uses [Fidget-friendly math operations](crate::context::Op).
//!
//! Scripts are evaluated in a Rhai context that includes [`core.rhai`](core),
//! which defines a few simple shapes and transforms.
use std::sync::{Arc, Mutex};

use crate::{
    context::{Context, Node},
    Error,
};

/// Engine for evaluating a Rhai script with Fidget-specific bindings
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
    /// Constructs a script evaluation engine with Fidget bindings
    ///
    /// The context includes a variety of functions that operate on [`Node`]
    /// handles, using a global [`Context`].
    ///
    /// In addition, it includes everything in [`core.rhai`](crate::rhai::core),
    /// which is effectively our standard library.
    pub fn new() -> Self {
        let mut engine = rhai::Engine::new();
        engine.register_type_with_name::<Node>("Node");
        engine.register_fn("__var_x", var_x);
        engine.register_fn("__var_y", var_y);
        engine.register_fn("__draw", draw);
        engine.register_fn("__draw_rgb", draw_rgb);

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
        register_binary_fns!("/", div, engine);
        register_binary_fns!("min", min, engine);
        register_binary_fns!("max", max, engine);
        register_unary_fns!("sqrt", sqrt, engine);
        register_unary_fns!("square", square, engine);
        register_unary_fns!("sin", sin, engine);
        register_unary_fns!("cos", cos, engine);
        register_unary_fns!("tan", tan, engine);
        register_unary_fns!("asin", asin, engine);
        register_unary_fns!("acos", acos, engine);
        register_unary_fns!("atan", atan, engine);
        register_unary_fns!("exp", exp, engine);
        register_unary_fns!("ln", ln, engine);
        register_unary_fns!("-", neg, engine);

        engine.set_fast_operators(false);

        let context = Arc::new(Mutex::new(ScriptContext::new()));
        engine.set_default_tag(rhai::Dynamic::from(context.clone()));
        engine.set_max_expr_depths(64, 32);

        let ast = engine.compile(include_str!("core.rhai")).unwrap();
        let module =
            rhai::Module::eval_ast_as_new(rhai::Scope::new(), &ast, &engine)
                .unwrap();
        engine.register_global_module(rhai::Shared::new(module));

        Self { engine, context }
    }

    /// Executes a full script
    pub fn run(&mut self, script: &str) -> Result<ScriptContext, Error> {
        self.context.lock().unwrap().clear();
        self.engine.run(script).map_err(|e| *e)?;

        // Steal the ScriptContext's contents
        let mut lock = self.context.lock().unwrap();
        Ok(std::mem::take(&mut lock))
    }

    /// Evaluates a single expression, in terms of `x`, `y`, and `z`
    pub fn eval(&mut self, script: &str) -> Result<(Node, Context), Error> {
        let mut scope = {
            let mut ctx = self.context.lock().unwrap();
            ctx.clear();

            // Create initialized scope with x/y/z
            let mut scope = rhai::Scope::new();
            scope.push("x", ctx.context.x());
            scope.push("y", ctx.context.y());
            scope.push("z", ctx.context.z());
            scope
        };

        let out = self
            .engine
            .eval_expression_with_scope::<Node>(&mut scope, script)
            .map_err(|e| *e)?;

        // Steal the ScriptContext's contents
        let mut lock = self.context.lock().unwrap();
        let ctx: ScriptContext = std::mem::take(&mut lock);

        Ok((out, ctx.context))
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Shape to render
///
/// Populated by calls to `draw(...)` or `draw_rgb(...)` in a Rhai script
pub struct DrawShape {
    /// Shape to render
    pub shape: Node,
    /// Color to use when drawing the shape
    pub color_rgb: [u8; 3],
}

/// Context for shape evaluation
///
/// This object includes a [`Context`] and a set of shapes (written with `draw`
/// or `draw_rgb`).
pub struct ScriptContext {
    /// Fidget context in which to accumulate shapes
    pub context: Context,
    /// List of shapes populated since the last call to [`clear`](Self::clear)
    pub shapes: Vec<DrawShape>,
}

impl Default for ScriptContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptContext {
    /// Builds a new empty script context
    pub fn new() -> Self {
        Self {
            context: Context::new(),
            shapes: vec![],
        }
    }
    /// Resets the script context
    pub fn clear(&mut self) {
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
// Functions injected into the Rhai context

fn var_x(ctx: rhai::NativeCallContext) -> Node {
    ctx.with_fidget_context(|c| c.x())
}
fn var_y(ctx: rhai::NativeCallContext) -> Node {
    ctx.with_fidget_context(|c| c.y())
}

fn draw(ctx: rhai::NativeCallContext, node: Node) {
    let ctx = ctx.tag().unwrap().clone_cast::<Arc<Mutex<ScriptContext>>>();
    ctx.lock().unwrap().shapes.push(DrawShape {
        shape: node,
        color_rgb: [u8::MAX; 3],
    });
}

fn draw_rgb(ctx: rhai::NativeCallContext, node: Node, r: f64, g: f64, b: f64) {
    let ctx = ctx.tag().unwrap().clone_cast::<Arc<Mutex<ScriptContext>>>();
    let f = |a| {
        if a < 0.0 {
            0
        } else if a > 1.0 {
            255
        } else {
            (a * 255.0) as u8
        }
    };
    ctx.lock().unwrap().shapes.push(DrawShape {
        shape: node,
        color_rgb: [f(r), f(g), f(b)],
    });
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
define_binary_fns!(div);
define_binary_fns!(min);
define_binary_fns!(max);
define_unary_fns!(sqrt);
define_unary_fns!(square);
define_unary_fns!(neg);
define_unary_fns!(sin);
define_unary_fns!(cos);
define_unary_fns!(tan);
define_unary_fns!(asin);
define_unary_fns!(acos);
define_unary_fns!(atan);
define_unary_fns!(exp);
define_unary_fns!(ln);

////////////////////////////////////////////////////////////////////////////////

/// One-shot evaluation of a single expression, in terms of `x, y, z`
pub fn eval(s: &str) -> Result<(Node, Context), Error> {
    let mut engine = Engine::new();
    engine.eval(s)
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bind() {
        let mut engine = Engine::new();
        let out = engine.run("draw(|x, y| x + y)").unwrap();
        assert_eq!(out.shapes.len(), 1);
    }

    #[test]
    fn test_eval() {
        let mut engine = Engine::new();
        let (sum, ctx) = engine.eval("x + y").unwrap();
        assert_eq!(ctx.eval_xyz(sum, 1.0, 2.0, 0.0).unwrap(), 3.0);
    }
}

pub mod core;
