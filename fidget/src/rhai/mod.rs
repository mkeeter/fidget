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
//! let tree = fidget::rhai::eval("x + y")?;
//! let shape = VmShape::from_tree(&tree);
//! let mut eval = VmShape::new_point_eval();
//! let tape = shape.ez_point_tape();
//! assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0)?.0, 3.0);
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
//! let out = engine.run("draw(x + y - 1)")?;
//!
//! assert_eq!(out.shapes.len(), 1);
//! let shape = VmShape::from_tree(&out.shapes[0].tree);
//! let mut eval = VmShape::new_point_eval();
//! let tape = shape.ez_point_tape();
//! assert_eq!(eval.eval(&tape, 0.5, 2.0, 0.0)?.0, 1.5);
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! Within a call to [`Engine::run`], `draw` and `draw_rgb` insert shapes into
//! [`ScriptContext::shapes`], which is returned after script evaluation is
//! complete.
//!
//! Scripts are evaluated in a Rhai context that includes [`core.rhai`](core),
//! which defines a few simple shapes and transforms.  `x`, `y`, and `z` are
//! defined in the root scope, and `axes()` returns an object with `x`/`y`/`z`
//! members.
use std::sync::{Arc, Mutex};

use crate::{context::Tree, Error};
use rhai::{CustomType, TypeBuilder};

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
    /// The context includes a variety of functions that operate on [`Tree`]
    /// handles.
    ///
    /// In addition, it includes everything in [`core.rhai`](crate::rhai::core),
    /// which is effectively our standard library.
    pub fn new() -> Self {
        let mut engine = rhai::Engine::new();
        engine
            .register_type::<Tree>()
            .register_fn("remap_xyz", remap_xyz);

        engine.build_type::<Axes>();
        engine.register_fn("axes", axes);
        engine.register_fn("draw", draw);
        engine.register_fn("draw_rgb", draw_rgb);

        macro_rules! register_binary_fns {
            ($op:literal, $name:ident, $engine:ident) => {
                $engine.register_fn($op, $name::node_dyn);
                $engine.register_fn($op, $name::dyn_node);
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
        register_binary_fns!("%", modulo, engine);
        register_binary_fns!("min", min, engine);
        register_binary_fns!("max", max, engine);
        register_binary_fns!("compare", compare, engine);
        register_binary_fns!("and", and, engine);
        register_binary_fns!("or", or, engine);
        register_unary_fns!("abs", abs, engine);
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
        register_unary_fns!("not", not, engine);
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

        let mut scope = rhai::Scope::new();
        scope.push("x", Tree::x());
        scope.push("y", Tree::y());
        scope.push("z", Tree::z());
        self.engine
            .run_with_scope(&mut scope, script)
            .map_err(|e| *e)?;

        // Steal the ScriptContext's contents
        let mut lock = self.context.lock().unwrap();
        Ok(std::mem::take(&mut lock))
    }

    /// Evaluates a single expression, in terms of `x`, `y`, and `z`
    pub fn eval(&mut self, script: &str) -> Result<Tree, Error> {
        let mut scope = {
            let mut ctx = self.context.lock().unwrap();
            ctx.clear();

            // Create initialized scope with x/y/z
            let mut scope = rhai::Scope::new();
            scope.push("x", Tree::x());
            scope.push("y", Tree::y());
            scope.push("z", Tree::z());
            scope
        };

        let out = self
            .engine
            .eval_expression_with_scope::<Tree>(&mut scope, script)
            .map_err(|e| *e)?;

        Ok(out)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Shape to render
///
/// Populated by calls to `draw(...)` or `draw_rgb(...)` in a Rhai script
pub struct DrawShape {
    /// Tree to render
    pub tree: Tree,
    /// Color to use when drawing the shape
    pub color_rgb: [u8; 3],
}

/// Context for shape evaluation
///
/// This object stores a set of shapes, which is populated by calls to `draw` or
/// `draw_rgb` during script evaluation.
pub struct ScriptContext {
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
        Self { shapes: vec![] }
    }
    /// Resets the script context
    pub fn clear(&mut self) {
        self.shapes.clear();
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functions injected into the Rhai context

#[derive(Clone, CustomType)]
struct Axes {
    #[rhai_type(readonly)]
    x: Tree,
    #[rhai_type(readonly)]
    y: Tree,
    #[rhai_type(readonly)]
    z: Tree,
}

fn axes(_ctx: rhai::NativeCallContext) -> Axes {
    let (x, y, z) = Tree::axes();
    Axes { x, y, z }
}

fn remap_xyz(shape: Tree, x: Tree, y: Tree, z: Tree) -> Tree {
    shape.remap_xyz(x, y, z)
}

fn draw(ctx: rhai::NativeCallContext, tree: Tree) {
    let ctx = ctx.tag().unwrap().clone_cast::<Arc<Mutex<ScriptContext>>>();
    ctx.lock().unwrap().shapes.push(DrawShape {
        tree,
        color_rgb: [u8::MAX; 3],
    });
}

fn draw_rgb(ctx: rhai::NativeCallContext, tree: Tree, r: f64, g: f64, b: f64) {
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
        tree,
        color_rgb: [f(r), f(g), f(b)],
    });
}

macro_rules! define_binary_fns {
    ($name:ident $(, $op:ident)?) => {
        mod $name {
            use super::*;
            use rhai::NativeCallContext;
            $(
            use std::ops::$op;
            )?
            pub fn node_dyn(
                _ctx: NativeCallContext,
                a: Tree,
                b: rhai::Dynamic,
            ) -> Result<Tree, Box<rhai::EvalAltResult>> {
                let b = if let Some(v) = b.clone().try_cast::<f64>() {
                    Tree::constant(v)
                } else if let Some(v) = b.clone().try_cast::<i64>() {
                    Tree::constant(v as f64)
                } else if let Some(t) = b.clone().try_cast::<Tree>() {
                    t
                } else {
                    let e = format!(
                        "invalid type for {}(Tree, rhs): {}",
                        stringify!($name),
                        b.type_name()
                    );
                    return Err(e.into());
                };
                Ok(a.$name(b))
            }
            pub fn dyn_node(
                _ctx: NativeCallContext,
                a: rhai::Dynamic,
                b: Tree,
            ) -> Result<Tree, Box<rhai::EvalAltResult>> {
                let a = if let Some(v) = a.clone().try_cast::<f64>() {
                    Tree::constant(v)
                } else if let Some(v) = a.clone().try_cast::<i64>() {
                    Tree::constant(v as f64)
                } else if let Some(t) = a.clone().try_cast::<Tree>() {
                    t
                } else {
                    let e = format!(
                        "invalid type for {}(lhs, Tree): {}",
                        stringify!($name),
                        a.type_name()
                    );
                    return Err(e.into());
                };
                Ok(a.$name(b))
            }
        }
    };
}

macro_rules! define_unary_fns {
    ($name:ident) => {
        mod $name {
            use super::*;
            use rhai::NativeCallContext;
            pub fn node(_ctx: NativeCallContext, a: Tree) -> Tree {
                a.$name()
            }
        }
    };
}

define_binary_fns!(add, Add);
define_binary_fns!(sub, Sub);
define_binary_fns!(mul, Mul);
define_binary_fns!(div, Div);
define_binary_fns!(min);
define_binary_fns!(max);
define_binary_fns!(compare);
define_binary_fns!(modulo);
define_binary_fns!(and);
define_binary_fns!(or);
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
define_unary_fns!(not);
define_unary_fns!(abs);

////////////////////////////////////////////////////////////////////////////////

/// One-shot evaluation of a single expression, in terms of `x, y, z`
pub fn eval(s: &str) -> Result<Tree, Error> {
    let mut engine = Engine::new();
    engine.eval(s)
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::Context;

    #[test]
    fn test_bind() {
        let mut engine = Engine::new();
        let out = engine.run("draw(x + y)").unwrap();
        assert_eq!(out.shapes.len(), 1);
    }

    #[test]
    fn test_eval() {
        let mut engine = Engine::new();
        let t = engine.eval("x + y").unwrap();
        let mut ctx = Context::new();
        let sum = ctx.import(&t);
        assert_eq!(ctx.eval_xyz(sum, 1.0, 2.0, 0.0).unwrap(), 3.0);
    }

    #[test]
    fn test_simple_script() {
        let mut engine = Engine::new();
        let out = engine
            .run(
                "
                let s = circle(0, 0, 2);
                draw(move_xy(s, 1, 3));
                ",
            )
            .unwrap();
        assert_eq!(out.shapes.len(), 1);
        let mut ctx = Context::new();
        let sum = ctx.import(&out.shapes[0].tree);
        assert_eq!(ctx.eval_xyz(sum, 1.0, 3.0, 0.0).unwrap(), -2.0);
    }
}

pub mod core;
