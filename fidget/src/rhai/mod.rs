//! Rhai bindings to Fidget
//!
//! There are two main ways to use these bindings.
//!
//! The simplest option is to call [`eval`], which evaluates a single expression
//! with pre-defined variables `x`, `y`, `z`.
//!
//! ```
//! use fidget::{
//!     shape::EzShape,
//!     vm::VmShape,
//! };
//!
//! let tree = fidget::rhai::eval("x + y")?;
//! let shape = VmShape::from(tree);
//! let mut eval = VmShape::new_point_eval();
//! let tape = shape.ez_point_tape();
//! assert_eq!(eval.eval(&tape, 1.0, 2.0, 0.0)?.0, 3.0);
//! # Ok::<(), fidget::Error>(())
//! ```
//!
//! `eval` returns a single value.  To evaluate a script with multiple outputs,
//! construct an [`Engine`] then call [`Engine::run`]:
//!
//! ```
//! use fidget::{
//!     shape::EzShape,
//!     vm::VmShape,
//!     rhai::Engine
//! };
//!
//! let mut engine = Engine::new();
//! let mut out = engine.run("draw(x + y - 1)")?;
//!
//! assert_eq!(out.shapes.len(), 1);
//! let shape = VmShape::from(out.shapes.pop().unwrap().tree);
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
use rhai::{CustomType, EvalAltResult, NativeCallContext, TypeBuilder};

pub mod shapes;
pub mod tree;
pub mod vec;

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

        tree::register(&mut engine);
        vec::register(&mut engine);
        shapes::register(&mut engine);

        engine.build_type::<Axes>();
        engine.register_fn("axes", axes);
        engine.register_fn("draw", draw);
        engine.register_fn("draw_rgb", draw_rgb);

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

    /// Sets the operation limit (e.g. for untrusted scripts)
    pub fn set_limit(&mut self, limit: u64) {
        self.engine.on_progress(move |count| {
            if count > limit {
                Some("script runtime exceeded".into())
            } else {
                None
            }
        });
    }

    /// Executes a full script
    pub fn run(&mut self, script: &str) -> Result<ScriptContext, Error> {
        self.context.lock().unwrap().clear();

        let mut scope = rhai::Scope::new();
        scope.push("x", Tree::x());
        scope.push("y", Tree::y());
        scope.push("z", Tree::z());
        self.engine.run_with_scope(&mut scope, script)?;

        // Steal the ScriptContext's contents
        let mut lock = self.context.lock().unwrap();
        Ok(std::mem::take(&mut lock))
    }

    /// Evaluates a single expression, in terms of `x`, `y`, and `z`
    pub fn eval(&mut self, script: &str) -> Result<Tree, Error> {
        self.context.lock().unwrap().clear();

        let mut scope = rhai::Scope::new();
        scope.push("x", Tree::x());
        scope.push("y", Tree::y());
        scope.push("z", Tree::z());

        let ast = self.engine.compile(script)?;
        let out = self.engine.eval_ast_with_scope::<Tree>(&mut scope, &ast)?;

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

fn axes(_ctx: NativeCallContext) -> Axes {
    let (x, y, z) = Tree::axes();
    Axes { x, y, z }
}

fn draw(
    ctx: NativeCallContext,
    tree: rhai::Dynamic,
) -> Result<(), Box<EvalAltResult>> {
    let tree = Tree::from_dynamic(&ctx, tree)?;
    let ctx = ctx.tag().unwrap().clone_cast::<Arc<Mutex<ScriptContext>>>();
    ctx.lock().unwrap().shapes.push(DrawShape {
        tree,
        color_rgb: [u8::MAX; 3],
    });
    Ok(())
}

fn draw_rgb(
    ctx: NativeCallContext,
    tree: rhai::Dynamic,
    r: f64,
    g: f64,
    b: f64,
) -> Result<(), Box<EvalAltResult>> {
    let tree = Tree::from_dynamic(&ctx, tree)?;
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
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////

/// One-shot evaluation of a single expression, in terms of `x, y, z`
pub fn eval(s: &str) -> Result<Tree, Error> {
    let mut engine = Engine::new();
    engine.eval(s)
}

////////////////////////////////////////////////////////////////////////////////

/// Helper trait to go from a Rhai dynamic object to a particular type
trait FromDynamic
where
    Self: Sized,
{
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        v: rhai::Dynamic,
    ) -> Result<Self, Box<EvalAltResult>>;
}

impl FromDynamic for Tree {
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        d: rhai::Dynamic,
    ) -> Result<Self, Box<EvalAltResult>> {
        if let Some(t) = d.clone().try_cast::<Tree>() {
            Ok(t)
        } else if let Ok(v) = f64::from_dynamic(ctx, d.clone()) {
            Ok(Tree::constant(v))
        } else if let Ok(v) = <Vec<Tree>>::from_dynamic(ctx, d.clone()) {
            Ok(crate::shapes::Union { input: v }.into())
        } else {
            Err(Box::new(rhai::EvalAltResult::ErrorMismatchDataType(
                "tree".to_string(),
                d.type_name().to_string(),
                ctx.position(),
            )))
        }
    }
}

impl FromDynamic for Vec<Tree> {
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        d: rhai::Dynamic,
    ) -> Result<Self, Box<EvalAltResult>> {
        if let Ok(d) = d.clone().into_array() {
            d.into_iter()
                .map(|v| Tree::from_dynamic(ctx, v))
                .collect::<Result<Vec<_>, _>>()
        } else {
            Err(Box::new(rhai::EvalAltResult::ErrorMismatchDataType(
                "Vec<tree>".to_string(),
                d.type_name().to_string(),
                ctx.position(),
            )))
        }
    }
}

impl FromDynamic for f64 {
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        d: rhai::Dynamic,
    ) -> Result<Self, Box<EvalAltResult>> {
        let ty = d.type_name();
        d.clone()
            .try_cast::<f64>()
            .or_else(|| d.try_cast::<i64>().map(|f| f as f64))
            .ok_or_else(|| {
                EvalAltResult::ErrorMismatchDataType(
                    "float".to_string(),
                    ty.to_string(),
                    ctx.position(),
                )
                .into()
            })
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        context::{BinaryOpcode, Op},
        Context,
    };

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
    fn test_eval_multiline() {
        let mut engine = Engine::new();
        let t = engine.eval("let foo = x; foo + y").unwrap();
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

    #[test]
    fn test_no_comparison() {
        let mut engine = Engine::new();
        let out = engine.run("x < 0");
        assert!(out.is_err());
    }

    #[test]
    fn test_gyroid_sphere() {
        let mut engine = Engine::new();
        let s = include_str!("../../../models/gyroid-sphere.rhai");
        let out = engine.run(s).unwrap();
        assert_eq!(out.shapes.len(), 1);
        let mut ctx = Context::new();
        let sphere = ctx.import(&out.shapes[0].tree);
        assert!(matches!(
            ctx.get_op(sphere).unwrap(),
            Op::Binary(BinaryOpcode::Max, _, _)
        ));
    }
}

pub mod core;
