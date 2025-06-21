//! Rhai bindings to Fidget
//!
//! The [`engine`] function lets you construct a [`rhai::Engine`] with
//! Fidget-specific bindings.  The rest of this documentation explains the
//! behavior of those bindings; for low-level details, see the [`engine`]
//! docstring.
//!
//! # Introduction
//! Rhai is a general-purpose scripting language embedded in Rust.  When used
//! for Fidget scripting, we use the Rhai script to capture a math expression.
//! This math expression can then be processed by Fidget's optimized evaluators.
//!
//! It's important to distinguish between the Rhai script and the target math
//! expression:
//!
//! - The Rhai script is general-purpose, evaluated a single time to compute
//!   math expressions, and supports language features like conditionals and
//!   loops
//! - The math expression is closed-form arithmetic, evaluated _many_ times
//!   over the course of rendering, and only supports operations from
//!   [`TreeOp`](crate::context::TreeOp)
//!
//! The purpose of evaluating the Rhai script is to capture a _trace_ of a math
//! expression.  Evaluating `x + y` in a Rhai script does not actually do any
//! arithmetic; instead, it creates a math expression `Add(Var::X, Var::Y)`.
//!
//! Operator overloading makes this ergonomic, but can mask the fact that the
//! Rhai script is not the math expression!
//!
//! # Trees
//! The basic type for math expressions is a `Tree`, which is equivalent to
//! [`fidget::context::Tree`](crate::context::Tree).  Trees are typically built
//! from `(x, y, z)` primitives, which can be constructed with the `axes()`
//! function:
//!
//! ```
//! # fidget::rhai::engine().run("
//! let xyz = axes();
//! xyz.x + xyz.y
//! # ").unwrap();
//! ```
//!
//! `x, y, z` variables are also automatically injected into the `Engine`'s
//! context before evaluation.
//!
//! # Vector types
//! The Rhai context includes `vec2`, `vec3`, and `vec4` types, which are
//! roughly analogous to their GLSL equivalents (for floating-point only).
//! Many (but not all) functions are implemented and overloaded for these types;
//! if you encounter something missing, feel free to open an issue.
//!
//! # Shapes
//! In Rhai scripts, shapes can be constructed using object map notation:
//! ```
//! # fidget::rhai::engine().run("
//! circle(#{ center: vec2(1.0, 2.0), radius: 3.0 })
//! # ").unwrap();
//! ```
//!
//! This works for any object type; in addition, there are a bunch of ergonomic
//! improvements on top of this low-level syntax.
//!
//! ## Type coercions
//! Shapes are built from a set of Rust primitives, with generous conversions
//! from Rhai's native types:
//!
//! - Scalar values (`f64`)
//!     - Both floating-point and integer Rhai values will be accepted
//! - Vectors (`vec2` and `vec3`)
//!     - These may be explicitly constructed with `vec2(x, y)` and
//!       `vec3(x, y, z)`
//!     - Appropriately-sized arrays of numbers will be automatically converted
//!     - A `vec2` (or something convertible into a `vec2`) will be converted
//!       into a `vec3` with a default `z` value.  This default value is
//!       shape-specific, e.g. it will be 0 for a position and 1 for a scale.
//! ```
//! # fidget::rhai::engine().run("
//! // array -> vec2
//! let c = circle(#{ center: [1, 2], radius: 3 });
//!
//! // array -> vec3
//! let s = sphere(#{ center: [1, 2, 4], radius: 3 });
//!
//! // array -> vec2 -> vec3
//! move(#{ shape: c, offset: [1, 1] });
//! # ").unwrap();
//! ```
//!
//! ## Default values
//! Many shape fields have sensibly-defined default values; these are usually
//! either 0 or 1 (or the equivalent `VecX` values).  Fields with default values
//! may be omitted from the map:
//!
//! ```
//! # fidget::rhai::engine().run("
//! let c = circle(#{ center: [1, 2] }); // radius = 1
//! let s = sphere(#{ radius: 3 }); // center = [0, 0, 0]
//! # ").unwrap();
//! ```
//!
//! ## Uniquely typed functions
//! Any shape with unique arguments may skip the object map and pass arguments
//! directly; order doesn't matter, because the type is unambiguous.
//!
//! ```
//! # fidget::rhai::engine().run("
//! // array -> vec2
//! let c1 = circle([1, 2], 3);
//! let c2 = circle(3, [1, 2]); // order doesn't matter!
//! # ").unwrap();
//! ```
//!
//! In addition, fields with default values may be skipped:
//!
//! ```
//! # fidget::rhai::engine().run("
//! // array -> vec2
//! let c1 = circle([1, 2]); // radius = 1
//! let c2 = circle(); // center = [0, 0], radius = 1
//! # ").unwrap();
//! ```
//!
//! Note that some kinds of type coercion will not work in this regime, e.g.
//! `vec2 -> vec3`:
//!
//! ```should_panic
//! # fidget::rhai::engine().run("
//! // array -> vec2
//! let c1 = sphere([1, 1], 4); // no vec2 -> vec3 conversion!
//! # ").unwrap();
//! ```
//!
//! ## Function chaining
//! Shapes with a single initial `Tree` member are typically transforms (e.g.
//! `move` from above).  These functions may be called with the tree as their
//! first (unnamed) argument, followed by an object map of remaining parameters.
//!
//! ```
//! # fidget::rhai::engine().run("
//! let c = circle(#{ center: [1, 2], radius: 3 });
//! move(c, #{ offset: [1, 1] });
//! # ").unwrap();
//! ```
//!
//! Given Rhai's dispatch strategy, this can also be written as a function
//! chain, which is more ergonomic for a string of transforms:
//!
//! ```
//! # fidget::rhai::engine().run("
//! circle(#{ center: [1, 2], radius: 3 })
//!     .move(#{ offset: [1, 1] });
//! # ").unwrap();
//! ```
//!
//! A transform which only take a single argument may skip the object map:
//!
//! ```
//! # fidget::rhai::engine().run("
//! circle(#{ center: [1, 2], radius: 3 })
//!     .move([1, 1]);
//! # ").unwrap();
//! ```
//!
//! ## Functions of two trees
//! Shapes which take two trees can be called with two (unnamed) arguments:
//!
//! ```
//! # fidget::rhai::engine().run("
//! let a = circle(#{ center: [0, 0], radius: 1 });
//! let b = circle(#{ center: [1, 0], radius: 0.5 });
//! difference(a, b);
//! # ").unwrap();
//! ```
//!
//! ## Tree reduction functions
//! Any function which takes a single `Vec<Tree>` will accept both an array of
//! trees or individual tree arguments (up to an 8-tuple).
//!
//! ```
//! # fidget::rhai::engine().run("
//! let a = circle(#{ center: [1, 1], radius: 3 });
//! let b = circle(#{ center: [2, 2], radius: 3 });
//! let c = circle(#{ center: [3, 3], radius: 3 });
//! union([a, b, c]);
//! union(a, b, c);
//! union(a, b, c, a, b, c, a, b);
//! # ").unwrap();
//! ```
//!
//! ## Automatic tree reduction
//! Any shape which takes a `Tree` will also accept an array of trees, which are
//! automatically reduced with a union operation.
//!
//! ```
//! # fidget::rhai::engine().run("
//! [
//!     circle(#{ center: [0, 0], radius: 3 }),
//!     circle(#{ center: [2, 2], radius: 3 }),
//! ]
//! .move(#{ offset: [1, 1] });
//! # ").unwrap();
//! ```
pub mod shapes;
pub mod tree;
pub mod types;

use crate::context::Tree;

/// Build a new engine with Fidget-specific bindings
///
/// The bindings are as follows:
///
/// - `Tree`-specific type, overloads, and `axes()` ([`tree::register`])
/// - Custom types (e.g. GLSL-style vectors), provided by [`types::register`]
/// - Shapes and transforms ([`shapes::register`])
/// - An `on_progress` limit of 50,000 steps (chosen arbitrarily)
/// - A custom resolver ([`resolver`]) which provides fallbacks for `x`, `y`,
///   and `z` (if not defined)
pub fn engine() -> rhai::Engine {
    let mut engine = rhai::Engine::new();

    tree::register(&mut engine);
    types::register(&mut engine);
    shapes::register(&mut engine);

    engine.on_progress(move |count| {
        if count > 50_000 {
            Some("script runtime exceeded".into())
        } else {
            None
        }
    });

    #[allow(deprecated, reason = "not actually deprecated, just unstable")]
    engine.on_var(resolver);

    engine
}

/// Variable resolver which provides `x`, `y`, `z` if not found
pub fn resolver(
    name: &str,
    _index: usize,
    ctx: rhai::EvalContext,
) -> Result<Option<rhai::Dynamic>, Box<rhai::EvalAltResult>> {
    if let Some(out) = ctx.scope().get_value(name) {
        Ok(Some(out))
    } else {
        match name {
            "x" => Ok(Some(rhai::Dynamic::from(Tree::x()))),
            "y" => Ok(Some(rhai::Dynamic::from(Tree::y()))),
            "z" => Ok(Some(rhai::Dynamic::from(Tree::z()))),
            _ => Ok(None),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Helper trait to go from a Rhai dynamic object to a particular type
pub trait FromDynamic
where
    Self: Sized,
{
    /// Build an object from a dynamic value and optional default
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        v: rhai::Dynamic,
        default: Option<&Self>,
    ) -> Result<Self, Box<rhai::EvalAltResult>>;
}

impl FromDynamic for f64 {
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        d: rhai::Dynamic,
        _default: Option<&f64>,
    ) -> Result<Self, Box<rhai::EvalAltResult>> {
        let ty = d.type_name();
        d.clone()
            .try_cast::<f64>()
            .or_else(|| d.try_cast::<i64>().map(|f| f as f64))
            .ok_or_else(|| {
                rhai::EvalAltResult::ErrorMismatchDataType(
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
    use crate::Context;

    #[test]
    fn test_eval() {
        let engine = engine();
        let t = engine.eval("x + y").unwrap();
        let mut ctx = Context::new();
        let sum = ctx.import(&t);
        assert_eq!(ctx.eval_xyz(sum, 1.0, 2.0, 0.0).unwrap(), 3.0);
    }

    #[test]
    fn test_eval_multiline() {
        let engine = engine();
        let t = engine.eval("let foo = x; foo + y").unwrap();
        let mut ctx = Context::new();
        let sum = ctx.import(&t);
        assert_eq!(ctx.eval_xyz(sum, 1.0, 2.0, 0.0).unwrap(), 3.0);
    }

    #[test]
    fn test_no_comparison() {
        let engine = engine();
        let out = engine.run("x < 0");
        assert!(out.is_err());
    }

    #[test]
    fn resolver_fallback() {
        let engine = engine();
        let out: bool = engine.eval("let x = 1; x < 0").unwrap();
        assert!(!out);
    }
}
