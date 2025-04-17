//! Tools for using [`fidget::shapes`](crate::shapes) in Rhai
//!
//! In Rhai scripts, shapes can be constructed using object map notation:
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! circle(#{ center: vec2(1.0, 2.0), radius: 3.0 })
//! # ").unwrap();
//! ```
//!
//! This works for any object type; in addition, there are a bunch of ergonomic
//! improvements on top of this low-level syntax.
//!
//! # Type coercions
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
//!       into a `vec3` with a `z` value of 0.
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
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
//! # Function chaining
//! Shapes which have a single `Tree` member are typically transforms (e.g.
//! `move` from above).  These functions may be called with the tree as their
//! first (unnamed) argument, followed by an object map of remaining parameters.
//!
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! let c = circle(#{ center: [1, 2], radius: 3 });
//! move(c, #{ offset: [1, 1] });
//! # ").unwrap();
//! ```
//!
//! Given Rhai's dispatch strategy, this can also be written as a function
//! chain, which is more ergonomic for a string of transforms:
//!
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! circle(#{ center: [1, 2], radius: 3 })
//!     .move(#{ offset: [1, 1] });
//! # ").unwrap();
//! ```
//!
//! # Tree reduction functions
//! Any function which takes a single `Vec<Tree>` will accept both an array of
//! trees or individual tree arguments (up to an 8-tuple).
//!
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! let a = circle(#{ center: [1, 1], radius: 3 });
//! let b = circle(#{ center: [2, 2], radius: 3 });
//! let c = circle(#{ center: [3, 3], radius: 3 });
//! union([a, b, c]);
//! union(a, b, c);
//! union(a, b, c, a, b, c, a, b);
//! # ").unwrap();
//! ```
//!
//! # Automatic tree reduction
//! Any shape which takes a `Tree` will also accept an array of trees, which are
//! automatically reduced with a union operation.
//!
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! [
//!     circle(#{ center: [0, 0], radius: 3 }),
//!     circle(#{ center: [2, 2], radius: 3 }),
//! ]
//! .move(#{ offset: [1, 1] });
//! # ").unwrap();
//! ```
use crate::{
    context::Tree,
    rhai::FromDynamic,
    shapes::{Vec2, Vec3},
};
use facet::{ConstTypeId, Facet};
use rhai::{EvalAltResult, NativeCallContext};

pub(crate) fn register(engine: &mut rhai::Engine) {
    use crate::shapes::*;

    register_one::<Circle>(engine);
    register_one::<Sphere>(engine);
    register_one::<Move>(engine);
    register_one::<Union>(engine);
}

fn register_one<T: Facet + Clone + Send + Sync + Into<Tree> + 'static>(
    engine: &mut rhai::Engine,
) {
    let s = validate_type::<T>(); // panic if the type is invalid

    use heck::ToSnakeCase;
    use std::io::Write;

    // Get shape type (CamelCase) and builder (snake_case) names
    let mut writer = std::io::BufWriter::new(Vec::new());
    write!(&mut writer, "{}", T::SHAPE).unwrap();
    let name = std::str::from_utf8(writer.buffer()).unwrap();
    let name_lower = name.to_snake_case();

    engine
        .register_type_with_name::<T>(name)
        .register_fn(&name_lower, build_from_map::<T>)
        .register_fn("build", |_ctx: NativeCallContext, t: T| -> Tree {
            t.into()
        });

    // Special handling for transform-shaped functions
    let tree_count = s
        .fields
        .iter()
        .filter(|t| t.shape().id == ConstTypeId::of::<Tree>())
        .count();
    if tree_count == 1
        && s.fields
            .iter()
            .all(|f| f.shape().id != ConstTypeId::of::<Vec<Tree>>())
    {
        engine.register_fn(&name_lower, build_transform::<T>);
    }

    // Pure tree reduction functions
    if s.fields.len() == 1
        && s.fields[0].shape().id == ConstTypeId::of::<Vec<Tree>>()
    {
        engine.register_fn(&name_lower, build_reduce1::<T>);
        engine.register_fn(&name_lower, build_reduce2::<T>);
        engine.register_fn(&name_lower, build_reduce3::<T>);
        engine.register_fn(&name_lower, build_reduce4::<T>);
        engine.register_fn(&name_lower, build_reduce5::<T>);
        engine.register_fn(&name_lower, build_reduce6::<T>);
        engine.register_fn(&name_lower, build_reduce7::<T>);
        engine.register_fn(&name_lower, build_reduce8::<T>);
    }
}

/// Checks whether `T` has fields of known types
fn validate_type<T: Facet>() -> facet::Struct {
    let facet::Def::Struct(s) = T::SHAPE.def else {
        panic!("must be a struct-shaped type");
    };
    // Linear search is faster than anything fancy for small N
    // NOTE: if you add a new type here, also add it to build_from_map
    let known_types = [
        ConstTypeId::of::<f64>(),
        ConstTypeId::of::<Vec2>(),
        ConstTypeId::of::<Vec3>(),
        ConstTypeId::of::<Tree>(),
        ConstTypeId::of::<Vec<Tree>>(),
    ];
    for f in s.fields {
        assert!(
            known_types.contains(&f.shape().id),
            "unkonwn type {}",
            f.shape()
        );
    }
    s
}

/// Builds a transform-shaped`T` from a Rhai map
fn build_transform<T: Facet>(
    ctx: NativeCallContext,
    t: rhai::Dynamic,
    m: rhai::Map,
) -> Result<T, Box<EvalAltResult>> {
    let mut t = Some(Tree::from_dynamic(&ctx, t)?);

    let mut builder = facet::Wip::alloc::<T>();
    let facet::Def::Struct(shape) = T::SHAPE.def else {
        panic!("must build a struct");
    };

    for (i, f) in shape.fields.iter().enumerate() {
        let field_shape = f.shape();
        if field_shape.id == ConstTypeId::of::<Tree>() {
            let t = t.take().unwrap();
            builder = builder.field(i).unwrap().put(t).unwrap().pop().unwrap();
            continue;
        }

        let Some(v) = m.get(f.name).cloned() else {
            return Err(EvalAltResult::ErrorRuntime(
                format!("field {} must be provided for {}", f.name, T::SHAPE)
                    .into(),
                ctx.position(),
            )
            .into());
        };

        // NOTE: if you add a new type here, also add it to validate_type
        builder = if field_shape.id == ConstTypeId::of::<f64>() {
            let v = f64::from_dynamic(&ctx, v)?;
            builder.field(i).unwrap().put(v).unwrap().pop().unwrap()
        } else if field_shape.id == ConstTypeId::of::<Vec2>() {
            let v = Vec2::from_dynamic(&ctx, v)?;
            builder.field(i).unwrap().put(v).unwrap().pop().unwrap()
        } else if field_shape.id == ConstTypeId::of::<Vec3>() {
            let v = Vec3::from_dynamic(&ctx, v)?;
            builder.field(i).unwrap().put(v).unwrap().pop().unwrap()
        } else {
            panic!("unknown type {}", field_shape);
        }
    }

    // This is quadratic, but N is small
    for k in m.keys() {
        if !shape.fields.iter().any(|p| p.name == k.as_str()) {
            return Err(EvalAltResult::ErrorRuntime(
                format!("field {k} is not present in {}", T::SHAPE).into(),
                ctx.position(),
            )
            .into());
        }
    }
    Ok(builder.build().unwrap().materialize().unwrap())
}

/// Builds a `T` from a Rhai map
fn build_from_map<T: Facet>(
    ctx: NativeCallContext,
    m: rhai::Map,
) -> Result<T, Box<EvalAltResult>> {
    let mut builder = facet::Wip::alloc::<T>();
    let facet::Def::Struct(shape) = T::SHAPE.def else {
        panic!("must build a struct");
    };
    for (i, f) in shape.fields.iter().enumerate() {
        let Some(v) = m.get(f.name).cloned() else {
            return Err(EvalAltResult::ErrorRuntime(
                format!("field {} must be provided for {}", f.name, T::SHAPE)
                    .into(),
                ctx.position(),
            )
            .into());
        };

        // NOTE: if you add a new type here, also add it to validate_type
        let field_shape = f.shape();
        builder = if field_shape.id == ConstTypeId::of::<f64>() {
            let v = f64::from_dynamic(&ctx, v)?;
            builder.field(i).unwrap().put(v).unwrap().pop().unwrap()
        } else if field_shape.id == ConstTypeId::of::<Vec2>() {
            let v = Vec2::from_dynamic(&ctx, v)?;
            builder.field(i).unwrap().put(v).unwrap().pop().unwrap()
        } else if field_shape.id == ConstTypeId::of::<Vec3>() {
            let v = Vec3::from_dynamic(&ctx, v)?;
            builder.field(i).unwrap().put(v).unwrap().pop().unwrap()
        } else if field_shape.id == ConstTypeId::of::<Tree>() {
            let v = Tree::from_dynamic(&ctx, v)?;
            builder.field(i).unwrap().put(v).unwrap().pop().unwrap()
        } else if field_shape.id == ConstTypeId::of::<Vec<Tree>>() {
            let v = <Vec<Tree>>::from_dynamic(&ctx, v)?;
            builder.field(i).unwrap().put(v).unwrap().pop().unwrap()
        } else {
            panic!("unknown type {}", field_shape);
        }
    }

    // This is quadratic, but N is small
    for k in m.keys() {
        if !shape.fields.iter().any(|p| p.name == k.as_str()) {
            return Err(EvalAltResult::ErrorRuntime(
                format!("field {k} is not present in {}", T::SHAPE).into(),
                ctx.position(),
            )
            .into());
        }
    }
    Ok(builder.build().unwrap().materialize().unwrap())
}

macro_rules! reducer {
    ($name:ident, $($v:ident),*) => {
        #[allow(clippy::too_many_arguments)]
        fn $name<T: Facet>(
            ctx: NativeCallContext,
            $($v: rhai::Dynamic),*
        ) -> Result<T, Box<EvalAltResult>> {
            let mut builder = facet::Wip::alloc::<T>();
            let facet::Def::Struct(shape) = T::SHAPE.def else {
                panic!("must build a struct");
            };
            assert_eq!(shape.fields[0].shape().id, ConstTypeId::of::<Vec<Tree>>());
            assert_eq!(shape.fields.len(), 1);
            let v = vec![$(
                Tree::from_dynamic(&ctx, $v)?
            ),*];
            builder = builder.field(0).unwrap().put(v).unwrap().pop().unwrap();

            Ok(builder.build().unwrap().materialize().unwrap())
        }
    }
}
reducer!(build_reduce1, a);
reducer!(build_reduce2, a, b);
reducer!(build_reduce3, a, b, c);
reducer!(build_reduce4, a, b, c, d);
reducer!(build_reduce5, a, b, c, d, e);
reducer!(build_reduce6, a, b, c, d, e, f);
reducer!(build_reduce7, a, b, c, d, e, f, g);
reducer!(build_reduce8, a, b, c, d, e, f, g, h);

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::shapes::*;

    #[test]
    fn circle_builder() {
        let mut e = rhai::Engine::new();
        register_one::<Circle>(&mut e);
        e.build_type::<Vec2>();
        let c: Circle = e
            .eval("circle(#{ center: vec2(1, 2), radius: 3 })")
            .unwrap();
        assert_eq!(c.center.x, 1.0);
        assert_eq!(c.center.y, 2.0);
        assert_eq!(c.radius, 3.0);

        assert!(e
            .eval::<Circle>("circle(#{ center: 3.0, radius: 3 })")
            .is_err());
        assert!(e
            .eval::<Circle>(
                "circle(#{ center: vec2(\"omg\", \"wtf\"), radius: 3 })"
            )
            .is_err());
        assert!(e.eval::<Circle>("circle(#{ radius: 4 })").is_err());
        assert!(e
            .eval::<Circle>("circle(#{ radius: 4, xy: vec2(1, 2) })")
            .is_err());
    }
}
