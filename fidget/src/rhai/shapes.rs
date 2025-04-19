//! Tools for using [`fidget::shapes`](crate::shapes) in Rhai
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
    register_one::<Scale>(engine);

    register_one::<Union>(engine);
    register_one::<Intersection>(engine);
    register_one::<Difference>(engine);
    register_one::<Inverse>(engine);
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

    engine.register_fn(&name_lower, build_from_map::<T>);

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
        if s.fields.len() == 2 {
            engine.register_fn(&name_lower, build_transform_one::<T>);
        }
    }
    if tree_count == 2 && s.fields.len() == 2 {
        engine.register_fn(&name_lower, build_binary::<T>);
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
            "unknown type {}",
            f.shape()
        );
    }
    s
}

/// Builds a transform-shaped `T` from a Rhai map
fn build_transform<T: Facet + Into<Tree>>(
    ctx: NativeCallContext,
    t: rhai::Dynamic,
    m: rhai::Map,
) -> Result<Tree, Box<EvalAltResult>> {
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
    let t: T = builder.build().unwrap().materialize().unwrap();
    Ok(t.into())
}

/// Builds a transform-shaped `T` from a Rhai map
fn build_transform_one<T: Facet + Into<Tree>>(
    ctx: NativeCallContext,
    t: rhai::Dynamic,
    arg: rhai::Dynamic,
) -> Result<Tree, Box<EvalAltResult>> {
    let mut t = Some(Tree::from_dynamic(&ctx, t)?);
    let mut arg = Some(arg);

    let mut builder = facet::Wip::alloc::<T>();
    let facet::Def::Struct(shape) = T::SHAPE.def else {
        panic!("must build a struct");
    };

    for (i, f) in shape.fields.iter().enumerate() {
        let field_shape = f.shape();
        if field_shape.id == ConstTypeId::of::<Tree>() {
            let t = t.take().unwrap();
            builder = builder.field(i).unwrap().put(t).unwrap().pop().unwrap();
        } else {
            let v = arg.take().unwrap();

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
    }

    let t: T = builder.build().unwrap().materialize().unwrap();
    Ok(t.into())
}

/// Builds a binary function of two trees
fn build_binary<T: Facet + Into<Tree>>(
    ctx: NativeCallContext,
    a: rhai::Dynamic,
    b: rhai::Dynamic,
) -> Result<Tree, Box<EvalAltResult>> {
    let a = Tree::from_dynamic(&ctx, a)?;
    let b = Tree::from_dynamic(&ctx, b)?;

    let mut builder = facet::Wip::alloc::<T>();
    let facet::Def::Struct(shape) = T::SHAPE.def else {
        panic!("must build a struct");
    };

    assert_eq!(shape.fields.len(), 2);
    for f in shape.fields.iter() {
        let field_shape = f.shape();
        assert_eq!(field_shape.id, ConstTypeId::of::<Tree>());
    }

    builder = builder.field(0).unwrap().put(a).unwrap().pop().unwrap();
    builder = builder.field(1).unwrap().put(b).unwrap().pop().unwrap();

    let t: T = builder.build().unwrap().materialize().unwrap();
    Ok(t.into())
}

/// Builds a `T` from a Rhai map
fn build_from_map<T: Facet + Into<Tree>>(
    ctx: NativeCallContext,
    m: rhai::Map,
) -> Result<Tree, Box<EvalAltResult>> {
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
    let t: T = builder.build().unwrap().materialize().unwrap();
    Ok(t.into())
}

macro_rules! reducer {
    ($name:ident, $($v:ident),*) => {
        #[allow(clippy::too_many_arguments)]
        fn $name<T: Facet + Into<Tree>>(
            ctx: NativeCallContext,
            $($v: rhai::Dynamic),*
        ) -> Result<Tree, Box<EvalAltResult>> {
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

            let t: T = builder.build().unwrap().materialize().unwrap();
            Ok(t.into())
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
        assert!(e
            .eval::<Tree>("circle(#{ center: vec2(1, 2), radius: 3 })")
            .is_ok());

        assert!(e
            .eval::<Tree>("circle(#{ center: 3.0, radius: 3 })")
            .is_err());
        assert!(e
            .eval::<Tree>(
                "circle(#{ center: vec2(\"omg\", \"wtf\"), radius: 3 })"
            )
            .is_err());
        assert!(e.eval::<Tree>("circle(#{ radius: 4 })").is_err());
        assert!(e
            .eval::<Tree>("circle(#{ radius: 4, xy: vec2(1, 2) })")
            .is_err());
    }
}
