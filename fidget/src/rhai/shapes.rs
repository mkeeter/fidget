//! Tools for using [`fidget::shapes`](crate::shapes) in Rhai
use crate::{
    context::Tree,
    shapes::{Vec2, Vec3},
};
use facet::{ConstTypeId, Facet};
use rhai::{CustomType, EvalAltResult, TypeBuilder};

impl CustomType for Vec2 {
    fn build(mut builder: TypeBuilder<Self>) {
        builder
            .with_name("Vec2")
            .with_fn(
                "vec2",
                |ctx: rhai::NativeCallContext,
                 x: rhai::Dynamic,
                 y: rhai::Dynamic|
                 -> Result<Self, Box<EvalAltResult>> {
                    let x = f64::from_dynamic(&ctx, x)?;
                    let y = f64::from_dynamic(&ctx, y)?;
                    Ok(Self { x, y })
                },
            )
            .with_get_set("x", |v: &mut Self| v.x, |v: &mut Self, x| v.x = x)
            .with_get_set("y", |v: &mut Self| v.y, |v: &mut Self, y| v.y = y);
    }
}

impl CustomType for Vec3 {
    fn build(mut builder: TypeBuilder<Self>) {
        builder
            .with_name("Vec3")
            .with_fn(
                "vec3",
                |ctx: rhai::NativeCallContext,
                 x: rhai::Dynamic,
                 y: rhai::Dynamic,
                 z: rhai::Dynamic|
                 -> Result<Self, Box<EvalAltResult>> {
                    let x = f64::from_dynamic(&ctx, x)?;
                    let y = f64::from_dynamic(&ctx, y)?;
                    let z = f64::from_dynamic(&ctx, z)?;
                    Ok(Self { x, y, z })
                },
            )
            .with_get_set("x", |v: &mut Self| v.x, |v: &mut Self, x| v.x = x)
            .with_get_set("y", |v: &mut Self| v.y, |v: &mut Self, y| v.y = y)
            .with_get_set("z", |v: &mut Self| v.z, |v: &mut Self, z| v.z = z);
    }
}

impl FromDynamic for Vec2 {
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        d: rhai::Dynamic,
    ) -> Result<Self, Box<EvalAltResult>> {
        if let Some(v) = d.clone().try_cast() {
            Ok(v)
        } else {
            let array = d.into_array().map_err(|ty| {
                EvalAltResult::ErrorMismatchDataType(
                    "array".to_string(),
                    ty.to_string(),
                    ctx.position(),
                )
            })?;
            match array.len() {
                2 => {
                    let x = f64::from_dynamic(ctx, array[0].clone())?;
                    let y = f64::from_dynamic(ctx, array[1].clone())?;
                    Ok(Vec2 { x, y })
                }
                n => Err(EvalAltResult::ErrorMismatchDataType(
                    "[float; 2]".to_string(),
                    format!("[dynamic; {n}]"),
                    ctx.position(),
                )
                .into()),
            }
        }
    }
}

impl FromDynamic for Vec3 {
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        d: rhai::Dynamic,
    ) -> Result<Self, Box<EvalAltResult>> {
        if let Some(v) = d.clone().try_cast() {
            Ok(v)
        } else {
            let array = d.into_array().map_err(|ty| {
                EvalAltResult::ErrorMismatchDataType(
                    "array".to_string(),
                    ty.to_string(),
                    ctx.position(),
                )
            })?;
            match array.len() {
                3 => {
                    let x = f64::from_dynamic(ctx, array[0].clone())?;
                    let y = f64::from_dynamic(ctx, array[1].clone())?;
                    let z = f64::from_dynamic(ctx, array[2].clone())?;
                    Ok(Vec3 { x, y, z })
                }
                n => Err(EvalAltResult::ErrorMismatchDataType(
                    "[float; 3]".to_string(),
                    format!("[dynamic; {n}]"),
                    ctx.position(),
                )
                .into()),
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Installs shapes and helper types into the engine
pub fn register_types(engine: &mut rhai::Engine) {
    use crate::shapes::*;

    engine.build_type::<Vec2>();
    engine.build_type::<Vec3>();
    register::<Circle>(engine);
}

fn register<T: Facet + Clone + Send + Sync + Into<Tree> + 'static>(
    engine: &mut rhai::Engine,
) {
    validate_type::<T>(); // panic if the type is invalid

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
        .register_fn("build", |_ctx: rhai::NativeCallContext, t: T| -> Tree {
            t.into()
        });
}

/// Checks whether `T` has fields of known types
fn validate_type<T: Facet>() {
    let facet::Def::Struct(s) = T::SHAPE.def else {
        panic!("must be a struct-shaped type");
    };
    // Linear search is faster than anything fancy for small N
    // NOTE: if you add a new type here, also add it to build_from_map
    let known_types = [ConstTypeId::of::<f64>(), ConstTypeId::of::<Vec2>()];
    for f in s.fields {
        assert!(known_types.contains(&f.shape.id));
    }
}

/// Builds a `T` from a Rhai map
fn build_from_map<T: Facet>(
    ctx: rhai::NativeCallContext,
    m: rhai::Map,
) -> Result<T, Box<EvalAltResult>> {
    let (poke, guard) = facet::PokeUninit::alloc::<T>();
    let mut poke = poke.into_struct();
    let facet::Def::Struct(shape) = poke.shape().def else {
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
        if f.shape.id == ConstTypeId::of::<f64>() {
            let v = f64::from_dynamic(&ctx, v)?;
            poke.set(i, v).unwrap();
        } else if f.shape.id == ConstTypeId::of::<Vec2>() {
            let v = Vec2::from_dynamic(&ctx, v)?;
            poke.set(i, v).unwrap();
        } else {
            panic!("unknown type {}", f.shape);
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
    Ok(poke.build(Some(guard)))
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::shapes::*;

    #[test]
    fn circle_builder() {
        let mut e = rhai::Engine::new();
        register::<Circle>(&mut e);
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
