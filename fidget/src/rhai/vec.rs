use crate::{
    rhai::FromDynamic,
    shapes::{Vec2, Vec3},
};
use rhai::{CustomType, EvalAltResult, TypeBuilder};

pub(crate) fn register(engine: &mut rhai::Engine) {
    engine.build_type::<Vec2>();
    engine.build_type::<Vec3>();
}

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
