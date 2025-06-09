//! Rhai bindings for Fidget's 2D and 3D vector types
use crate::{
    rhai::FromDynamic,
    shapes::{Vec2, Vec3, Vec4},
};
use rhai::{CustomType, EvalAltResult, TypeBuilder};

macro_rules! register_all {
    ($engine:ident, $ty:ident) => {
        register_binary!($engine, $ty, "+", add, Add);
        register_binary!($engine, $ty, "*", mul, Mul);
        register_binary!($engine, $ty, "-", sub, Sub);
        register_binary!($engine, $ty, "/", div, Div);

        register_binary!($engine, $ty, min);
        register_binary!($engine, $ty, max);

        register_unary!($engine, $ty, sqrt);
        register_unary!($engine, $ty, abs);
    };
}

macro_rules! register_binary {
    ($engine:ident, $ty:ident, $rop:expr, $base_fn:ident $(, $op:ident)?) => {
        $engine.register_fn($rop, |a: $ty, b: $ty| -> $ty {
            $( use std::ops::$op; )?
            a.$base_fn(b)
        });
        $engine.register_fn($rop, |a: $ty, b: f64| -> $ty {
            $( use std::ops::$op; )?
            a.$base_fn(b)
        });
        $engine.register_fn($rop, |a: $ty, b: i64| -> $ty {
            $( use std::ops::$op; )?
            a.$base_fn(b as f64)
        });
        $engine.register_fn($rop, |a: f64, b: $ty| -> $ty {
            $( use std::ops::$op; )?
            $ty::from(a).$base_fn(b)
        });
        $engine.register_fn($rop, |a: i64, b: $ty| -> $ty {
            $( use std::ops::$op; )?
            $ty::from(a as f64).$base_fn(b)
        });
    };
    ($engine:ident, $ty:ident, $base_fn:ident) => {
        register_binary!($engine, $ty, stringify!($base_fn), $base_fn)
    };
}

macro_rules! register_unary {
    ($engine:ident, $ty:ident, $base_fn:ident) => {
        $engine.register_fn(stringify!($base_fn), |a: $ty| -> $ty {
            a.$base_fn()
        });
    };
}

/// Installs the `Vec2` and `Vec3` types into a Rhai engine
pub fn register(engine: &mut rhai::Engine) {
    engine.build_type::<Vec2>();
    engine.build_type::<Vec3>();
    register_all!(engine, Vec2);
    register_all!(engine, Vec3);
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
                    let x = f64::from_dynamic(&ctx, x, None)?;
                    let y = f64::from_dynamic(&ctx, y, None)?;
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
                    let x = f64::from_dynamic(&ctx, x, None)?;
                    let y = f64::from_dynamic(&ctx, y, None)?;
                    let z = f64::from_dynamic(&ctx, z, None)?;
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
        _default: Option<&Vec2>,
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
                    let x = f64::from_dynamic(ctx, array[0].clone(), None)?;
                    let y = f64::from_dynamic(ctx, array[1].clone(), None)?;
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
        default: Option<&Vec3>,
    ) -> Result<Self, Box<EvalAltResult>> {
        if let Ok(v) = Vec2::from_dynamic(ctx, d.clone(), None) {
            Ok(Vec3 {
                x: v.x,
                y: v.y,
                z: default.map(|d| d.z).unwrap_or(0.0),
            })
        } else if let Some(v) = d.clone().try_cast() {
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
                    let x = f64::from_dynamic(ctx, array[0].clone(), None)?;
                    let y = f64::from_dynamic(ctx, array[1].clone(), None)?;
                    let z = f64::from_dynamic(ctx, array[2].clone(), None)?;
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

impl FromDynamic for Vec4 {
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        d: rhai::Dynamic,
        _default: Option<&Vec4>,
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
                4 => {
                    let x = f64::from_dynamic(ctx, array[0].clone(), None)?;
                    let y = f64::from_dynamic(ctx, array[1].clone(), None)?;
                    let z = f64::from_dynamic(ctx, array[2].clone(), None)?;
                    let w = f64::from_dynamic(ctx, array[3].clone(), None)?;
                    Ok(Vec4 { x, y, z, w })
                }
                n => Err(EvalAltResult::ErrorMismatchDataType(
                    "[float; 4]".to_string(),
                    format!("[dynamic; {n}]"),
                    ctx.position(),
                )
                .into()),
            }
        }
    }
}
