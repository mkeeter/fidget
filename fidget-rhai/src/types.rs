//! Rhai bindings for Fidget's 2D and 3D vector types
use crate::FromDynamic;
use fidget_core::{
    context::{Tree, TreeOp},
    var::Var,
};
use fidget_shapes::types::{Axis, Plane, Vec2, Vec3, Vec4};
use rhai::EvalAltResult;

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
        $engine.register_fn("-", |v: $ty| -v);
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

/// Installs common types (from [`fidget::shapes`](crate::shapes)) into the engine
pub fn register(engine: &mut rhai::Engine) {
    register_vec2(engine);
    register_vec3(engine);
    register_all!(engine, Vec2);
    register_all!(engine, Vec3);

    register_axis(engine);
    register_plane(engine);
}

fn register_vec2(engine: &mut rhai::Engine) {
    engine
        .register_type_with_name::<Vec2>("Vec2")
        .register_fn("to_string", |t: &mut Vec2| format!("[{}, {}]", t.x, t.y))
        .register_fn("vec2", |v: Vec2| v) // idempotent
        .register_fn(
            "vec2",
            |ctx: rhai::NativeCallContext,
             x: rhai::Dynamic,
             y: rhai::Dynamic|
             -> Result<Vec2, Box<EvalAltResult>> {
                let x = f64::from_dynamic(&ctx, x, None)?;
                let y = f64::from_dynamic(&ctx, y, None)?;
                Ok(Vec2 { x, y })
            },
        )
        .register_fn(
            "vec2",
            |ctx: rhai::NativeCallContext,
             array: rhai::Array|
             -> Result<Vec2, Box<EvalAltResult>> {
                vec2_from_rhai_array(&ctx, array)
            },
        )
        .register_get_set("x", |v: &mut Vec2| v.x, |v: &mut Vec2, x| v.x = x)
        .register_get_set("y", |v: &mut Vec2| v.y, |v: &mut Vec2, y| v.y = y);
}

fn register_vec3(engine: &mut rhai::Engine) {
    engine
        .register_type_with_name::<Vec3>("Vec3")
        .register_fn("to_string", |t: &mut Vec3| {
            format!("[{}, {}, {}]", t.x, t.y, t.z)
        })
        .register_fn("vec3", |v: Vec3| v) // idempotent
        .register_fn(
            "vec3",
            |ctx: rhai::NativeCallContext,
             x: rhai::Dynamic,
             y: rhai::Dynamic,
             z: rhai::Dynamic|
             -> Result<Vec3, Box<EvalAltResult>> {
                let x = f64::from_dynamic(&ctx, x, None)?;
                let y = f64::from_dynamic(&ctx, y, None)?;
                let z = f64::from_dynamic(&ctx, z, None)?;
                Ok(Vec3 { x, y, z })
            },
        )
        .register_fn(
            "vec3",
            |ctx: rhai::NativeCallContext,
             array: rhai::Array|
             -> Result<Vec3, Box<EvalAltResult>> {
                vec3_from_rhai_array(&ctx, array, None)
            },
        )
        .register_get_set("x", |v: &mut Vec3| v.x, |v: &mut Vec3, x| v.x = x)
        .register_get_set("y", |v: &mut Vec3| v.y, |v: &mut Vec3, y| v.y = y)
        .register_get_set("z", |v: &mut Vec3| v.z, |v: &mut Vec3, z| v.z = z);
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
            vec2_from_rhai_array(ctx, array)
        }
    }
}

fn vec2_from_rhai_array(
    ctx: &rhai::NativeCallContext,
    array: rhai::Array,
) -> Result<Vec2, Box<EvalAltResult>> {
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
            vec3_from_rhai_array(ctx, array, default)
        }
    }
}

fn vec3_from_rhai_array(
    ctx: &rhai::NativeCallContext,
    array: rhai::Array,
    default: Option<&Vec3>,
) -> Result<Vec3, Box<EvalAltResult>> {
    match array.len() {
        2 => {
            let x = f64::from_dynamic(ctx, array[0].clone(), None)?;
            let y = f64::from_dynamic(ctx, array[1].clone(), None)?;
            let z = default.map(|d| d.z).unwrap_or(0.0);
            Ok(Vec3 { x, y, z })
        }
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

impl FromDynamic for Axis {
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        d: rhai::Dynamic,
        _default: Option<&Self>,
    ) -> Result<Self, Box<EvalAltResult>> {
        let out = if let Some(v) = d.clone().try_cast() {
            Some(v)
        } else if let Ok(v) = Vec3::from_dynamic(ctx, d.clone(), None) {
            let v = v.try_into().map_err(|e| {
                Box::new(EvalAltResult::ErrorMismatchDataType(
                    format!("conversion failed: {e}"),
                    "vec3 with reasonable length".to_string(),
                    ctx.position(),
                ))
            })?;
            Some(v)
        } else if let Ok(s) = d.clone().into_immutable_string() {
            match s.as_str() {
                "x" | "X" => Some(Axis::X),
                "y" | "Y" => Some(Axis::Y),
                "z" | "Z" => Some(Axis::Z),
                _ => None,
            }
        } else if let Ok(c) = d.clone().as_char() {
            match c {
                'x' | 'X' => Some(Axis::X),
                'y' | 'Y' => Some(Axis::Y),
                'z' | 'Z' => Some(Axis::Z),
                _ => None,
            }
        } else if let Some(t) = d.clone().try_cast::<Tree>() {
            match &*t {
                TreeOp::Input(Var::X) => Some(Axis::X),
                TreeOp::Input(Var::Y) => Some(Axis::Y),
                TreeOp::Input(Var::Z) => Some(Axis::Z),
                _ => None,
            }
        } else {
            None
        };

        out.ok_or_else(|| {
            EvalAltResult::ErrorMismatchDataType(
                "vec3 or [float; 3]".to_string(),
                d.type_name().to_owned(),
                ctx.position(),
            )
            .into()
        })
    }
}

fn print_axis(t: Axis) -> String {
    if t == Axis::X {
        "axis(\"x\")".to_owned()
    } else if t == Axis::Y {
        "axis(\"y\")".to_owned()
    } else if t == Axis::Z {
        "axis(\"z\")".to_owned()
    } else {
        let v = t.vec();
        format!("axis([{}, {}, {}])", v.x, v.y, v.z)
    }
}

fn register_axis(engine: &mut rhai::Engine) {
    engine
        .register_type_with_name::<Axis>("Axis")
        .register_fn("to_string", |t: &mut Axis| print_axis(*t))
        .register_fn(
            "axis",
            |ctx: rhai::NativeCallContext,
             v: rhai::Dynamic|
             -> Result<Axis, Box<EvalAltResult>> {
                Axis::from_dynamic(&ctx, v, None)
            },
        );
}

impl FromDynamic for Plane {
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        d: rhai::Dynamic,
        _default: Option<&Self>,
    ) -> Result<Self, Box<EvalAltResult>> {
        let r = if let Some(v) = d.clone().try_cast() {
            Some(v)
        } else if let Ok(axis) = Axis::from_dynamic(ctx, d.clone(), None) {
            Some(Self { axis, offset: 0.0 })
        } else if let Ok(s) = d.clone().into_immutable_string() {
            match s.as_str() {
                "xy" | "XY" => Some(Plane::XY),
                "yz" | "YZ" => Some(Plane::YZ),
                "zx" | "ZX" => Some(Plane::ZX),
                _ => None,
            }
        } else {
            None
        };

        r.ok_or_else(|| {
            EvalAltResult::ErrorMismatchDataType(
                "axis or plane name".to_owned(),
                d.type_name().to_owned(),
                ctx.position(),
            )
            .into()
        })
    }
}

fn register_plane(engine: &mut rhai::Engine) {
    engine
        .register_type_with_name::<Plane>("Plane")
        .register_fn("to_string", |t: &mut Plane| {
            if t == &Plane::XY {
                "plane(\"xy\")".to_owned()
            } else if t == &Plane::YZ {
                "plane(\"yz\")".to_owned()
            } else if t == &Plane::ZX {
                "plane(\"zx\")".to_owned()
            } else {
                let ax = print_axis(t.axis);
                if t.offset == 0.0 {
                    format!("plane({ax})")
                } else {
                    format!("plane({ax}, {})", t.offset)
                }
            }
        })
        .register_fn(
            "plane",
            |ctx: rhai::NativeCallContext,
             v: rhai::Dynamic|
             -> Result<Plane, Box<EvalAltResult>> {
                Plane::from_dynamic(&ctx, v, None)
            },
        )
        .register_fn(
            "plane",
            |ctx: rhai::NativeCallContext,
             v: rhai::Dynamic,
             offset: f64|
             -> Result<Plane, Box<EvalAltResult>> {
                let plane = Plane::from_dynamic(&ctx, v, None)?;
                Ok(Plane {
                    axis: plane.axis,
                    offset,
                })
            },
        );
}

#[cfg(test)]
mod test {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn type_constructors() {
        let mut e = rhai::Engine::new();
        register(&mut e);
        assert_eq!(
            e.eval::<Vec2>("vec2([5, 10.0])").unwrap(),
            Vec2::new(5.0, 10.0),
        );
        assert_eq!(
            e.eval::<Vec3>("vec3([1, 2, 3])").unwrap(),
            Vec3::new(1.0, 2.0, 3.0),
        );
        assert_eq!(
            e.eval::<Vec3>("vec3([1, 2])").unwrap(),
            Vec3::new(1.0, 2.0, 0.0),
        );

        assert_eq!(e.eval::<Axis>("axis([1, 0])").unwrap(), Axis::X);
        assert_eq!(e.eval::<Axis>("axis([0, 0, 1])").unwrap(), Axis::Z);
        assert_eq!(e.eval::<Axis>("axis('z')").unwrap(), Axis::Z);
        assert_eq!(e.eval::<Axis>("axis(\"z\")").unwrap(), Axis::Z);
        assert!(e.eval::<Axis>("axis([0, 0, 0])").is_err());

        assert_eq!(e.eval::<Plane>("plane([1, 0])").unwrap(), Plane::YZ);
        assert_eq!(
            e.eval::<Plane>("plane([1, 0], 0.5)").unwrap(),
            Plane {
                axis: Axis::X,
                offset: 0.5
            }
        );
        assert_eq!(e.eval::<Plane>("plane(\"yz\")").unwrap(), Plane::YZ);
    }

    #[test]
    fn type_printing() {
        let mut e = rhai::Engine::new();
        register(&mut e);
        let lines = Arc::new(Mutex::new(vec![]));
        let lines_ = lines.clone();
        e.on_print(move |s| lines_.lock().unwrap().push(s.to_string()));

        e.eval::<()>("print(vec2(1, 0))").unwrap();
        assert_eq!(
            lines.lock().unwrap().last().map(|s| s.as_str()),
            Some("[1, 0]")
        );

        e.eval::<()>("print(vec3(1, 2, 3))").unwrap();
        assert_eq!(
            lines.lock().unwrap().last().map(|s| s.as_str()),
            Some("[1, 2, 3]")
        );

        e.eval::<()>("print(axis([1, 0]))").unwrap();
        assert_eq!(
            lines.lock().unwrap().last().map(|s| s.as_str()),
            Some("axis(\"x\")")
        );

        e.eval::<()>("print(plane([1, 0]))").unwrap();
        assert_eq!(
            lines.lock().unwrap().last().map(|s| s.as_str()),
            Some("plane(\"yz\")")
        );
    }

    #[test]
    fn type_ops() {
        let mut e = rhai::Engine::new();
        register(&mut e);
        let v = e.eval::<Vec2>("-vec2(1, 2)").unwrap();
        assert_eq!(v.x, -1.0);
        assert_eq!(v.y, -2.0);
    }
}
