//! Rhai bindings for the Fidget [`Tree`] type
use crate::FromDynamic;
use fidget_core::{
    context::{Tree, TreeOp},
    var::Var,
};
use rhai::{EvalAltResult, NativeCallContext};

impl FromDynamic for Tree {
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        d: rhai::Dynamic,
        _default: Option<&Tree>,
    ) -> Result<Self, Box<EvalAltResult>> {
        if let Some(t) = d.clone().try_cast::<Tree>() {
            Ok(t)
        } else if let Ok(v) = f64::from_dynamic(ctx, d.clone(), None) {
            Ok(Tree::constant(v))
        } else if let Ok(v) = <Vec<Tree>>::from_dynamic(ctx, d.clone(), None) {
            Ok(fidget_shapes::Union { input: v }.into())
        } else {
            Err(Box::new(rhai::EvalAltResult::ErrorMismatchDataType(
                "tree".to_string(),
                d.type_name().to_string(),
                ctx.call_position(),
            )))
        }
    }
}

impl FromDynamic for Vec<Tree> {
    fn from_dynamic(
        ctx: &rhai::NativeCallContext,
        d: rhai::Dynamic,
        _default: Option<&Vec<Tree>>,
    ) -> Result<Self, Box<EvalAltResult>> {
        if let Ok(d) = d.clone().into_array() {
            d.into_iter()
                .map(|v| Tree::from_dynamic(ctx, v, None))
                .collect::<Result<Vec<_>, _>>()
        } else {
            Err(Box::new(rhai::EvalAltResult::ErrorMismatchDataType(
                "Vec<tree>".to_string(),
                d.type_name().to_string(),
                ctx.call_position(),
            )))
        }
    }
}

fn register_tree(engine: &mut rhai::Engine) {
    engine
        .register_type_with_name::<Tree>("Tree")
        .register_fn("to_string", |t: &mut Tree| {
            match &**t {
                TreeOp::Input(Var::X) => "x",
                TreeOp::Input(Var::Y) => "y",
                TreeOp::Input(Var::Z) => "z",
                _ => "Tree(..)",
            }
            .to_owned()
        })
        .register_fn("remap", remap_xyz)
        .register_fn("remap", remap_xy);
}

/// Installs the [`Tree`] type into a Rhai engine, with various overloads
///
/// Also installs `axes() -> {x, y, z}`
pub fn register(engine: &mut rhai::Engine) {
    register_tree(engine);
    engine.register_fn("axes", || {
        use fidget_core::context::Tree;
        let mut out = rhai::Map::new();
        out.insert("x".into(), rhai::Dynamic::from(Tree::x()));
        out.insert("y".into(), rhai::Dynamic::from(Tree::y()));
        out.insert("z".into(), rhai::Dynamic::from(Tree::z()));
        out
    });

    macro_rules! register_binary_fns {
        ($op:literal, $name:ident, $engine:ident) => {
            $engine.register_fn($op, $name::tree_dyn);
            $engine.register_fn($op, $name::dyn_tree);
        };
    }
    macro_rules! register_unary_fns {
        ($op:literal, $name:ident, $engine:ident) => {
            $engine.register_fn($op, $name::tree);
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
    register_binary_fns!("atan2", atan2, engine);
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
    register_unary_fns!("ceil", ceil, engine);
    register_unary_fns!("floor", floor, engine);
    register_unary_fns!("round", round, engine);
    register_unary_fns!("-", neg, engine);

    // Ban comparison operators
    for op in ["==", "!=", "<", ">", "<=", ">="] {
        engine.register_fn(op, bad_cmp_tree_dyn);
        engine.register_fn(op, bad_cmp_dyn_tree);
    }
}

fn remap_xyz(
    ctx: NativeCallContext,
    shape: rhai::Dynamic,
    x: Tree,
    y: Tree,
    z: Tree,
) -> Result<Tree, Box<EvalAltResult>> {
    let shape = Tree::from_dynamic(&ctx, shape, None)?;
    Ok(shape.remap_xyz(x, y, z))
}

fn remap_xy(
    ctx: NativeCallContext,
    shape: rhai::Dynamic,
    x: Tree,
    y: Tree,
) -> Result<Tree, Box<EvalAltResult>> {
    let shape = Tree::from_dynamic(&ctx, shape, None)?;
    Ok(shape.remap_xyz(x, y, Tree::z()))
}

macro_rules! define_binary_fns {
    ($name:ident $(, $op:ident)?) => {
        mod $name {
            use super::*;
            use NativeCallContext;
            $(
            use std::ops::$op;
            )?
            pub fn tree_dyn(
                ctx: NativeCallContext,
                a: Tree,
                b: rhai::Dynamic,
            ) -> Result<Tree, Box<rhai::EvalAltResult>> {
                let b = Tree::from_dynamic(&ctx, b, None)?;
                Ok(a.$name(b))
            }
            pub fn dyn_tree(
                ctx: NativeCallContext,
                a: rhai::Dynamic,
                b: Tree,
            ) -> Result<Tree, Box<rhai::EvalAltResult>> {
                let a = Tree::from_dynamic(&ctx, a, None)?;
                Ok(a.$name(b))
            }
        }
    };
}

macro_rules! define_unary_fns {
    ($name:ident) => {
        mod $name {
            use super::*;
            pub fn tree(
                ctx: NativeCallContext,
                a: rhai::Dynamic,
            ) -> Result<Tree, Box<EvalAltResult>> {
                let a = Tree::from_dynamic(&ctx, a, None)?;
                Ok(a.$name())
            }
        }
    };
}

fn bad_cmp_tree_dyn(
    _ctx: NativeCallContext,
    _a: Tree,
    _b: rhai::Dynamic,
) -> Result<Tree, Box<rhai::EvalAltResult>> {
    let e = "cannot compare Tree types during function tracing";
    Err(e.into())
}

fn bad_cmp_dyn_tree(
    _ctx: NativeCallContext,
    _a: rhai::Dynamic,
    _b: Tree,
) -> Result<Tree, Box<rhai::EvalAltResult>> {
    let e = "cannot compare Tree types during function tracing";
    Err(e.into())
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
define_binary_fns!(atan2);
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
define_unary_fns!(floor);
define_unary_fns!(ceil);
define_unary_fns!(round);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn tree_build_print() {
        let mut e = rhai::Engine::new();
        register(&mut e);
        assert_eq!(e.eval::<String>("to_string(axes().x)").unwrap(), "x");
        assert_eq!(
            e.eval::<String>("to_string(axes().x + 1)").unwrap(),
            "Tree(..)"
        );
    }
}
