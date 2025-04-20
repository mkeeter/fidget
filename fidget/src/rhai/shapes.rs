//! Tools for using [`fidget::shapes`](crate::shapes) in Rhai
use crate::{
    context::Tree,
    rhai::FromDynamic,
    shapes::{Vec2, Vec3, Vec4},
};
use facet::{ConstTypeId, Facet};
use rhai::{EvalAltResult, NativeCallContext};

/// Register all shapes with the Rhai engine
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

/// Type used for Rust-Rhai interop
#[derive(strum::EnumDiscriminants)]
#[strum_discriminants(name(TypeTag), derive(enum_map::Enum))]
enum Type {
    Float(f64),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
    Tree(Tree),
    VecTree(Vec<Tree>),
}

/// Convert from a Facet type to a tag
impl TryFrom<facet::ConstTypeId> for TypeTag {
    type Error = facet::ConstTypeId;
    fn try_from(t: facet::ConstTypeId) -> Result<Self, Self::Error> {
        if t == ConstTypeId::of::<f64>() {
            Ok(Self::Float)
        } else if t == ConstTypeId::of::<Vec2>() {
            Ok(Self::Vec2)
        } else if t == ConstTypeId::of::<Vec3>() {
            Ok(Self::Vec3)
        } else if t == ConstTypeId::of::<Vec4>() {
            Ok(Self::Vec4)
        } else if t == ConstTypeId::of::<Tree>() {
            Ok(Self::Tree)
        } else if t == ConstTypeId::of::<Vec<Tree>>() {
            Ok(Self::VecTree)
        } else {
            Err(t)
        }
    }
}

impl Type {
    /// Build a [`Type`] from a dynamic value and tag hint
    ///
    /// The resulting type is guaranteed to match the tag.
    fn from_dynamic(
        ctx: &NativeCallContext,
        v: rhai::Dynamic,
        hint: TypeTag,
    ) -> Result<Self, Box<EvalAltResult>> {
        let out = match hint {
            TypeTag::Float => {
                let v = f64::from_dynamic(ctx, v)?;
                Self::Float(v)
            }
            TypeTag::Vec2 => {
                let v = Vec2::from_dynamic(ctx, v)?;
                Self::Vec2(v)
            }
            TypeTag::Vec3 => {
                let v = Vec3::from_dynamic(ctx, v)?;
                Self::Vec3(v)
            }
            TypeTag::Vec4 => {
                let v = Vec4::from_dynamic(ctx, v)?;
                Self::Vec4(v)
            }
            TypeTag::Tree => {
                let v = Tree::from_dynamic(ctx, v)?;
                Self::Tree(v)
            }
            TypeTag::VecTree => {
                let v = <Vec<Tree>>::from_dynamic(ctx, v)?;
                Self::VecTree(v)
            }
        };
        Ok(out)
    }

    /// Puts the type into an in-progress builder
    ///
    /// The builder must have a selected field, which is then popped
    ///
    /// # Panics
    /// If the currently-selected builder field does not match our type
    fn put(self, builder: facet::Wip) -> facet::Wip {
        match self {
            Type::Float(v) => builder.put(v),
            Type::Vec2(v) => builder.put(v),
            Type::Vec3(v) => builder.put(v),
            Type::Vec4(v) => builder.put(v),
            Type::Tree(v) => builder.put(v),
            Type::VecTree(v) => builder.put(v),
        }
        .unwrap()
        .pop()
        .unwrap()
    }
}

/// Register a type into a Rhai runtime
fn register_one<T: Facet + Clone + Send + Sync + Into<Tree> + 'static>(
    engine: &mut rhai::Engine,
) {
    let s = validate_type::<T>(); // panic if the type is invalid

    use heck::ToSnakeCase;

    // Get shape type (CamelCase) and builder (snake_case) names
    let name = T::SHAPE.to_string();
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

/// Checks whether `T`'s fields are all [`Type`]-compatible.
fn validate_type<T: Facet>() -> facet::Struct {
    // TODO could we make this `const`?
    let facet::Def::Struct(s) = T::SHAPE.def else {
        panic!("must be a struct-shaped type");
    };
    for f in s.fields {
        if TypeTag::try_from(f.shape().id).is_err() {
            panic!("unknown type {}", f.shape());
        }
    }
    s
}

/// Builds a transform-shaped `T` from a Rhai map
///
/// # Panics
/// `T` must be a `struct` with exactly one `Tree` field; the other fields must
/// be [`Type`]-compatible.
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
        let tag = TypeTag::try_from(f.shape().id).unwrap();
        if matches!(tag, TypeTag::Tree) {
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
        let v = Type::from_dynamic(&ctx, v, tag)?;
        builder = v.put(builder.field(i).unwrap())
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
///
/// # Panics
/// `T` must be a `struct` with two fields; one must be a `Tree`, and the other
/// must be [`Type`]-compatible.
fn build_transform_one<T: Facet + Into<Tree>>(
    ctx: NativeCallContext,
    t: rhai::Dynamic,
    arg: rhai::Dynamic,
) -> Result<Tree, Box<EvalAltResult>> {
    let facet::Def::Struct(shape) = T::SHAPE.def else {
        panic!("must build a struct");
    };
    assert_eq!(shape.fields.len(), 2);

    let mut t = Some(Tree::from_dynamic(&ctx, t)?);
    let mut arg = Some(arg);

    let mut builder = facet::Wip::alloc::<T>();

    for (i, f) in shape.fields.iter().enumerate() {
        let tag = TypeTag::try_from(f.shape().id).unwrap();
        if matches!(tag, TypeTag::Tree) {
            let t = t.take().unwrap();
            builder = builder.field(i).unwrap().put(t).unwrap().pop().unwrap();
        } else {
            let v = Type::from_dynamic(&ctx, arg.take().unwrap(), tag)?;
            builder = v.put(builder.field(i).unwrap())
        }
    }

    let t: T = builder.build().unwrap().materialize().unwrap();
    Ok(t.into())
}

/// Builds a `T`, which should be a binary function of two trees
///
/// # Panics
/// `T` must be a `struct` with two fields, which both must be `Tree`s
fn build_binary<T: Facet + Into<Tree>>(
    ctx: NativeCallContext,
    a: rhai::Dynamic,
    b: rhai::Dynamic,
) -> Result<Tree, Box<EvalAltResult>> {
    let facet::Def::Struct(shape) = T::SHAPE.def else {
        panic!("must build a struct");
    };
    assert_eq!(shape.fields.len(), 2);
    assert!(shape
        .fields
        .iter()
        .all(|f| f.shape().id == ConstTypeId::of::<Tree>()));

    let a = Tree::from_dynamic(&ctx, a)?;
    let b = Tree::from_dynamic(&ctx, b)?;

    let mut builder = facet::Wip::alloc::<T>();

    builder = builder.field(0).unwrap().put(a).unwrap().pop().unwrap();
    builder = builder.field(1).unwrap().put(b).unwrap().pop().unwrap();

    let t: T = builder.build().unwrap().materialize().unwrap();
    Ok(t.into())
}

/// Builds a `T` from a Rhai map
///
/// # Panics
/// `T` must be a `struct` with fields that are [`Type`]-compatible
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

        let tag = TypeTag::try_from(f.shape().id).unwrap();
        let v = Type::from_dynamic(&ctx, v, tag)?;
        builder = v.put(builder.field(i).unwrap())
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
        /// Builds a `T` from a list of Rhai values
        ///
        /// # Panics
        /// `T` must be a `struct` one `Vec<Tree>` field
        #[allow(clippy::too_many_arguments)]
        fn $name<T: Facet + Into<Tree>>(
            ctx: NativeCallContext,
            $($v: rhai::Dynamic),*
        ) -> Result<Tree, Box<EvalAltResult>> {
            let facet::Def::Struct(shape) = T::SHAPE.def else {
                panic!("must build a struct");
            };
            assert_eq!(shape.fields[0].shape().id, ConstTypeId::of::<Vec<Tree>>());
            assert_eq!(shape.fields.len(), 1);

            let mut builder = facet::Wip::alloc::<T>();
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
