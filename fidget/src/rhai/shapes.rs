//! Tools for using [`fidget::shapes`](crate::shapes) in Rhai
use crate::{
    context::Tree,
    rhai::FromDynamic,
    shapes::{Vec2, Vec3, Vec4},
};
use facet::{ConstTypeId, Facet};
use rhai::{EvalAltResult, NativeCallContext};
use strum::IntoDiscriminant;

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

impl TypeTag {
    /// Build a [`Type`] from a dynamic value and tag hint
    ///
    /// The resulting type is guaranteed to match the tag.
    fn into_type(
        self,
        ctx: &NativeCallContext,
        v: rhai::Dynamic,
    ) -> Result<Type, Box<EvalAltResult>> {
        let out = match self {
            TypeTag::Float => Type::Float(<_>::from_dynamic(ctx, v)?),
            TypeTag::Vec2 => Type::Vec2(<_>::from_dynamic(ctx, v)?),
            TypeTag::Vec3 => Type::Vec3(<_>::from_dynamic(ctx, v)?),
            TypeTag::Vec4 => Type::Vec4(<_>::from_dynamic(ctx, v)?),
            TypeTag::Tree => Type::Tree(<_>::from_dynamic(ctx, v)?),
            TypeTag::VecTree => Type::VecTree(<_>::from_dynamic(ctx, v)?),
        };
        Ok(out)
    }
}

impl Type {
    fn from_dynamic(
        ctx: &NativeCallContext,
        v: rhai::Dynamic,
    ) -> Result<Type, Box<EvalAltResult>> {
        // This chain is ordered to prevent implicit conversions, e.g. we check
        // `Vec<Tree>` before `Tree` becaues Tree::from_dynamic` will
        // automaticalyl collapse a `[Tree]` list.
        <_>::from_dynamic(ctx, v.clone())
            .map(Type::Float)
            .or_else(|_| <_>::from_dynamic(ctx, v.clone()).map(Type::Vec2))
            .or_else(|_| <_>::from_dynamic(ctx, v.clone()).map(Type::Vec3))
            .or_else(|_| <_>::from_dynamic(ctx, v.clone()).map(Type::Vec4))
            .or_else(|_| <_>::from_dynamic(ctx, v.clone()).map(Type::VecTree))
            .or_else(|_| <_>::from_dynamic(ctx, v.clone()).map(Type::Tree))
            .map_err(|_| {
                Box::new(rhai::EvalAltResult::ErrorMismatchDataType(
                    "any Rust-compatible Type".to_string(),
                    v.type_name().to_string(),
                    ctx.position(),
                ))
            })
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
    let mut skip_build_unique = false;
    let tree_count = s
        .fields
        .iter()
        .filter(|t| t.shape().id == ConstTypeId::of::<Tree>())
        .count();
    if tree_count == 1
        && s.fields[0].shape().id == ConstTypeId::of::<Tree>()
        && s.fields
            .iter()
            .all(|f| f.shape().id != ConstTypeId::of::<Vec<Tree>>())
    {
        engine.register_fn(&name_lower, build_transform::<T>);
        if s.fields.len() == 2 {
            engine.register_fn(&name_lower, build_transform_one::<T>);
            skip_build_unique = true;
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

    let mut count = enum_map::EnumMap::<TypeTag, usize>::default();
    for f in s.fields {
        let t = TypeTag::try_from(f.shape().id).unwrap();
        count[t] += 1;
    }
    if !skip_build_unique && count.iter().all(|(_k, v)| *v <= 1) {
        match s.fields.len() {
            1 => engine.register_fn(&name_lower, build_unique1::<T>),
            2 => engine.register_fn(&name_lower, build_unique2::<T>),
            3 => engine.register_fn(&name_lower, build_unique3::<T>),
            4 => engine.register_fn(&name_lower, build_unique4::<T>),
            5 => engine.register_fn(&name_lower, build_unique5::<T>),
            6 => engine.register_fn(&name_lower, build_unique6::<T>),
            7 => engine.register_fn(&name_lower, build_unique7::<T>),
            8 => engine.register_fn(&name_lower, build_unique8::<T>),
            _ => engine,
        };
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
        let v = tag.into_type(&ctx, v)?;
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
            let v = tag.into_type(&ctx, arg.take().unwrap())?;
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
        let v = tag.into_type(&ctx, v)?;
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
        /// `T` must be a `struct` with a single `Vec<Tree>` field
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

macro_rules! unique {
    ($name:ident, $($v:ident),*) => {
        /// Builds a `T` from a list of Rhai values of unique types
        ///
        /// # Panics
        /// `T` must be a `struct` with [`Type`]-compatible fields
        #[allow(clippy::too_many_arguments)]
        fn $name<T: Facet + Into<Tree>>(
            ctx: NativeCallContext,
            $($v: rhai::Dynamic),*
        ) -> Result<Tree, Box<EvalAltResult>> {
            let facet::Def::Struct(shape) = T::SHAPE.def else {
                panic!("must build a struct");
            };

            // Build a map of our known values
            let mut vs = enum_map::EnumMap::<TypeTag, Option<Type>>::default();
            $(
                let $v = Type::from_dynamic(&ctx, $v)?;
                let tag = $v.discriminant();
                vs[tag] = Some($v);
            )*

            let mut builder = facet::Wip::alloc::<T>();
            for (i, f) in shape.fields.iter().enumerate() {
                let t = TypeTag::try_from(f.shape().id).unwrap();
                let Some(v) = vs[t].take() else {
                    return Err(EvalAltResult::ErrorRuntime(
                        format!("missing argument of type {}", f.shape())
                            .into(),
                        ctx.position(),
                    )
                    .into());
                };
                builder = v.put(builder.field(i).unwrap())
            }

            let t: T = builder.build().unwrap().materialize().unwrap();
            Ok(t.into())
        }
    }
}

unique!(build_unique1, a);
unique!(build_unique2, a, b);
unique!(build_unique3, a, b, c);
unique!(build_unique4, a, b, c, d);
unique!(build_unique5, a, b, c, d, e);
unique!(build_unique6, a, b, c, d, e, f);
unique!(build_unique7, a, b, c, d, e, f, g);
unique!(build_unique8, a, b, c, d, e, f, g, h);

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
