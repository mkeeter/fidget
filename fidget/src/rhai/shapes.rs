//! Tools for using [`fidget::shapes`](crate::shapes) in Rhai
use crate::{
    context::Tree,
    rhai::FromDynamic,
    shapes::{
        ShapeVisitor,
        types::{Plane, Type, Value, Vec3, validate},
        visit_shapes,
    },
};
use facet::{ConstTypeId, Facet};
use rhai::{EvalAltResult, NativeCallContext};
use strum::IntoDiscriminant;

/// Registers [all shapes](crate::shapes) with the engine
pub fn register(engine: &mut rhai::Engine) {
    struct EngineVisitor<'a>(&'a mut rhai::Engine);
    impl ShapeVisitor for EngineVisitor<'_> {
        fn visit<
            T: Facet<'static> + Clone + Send + Sync + Into<Tree> + 'static,
        >(
            &mut self,
        ) {
            register_shape::<T>(self.0);
        }
    }
    let mut v = EngineVisitor(engine);
    visit_shapes(&mut v);
}

impl Type {
    /// Build a [`Value`] from a dynamic value and tag hint
    ///
    /// The resulting type is guaranteed to match the tag.
    fn into_type(
        self,
        ctx: &NativeCallContext,
        v: rhai::Dynamic,
        default: Option<Value>,
    ) -> Result<Value, Box<EvalAltResult>> {
        let default = default.as_ref();
        let out = match self {
            Type::Float => {
                from_dynamic_with_hint(ctx, v, default, Value::Float)?
            }
            Type::Vec2 => from_dynamic_with_hint(ctx, v, default, Value::Vec2)?,
            Type::Vec3 => from_dynamic_with_hint(ctx, v, default, Value::Vec3)?,
            Type::Vec4 => from_dynamic_with_hint(ctx, v, default, Value::Vec4)?,
            Type::Tree => from_dynamic_with_hint(ctx, v, default, Value::Tree)?,
            Type::Axis => from_dynamic_with_hint(ctx, v, default, Value::Axis)?,
            Type::Plane => {
                from_dynamic_with_hint(ctx, v, default, Value::Plane)?
            }
            Type::VecTree => {
                from_dynamic_with_hint(ctx, v, default, Value::VecTree)?
            }
        };
        Ok(out)
    }
}

fn from_dynamic_with_hint<T: FromDynamic>(
    ctx: &NativeCallContext,
    v: rhai::Dynamic,
    default: Option<&Value>,
    b: fn(T) -> Value,
) -> Result<Value, Box<EvalAltResult>>
where
    for<'a> &'a Value: TryInto<&'a T>,
{
    <_>::from_dynamic(ctx, v, default.and_then(|d| d.try_into().ok())).map(b)
}

impl Value {
    /// Converts an arbitrary [`rhai::Dynamic`] value into a [`Value`]
    fn from_dynamic(
        ctx: &NativeCallContext,
        v: rhai::Dynamic,
        default: Option<Value>,
    ) -> Result<Value, Box<EvalAltResult>> {
        // This chain is ordered to prevent implicit conversions, e.g. we check
        // `Vec<Tree>` before `Tree` becaues Tree::from_dynamic` will
        // automatically collapse a `[Tree]` list.
        let default = default.as_ref();
        from_dynamic_with_hint(ctx, v.clone(), default, Value::Float)
            .or_else(|_| {
                from_dynamic_with_hint(ctx, v.clone(), default, Value::Vec2)
            })
            .or_else(|_| {
                from_dynamic_with_hint(ctx, v.clone(), default, Value::Vec3)
            })
            .or_else(|_| {
                from_dynamic_with_hint(ctx, v.clone(), default, Value::Vec4)
            })
            .or_else(|_| {
                from_dynamic_with_hint(ctx, v.clone(), default, Value::VecTree)
            })
            .or_else(|_| {
                from_dynamic_with_hint(ctx, v.clone(), default, Value::Tree)
            })
            .or_else(|_| {
                from_dynamic_with_hint(ctx, v.clone(), default, Value::Axis)
            })
            .or_else(|_| {
                from_dynamic_with_hint(ctx, v.clone(), default, Value::Plane)
            })
            .map_err(|_| {
                Box::new(rhai::EvalAltResult::ErrorMismatchDataType(
                    "any Type-compatible value".to_string(),
                    v.type_name().to_string(),
                    ctx.position(),
                ))
            })
    }
}

/// Register a shape-building type into a Rhai runtime
fn register_shape<
    T: Facet<'static> + Clone + Send + Sync + Into<Tree> + 'static,
>(
    engine: &mut rhai::Engine,
) {
    let s = validate::<T>(); // panic if the type is invalid

    use heck::ToSnakeCase;

    // Get shape type (CamelCase) and builder (snake_case) names
    let name = T::SHAPE.to_string();
    let name_lower = name.to_snake_case();

    engine.register_fn(&name_lower, build_from_map::<T>);
    let mut skip_ordered_builder = false;

    // Special handling for transform-shaped functions
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
    }
    if tree_count == 2 && s.fields.len() == 2 {
        engine.register_fn(&name_lower, build_binary::<T>);
        skip_ordered_builder = true;
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
        skip_ordered_builder = true;
    }

    let mut count = enum_map::EnumMap::<Type, usize>::default();
    let mut default_count = 0;
    for f in s.fields {
        let t = Type::try_from(f.shape().id).unwrap();
        count[t] += 1;
        if f.vtable.default_fn.is_some() {
            default_count += 1;
        }
    }
    if count.iter().all(|(_k, v)| *v <= 1) {
        // Generate typed unique builders for all possible argument counts
        let field_count = s.fields.len();
        let min_field_count = field_count - default_count;
        for n in min_field_count..=field_count {
            match n {
                0 => engine.register_fn(&name_lower, build_unique0::<T>),
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
        skip_ordered_builder = true;
    }

    if !skip_ordered_builder {
        match s.fields.len() {
            1 => engine.register_fn(&name_lower, build_ordered1::<T>),
            2 => engine.register_fn(&name_lower, build_ordered2::<T>),
            3 => engine.register_fn(&name_lower, build_ordered3::<T>),
            4 => engine.register_fn(&name_lower, build_ordered4::<T>),
            5 => engine.register_fn(&name_lower, build_ordered5::<T>),
            6 => engine.register_fn(&name_lower, build_ordered6::<T>),
            7 => engine.register_fn(&name_lower, build_ordered7::<T>),
            8 => engine.register_fn(&name_lower, build_ordered8::<T>),
            _ => engine,
        };
    }
}

/// Builds a transform-shaped `T` from a Rhai map
///
/// # Panics
/// `T` must be a `struct` with exactly one `Tree` field; the other fields must
/// be [`Type`]-compatible.
fn build_transform<T: Facet<'static> + Into<Tree>>(
    ctx: NativeCallContext,
    t: rhai::Dynamic,
    m: rhai::Map,
) -> Result<Tree, Box<EvalAltResult>> {
    let mut t = Some(Tree::from_dynamic(&ctx, t, None)?);

    let mut builder = facet::Partial::alloc_shape(T::SHAPE).unwrap();
    let facet::Type::User(facet::UserType::Struct(shape)) = T::SHAPE.ty else {
        panic!("must build a struct");
    };

    for (i, f) in shape.fields.iter().enumerate() {
        let tag = Type::try_from(f.shape().id).unwrap();
        if matches!(tag, Type::Tree) {
            let t = t.take().unwrap();
            builder.set_nth_field(i, t).unwrap();
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
        let d = f
            .vtable
            .default_fn
            .map(|df| unsafe { tag.build_from_default_fn(df) });
        let v = tag.into_type(&ctx, v, d)?;
        v.put(&mut builder, i);
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

/// Builds a `T`, which should be a binary function of two trees
///
/// # Panics
/// `T` must be a `struct` with two fields, which both must be `Tree`s
fn build_binary<T: Facet<'static> + Into<Tree>>(
    ctx: NativeCallContext,
    a: rhai::Dynamic,
    b: rhai::Dynamic,
) -> Result<Tree, Box<EvalAltResult>> {
    let facet::Type::User(facet::UserType::Struct(shape)) = T::SHAPE.ty else {
        panic!("must build a struct");
    };
    assert_eq!(shape.fields.len(), 2);
    assert!(
        shape
            .fields
            .iter()
            .all(|f| f.shape().id == ConstTypeId::of::<Tree>())
    );

    let a = Tree::from_dynamic(&ctx, a, None)?;
    let b = Tree::from_dynamic(&ctx, b, None)?;

    let mut builder = facet::Partial::alloc_shape(T::SHAPE).unwrap();

    builder.set_nth_field(0, a).unwrap();
    builder.set_nth_field(1, b).unwrap();

    let t: T = builder.build().unwrap().materialize().unwrap();
    Ok(t.into())
}

/// Builds a `T` from a Rhai map
///
/// # Panics
/// `T` must be a `struct` with fields that are [`Type`]-compatible
fn build_from_map<T: Facet<'static> + Into<Tree>>(
    ctx: NativeCallContext,
    m: rhai::Map,
) -> Result<Tree, Box<EvalAltResult>> {
    let mut builder = facet::Partial::alloc_shape(T::SHAPE).unwrap();
    let facet::Type::User(facet::UserType::Struct(shape)) = T::SHAPE.ty else {
        panic!("must build a struct");
    };
    for (i, f) in shape.fields.iter().enumerate() {
        let tag = Type::try_from(f.shape().id).unwrap();

        let d = f
            .vtable
            .default_fn
            .map(|df| unsafe { tag.build_from_default_fn(df) });
        let v = if let Some(v) = m.get(f.name).cloned() {
            tag.into_type(&ctx, v, d)?
        } else if let Some(v) = d {
            v
        } else {
            return Err(EvalAltResult::ErrorRuntime(
                format!("field {} must be provided for {}", f.name, T::SHAPE)
                    .into(),
                ctx.position(),
            )
            .into());
        };

        v.put(&mut builder, i);
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
        fn $name<T: Facet<'static> + Into<Tree>>(
            ctx: NativeCallContext,
            $($v: rhai::Dynamic),*
        ) -> Result<Tree, Box<EvalAltResult>> {
            let facet::Type::User(facet::UserType::Struct(shape)) = T::SHAPE.ty
            else {
                panic!("must build a struct");
            };
            assert_eq!(shape.fields[0].shape().id, ConstTypeId::of::<Vec<Tree>>());
            assert_eq!(shape.fields.len(), 1);

            let mut builder = facet::Partial::alloc_shape(&T::SHAPE).unwrap();
            let v = vec![$(
                Tree::from_dynamic(&ctx, $v, None)?
            ),*];
            builder.set_nth_field(0, v).unwrap();

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
    ($name:ident$(,)? $($v:ident),*) => {
        /// Builds a `T` from a list of Rhai values of unique types
        ///
        /// # Panics
        /// `T` must be a `struct` with [`Type`]-compatible fields
        #[allow(clippy::too_many_arguments)]
        fn $name<T: Facet<'static> + Into<Tree>>(
            ctx: NativeCallContext,
            $($v: rhai::Dynamic),*
        ) -> Result<Tree, Box<EvalAltResult>> {
            // Build a map of our known values
            #[allow(unused_mut, reason = "0-item constructor")]
            let mut vs = enum_map::EnumMap::<Type, Option<Value>>::default();
            $(
                let v = Value::from_dynamic(&ctx, $v.clone(), None)?;
                let tag = v.discriminant();
                vs[tag] = Some(v);
            )*

            from_enum_map::<T>(ctx, vs)
        }
    }
}

fn from_enum_map<T: Facet<'static> + Into<Tree>>(
    ctx: NativeCallContext,
    mut vs: enum_map::EnumMap<Type, Option<Value>>,
) -> Result<Tree, Box<EvalAltResult>> {
    let facet::Type::User(facet::UserType::Struct(shape)) = T::SHAPE.ty else {
        panic!("must build a struct");
    };

    let mut builder = facet::Partial::alloc_shape(T::SHAPE).unwrap();

    // Track which fields the shape actually uses, for upgrades
    let mut has_ty = enum_map::EnumMap::<Type, bool>::default();
    for f in shape.fields.iter() {
        has_ty[Type::try_from(f.shape().id).unwrap()] = true;
    }

    for (i, f) in shape.fields.iter().enumerate() {
        let tag = Type::try_from(f.shape().id).unwrap();

        let d = f
            .vtable
            .default_fn
            .map(|df| unsafe { tag.build_from_default_fn(df) });
        let v = if let Some(v) = vs[tag].take() {
            v
        } else if tag == Type::Vec3 // Vec2 -> Vec3 upgrade
            && vs[Type::Vec2].is_some()
            && !has_ty[Type::Vec2]
            && d.is_some()
        {
            let Some(Value::Vec2(v)) = vs[Type::Vec2].take() else {
                unreachable!()
            };
            let Some(Value::Vec3(d)) = d else {
                unreachable!()
            };
            Value::Vec3(Vec3 {
                x: v.x,
                y: v.y,
                z: d.z,
            })
        } else if tag == Type::Plane // Plane -> Axis upgrade
            && vs[Type::Axis].is_some()
            && !has_ty[Type::Axis]
        {
            let Some(Value::Axis(axis)) = vs[Type::Axis].take() else {
                unreachable!()
            };
            Value::Plane(Plane { axis, offset: 0.0 })
        } else if let Some(v) = d {
            v
        } else {
            return Err(EvalAltResult::ErrorRuntime(
                format!("missing argument of type {}", f.shape()).into(),
                ctx.position(),
            )
            .into());
        };
        v.put(&mut builder, i);
    }

    if let Some((k, _v)) = vs.iter().find(|(_k, v)| v.is_some()) {
        return Err(EvalAltResult::ErrorRuntime(
            format!("shape does not have an argument of type {k:?}").into(),
            ctx.position(),
        )
        .into());
    }

    let t: T = builder.build().unwrap().materialize().unwrap();
    Ok(t.into())
}

unique!(build_unique0);
unique!(build_unique1, a);
unique!(build_unique2, a, b);
unique!(build_unique3, a, b, c);
unique!(build_unique4, a, b, c, d);
unique!(build_unique5, a, b, c, d, e);
unique!(build_unique6, a, b, c, d, e, f);
unique!(build_unique7, a, b, c, d, e, f, g);
unique!(build_unique8, a, b, c, d, e, f, g, h);

////////////////////////////////////////////////////////////////////////////////

macro_rules! ordered {
    ($name:ident$(,)? $($v:ident),*) => {
        /// Builds a `T` from an ordered list of Rhai values
        ///
        /// # Panics
        /// `T` must be a `struct` with [`Type`]-compatible fields
        #[allow(clippy::too_many_arguments)]
        fn $name<T: Facet<'static> + Into<Tree>>(
            ctx: NativeCallContext,
            $($v: rhai::Dynamic),*
        ) -> Result<Tree, Box<EvalAltResult>> {
            let mut vs = vec![];
            $(
                let v = Value::from_dynamic(&ctx, $v.clone(), None)?;
                vs.push(v);
            )*

            from_value_list::<T>(ctx, vs)
        }
    }
}

fn from_value_list<T: Facet<'static> + Into<Tree>>(
    ctx: NativeCallContext,
    vs: Vec<Value>,
) -> Result<Tree, Box<EvalAltResult>> {
    let facet::Type::User(facet::UserType::Struct(shape)) = T::SHAPE.ty else {
        panic!("must build a struct");
    };
    assert_eq!(vs.len(), shape.fields.len(), "invalid field count");

    let mut builder = facet::Partial::alloc_shape(T::SHAPE).unwrap();

    for (i, (f, v)) in shape.fields.iter().zip(vs).enumerate() {
        let expected_tag = Type::try_from(f.shape().id).unwrap();
        let actual_tag: Type = Type::from(&v);
        if actual_tag != expected_tag {
            return Err(EvalAltResult::ErrorMismatchDataType(
                expected_tag.to_string(),
                actual_tag.to_string(),
                ctx.position(),
            )
            .into());
        }

        v.put(&mut builder, i);
    }

    let t: T = builder.build().unwrap().materialize().unwrap();
    Ok(t.into())
}

ordered!(build_ordered1, a);
ordered!(build_ordered2, a, b);
ordered!(build_ordered3, a, b, c);
ordered!(build_ordered4, a, b, c, d);
ordered!(build_ordered5, a, b, c, d, e);
ordered!(build_ordered6, a, b, c, d, e, f);
ordered!(build_ordered7, a, b, c, d, e, f, g);
ordered!(build_ordered8, a, b, c, d, e, f, g, h);

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::shapes::{types::Vec2, *};
    use crate::{Context, context::Op, var::Var};

    #[test]
    fn circle_builder() {
        let mut e = rhai::Engine::new();
        register_shape::<Circle>(&mut e);
        e.build_type::<Vec2>();
        assert!(
            e.eval::<Tree>("circle(#{ center: vec2(1, 2), radius: 3 })")
                .is_ok()
        );

        assert!(
            e.eval::<Tree>("circle(#{ center: 3.0, radius: 3 })")
                .is_err()
        );
        assert!(
            e.eval::<Tree>(
                "circle(#{ center: vec2(\"omg\", \"wtf\"), radius: 3 })"
            )
            .is_err()
        );
        assert!(
            e.eval::<Tree>("circle(#{ radius: 4, xy: vec2(1, 2) })")
                .is_err()
        );

        assert!(e.eval::<Tree>("circle([1, 2], 3)").is_ok());
    }

    #[test]
    fn circle_builder_default() {
        let mut e = rhai::Engine::new();
        register_shape::<Circle>(&mut e);
        e.build_type::<Vec2>();
        assert!(e.eval::<Tree>("circle(#{ center: vec2(1, 2)})").is_ok());
        assert!(e.eval::<Tree>("circle(#{ radius: 1})").is_ok());

        assert!(e.eval::<Tree>("circle(3)").is_ok());
        assert!(e.eval::<Tree>("circle([1, 2])").is_ok());

        assert!(e.eval::<Tree>("circle()").is_ok());
    }

    #[test]
    fn scale_and_move_defaults() {
        // Move should default to 0 on the Z axis
        let e = crate::rhai::engine();
        let mut ctx = Context::new();
        let v = e.eval("z.move([1, 1])").unwrap();
        let root = ctx.import(&v);
        assert_eq!(ctx.get_op(root).unwrap(), &Op::Input(Var::Z));

        // Scale should default to 1 on the Z axis
        let v = e.eval("z.scale([1, 1])").unwrap();
        let root = ctx.import(&v);
        assert_eq!(ctx.get_op(root).unwrap(), &Op::Input(Var::Z));
    }

    #[test]
    fn string_to_plane() {
        let e = crate::rhai::engine();
        let v = e.eval("x.reflect(\"yz\")").unwrap(); // plane
        let mut ctx = Context::new();
        let root = ctx.import(&v);
        // XXX this could be reduced to -X, but that requires automatic affine
        // detection, which isn't yet implemented.
        let expected = ctx.import(&(Tree::x() - Tree::x() * 2));
        assert_eq!(root, expected);

        let v = e.eval("x.reflect(\"x\")").unwrap(); // axis -> plane
        let root = ctx.import(&v);
        assert_eq!(root, expected);
    }

    #[test]
    fn rect_builder_ordered() {
        let e = crate::rhai::engine();
        assert!(e.eval::<Tree>("rectangle([0,0], [1,1])").is_ok());
        assert!(e.eval::<Tree>("rectangle([0,0], [1,1,1])").is_err());
    }

    #[test]
    fn extrude_builder_ordered() {
        let e = crate::rhai::engine();
        assert!(e.eval::<Tree>("extrude_z(x, 0, 1)").is_ok());
    }
}
