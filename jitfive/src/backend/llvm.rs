use std::collections::BTreeMap;

use crate::{
    error::Error,
    op::{BinaryChoiceOpcode, BinaryOpcode, UnaryOpcode},
    stage0::{NodeIndex, Op},
    stage1::{GroupIndex, Source},
    stage4::Stage4,
};

use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::intrinsics::Intrinsic;
use inkwell::values::FunctionValue;
use inkwell::AddressSpace;
use inkwell::OptimizationLevel;
use inkwell::{builder::Builder, values::FloatValue};

type FloatFunc = unsafe extern "C" fn(f32, f32, *const i32) -> f32;

fn recurse<'a, 'ctx>(
    t: &'a Stage4,
    g: GroupIndex,
    intrinsics: &'ctx Intrinsics,
    context: &'ctx Context,
    function: &'ctx FunctionValue<'ctx>,
    builder: &'ctx Builder<'ctx>,
    values: &mut BTreeMap<NodeIndex, FloatValue<'ctx>>,
) {
    let group = &t.groups[g];
    let unconditional =
        group.choices.iter().any(|c| matches!(c, Source::Both(..)));

    if !unconditional {}

    for &g in &group.children {
        recurse(t, g, intrinsics, context, function, builder, values);
    }
    for &n in &group.nodes {
        let op = t.ops[n];
        let value = match op.op {
            Op::Var(v) => match t.vars.get_by_index(v).unwrap().as_str() {
                "X" => function.get_nth_param(0).unwrap().into_float_value(),
                "Y" => function.get_nth_param(1).unwrap().into_float_value(),
                v => panic!("Unexpected var {:?}", v),
            },
            Op::Const(f) => context.f32_type().const_float(f),
            Op::Binary(op, a, b) => {
                let f = match op {
                    BinaryOpcode::Add => Builder::build_float_add,
                    BinaryOpcode::Mul => Builder::build_float_mul,
                };
                f(
                    builder,
                    values[&a],
                    values[&b],
                    &format!("n{}", usize::from(n)),
                )
            }

            Op::BinaryChoice(op, a, b, _) => {
                let i = match op {
                    BinaryChoiceOpcode::Min => intrinsics.min,
                    BinaryChoiceOpcode::Max => intrinsics.max,
                };
                builder
                    .build_call(
                        i,
                        &[values[&a].into(), values[&b].into()],
                        &format!("n{}", usize::from(n)),
                    )
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value()
            }
            Op::Unary(op, a) => {
                let call_intrinsic = |i| {
                    builder
                        .build_call(
                            i,
                            &[values[&a].into()],
                            &format!("call_n{}", usize::from(n)),
                        )
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                        .into_float_value()
                };
                match op {
                    UnaryOpcode::Neg => builder.build_float_neg(
                        values[&a],
                        &format!("n{}", usize::from(n)),
                    ),
                    UnaryOpcode::Abs => call_intrinsic(intrinsics.abs),
                    UnaryOpcode::Sqrt => call_intrinsic(intrinsics.sqrt),
                    UnaryOpcode::Cos => call_intrinsic(intrinsics.cos),
                    UnaryOpcode::Sin => call_intrinsic(intrinsics.sin),
                    UnaryOpcode::Exp => call_intrinsic(intrinsics.exp),
                    UnaryOpcode::Ln => call_intrinsic(intrinsics.ln),
                    UnaryOpcode::Recip => builder.build_float_div(
                        context.f32_type().const_float(1.0),
                        values[&a],
                        &format!("n{}", usize::from(n)),
                    ),
                    /*
                    UnaryOpcode::Tan => Builder::build_float_tan,
                    UnaryOpcode::Asin => Builder::build_float_asin,
                    UnaryOpcode::Acos => Builder::build_float_acos,
                    UnaryOpcode::Atan => Builder::build_float_atan,
                    */
                    op => panic!("No such opcode {:?}", op),
                }
            }
        };
        values.insert(n, value);
    }
}

/// Helper struct to store LLVM intrinsics for various operations
#[derive(Debug)]
struct Intrinsics<'ctx> {
    abs: FunctionValue<'ctx>,
    sqrt: FunctionValue<'ctx>,
    sin: FunctionValue<'ctx>,
    cos: FunctionValue<'ctx>,
    exp: FunctionValue<'ctx>,
    ln: FunctionValue<'ctx>,
    max: FunctionValue<'ctx>,
    min: FunctionValue<'ctx>,
}

pub fn to_jit_fn<'a, 'b>(
    t: &'a Stage4,
    context: &'b Context,
) -> Result<JitFunction<'b, FloatFunc>, Error> {
    let module = context.create_module("sum");
    let builder = context.create_builder();
    let execution_engine =
        module.create_jit_execution_engine(OptimizationLevel::Aggressive)?;

    let f32_type = context.f32_type();

    // Create intrinsics for special functions
    let get_unary_intrinsic = |name| {
        let intrinsic = Intrinsic::find(&format!("llvm.{}", name)).unwrap();
        intrinsic
            .get_declaration(&module, &[f32_type.into()])
            .unwrap()
    };
    // Create intrinsics for special functions
    let get_binary_intrinsic = |name| {
        let intrinsic = Intrinsic::find(&format!("llvm.{}", name)).unwrap();
        intrinsic
            .get_declaration(&module, &[f32_type.into(), f32_type.into()])
            .unwrap()
    };

    let intrinsics = Intrinsics {
        abs: get_unary_intrinsic("abs"),
        sqrt: get_unary_intrinsic("sqrt"),
        sin: get_unary_intrinsic("sin"),
        cos: get_unary_intrinsic("cos"),
        exp: get_unary_intrinsic("exp"),
        ln: get_unary_intrinsic("log"),
        max: get_binary_intrinsic("maxnum"),
        min: get_binary_intrinsic("minnum"),
    };

    // Build our main function
    let i32_type = context.i32_type();
    let i32_ptr_type = i32_type.ptr_type(AddressSpace::Const);
    let fn_type = f32_type.fn_type(
        &[f32_type.into(), f32_type.into(), i32_ptr_type.into()],
        false,
    );
    let function = module.add_function("sum", fn_type, None);

    let basic_block = context.append_basic_block(function, "entry");

    builder.position_at_end(basic_block);
    let mut values = BTreeMap::new();
    recurse(
        t,
        t.ops[t.root].group,
        &intrinsics,
        context,
        &function,
        &builder,
        &mut values,
    );

    builder.build_return(Some(&values[&t.root]));
    module.print_to_stderr();

    let out = unsafe { execution_engine.get_function("sum")? };
    Ok(out)
}
