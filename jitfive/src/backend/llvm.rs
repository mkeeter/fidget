use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

use crate::{
    compiler::{Compiler, GroupIndex, NodeIndex, Op, Source},
    error::Error,
    op::{BinaryChoiceOpcode, BinaryOpcode, UnaryOpcode},
};

use inkwell::{
    attributes::{Attribute, AttributeLoc},
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    intrinsics::Intrinsic,
    module::{Linkage, Module},
    passes::PassBuilderOptions,
    targets::{
        CodeModel, InitializationConfig, RelocMode, Target, TargetMachine,
    },
    types::{
        BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FloatType,
        FunctionType, IntType, PointerType,
    },
    values::{
        BasicValue, BasicValueEnum, FloatValue, FunctionValue, IntValue,
        PointerValue,
    },
    AddressSpace, FloatPredicate, OptimizationLevel,
};
use log::info;

const LHS: u32 = 1;
const RHS: u32 = 2;

type FloatFunc = unsafe extern "C" fn(f32, f32, *const u32) -> f32;

/// Wrapper for an LLVM context
pub struct JitContext(Context);
impl JitContext {
    pub fn new() -> Self {
        Self(Context::create())
    }
}
impl Default for JitContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Common LLVM intrinsics
struct JitIntrinsics<'ctx> {
    abs: FunctionValue<'ctx>,
    sqrt: FunctionValue<'ctx>,
    minnum: FunctionValue<'ctx>,
    maxnum: FunctionValue<'ctx>,
}

/// Library of math functions, specialized to a given type
struct JitMath<'ctx, T> {
    /// `fn add(lhs: T, rhs: T) -> T`
    add: FunctionValue<'ctx>,

    /// `fn mul(lhs: T, rhs: T) -> T`
    mul: FunctionValue<'ctx>,

    /// `fn neg(a: T) -> T`
    neg: FunctionValue<'ctx>,

    /// `fn abs(a: T) -> T`
    abs: FunctionValue<'ctx>,

    /// `fn recip(a: T) -> T`
    recip: FunctionValue<'ctx>,

    /// `fn sqrt(a: T) -> T`
    sqrt: FunctionValue<'ctx>,

    /// `fn min(lhs: T, rhs: T, choice: u32, shift: u32) -> T`
    ///
    /// Right-shifts `choice` by `shift`, then checks its lowest two bits:
    /// - `0b01 => lhs`
    /// - `0b10 => rhs`
    /// - `0b11 => min(lhs, rhs)`
    /// (`0b00` is invalid, but will end up calling `fmin`)
    min: FunctionValue<'ctx>,

    /// `fn max(lhs: T, rhs: T, choice: u32, shift: u32) -> T`
    ///
    /// Right-shifts `choice` by `shift`, then checks its lowest two bits:
    /// - `0b01 => lhs`
    /// - `0b10 => rhs`
    /// - `0b11 => max(lhs, rhs)`
    /// (`0b00` is invalid, but will end up calling `fmax`)
    max: FunctionValue<'ctx>,

    _p: std::marker::PhantomData<&'ctx T>,
}

struct JitCore<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    float: JitType<'ctx>,
    always_inline: Attribute,
    i32_type: IntType<'ctx>,
}

/// Derived types associated with a particular evaluation mode
struct JitType<'ctx> {
    ty: BasicTypeEnum<'ctx>,
    binary_choice_fn: FunctionType<'ctx>,
    shape_fn: FunctionType<'ctx>,
}

impl<'ctx> JitType<'ctx> {
    fn new<T>(
        ty: T,
        i32_type: IntType<'ctx>,
        i32_const_ptr_type: PointerType<'ctx>,
    ) -> Self
    where
        T: BasicType<'ctx> + Copy + Clone,
        BasicMetadataTypeEnum<'ctx>: From<T>,
        BasicTypeEnum<'ctx>: From<T>,
    {
        let binary_choice_fn = ty.fn_type(
            &[
                ty.into(),       // lhs
                ty.into(),       // rhs
                i32_type.into(), // choice
                i32_type.into(), // shift
            ],
            false,
        );
        let shape_fn = ty
            .fn_type(&[ty.into(), ty.into(), i32_const_ptr_type.into()], false);
        Self {
            ty: ty.into(),
            binary_choice_fn,
            shape_fn,
        }
    }
}

impl<'ctx> JitCore<'ctx> {
    fn new(context: &'ctx Context) -> Self {
        let i32_type = context.i32_type();
        let f32_type = context.f32_type();
        let i32_const_ptr_type = i32_type.ptr_type(AddressSpace::Const);

        let float = JitType::new(f32_type, i32_type, i32_const_ptr_type);

        let module = context.create_module("shape");
        let builder = context.create_builder();

        let kind_id = Attribute::get_named_enum_kind_id("alwaysinline");
        let always_inline = context.create_enum_attribute(kind_id, 0);

        Self {
            context,
            module,
            builder,
            float,
            always_inline,
            i32_type,
        }
    }

    fn f32_type(&self) -> FloatType<'ctx> {
        self.float.ty.into_float_type()
    }

    fn get_unary_intrinsic_f(&self, name: &str) -> FunctionValue<'ctx> {
        let intrinsic = Intrinsic::find(&format!("llvm.{}", name)).unwrap();
        intrinsic
            .get_declaration(&self.module, &[self.f32_type().into()])
            .unwrap()
    }

    fn get_binary_intrinsic_f(&self, name: &str) -> FunctionValue<'ctx> {
        let intrinsic = Intrinsic::find(&format!("llvm.{}", name)).unwrap();
        intrinsic
            .get_declaration(
                &self.module,
                &[self.f32_type().into(), self.f32_type().into()],
            )
            .unwrap()
    }

    fn get_math_f(
        &self,
        intrinsics: &JitIntrinsics<'ctx>,
    ) -> JitMath<'ctx, Float<'ctx>> {
        let (f_min, lhs, rhs, _choice, _out) =
            self.binary_choice_op_prelude::<Float>("f_min");
        let fmin_result = self
            .builder
            .build_call(
                intrinsics.minnum,
                &[lhs.value.into(), rhs.value.into()],
                "both",
            )
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        self.builder.build_return(Some(&fmin_result));

        let (f_max, lhs, rhs, _choice, _out) =
            self.binary_choice_op_prelude::<Float>("f_max");
        let fmax_result = self
            .builder
            .build_call(
                intrinsics.maxnum,
                &[lhs.value.into(), rhs.value.into()],
                "both",
            )
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        self.builder.build_return(Some(&fmax_result));

        let (f_add, lhs, rhs) = self.binary_op_prelude::<Float>("f_add");
        let sum = self.builder.build_float_add(lhs.value, rhs.value, "sum");
        self.builder.build_return(Some(&sum));

        let (f_mul, lhs, rhs) = self.binary_op_prelude::<Float>("f_mul");
        let prod = self.builder.build_float_mul(lhs.value, rhs.value, "prod");
        self.builder.build_return(Some(&prod));

        let (f_neg, lhs) = self.unary_op_prelude::<Float>("f_neg");
        let neg = self.builder.build_float_neg(lhs.value, "neg");
        self.builder.build_return(Some(&neg));

        let (f_abs, lhs) = self.unary_op_prelude::<Float>("f_abs");
        let abs = self
            .builder
            .build_call(intrinsics.abs, &[lhs.value.into()], "abs")
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        self.builder.build_return(Some(&abs));

        let (f_sqrt, lhs) = self.unary_op_prelude::<Float>("f_sqrt");
        let sqrt = self
            .builder
            .build_call(intrinsics.sqrt, &[lhs.value.into()], "sqrt")
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        self.builder.build_return(Some(&sqrt));

        let (f_recip, lhs) = self.unary_op_prelude::<Float>("f_recip");
        let recip = self.builder.build_float_div(
            self.f32_type().const_float(1.0),
            lhs.value,
            "recip",
        );
        self.builder.build_return(Some(&recip));

        JitMath {
            add: f_add,
            min: f_min,
            max: f_max,
            mul: f_mul,
            neg: f_neg,
            abs: f_abs,
            sqrt: f_sqrt,
            recip: f_recip,
            _p: std::marker::PhantomData,
        }
    }

    fn get_math_i(
        &self,
        intrinsics: &JitIntrinsics<'ctx>,
    ) -> JitMath<'ctx, Interval<'ctx>> {
        let i_add = self.build_i_add();
        let i_mul = self.build_i_mul(intrinsics.minnum, intrinsics.maxnum);
        let i_neg = self.build_i_neg();
        let i_abs = self.build_i_abs(intrinsics.maxnum);
        let i_sqrt = self.build_i_sqrt(intrinsics.sqrt);

        JitMath {
            add: i_add,
            mul: i_mul,
            max: i_mul, // TODO
            min: i_mul, // TODO
            neg: i_neg,
            abs: i_abs,
            sqrt: i_sqrt,
            recip: i_mul, // TODO
            _p: std::marker::PhantomData,
        }
    }

    fn get_intrinsics(&self) -> JitIntrinsics<'ctx> {
        let abs = self.get_unary_intrinsic_f("fabs");
        let sqrt = self.get_unary_intrinsic_f("sqrt");
        let maxnum = self.get_binary_intrinsic_f("maxnum");
        let minnum = self.get_binary_intrinsic_f("minnum");

        JitIntrinsics {
            abs,
            sqrt,
            maxnum,
            minnum,
        }
    }

    fn build_i_add(&self) -> FunctionValue<'ctx> {
        let (f, lhs, rhs) = self.binary_op_prelude::<Interval>("i_add");
        let out_lo =
            self.builder.build_float_add(lhs.lower, rhs.lower, "out_lo");
        let out_hi =
            self.builder.build_float_add(lhs.upper, rhs.upper, "out_hi");
        self.build_return_i(Interval {
            lower: out_lo,
            upper: out_hi,
        });
        f
    }

    fn build_i_mul(
        &self,
        fmin: FunctionValue<'ctx>,
        fmax: FunctionValue<'ctx>,
    ) -> FunctionValue<'ctx> {
        // It ain't pretty, but it works: this is a manual port of
        // boost::interval's multiplication code into LLVM IR.
        let (f, lhs, rhs) = self.binary_op_prelude::<Interval>("i_mul");
        let zero = self.f32_type().const_float(0.0);

        let a0_lt_0 = || {
            self.builder.build_float_compare(
                FloatPredicate::OLT,
                lhs.lower,
                zero,
                "a0_lt_0",
            )
        };
        let a1_gt_0 = || {
            self.builder.build_float_compare(
                FloatPredicate::OGT,
                lhs.upper,
                zero,
                "a1_gt_0",
            )
        };
        let b1_gt_0 = || {
            self.builder.build_float_compare(
                FloatPredicate::OGT,
                rhs.upper,
                zero,
                "b1_gt_0",
            )
        };
        let b0_lt_0 = || {
            self.builder.build_float_compare(
                FloatPredicate::OLT,
                rhs.lower,
                zero,
                "b0_lt_0",
            )
        };
        let a0b1 =
            || self.builder.build_float_mul(lhs.lower, rhs.upper, "a0b1");
        let a1b0 =
            || self.builder.build_float_mul(lhs.upper, rhs.lower, "a1b0");
        let a0b0 =
            || self.builder.build_float_mul(lhs.lower, rhs.lower, "a0b0");
        let a1b1 =
            || self.builder.build_float_mul(lhs.upper, rhs.upper, "a1b1");

        let a0_lt_0_block = self.context.append_basic_block(f, "a0_lt_0_block");
        let a0_ge_0_block = self.context.append_basic_block(f, "a0_ge_0_block");
        self.builder.build_conditional_branch(
            a0_lt_0(),
            a0_lt_0_block,
            a0_ge_0_block,
        );
        {
            self.builder.position_at_end(a0_lt_0_block);
            let a1_gt_0_block =
                self.context.append_basic_block(f, "a1_gt_0_block");
            let a1_le_0_block =
                self.context.append_basic_block(f, "a1_le_0_block");
            self.builder.build_conditional_branch(
                a1_gt_0(),
                a1_gt_0_block,
                a1_le_0_block,
            );
            {
                self.builder.position_at_end(a1_gt_0_block);
                let b0_lt_0_block =
                    self.context.append_basic_block(f, "b0_lt_0_block");
                let b0_ge_0_block =
                    self.context.append_basic_block(f, "b0_ge_0_block");
                self.builder.build_conditional_branch(
                    b0_lt_0(),
                    b0_lt_0_block,
                    b0_ge_0_block,
                );
                {
                    self.builder.position_at_end(b0_lt_0_block);
                    let b1_gt_0_block =
                        self.context.append_basic_block(f, "b1_gt_0_block");
                    let b1_le_0_block =
                        self.context.append_basic_block(f, "b1_le_0_block");
                    self.builder.build_conditional_branch(
                        b1_gt_0(),
                        b1_gt_0_block,
                        b1_le_0_block,
                    );
                    {
                        // M * M
                        self.builder.position_at_end(b1_gt_0_block);

                        let lower = self
                            .builder
                            .build_call(
                                fmin,
                                &[a0b1().into(), a1b0().into()],
                                "min_result",
                            )
                            .try_as_basic_value()
                            .left()
                            .unwrap()
                            .into_float_value();
                        let upper = self
                            .builder
                            .build_call(
                                fmax,
                                &[a0b0().into(), a1b1().into()],
                                "max_result",
                            )
                            .try_as_basic_value()
                            .left()
                            .unwrap()
                            .into_float_value();

                        self.build_return_i(Interval { lower, upper });
                    }
                    {
                        // M * N
                        self.builder.position_at_end(b1_le_0_block);
                        self.build_return_i(Interval {
                            lower: a1b0(),
                            upper: a0b0(),
                        });
                    }
                }
                {
                    self.builder.position_at_end(b0_ge_0_block);
                    let b1_gt_0_block =
                        self.context.append_basic_block(f, "b1_gt_0_block");
                    let b1_le_0_block =
                        self.context.append_basic_block(f, "b1_le_0_block");
                    self.builder.build_conditional_branch(
                        b1_gt_0(),
                        b1_gt_0_block,
                        b1_le_0_block,
                    );
                    {
                        // M * P
                        self.builder.position_at_end(b1_gt_0_block);
                        self.build_return_i(Interval {
                            lower: a0b1(),
                            upper: a1b1(),
                        });
                    }
                    {
                        // M * Z
                        self.builder.position_at_end(b1_le_0_block);
                        self.build_return_i(Interval {
                            lower: zero,
                            upper: zero,
                        });
                    }
                }
            }
            {
                self.builder.position_at_end(a1_le_0_block);
                let b0_lt_0_block =
                    self.context.append_basic_block(f, "b0_lt_0_block");
                let b0_ge_0_block =
                    self.context.append_basic_block(f, "b0_ge_0_block");
                self.builder.build_conditional_branch(
                    b0_lt_0(),
                    b0_lt_0_block,
                    b0_ge_0_block,
                );
                {
                    self.builder.position_at_end(b0_lt_0_block);
                    let b1_gt_0_block =
                        self.context.append_basic_block(f, "b1_gt_0_block");
                    let b1_le_0_block =
                        self.context.append_basic_block(f, "b1_le_0_block");
                    self.builder.build_conditional_branch(
                        b1_gt_0(),
                        b1_gt_0_block,
                        b1_le_0_block,
                    );
                    {
                        // N * M
                        self.builder.position_at_end(b1_gt_0_block);
                        self.build_return_i(Interval {
                            lower: a0b1(),
                            upper: a0b0(),
                        });
                    }
                    {
                        // N * N
                        self.builder.position_at_end(b1_le_0_block);
                        self.build_return_i(Interval {
                            lower: a1b1(),
                            upper: a0b0(),
                        });
                    }
                }
                {
                    self.builder.position_at_end(b0_ge_0_block);
                    let b1_gt_0_block =
                        self.context.append_basic_block(f, "b1_gt_0_block");
                    let b1_le_0_block =
                        self.context.append_basic_block(f, "b1_le_0_block");
                    self.builder.build_conditional_branch(
                        b1_gt_0(),
                        b1_gt_0_block,
                        b1_le_0_block,
                    );
                    {
                        // N * P
                        self.builder.position_at_end(b1_gt_0_block);
                        self.build_return_i(Interval {
                            lower: a0b1(),
                            upper: a1b0(),
                        });
                    }
                    {
                        // N * Z
                        self.builder.position_at_end(b1_le_0_block);
                        self.build_return_i(Interval {
                            lower: zero,
                            upper: zero,
                        });
                    }
                }
            }
        }
        {
            self.builder.position_at_end(a0_ge_0_block);
            let a1_gt_0_block =
                self.context.append_basic_block(f, "a1_gt_0_block");
            let a1_le_0_block =
                self.context.append_basic_block(f, "a1_le_0_block");
            self.builder.build_conditional_branch(
                a1_gt_0(),
                a1_gt_0_block,
                a1_le_0_block,
            );
            {
                self.builder.position_at_end(a1_gt_0_block);
                let b0_lt_0_block =
                    self.context.append_basic_block(f, "b0_lt_0_block");
                let b0_ge_0_block =
                    self.context.append_basic_block(f, "b0_ge_0_block");
                self.builder.build_conditional_branch(
                    b0_lt_0(),
                    b0_lt_0_block,
                    b0_ge_0_block,
                );
                {
                    self.builder.position_at_end(b0_lt_0_block);
                    let b1_gt_0_block =
                        self.context.append_basic_block(f, "b1_gt_0_block");
                    let b1_le_0_block =
                        self.context.append_basic_block(f, "b1_le_0_block");
                    self.builder.build_conditional_branch(
                        b1_gt_0(),
                        b1_gt_0_block,
                        b1_le_0_block,
                    );
                    {
                        // P * M
                        self.builder.position_at_end(b1_gt_0_block);
                        self.build_return_i(Interval {
                            lower: a1b0(),
                            upper: a1b1(),
                        });
                    }
                    {
                        // P * N
                        self.builder.position_at_end(b1_le_0_block);
                        self.build_return_i(Interval {
                            lower: a1b0(),
                            upper: a0b1(),
                        });
                    }
                }
                {
                    self.builder.position_at_end(b0_ge_0_block);
                    let b1_gt_0_block =
                        self.context.append_basic_block(f, "b1_gt_0_block");
                    let b1_le_0_block =
                        self.context.append_basic_block(f, "b1_le_0_block");
                    self.builder.build_conditional_branch(
                        b1_gt_0(),
                        b1_gt_0_block,
                        b1_le_0_block,
                    );
                    {
                        // P * P
                        self.builder.position_at_end(b1_gt_0_block);
                        self.build_return_i(Interval {
                            lower: a0b0(),
                            upper: a1b1(),
                        });
                    }
                    {
                        // P * Z
                        self.builder.position_at_end(b1_le_0_block);
                        self.build_return_i(Interval {
                            lower: zero,
                            upper: zero,
                        });
                    }
                }
            }
            {
                // Z * ?
                self.builder.position_at_end(a1_le_0_block);
                self.build_return_i(Interval {
                    lower: zero,
                    upper: zero,
                });
            }
        }
        f
    }

    fn build_i_neg(&self) -> FunctionValue<'ctx> {
        let (f, a) = self.unary_op_prelude::<Interval>("i_neg");
        let upper = self.builder.build_float_neg(a.lower, "upper");
        let lower = self.builder.build_float_neg(a.upper, "lower");
        self.build_return_i(Interval { lower, upper });
        f
    }

    fn build_i_abs(&self, fmax: FunctionValue<'ctx>) -> FunctionValue<'ctx> {
        let (f, a) = self.unary_op_prelude::<Interval>("i_abs");
        let zero = self.f32_type().const_float(0.0);
        let lower_ge_0 = self.builder.build_float_compare(
            FloatPredicate::OGE,
            a.lower,
            zero,
            "lower_ge_0",
        );
        let lower_ge_0_block = self.context.append_basic_block(f, "lower_ge_0");
        let lower_lt_0_block = self.context.append_basic_block(f, "lower_lt_0");
        self.builder.build_conditional_branch(
            lower_ge_0,
            lower_ge_0_block,
            lower_lt_0_block,
        );
        self.builder.position_at_end(lower_ge_0_block);
        self.build_return_i(a);
        self.builder.position_at_end(lower_lt_0_block);
        let upper_le_0 = self.builder.build_float_compare(
            FloatPredicate::OLE,
            a.upper,
            zero,
            "upper_le_0",
        );
        let upper_le_0_block = self.context.append_basic_block(f, "upper_le_0");
        let upper_gt_0_block = self.context.append_basic_block(f, "upper_gt_0");
        self.builder.build_conditional_branch(
            upper_le_0,
            upper_le_0_block,
            upper_gt_0_block,
        );
        self.builder.position_at_end(upper_le_0_block);
        let upper = self.builder.build_float_neg(a.lower, "upper");
        let lower = self.builder.build_float_neg(a.upper, "lower");
        self.build_return_i(Interval { lower, upper });

        self.builder.position_at_end(upper_gt_0_block);
        let neg_lower = self.builder.build_float_neg(a.lower, "neg_lower");
        let upper = self
            .builder
            .build_call(fmax, &[neg_lower.into(), a.upper.into()], "max")
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        self.build_return_i(Interval { lower: zero, upper });
        f
    }

    fn build_i_sqrt(&self, fsqrt: FunctionValue<'ctx>) -> FunctionValue<'ctx> {
        let (f, a) = self.unary_op_prelude::<Interval>("i_sqrt");
        let zero = self.f32_type().const_float(0.0);
        let upper_lt_0 = self.builder.build_float_compare(
            FloatPredicate::OLT,
            a.upper,
            zero,
            "upper_lt_0",
        );
        let upper_lt_0_block = self.context.append_basic_block(f, "upper_lt_0");
        let upper_ge_0_block = self.context.append_basic_block(f, "upper_ge_0");
        self.builder.build_conditional_branch(
            upper_lt_0,
            upper_lt_0_block,
            upper_ge_0_block,
        );
        self.builder.position_at_end(upper_lt_0_block);
        let nan = self.f32_type().const_float(f64::NAN);
        self.build_return_i(Interval {
            lower: nan,
            upper: nan,
        });

        self.builder.position_at_end(upper_ge_0_block);
        let lower_le_0 = self.builder.build_float_compare(
            FloatPredicate::OLE,
            a.lower,
            zero,
            "lower_le_0",
        );
        let lower_le_0_block = self.context.append_basic_block(f, "lower_le_0");
        let lower_gt_0_block = self.context.append_basic_block(f, "lower_gt_0");
        self.builder.build_conditional_branch(
            lower_le_0,
            lower_le_0_block,
            lower_gt_0_block,
        );

        self.builder.position_at_end(lower_le_0_block);
        let upper = self
            .builder
            .build_call(fsqrt, &[a.upper.into()], "sqrt_upper")
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        self.build_return_i(Interval { lower: zero, upper });

        self.builder.position_at_end(lower_gt_0_block);
        let lower = self
            .builder
            .build_call(fsqrt, &[a.lower.into()], "sqrt_lower")
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let upper = self
            .builder
            .build_call(fsqrt, &[a.upper.into()], "sqrt_upper")
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        self.build_return_i(Interval { lower, upper });

        f
    }

    fn unary_op_prelude<T: JitValue<'ctx>>(
        &self,
        name: &str,
    ) -> (FunctionValue<'ctx>, T) {
        let ty = T::ty(self);
        let fn_ty = ty.fn_type(&[ty.into()], false);
        let function = self.module.add_function(name, fn_ty, None);
        function.add_attribute(AttributeLoc::Function, self.always_inline);
        let entry_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_block);
        let a = function.get_nth_param(0).unwrap();
        let i = T::from_basic_value_enum(self, a);
        (function, i)
    }

    /// Builds a prelude for a function of the form
    /// ```
    /// T op(lhs: T, rhs: T, choice: u32, shift: u32, out: *u32) {
    ///     let c = (choice >> shift) & 3;
    ///     if c == LHS {
    ///         return lhs;
    ///     } else if c == RHS {
    ///         return rhs;
    ///     } else {
    ///         // builder is positioned here
    ///     }
    /// ```
    ///
    /// Returns `(f, lhs, rhs, shift, out)`
    fn binary_choice_op_prelude<T: JitValue<'ctx>>(
        &self,
        name: &str,
    ) -> (FunctionValue<'ctx>, T, T, IntValue, PointerValue) {
        let ty = T::ty(self);
        let i32_type = self.context.i32_type();
        let i32_ptr_type = i32_type.ptr_type(AddressSpace::Local); // XXX ?
        let fn_ty = ty.fn_type(
            &[
                ty.into(),
                ty.into(),
                i32_type.into(),
                i32_type.into(),
                i32_ptr_type.into(),
            ],
            false,
        );
        let function =
            self.module
                .add_function(name, fn_ty, Some(Linkage::Private));
        function.add_attribute(AttributeLoc::Function, self.always_inline);
        let entry_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_block);

        let a = function.get_nth_param(0).unwrap();
        let lhs = T::from_basic_value_enum(self, a);
        let b = function.get_nth_param(1).unwrap();
        let rhs = T::from_basic_value_enum(self, b);
        let choice = function.get_nth_param(2).unwrap().into_int_value();
        let shift = function.get_nth_param(3).unwrap().into_int_value();
        let out_ptr = function.get_nth_param(4).unwrap().into_pointer_value();

        let left_block =
            self.context.append_basic_block(function, "choice_left");
        let right_block =
            self.context.append_basic_block(function, "choice_right");
        let both_block =
            self.context.append_basic_block(function, "choice_both");
        let end_block = self.context.append_basic_block(function, "choice_end");

        let choice_value_shr =
            self.builder
                .build_right_shift(choice, shift, false, "choice_shr");
        let choice_value_masked = self.builder.build_and(
            choice_value_shr,
            self.i32_type.const_int(3, false),
            "choice_masked",
        );

        self.builder.build_unconditional_branch(left_block);
        self.builder.position_at_end(left_block);

        let left_cmp = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            choice_value_masked,
            self.i32_type.const_int(LHS as u64, true),
            "left_cmp",
        );
        self.builder
            .build_conditional_branch(left_cmp, end_block, right_block);

        self.builder.position_at_end(right_block);
        let right_cmp = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            choice_value_masked,
            self.i32_type.const_int(RHS as u64, true),
            "right_cmp",
        );
        self.builder
            .build_conditional_branch(right_cmp, end_block, both_block);

        // Handle the early exits, which both jump to the choice_end block
        self.builder.position_at_end(end_block);
        let out = self.builder.build_phi(self.f32_type(), "out");
        out.add_incoming(&[(&a, left_block), (&b, right_block)]);

        let out = out.as_basic_value();
        self.builder.build_return(Some(&out));

        // Allow the caller to write code in the choice_both block
        self.builder.position_at_end(both_block);

        (function, lhs, rhs, shift, out_ptr)
    }

    fn binary_op_prelude<T: JitValue<'ctx>>(
        &self,
        name: &str,
    ) -> (FunctionValue<'ctx>, T, T) {
        let ty = T::ty(self);
        let fn_ty = ty.fn_type(&[ty.into(), ty.into()], false);
        let function = self.module.add_function(name, fn_ty, None);
        function.add_attribute(AttributeLoc::Function, self.always_inline);
        let entry_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_block);
        let a = function.get_nth_param(0).unwrap();
        let lhs = T::from_basic_value_enum(self, a);
        let b = function.get_nth_param(1).unwrap();
        let rhs = T::from_basic_value_enum(self, b);
        (function, lhs, rhs)
    }

    fn build_return_i(&self, out: Interval<'ctx>) {
        let out = out.to_basic_value_enum(self);
        self.builder.build_return(Some(&out));
    }
}

#[derive(Copy, Clone)]
enum EvalType {
    Interval,
    Float,
}

struct Jit<'a, 'ctx> {
    /// Compiled math expression to be JIT'ed
    t: &'a Compiler,

    /// Basic LLVM structs and types
    core: JitCore<'ctx>,

    /// shape function
    function: FunctionValue<'ctx>,

    /// Functions to use when rendering
    math_f: JitMath<'ctx, Float<'ctx>>,
    math_i: JitMath<'ctx, Interval<'ctx>>,

    /// Last known use of a particular node
    ///
    /// Stores `FloatValues` when building a floating-point evaluator, and
    /// `ArrayValue` when building an interval evaluator.
    ///
    /// A `NodeIndex` may be stored in multiple locations if it needs to escape
    /// from recursive blocks via phi nodes.
    values: BTreeMap<NodeIndex, BasicValueEnum<'ctx>>,

    /// Values in the `choices` array, loaded in at the `entry` point to
    /// reduce code size.
    choices: Vec<IntValue<'ctx>>,

    /// Index in the opcode tape
    i: usize,
}

impl<'a, 'ctx> Jit<'a, 'ctx> {
    fn build(t: &'a Compiler, core: JitCore<'ctx>) -> Self {
        let intrinsics = core.get_intrinsics();
        let math_i = core.get_math_i(&intrinsics);
        let math_f = core.get_math_f(&intrinsics);

        // Build our main function
        let function =
            core.module.add_function("shape", core.float.shape_fn, None);
        let entry_block = core.context.append_basic_block(function, "entry");
        core.builder.position_at_end(entry_block);

        let choice_array_size = (t.num_choices + 15) / 16;
        let choice_base_ptr =
            function.get_nth_param(2).unwrap().into_pointer_value();

        let choices = (0..choice_array_size)
            .map(|i| {
                let choice_ptr = unsafe {
                    core.builder.build_gep(
                        choice_base_ptr,
                        &[core.i32_type.const_int(i.try_into().unwrap(), true)],
                        &format!("choice_ptr_{}", i),
                    )
                };
                core.builder
                    .build_load(choice_ptr, &format!("choice_{}", i))
                    .into_int_value()
            })
            .collect();

        let root = t.root;
        let mut worker = Self {
            t,
            core,
            function,
            math_i,
            math_f,
            values: BTreeMap::new(),
            choices,
            i: 0,
        };
        worker.recurse(t.op_group[root], EvalType::Float);
        worker
            .core
            .builder
            .build_return(Some(&worker.values[&root]));

        info!(
            "LLVM module is {} characters (pre-optimization)",
            worker.core.module.to_string().len()
        );
        worker.core.module.print_to_file("jit.ll").unwrap();

        // Run an optimization pass on the IR, which is mostly helpful to inline
        // function calls.
        Target::initialize_native(&InitializationConfig::default()).unwrap();
        let target = Target::from_name("arm64").unwrap();
        let target_machine = target
            .create_target_machine(
                &TargetMachine::get_default_triple(),
                TargetMachine::get_host_cpu_name().to_str().unwrap(),
                TargetMachine::get_host_cpu_features().to_str().unwrap(),
                OptimizationLevel::Default,
                RelocMode::Default,
                CodeModel::JITDefault,
            )
            .unwrap();
        worker
            .core
            .module
            .run_passes(
                "default<O1>",
                &target_machine,
                PassBuilderOptions::create(),
            )
            .unwrap();

        worker.core.module.print_to_file("jit.opt.ll").unwrap();
        info!(
            "LLVM module is {} characters (post-optimization)",
            worker.core.module.to_string().len()
        );

        worker
    }

    /// Recurses into the given group, building its children then its nodes
    ///
    /// Returns a set of active nodes
    fn recurse(
        &mut self,
        g: GroupIndex,
        mode: EvalType,
    ) -> BTreeSet<(usize, NodeIndex)> {
        let mut active = BTreeSet::new();
        let group = &self.t.groups[g];
        for &c in &group.children {
            let child_active = self.build_child_group(g, c, mode);
            for c in child_active.into_iter() {
                active.insert(c);
            }
        }

        for &n in &group.nodes {
            let node_name = format!("g{}n{}", usize::from(g), usize::from(n));
            let op = self.t.ops[n];
            let v = match mode {
                EvalType::Float => self.build_op(&node_name, op, &self.math_f),
                EvalType::Interval => {
                    self.build_op(&node_name, op, &self.math_i)
                }
            };
            self.values.insert(n, v);
            self.i += 1;

            // This node is now active!
            active.insert((self.t.last_use[n], n));
        }

        // Drop any nodes which are no longer active
        // This could be cleaner once #62924 map_first_last is stabilized
        while let Some((index, node)) = active.iter().next().cloned() {
            if index >= self.i {
                break;
            }
            active.remove(&(index, node));
        }
        active
    }

    /// Builds a single node's operation
    fn build_op<T: JitValue<'ctx>>(
        &self,
        node_name: &str,
        op: Op,
        math: &JitMath<'ctx, T>,
    ) -> BasicValueEnum<'ctx> {
        match op {
            Op::Var(v) => {
                let i = match self.t.vars.get_by_index(v).unwrap().as_str() {
                    "X" => 0,
                    "Y" => 1,
                    v => panic!("Unexpected var {:?}", v),
                };
                self.function.get_nth_param(i).unwrap()
            }
            Op::Const(f) => T::const_value(&self.core, f),
            Op::Binary(op, a, b) => {
                let f = match op {
                    BinaryOpcode::Add => math.add,
                    BinaryOpcode::Mul => math.mul,
                };
                let lhs = self.values[&a];
                let rhs = self.values[&b];
                self.core
                    .builder
                    .build_call(f, &[lhs.into(), rhs.into()], node_name)
                    .try_as_basic_value()
                    .left()
                    .unwrap()
            }

            Op::BinaryChoice(op, a, b, c) => {
                let c = usize::from(c);

                let f = match op {
                    BinaryChoiceOpcode::Min => math.min,
                    BinaryChoiceOpcode::Max => math.max,
                };
                let lhs = self.values[&a];
                let rhs = self.values[&b];
                let choice = self.choices[c / 16];
                let shift = (c % 16) * 2;
                let shift = self.core.i32_type.const_int(shift as u64, false);

                self.core
                    .builder
                    .build_call(
                        f,
                        &[
                            lhs.into(),
                            rhs.into(),
                            choice.into(),
                            shift.into(),
                            self.core
                                .i32_type
                                .ptr_type(AddressSpace::Local)
                                .const_null()
                                .into(),
                        ],
                        node_name,
                    )
                    .try_as_basic_value()
                    .left()
                    .unwrap()
            }
            Op::Unary(op, a) => {
                let f = match op {
                    UnaryOpcode::Neg => math.neg,
                    UnaryOpcode::Abs => math.abs,
                    UnaryOpcode::Sqrt => math.sqrt,
                    UnaryOpcode::Recip => math.recip,
                };
                self.core
                    .builder
                    .build_call(f, &[self.values[&a].into()], node_name)
                    .try_as_basic_value()
                    .left()
                    .unwrap()
            }
        }
    }

    /// Builds the conditional chain and recursion into a child group
    ///
    /// Returns the set of active nodes, which have been written but not used
    /// for the last time.
    fn build_child_group(
        &mut self,
        group_index: GroupIndex,
        child_index: GroupIndex,
        mode: EvalType,
    ) -> BTreeSet<(usize, NodeIndex)> {
        let child_group = &self.t.groups[child_index];
        let has_both_or_root = child_group
            .choices
            .iter()
            .any(|c| matches!(c, Source::Both(..) | Source::Root));
        let unconditional = has_both_or_root
            || (child_group.child_weight < child_group.choices.len());

        let last_cond = if unconditional {
            // Nothing to do here, proceed straight into child
            None
        } else {
            // Collect choices into array of u32 masks
            let mut choice_u32s = BTreeMap::new();
            for c in child_group.choices.iter() {
                let (c, v) = match c {
                    Source::Left(c) => (c, LHS),
                    Source::Right(c) => (c, RHS),
                    _ => unreachable!(),
                };
                let e = choice_u32s.entry(usize::from(*c) / 16).or_insert(0);
                *e |= v << ((usize::from(*c) % 16) * 2);
            }
            assert!(!choice_u32s.is_empty());

            let cond_blocks = (0..choice_u32s.len())
                .map(|i| {
                    self.core.context.append_basic_block(
                        self.function,
                        &format!("g{}_{}", usize::from(child_index), i),
                    )
                })
                .collect::<Vec<_>>();
            let recurse_block = self.core.context.append_basic_block(
                self.function,
                &format!("g{}_recurse", usize::from(child_index)),
            );

            /*
             * We unpack choices into roughly the following pseudocode:
             *  load choices[$c] => $choice
             *  and $choice $mask => $out
             *  cmp $out 0
             *  branch recurse next
             *  ...etc
             *  branch recurse done
             *  recurse:
             *      Do recursion here
             *      branch done
             *  done:
             *  Start the next thing
             */
            for (i, (c, v)) in choice_u32s.iter().enumerate() {
                let label = format!("g{}_{}", usize::from(child_index), i);
                let choice_value_masked = self.core.builder.build_and(
                    self.choices[*c],
                    self.core.i32_type.const_int(*v as u64, false),
                    &format!("choice_masked_{}", label),
                );
                let choice_cmp = self.core.builder.build_int_compare(
                    inkwell::IntPredicate::NE,
                    choice_value_masked,
                    self.core.i32_type.const_int(0, true),
                    &format!("choice_cmp_{}", label),
                );
                let next_block = cond_blocks[i];
                self.core.builder.build_conditional_branch(
                    choice_cmp,
                    recurse_block,
                    next_block,
                );
                self.core.builder.position_at_end(next_block);
            }
            // At this point, we begin constructing the recursive call.
            // We'll need to patch up the last cond_block later, after
            // we've constructed the done block.
            self.core.builder.position_at_end(recurse_block);

            Some(*cond_blocks.last().unwrap())
        };

        // Do the recursion
        let active = self.recurse(child_index, mode);
        let recurse_block = self.function.get_last_basic_block().unwrap();

        let done_block = self.core.context.append_basic_block(
            self.function,
            &format!("g{}_done", usize::from(child_index)),
        );
        self.core.builder.build_unconditional_branch(done_block);

        // Stitch the final conditional into the done block
        if let Some(last_block) = last_cond {
            self.core.builder.position_at_end(last_block);
            self.core.builder.build_unconditional_branch(done_block);

            self.core.builder.position_at_end(done_block);
            // Construct phi nodes which extract outputs from the recursive
            // block into the parent block's context for active nodes
            for n in active.iter().map(|(_, n)| *n) {
                let out = self.core.builder.build_phi(
                    self.core.f32_type(),
                    &format!(
                        "g{}n{}",
                        usize::from(group_index),
                        usize::from(n)
                    ),
                );
                out.add_incoming(&[
                    (&self.values[&n], recurse_block),
                    (&self.core.f32_type().get_undef(), last_block),
                ]);
                self.values.insert(n, out.as_basic_value());
            }
        } else {
            self.core.builder.position_at_end(done_block);
        }
        active
    }
}

#[derive(Copy, Clone)]
struct Interval<'ctx> {
    lower: FloatValue<'ctx>,
    upper: FloatValue<'ctx>,
}

#[derive(Copy, Clone)]
struct Float<'ctx> {
    value: FloatValue<'ctx>,
}

trait JitValue<'ctx> {
    fn const_value<'a>(core: &'a JitCore<'ctx>, f: f64)
        -> BasicValueEnum<'ctx>;
    fn from_basic_value_enum<'a>(
        core: &'a JitCore<'ctx>,
        v: BasicValueEnum<'ctx>,
    ) -> Self;
    fn to_basic_value_enum<'a>(
        &self,
        core: &'a JitCore<'ctx>,
    ) -> BasicValueEnum<'ctx>;
    fn ty(core: &JitCore<'ctx>) -> BasicTypeEnum<'ctx>;
}

impl<'ctx> JitValue<'ctx> for Interval<'ctx> {
    fn const_value<'a>(
        core: &'a JitCore<'ctx>,
        f: f64,
    ) -> BasicValueEnum<'ctx> {
        let v = core.f32_type().const_float(f);
        core.f32_type().const_array(&[v, v]).as_basic_value_enum()
    }
    fn from_basic_value_enum<'a>(
        core: &'a JitCore<'ctx>,
        v: BasicValueEnum<'ctx>,
    ) -> Self {
        let v = v.into_array_value();
        let lower = core
            .builder
            .build_extract_value(v, 0, "lower")
            .unwrap()
            .into_float_value();
        let upper = core
            .builder
            .build_extract_value(v, 1, "upper")
            .unwrap()
            .into_float_value();
        Self { lower, upper }
    }
    fn ty(core: &JitCore<'ctx>) -> BasicTypeEnum<'ctx> {
        core.f32_type().array_type(2).into()
    }
    fn to_basic_value_enum(
        &self,
        core: &JitCore<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        let out0 = core.f32_type().array_type(2).const_zero();
        let out1 = core
            .builder
            .build_insert_value(out0, self.lower, 0, "lower")
            .unwrap();
        let out2 = core
            .builder
            .build_insert_value(out1, self.upper, 1, "upper")
            .unwrap();
        out2.as_basic_value_enum()
    }
}

impl<'ctx> JitValue<'ctx> for Float<'ctx> {
    fn const_value<'a>(
        core: &'a JitCore<'ctx>,
        f: f64,
    ) -> BasicValueEnum<'ctx> {
        core.f32_type().const_float(f).as_basic_value_enum()
    }
    fn from_basic_value_enum<'a>(
        _core: &'a JitCore<'ctx>,
        v: BasicValueEnum<'ctx>,
    ) -> Self {
        let value = v.into_float_value();
        Self { value }
    }
    fn ty(core: &JitCore<'ctx>) -> BasicTypeEnum<'ctx> {
        core.f32_type().into()
    }
    fn to_basic_value_enum(
        &self,
        _core: &JitCore<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        self.value.as_basic_value_enum()
    }
}

pub fn to_jit_fn<'a, 'b>(
    t: &'a Compiler,
    context: &'b JitContext,
) -> Result<JitFunction<'b, FloatFunc>, Error> {
    let now = Instant::now();
    info!("Building JIT function");
    let jit_core = JitCore::new(&context.0);
    let jit = Jit::build(t, jit_core);
    info!("Finished building JIT function in {:?}", now.elapsed());

    let now = Instant::now();
    info!("Compiling...");
    let execution_engine = jit
        .core
        .module
        .create_jit_execution_engine(OptimizationLevel::Default)?;
    let out = unsafe { execution_engine.get_function("shape")? };
    info!("Extracted JIT function in {:?}", now.elapsed());
    Ok(out)
}
