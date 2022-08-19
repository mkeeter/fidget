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
    types::{ArrayType, FloatType, FunctionType, IntType},
    values::FunctionValue,
    values::{FloatValue, IntValue},
    AddressSpace, OptimizationLevel,
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

struct JitIntrinsics<'ctx> {
    f_abs: FunctionValue<'ctx>,
    f_sqrt: FunctionValue<'ctx>,
    f_max: FunctionValue<'ctx>,
    f_min: FunctionValue<'ctx>,
    fc_min: FunctionValue<'ctx>,
    fc_max: FunctionValue<'ctx>,
    i_add: FunctionValue<'ctx>,
}

struct JitCore<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    f32_type: FloatType<'ctx>,
    i32_type: IntType<'ctx>,
    interval_type: ArrayType<'ctx>,
    binary_fn_type_i: FunctionType<'ctx>,
    binary_choice_fn_type_f: FunctionType<'ctx>,
    shape_fn_type: FunctionType<'ctx>,
}

impl<'ctx> JitCore<'ctx> {
    fn new(context: &'ctx Context) -> Self {
        let i32_type = context.i32_type();
        let f32_type = context.f32_type();
        let interval_type = f32_type.array_type(2);

        let module = context.create_module("shape");
        let builder = context.create_builder();

        let binary_fn_type_i = interval_type
            .fn_type(&[interval_type.into(), interval_type.into()], false);

        let binary_choice_fn_type_f = f32_type.fn_type(
            &[
                f32_type.into(), // lhs
                f32_type.into(), // rhs
                i32_type.into(), // choice
                i32_type.into(), // shift
            ],
            false,
        );

        let i32_const_ptr_type = i32_type.ptr_type(AddressSpace::Const);
        let shape_fn_type = f32_type.fn_type(
            &[f32_type.into(), f32_type.into(), i32_const_ptr_type.into()],
            false,
        );

        Self {
            context,
            module,
            builder,
            f32_type,
            i32_type,
            interval_type,
            binary_fn_type_i,
            binary_choice_fn_type_f,
            shape_fn_type,
        }
    }

    fn get_unary_intrinsic_f(&self, name: &str) -> FunctionValue<'ctx> {
        let intrinsic = Intrinsic::find(&format!("llvm.{}", name)).unwrap();
        intrinsic
            .get_declaration(&self.module, &[self.f32_type.into()])
            .unwrap()
    }

    fn get_binary_intrinsic_f(&self, name: &str) -> FunctionValue<'ctx> {
        let intrinsic = Intrinsic::find(&format!("llvm.{}", name)).unwrap();
        intrinsic
            .get_declaration(
                &self.module,
                &[self.f32_type.into(), self.f32_type.into()],
            )
            .unwrap()
    }

    fn get_intrinsics(&self) -> JitIntrinsics<'ctx> {
        let f_abs = self.get_unary_intrinsic_f("abs");
        let f_sqrt = self.get_unary_intrinsic_f("sqrt");
        let f_max = self.get_binary_intrinsic_f("maxnum");
        let f_min = self.get_binary_intrinsic_f("minnum");
        let fc_min = self.build_min_max("f_min", f_min);
        let fc_max = self.build_min_max("f_max", f_max);

        let i_add = self.build_i_add();

        JitIntrinsics {
            f_abs,
            f_sqrt,
            f_max,
            f_min,
            i_add,
            fc_min,
            fc_max,
        }
    }

    fn build_min_max(
        &self,
        name: &str,
        f: FunctionValue<'ctx>,
    ) -> FunctionValue<'ctx> {
        let function = self.module.add_function(
            name,
            self.binary_choice_fn_type_f,
            Some(Linkage::Private),
        );
        let kind_id = Attribute::get_named_enum_kind_id("alwaysinline");
        let attr = self.context.create_enum_attribute(kind_id, 0);
        function.add_attribute(AttributeLoc::Function, attr);
        let entry_block = self.context.append_basic_block(function, "entry");

        let lhs = function.get_nth_param(0).unwrap().into_float_value();
        let rhs = function.get_nth_param(1).unwrap().into_float_value();
        let choice = function.get_nth_param(2).unwrap().into_int_value();
        let shift = function.get_nth_param(3).unwrap().into_int_value();

        self.builder.position_at_end(entry_block);
        let left_block = self
            .context
            .append_basic_block(function, &format!("minmax_left"));
        let right_block = self
            .context
            .append_basic_block(function, &format!("minmax_right"));
        let both_block = self
            .context
            .append_basic_block(function, &format!("minmax_both"));
        let end_block = self
            .context
            .append_basic_block(function, &format!("minmax_end"));

        let choice_value_shr = self.builder.build_right_shift(
            choice,
            shift,
            false,
            &format!("choice_shr"),
        );
        let choice_value_masked = self.builder.build_and(
            choice_value_shr,
            self.i32_type.const_int(3, false),
            &format!("choice_masked"),
        );

        self.builder.build_unconditional_branch(left_block);
        self.builder.position_at_end(left_block);

        let left_cmp = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            choice_value_masked,
            self.i32_type.const_int(LHS as u64, true),
            &format!("left_cmp"),
        );
        self.builder
            .build_conditional_branch(left_cmp, end_block, right_block);

        self.builder.position_at_end(right_block);
        let right_cmp = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            choice_value_masked,
            self.i32_type.const_int(RHS as u64, true),
            &format!("right_cmp"),
        );
        self.builder
            .build_conditional_branch(right_cmp, end_block, both_block);

        self.builder.position_at_end(both_block);
        let fmax_result = self
            .builder
            .build_call(f, &[lhs.into(), rhs.into()], &format!("both"))
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        self.builder.build_return(Some(&fmax_result));

        self.builder.position_at_end(end_block);
        let out = self.builder.build_phi(self.f32_type, "out");
        out.add_incoming(&[(&lhs, left_block), (&rhs, right_block)]);

        let out = out.as_basic_value().into_float_value();
        self.builder.build_return(Some(&out));
        function
    }

    fn build_i_add(&self) -> FunctionValue<'ctx> {
        let (f, lhs, rhs) = self.binary_op_prelude("i_add");
        let out_lo =
            self.builder.build_float_add(lhs.lower, rhs.lower, "out_lo");
        let out_hi =
            self.builder.build_float_add(lhs.upper, rhs.upper, "out_hi");
        self.binary_op_ret(Interval {
            lower: out_lo,
            upper: out_hi,
        });
        f
    }

    fn binary_op_prelude<'a>(
        &self,
        name: &str,
    ) -> (FunctionValue<'ctx>, Interval<'ctx>, Interval<'ctx>) {
        let function =
            self.module.add_function(name, self.binary_fn_type_i, None);
        let entry_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_block);
        let lhs = function.get_nth_param(0).unwrap().into_array_value();
        let rhs = function.get_nth_param(1).unwrap().into_array_value();
        let lhs_lo = self
            .builder
            .build_extract_value(lhs, 0, "lhs_lo")
            .unwrap()
            .into_float_value();
        let lhs_hi = self
            .builder
            .build_extract_value(lhs, 1, "lhs_hi")
            .unwrap()
            .into_float_value();
        let rhs_lo = self
            .builder
            .build_extract_value(rhs, 0, "rhs_lo")
            .unwrap()
            .into_float_value();
        let rhs_hi = self
            .builder
            .build_extract_value(rhs, 1, "rhs_hi")
            .unwrap()
            .into_float_value();
        (
            function,
            Interval {
                lower: lhs_lo,
                upper: lhs_hi,
            },
            Interval {
                lower: rhs_lo,
                upper: rhs_hi,
            },
        )
    }
    fn binary_op_ret(&self, out: Interval) {
        let out0 = self.interval_type.const_zero();
        let out1 = self
            .builder
            .build_insert_value(out0, out.lower, 0, "out1")
            .unwrap();
        let out2 = self
            .builder
            .build_insert_value(out1, out.upper, 1, "out2")
            .unwrap();
        self.builder.build_return(Some(&out2));
    }
}

struct Jit<'a, 'ctx> {
    /// Compiled math expression to be JIT'ed
    t: &'a Compiler,

    /// Basic LLVM structs and types
    core: JitCore<'ctx>,

    /// shape function
    function: FunctionValue<'ctx>,

    /// Functions used when rendering
    ///
    /// This includes LLVM intrinsics and our own interval math library
    intrinsics: JitIntrinsics<'ctx>,

    /// Last known use of a particular node
    ///
    /// A `NodeIndex` may be stored in multiple `FloatValue` locations if it
    /// needs to escape from recursive blocks via phi nodes.
    values: BTreeMap<NodeIndex, FloatValue<'ctx>>,

    /// Values in the `choices` array, loaded in at the `entry` point to
    /// reduce code size.
    choices: Vec<IntValue<'ctx>>,

    /// Index in the opcode tape
    i: usize,
}

impl<'a, 'ctx> Jit<'a, 'ctx> {
    fn build(t: &'a Compiler, core: JitCore<'ctx>) -> Self {
        let intrinsics = core.get_intrinsics();

        // Build our main function
        let function =
            core.module.add_function("shape", core.shape_fn_type, None);
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
                        &format!("choice_ptr_{}", usize::from(i)),
                    )
                };
                core.builder
                    .build_load(
                        choice_ptr,
                        &format!("choice_{}", usize::from(i)),
                    )
                    .into_int_value()
            })
            .collect();

        let root = t.root;
        let mut worker = Self {
            t,
            core,
            function,
            intrinsics,
            values: BTreeMap::new(),
            choices,
            i: 0,
        };
        worker.recurse(t.op_group[root]);
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
    fn recurse(&mut self, g: GroupIndex) -> BTreeSet<(usize, NodeIndex)> {
        let mut active = BTreeSet::new();
        let group = &self.t.groups[g];
        for &c in &group.children {
            let child_active = self.build_child_group(g, c);
            for c in child_active.into_iter() {
                active.insert(c);
            }
        }

        for &n in &group.nodes {
            self.build_op(g, n);

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
    fn build_op(&mut self, g: GroupIndex, n: NodeIndex) {
        let node_name = format!("g{}n{}", usize::from(g), usize::from(n));
        let op = self.t.ops[n];
        let value = match op {
            Op::Var(v) => {
                let i = match self.t.vars.get_by_index(v).unwrap().as_str() {
                    "X" => 0,
                    "Y" => 1,
                    v => panic!("Unexpected var {:?}", v),
                };
                self.function.get_nth_param(i).unwrap().into_float_value()
            }
            Op::Const(f) => self.core.f32_type.const_float(f),
            Op::Binary(op, a, b) => {
                let f = match op {
                    BinaryOpcode::Add => Builder::build_float_add,
                    BinaryOpcode::Mul => Builder::build_float_mul,
                };
                f(
                    &self.core.builder,
                    self.values[&a],
                    self.values[&b],
                    &node_name,
                )
            }

            Op::BinaryChoice(op, a, b, c) => {
                let c = usize::from(c);

                let f = match op {
                    BinaryChoiceOpcode::Min => self.intrinsics.fc_min,
                    BinaryChoiceOpcode::Max => self.intrinsics.fc_max,
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
                        &[lhs.into(), rhs.into(), choice.into(), shift.into()],
                        &format!("call_n{}", usize::from(n)),
                    )
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value()
            }
            Op::Unary(op, a) => {
                let call_intrinsic = |i| {
                    self.core
                        .builder
                        .build_call(
                            i,
                            &[self.values[&a].into()],
                            &format!("call_n{}", usize::from(n)),
                        )
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                        .into_float_value()
                };
                match op {
                    UnaryOpcode::Neg => self
                        .core
                        .builder
                        .build_float_neg(self.values[&a], &node_name),
                    UnaryOpcode::Abs => call_intrinsic(self.intrinsics.f_abs),
                    UnaryOpcode::Sqrt => call_intrinsic(self.intrinsics.f_sqrt),
                    UnaryOpcode::Recip => self.core.builder.build_float_div(
                        self.core.f32_type.const_float(1.0),
                        self.values[&a],
                        &node_name,
                    ),
                }
            }
        };
        self.values.insert(n, value);

        self.i += 1;
    }

    /// Builds the conditional chain and recursion into a child group
    ///
    /// Returns the set of active nodes, which have been written but not used
    /// for the last time.
    fn build_child_group(
        &mut self,
        group_index: GroupIndex,
        child_index: GroupIndex,
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
        let active = self.recurse(child_index);
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
                    self.core.f32_type,
                    &format!(
                        "g{}n{}",
                        usize::from(group_index),
                        usize::from(n)
                    ),
                );
                out.add_incoming(&[
                    (&self.values[&n], recurse_block),
                    (&self.core.f32_type.get_undef(), last_block),
                ]);
                self.values
                    .insert(n, out.as_basic_value().into_float_value());
            }
        } else {
            self.core.builder.position_at_end(done_block);
        }
        active
    }
}

struct Interval<'ctx> {
    lower: FloatValue<'ctx>,
    upper: FloatValue<'ctx>,
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
