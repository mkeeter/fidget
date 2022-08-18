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
    builder::Builder,
    context::Context,
    execution_engine::JitFunction,
    intrinsics::Intrinsic,
    module::Module,
    types::{FloatType, IntType},
    values::FloatValue,
    values::FunctionValue,
    AddressSpace, OptimizationLevel,
};
use log::info;

const LHS: u32 = 1;
const RHS: u32 = 2;

type FloatFunc = unsafe extern "C" fn(f32, f32, *const u32) -> f32;

struct Jit<'a, 'ctx> {
    t: &'a Compiler,
    context: &'ctx Context,
    intrinsics: Intrinsics<'ctx>,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    function: FunctionValue<'ctx>,
    values: BTreeMap<NodeIndex, FloatValue<'ctx>>,
    f32_type: FloatType<'ctx>,
    i32_type: IntType<'ctx>,

    /*
    /// Values in the `choices` array, loaded in at the `entry` point to
    /// reduce code size.
    choices: Vec<IntValue<'ctx>>,
    */
    /// Index in the opcode tape
    i: usize,
}

impl<'a, 'ctx> Jit<'a, 'ctx> {
    fn new(t: &'a Compiler, context: &'ctx Context) -> Self {
        let i32_type = context.i32_type();
        let f32_type = context.f32_type();

        let module = context.create_module("shape");
        let builder = context.create_builder();

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
        let i32_ptr_type = i32_type.ptr_type(AddressSpace::Const);
        let fn_type = f32_type.fn_type(
            &[f32_type.into(), f32_type.into(), i32_ptr_type.into()],
            false,
        );
        let function = module.add_function("shape", fn_type, None);

        Self {
            t,
            context,
            intrinsics,
            module,
            builder,
            function,
            values: BTreeMap::new(),
            f32_type,
            i32_type,

            i: 0,
        }
    }

    fn build(&mut self) {
        let basic_block =
            self.context.append_basic_block(self.function, "entry");
        self.builder.position_at_end(basic_block);
        let root = self.t.root;
        self.recurse(self.t.op_group[root]);
        self.builder.build_return(Some(&self.values[&root]));

        //self.module.print_to_stderr();
        self.module.print_to_file("jit.ll").unwrap();
        info!(
            "LLVM module is {} characters",
            self.module.to_string().len()
        );
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
            Op::Const(f) => self.f32_type.const_float(f),
            Op::Binary(op, a, b) => {
                let f = match op {
                    BinaryOpcode::Add => Builder::build_float_add,
                    BinaryOpcode::Mul => Builder::build_float_mul,
                };
                f(&self.builder, self.values[&a], self.values[&b], &node_name)
            }

            Op::BinaryChoice(op, a, b, c) => {
                let left_block = self.context.append_basic_block(
                    self.function,
                    &format!("minmax_left_{}", usize::from(n)),
                );
                let right_block = self.context.append_basic_block(
                    self.function,
                    &format!("minmax_right_{}", usize::from(n)),
                );
                let both_block = self.context.append_basic_block(
                    self.function,
                    &format!("minmax_both_{}", usize::from(n)),
                );
                let end_block = self.context.append_basic_block(
                    self.function,
                    &format!("minmax_end_{}", usize::from(n)),
                );

                let choice_base_ptr = self
                    .function
                    .get_nth_param(2)
                    .unwrap()
                    .into_pointer_value();
                let c = usize::from(c);
                let choice_ptr = unsafe {
                    self.builder.build_gep(
                        choice_base_ptr,
                        &[self.i32_type.const_int(c as u64 / 16, true)],
                        &format!("choice_ptr_{}", usize::from(n)),
                    )
                };

                let choice_value = self
                    .builder
                    .build_load(
                        choice_ptr,
                        &format!("choice_{}", usize::from(n)),
                    )
                    .into_int_value();
                let shift = (c % 16) * 2;
                let choice_value_masked = self.builder.build_and(
                    choice_value,
                    self.i32_type.const_int(3 << shift, false),
                    &format!("choice_masked_{}", usize::from(n)),
                );

                self.builder.build_unconditional_branch(left_block);
                self.builder.position_at_end(left_block);

                let left_cmp = self.builder.build_int_compare(
                    inkwell::IntPredicate::EQ,
                    choice_value_masked,
                    self.i32_type.const_int((LHS as u64) << shift, true),
                    &format!("left_cmp_{}", usize::from(n)),
                );
                self.builder.build_conditional_branch(
                    left_cmp,
                    end_block,
                    right_block,
                );

                self.builder.position_at_end(right_block);
                let right_cmp = self.builder.build_int_compare(
                    inkwell::IntPredicate::EQ,
                    choice_value_masked,
                    self.i32_type.const_int((RHS as u64) << shift, true),
                    &format!("right_cmp_{}", usize::from(n)),
                );
                self.builder
                    .build_conditional_branch(right_cmp, end_block, both_block);

                self.builder.position_at_end(both_block);
                let i = match op {
                    BinaryChoiceOpcode::Min => self.intrinsics.min,
                    BinaryChoiceOpcode::Max => self.intrinsics.max,
                };
                let fmax_result = self
                    .builder
                    .build_call(
                        i,
                        &[self.values[&a].into(), self.values[&b].into()],
                        &format!("{}_both", node_name),
                    )
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value();
                self.builder.build_unconditional_branch(end_block);

                self.builder.position_at_end(end_block);
                let out = self.builder.build_phi(self.f32_type, &node_name);
                out.add_incoming(&[
                    (&fmax_result, both_block),
                    (&self.values[&a], left_block),
                    (&self.values[&b], right_block),
                ]);
                out.as_basic_value().into_float_value()
            }
            Op::Unary(op, a) => {
                let call_intrinsic = |i| {
                    self.builder
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
                        .builder
                        .build_float_neg(self.values[&a], &node_name),
                    UnaryOpcode::Abs => call_intrinsic(self.intrinsics.abs),
                    UnaryOpcode::Sqrt => call_intrinsic(self.intrinsics.sqrt),
                    UnaryOpcode::Cos => call_intrinsic(self.intrinsics.cos),
                    UnaryOpcode::Sin => call_intrinsic(self.intrinsics.sin),
                    UnaryOpcode::Exp => call_intrinsic(self.intrinsics.exp),
                    UnaryOpcode::Ln => call_intrinsic(self.intrinsics.ln),
                    UnaryOpcode::Recip => self.builder.build_float_div(
                        self.f32_type.const_float(1.0),
                        self.values[&a],
                        &node_name,
                    ),
                    /*
                    UnaryOpcode::Tan => Builder::build_float_tan,
                    UnaryOpcode::Asin => Builder::build_float_asin,
                    UnaryOpcode::Acos => Builder::build_float_acos,
                    UnaryOpcode::Atan => Builder::build_float_atan,
                    */
                    op => panic!("Unimplemented opcode {:?}", op),
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
        let unconditional = child_group
            .choices
            .iter()
            .any(|c| matches!(c, Source::Both(..) | Source::Root));

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
                    self.context.append_basic_block(
                        self.function,
                        &format!("g{}_{}", usize::from(child_index), i),
                    )
                })
                .collect::<Vec<_>>();
            let recurse_block = self.context.append_basic_block(
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
                let choice_base_ptr = self
                    .function
                    .get_nth_param(2)
                    .unwrap()
                    .into_pointer_value();
                let choice_ptr = unsafe {
                    self.builder.build_gep(
                        choice_base_ptr,
                        &[self.i32_type.const_int(*c as u64 / 16, true)],
                        &format!("choice_ptr_{}", label),
                    )
                };
                let choice_value = self
                    .builder
                    .build_load(choice_ptr, &format!("choice_{}", label))
                    .into_int_value();

                let choice_value_masked = self.builder.build_and(
                    choice_value,
                    self.i32_type.const_int(*v as u64, false),
                    &format!("choice_masked_{}", label),
                );
                let choice_cmp = self.builder.build_int_compare(
                    inkwell::IntPredicate::NE,
                    choice_value_masked,
                    self.i32_type.const_int(0, true),
                    &format!("choice_cmp_{}", label),
                );
                let next_block = cond_blocks[i];
                self.builder.build_conditional_branch(
                    choice_cmp,
                    recurse_block,
                    next_block,
                );
                self.builder.position_at_end(next_block);
            }
            // At this point, we begin constructing the recursive call.
            // We'll need to patch up the last cond_block later, after
            // we've constructed the done block.
            self.builder.position_at_end(recurse_block);

            Some(*cond_blocks.last().unwrap())
        };

        // Do the recursion
        let active = self.recurse(child_index);
        let recurse_block = self.function.get_last_basic_block().unwrap();

        let done_block = self.context.append_basic_block(
            self.function,
            &format!("g{}_done", usize::from(child_index)),
        );
        self.builder.build_unconditional_branch(done_block);

        // Stitch the final conditional into the done block
        if let Some(last_block) = last_cond {
            self.builder.position_at_end(last_block);
            self.builder.build_unconditional_branch(done_block);

            self.builder.position_at_end(done_block);
            // Construct phi nodes which extract outputs from the recursive
            // block into the parent block's context for active nodes
            for n in active.iter().map(|(_, n)| *n) {
                let out = self.builder.build_phi(
                    self.f32_type,
                    &format!(
                        "g{}n{}",
                        usize::from(group_index),
                        usize::from(n)
                    ),
                );
                out.add_incoming(&[
                    (&self.values[&n], recurse_block),
                    (&self.f32_type.const_float(f64::NAN), last_block),
                ]);
                self.values
                    .insert(n, out.as_basic_value().into_float_value());
            }
        } else {
            self.builder.position_at_end(done_block);
        }
        active
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
    t: &'a Compiler,
    context: &'b Context,
) -> Result<JitFunction<'b, FloatFunc>, Error> {
    let now = Instant::now();
    info!("Building JIT function");
    let mut jit = Jit::new(t, context);
    jit.build();
    info!("Finished building JIT function in {:?}", now.elapsed());

    let now = Instant::now();
    info!("Compiling...");
    let execution_engine = jit
        .module
        .create_jit_execution_engine(OptimizationLevel::Default)?;
    let out = unsafe { execution_engine.get_function("shape")? };
    info!("Extracted JIT function in {:?}", now.elapsed());
    Ok(out)
}
