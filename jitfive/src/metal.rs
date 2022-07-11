use bytemuck::{Pod, Zeroable};
use indoc::formatdoc;
use std::collections::{BTreeMap, BTreeSet};

use crate::program::{Block, Choice, Config, Instruction, Program, RegIndex};

impl Choice {
    fn to_metal(self) -> &'static str {
        match self {
            Self::Left => "LHS",
            Self::Right => "RHS",
            Self::Both => "BOTH",
        }
    }
}

/// A generated function.
///
/// The opening of the function is omitted but can be reconstructed
/// from the index, inputs, and outputs.
struct Function {
    index: usize,
    body: String,
    root: bool,

    has_var: bool,
    has_choice: bool,

    /// Registers which are sourced externally to this block and unmodified
    inputs: BTreeSet<RegIndex>,
    /// Registers which are sourced externally to this block and modified
    outputs: BTreeSet<RegIndex>,
}

/// Shader mode
#[derive(Copy, Clone, Debug)]
pub enum Mode {
    Pixel,
    Interval,
}

impl Mode {
    fn function_prefix(&self) -> &str {
        match self {
            Mode::Pixel => "f",
            Mode::Interval => "i",
        }
    }

    fn vars_type(&self) -> &str {
        match self {
            Mode::Pixel => "const thread float*",
            Mode::Interval => "const thread float2*",
        }
    }
    fn local_type(&self) -> &str {
        match self {
            Mode::Pixel => "float",
            Mode::Interval => "float2",
        }
    }
    fn choice_type(&self) -> &str {
        match self {
            Mode::Pixel => "const constant uint32_t*",
            Mode::Interval => "thread uint32_t*",
        }
    }
}

impl Function {
    fn declaration(&self, mode: Mode) -> String {
        let mut out = String::new();
        out += &formatdoc!(
            "inline {} t_shape_{}(",
            if self.root { mode.local_type() } else { "void" },
            self.index
        );
        let mut args = vec![];
        if self.has_var {
            args.push(format!("{} vars", mode.vars_type()));
        }
        if self.has_choice {
            args.push(format!("{} choices", mode.choice_type()));
        }
        for i in &self.inputs {
            args.push(format!(
                "const {} v{}",
                mode.local_type(),
                usize::from(*i)
            ));
        }
        for i in &self.outputs {
            args.push(format!(
                "thread {}& v{}",
                mode.local_type(),
                usize::from(*i)
            ));
        }
        out += &args.join(", ");
        out += "\n)";
        out
    }
    /// Generates text to call a function
    fn call(&self) -> String {
        let mut out = String::new();
        out += &format!("t_shape_{}(", self.index);
        let mut args = vec![];
        if self.has_var {
            args.push("vars".to_owned());
        }
        if self.has_choice {
            args.push("choices".to_owned());
        }

        for i in &self.inputs {
            args.push(format!("v{}", usize::from(*i)));
        }
        for i in &self.outputs {
            args.push(format!("v{}", usize::from(*i)));
        }
        out += &args.join(", ");
        out += ");";
        out
    }
}

// Inject a `to_metal` function to `program::Instruction`
impl Instruction {
    fn to_metal(&self, mode: Mode) -> Option<String> {
        let out = usize::from(self.out_reg()?);
        let t = mode.function_prefix();
        Some(match self {
            Self::Var { var, .. } => {
                format!("v{out} = {t}_var(vars[{}]);", usize::from(*var))
            }
            Self::Const { value, .. } => {
                format!("v{out} = {t}_const({});", value)
            }
            Self::Mul { lhs, rhs, .. } | Self::Add { lhs, rhs, .. } => {
                format!(
                    "v{out} = {t}_{}(v{}, v{});",
                    self.name(),
                    usize::from(*lhs),
                    usize::from(*rhs)
                )
            }
            Self::Max {
                lhs, rhs, choice, ..
            }
            | Self::Min {
                lhs, rhs, choice, ..
            } => {
                let choice_u = usize::from(*choice);
                let c_slot = choice_u / 16;
                let c_shift = (choice_u % 16) * 2;
                let mut switch = formatdoc!(
                    "switch ((choices[{c_slot}] >> {c_shift}) & 3) {{
                        case LHS: v{out} = v{}; break;
                        case RHS: v{out} = v{}; break;
                        default: ",
                    usize::from(*lhs),
                    usize::from(*rhs),
                );
                switch += &match mode {
                    Mode::Pixel => format!(
                        "v{out} = {t}_{}(v{}, v{}); break;",
                        self.name(),
                        usize::from(*lhs),
                        usize::from(*rhs)
                    ),
                    Mode::Interval => {
                        let a = usize::from(*lhs);
                        let b = usize::from(*rhs);
                        match self {
                            Self::Max { .. } => formatdoc!(
                                "
                            if (v{a}[0] > v{b}[1]) {{
                                choices[{c_slot}] &= ~(RHS << {c_shift});
                                v{out} = v{a};
                            }} else if (v{b}[0] > v{a}[1]) {{
                                choices[{c_slot}] &= ~(LHS << {c_shift});
                                v{out} = v{b};
                            }} else {{
                                v{out} = i_max(v{a}, v{b});
                            }}
                            "
                            ),
                            Self::Min { .. } => formatdoc!(
                                "
                            if (v{a}[1] < v{b}[0]) {{
                                choices[{c_slot}] &= ~(RHS << {c_shift});
                                v{out} = v{a};
                            }} else if (v{b}[1] < v{a}[0]) {{
                                choices[{c_slot}] &= ~(LHS << {c_shift});
                                v{out} = v{b};
                            }} else {{
                                v{out} = i_min(v{a}, v{b});
                            }}
                            "
                            ),
                            _ => unreachable!(),
                        }
                    }
                };
                switch + "}\n"
            }
            Self::Ln { reg, .. }
            | Self::Exp { reg, .. }
            | Self::Atan { reg, .. }
            | Self::Acos { reg, .. }
            | Self::Asin { reg, .. }
            | Self::Tan { reg, .. }
            | Self::Cos { reg, .. }
            | Self::Sin { reg, .. }
            | Self::Sqrt { reg, .. }
            | Self::Recip { reg, .. }
            | Self::Abs { reg, .. }
            | Self::Neg { reg, .. } => {
                format!("v{out} = {t}_{}(v{});", self.name(), usize::from(*reg))
            }
            Self::Cond(..) => return None,
        })
    }
}

impl Program {
    /// Converts the program to a Metal shader
    pub fn to_metal(&self, mode: Mode) -> String {
        let mut out = formatdoc!(
            "
            {}
            ",
            METAL_PRELUDE,
        );

        // Global map from block paths to (function index, body)
        let mut functions: BTreeMap<Vec<usize>, Function> = BTreeMap::new();
        self.to_metal_inner(&self.tape, mode, &mut vec![], &mut functions);

        out += "\n// Function definitions\n";
        for f in functions.values().rev() {
            out += &format!("{} {{\n{}}}\n", f.declaration(mode), f.body);
        }
        out += "\n";
        out += &formatdoc!(
            "
        // Root function
        inline {} t_eval({} vars,
                         {} choices)
        {{
            return t_shape_{}(vars, choices);
        }}
        ",
            mode.local_type(),
            mode.vars_type(),
            mode.choice_type(),
            functions.get(&vec![]).unwrap().index
        );
        out += &formatdoc!(
            "
            #define VAR_COUNT {}
            #define CHOICE_BUF_SIZE {}
            ",
            self.config().var_count,
            (self.config().choice_count + 15) / 16,
        );
        out += match mode {
            Mode::Interval => METAL_KERNEL_INTERVALS,
            Mode::Pixel => METAL_KERNEL_PIXELS,
        };
        out
    }

    fn to_metal_inner(
        &self,
        block: &Block,
        mode: Mode,
        path: &mut Vec<usize>,
        functions: &mut BTreeMap<Vec<usize>, Function>,
    ) {
        let mut first = true;
        let mut out = String::new();
        for r in block.locals.iter() {
            if first {
                out += &format!("    {} ", mode.local_type());
                first = false;
            } else {
                out += ", ";
            }
            out += &format!("v{}", usize::from(*r));
        }
        if !first {
            out += ";\n"
        }
        for (index, instruction) in block.tape.iter().enumerate() {
            if let Some(i) = instruction.to_metal(mode) {
                for line in i.lines() {
                    out += &format!("    {}\n", line);
                }
            } else if let Instruction::Cond(cond, next) = &instruction {
                // Recurse!
                path.push(index);
                self.to_metal_inner(next, mode, path, functions);
                let f = functions.get(path).unwrap();
                path.pop();

                // Write out the conditional, calling the inner function
                out += "    if (";
                let mut grouped_cond: BTreeMap<_, BTreeSet<_>> =
                    BTreeMap::new();
                for c in cond {
                    let u = usize::from(c.0);
                    grouped_cond
                        .entry(u / 16)
                        .or_default()
                        .insert(((u % 16) * 2, c.1));
                }
                let mut first = true;
                for (c_slot, v) in &grouped_cond {
                    if first {
                        first = false;
                    } else {
                        out += " ||\n        ";
                    }
                    out += &format!("(choices[{c_slot}] & (");
                    let mut inner_first = true;
                    for (c_shift, mask) in v {
                        if inner_first {
                            inner_first = false;
                        } else {
                            out += " | ";
                        }
                        out += &format!("({} << {c_shift})", mask.to_metal());
                    }
                    out += "))";
                }
                out += ") {\n        ";
                out += &f.call();
                out += "\n    }\n";
            } else {
                panic!("Could not get out register or Cond block");
            }
        }
        let i = functions.len();
        let is_root = path.is_empty();
        if is_root {
            out += &format!("    return v{};\n", usize::from(self.root));
        }
        functions.insert(
            path.clone(),
            Function {
                index: i,
                body: out,
                root: is_root,
                has_var: block.has_var,
                has_choice: block.has_choice,
                inputs: block.inputs.clone(),
                outputs: block.outputs.clone(),
            },
        );
    }
}

////////////////////////////////////////////////////////////////////////////////

const METAL_PRELUDE: &str = include_str!("../shader/prelude.metal");
const METAL_KERNEL_INTERVALS: &str = include_str!("../shader/intervals.metal");
const METAL_KERNEL_PIXELS: &str = include_str!("../shader/pixels.metal");

////////////////////////////////////////////////////////////////////////////////

use piet_gpu_hal::{BindType, BufferUsage, ComputePassDescriptor, ShaderCode};

pub struct IntervalRenderBuffers {
    tile_count: u32,
    choice_count: u32,

    tiles: piet_gpu_hal::Buffer,
    choices: piet_gpu_hal::Buffer,
    out: piet_gpu_hal::Buffer, // RenderOut
}

impl IntervalRenderBuffers {
    fn new(
        tile_count: u32,
        choice_count: u32,
        session: &piet_gpu_hal::Session,
    ) -> Self {
        let tiles = session
            .create_buffer(
                (tile_count as usize * std::mem::size_of::<u32>())
                    .try_into()
                    .unwrap(),
                BufferUsage::STORAGE,
            )
            .unwrap();

        // We tightly pack 2-bit choices into a buffer of uint32_t
        let choice_buf_size = (choice_count + 15) / 16;
        let choices = session
            .create_buffer(
                ((tile_count * choice_buf_size) as usize
                    * std::mem::size_of::<u32>())
                .try_into()
                .unwrap(),
                BufferUsage::STORAGE,
            )
            .unwrap();

        // We need this to be readable because we load the number of active
        // tiles from the first item in the buffer, in order to run the
        // correct number of threads in the next pass.
        let out = session
            .create_buffer(
                ((tile_count as usize + 1) * std::mem::size_of::<u32>())
                    .try_into()
                    .unwrap(),
                BufferUsage::STORAGE | BufferUsage::MAP_READ,
            )
            .unwrap();

        Self {
            tile_count,
            choice_count,
            tiles,
            choices,
            out,
        }
    }

    fn resize_to_fit(
        &mut self,
        tile_count: u32,
        session: &piet_gpu_hal::Session,
    ) {
        if tile_count > self.tile_count {
            let mut next = Self::new(tile_count, self.choice_count, session);
            std::mem::swap(self, &mut next);
        }
    }
}

pub struct ImageBuffers {
    image_size: u32,

    image_64x64: piet_gpu_hal::Buffer,
    image_8x8: piet_gpu_hal::Buffer,
    image_1x1: piet_gpu_hal::Buffer,
    final_image: piet_gpu_hal::Buffer,
}

impl ImageBuffers {
    fn new(image_size: u32, session: &piet_gpu_hal::Session) -> Self {
        assert_eq!(image_size % 64, 0);
        let image_64x64 = session
            .create_buffer(
                u64::from((image_size / 64).pow(2)),
                BufferUsage::STORAGE,
            )
            .unwrap();
        let image_8x8 = session
            .create_buffer(
                u64::from((image_size / 8).pow(2)),
                BufferUsage::STORAGE,
            )
            .unwrap();
        let image_1x1 = session
            .create_buffer(u64::from(image_size.pow(2)), BufferUsage::STORAGE)
            .unwrap();
        let final_image = session
            .create_buffer(
                u64::from(4 * image_size.pow(2)),
                BufferUsage::STORAGE | BufferUsage::MAP_READ,
            )
            .unwrap();

        Self {
            image_size,
            image_64x64,
            image_8x8,
            image_1x1,
            final_image,
        }
    }
    fn resize_to_fit(
        &mut self,
        image_size: u32,
        session: &piet_gpu_hal::Session,
    ) {
        if image_size > self.image_size {
            let mut next = Self::new(image_size, session);
            std::mem::swap(self, &mut next);
        }
    }
}

pub struct Render {
    config: Config,

    /// Buffer that's sized to hold a [RenderConfig]
    config_buf: piet_gpu_hal::Buffer,

    /// Compute pipelines to initialize before the first interval evaluation
    init: piet_gpu_hal::Pipeline,

    /// Compute pipeline to perform interval evaluation
    interval: piet_gpu_hal::Pipeline,

    /// Compute pipelines to subdivide the results of interval evaluation
    subdivide: piet_gpu_hal::Pipeline,

    /// Compute pipelines to render individual pixels
    pixels: piet_gpu_hal::Pipeline,

    /// Compute pipeline to merge multiple mipmap levels
    merge: piet_gpu_hal::Pipeline,

    /// Compute pipeline to clear multiple mipmap levels
    clear: piet_gpu_hal::Pipeline,

    image_buf: Option<ImageBuffers>,
    stage0: Option<IntervalRenderBuffers>,
    stage1: Option<IntervalRenderBuffers>,
}

/// The configuration block passed to compute
///
/// Note: this should be kept in sync with the version in `METAL_PRELUDE`
/// above
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, Zeroable, Pod)]
pub struct RenderConfig {
    /// Total image size, in pixels.  This will be a multiple of `tile_size`.
    pub image_size: u32,

    /// Size of a render tile, in pixels
    pub tile_size: u32,

    /// Number of tiles being rendered in this pass.
    ///
    /// In interval evaluation, each tile corresponds to a single GPU thread;
    /// in pixels evaluation, each tile spawns `tile_size**2` threads.
    pub tile_count: u32,

    /// Index of the X variable in `vars`, or `u32::MAX` if not present
    pub var_index_x: u32,

    /// Index of the Y variable in `vars`, or `u32::MAX` if not present
    pub var_index_y: u32,

    /// Index of the Z variable in `vars`, or `u32::MAX` if not present
    pub var_index_z: u32,

    /// Number of choices in this program
    pub choice_buf_size: u32,
}

impl Render {
    pub fn new(prog: &Program, session: &piet_gpu_hal::Session) -> Self {
        let shader_f = prog.to_metal(Mode::Pixel);
        let shader_i = prog.to_metal(Mode::Interval);

        let config_buf = session
            .create_buffer(
                std::mem::size_of::<RenderConfig>().try_into().unwrap(),
                BufferUsage::STORAGE | BufferUsage::MAP_WRITE,
            )
            .unwrap();

        // SAFETY: it's doing GPU stuff, so who knows?
        unsafe {
            let init = session
                .create_compute_pipeline(
                    ShaderCode::Msl(concat!(
                        include_str!("../shader/prelude.metal"),
                        include_str!("../shader/init.metal")
                    )),
                    &[
                        BindType::BufReadOnly, // config
                        BindType::Buffer,      // tiles
                        BindType::Buffer,      // choices
                    ],
                )
                .unwrap();
            let pixels = session
                .create_compute_pipeline(
                    ShaderCode::Msl(&shader_f),
                    &[
                        BindType::BufReadOnly, // config
                        BindType::BufReadOnly, // tiles
                        BindType::BufReadOnly, // choices
                        BindType::Buffer,      // out
                    ],
                )
                .unwrap();
            let interval = session
                .create_compute_pipeline(
                    ShaderCode::Msl(&shader_i),
                    &[
                        BindType::BufReadOnly, // config
                        BindType::Buffer,      // tiles
                        BindType::Buffer,      // choices
                        BindType::Buffer,      // images
                        BindType::Buffer,      // out
                    ],
                )
                .unwrap();
            let subdivide = session
                .create_compute_pipeline(
                    ShaderCode::Msl(concat!(
                        include_str!("../shader/prelude.metal"),
                        include_str!("../shader/subdivide.metal")
                    )),
                    &[
                        BindType::BufReadOnly, // config
                        BindType::BufReadOnly, // prev
                        BindType::BufReadOnly, // choices_in
                        BindType::Buffer,      // subtiles
                        BindType::Buffer,      // choices_out
                    ],
                )
                .unwrap();
            let merge = session
                .create_compute_pipeline(
                    ShaderCode::Msl(concat!(
                        include_str!("../shader/prelude.metal"),
                        include_str!("../shader/merge.metal")
                    )),
                    &[
                        BindType::BufReadOnly, // config
                        BindType::BufReadOnly, // 64x64
                        BindType::BufReadOnly, // 8x8
                        BindType::BufReadOnly, // 1x1
                        BindType::Buffer,      // out
                    ],
                )
                .unwrap();
            let clear = session
                .create_compute_pipeline(
                    ShaderCode::Msl(concat!(
                        include_str!("../shader/prelude.metal"),
                        include_str!("../shader/clear.metal")
                    )),
                    &[
                        BindType::Buffer, // config
                        BindType::Buffer, // 64x64
                        BindType::Buffer, // 8x8
                        BindType::Buffer, // 1x1
                    ],
                )
                .unwrap();

            Self {
                config: prog.config().clone(),
                config_buf,
                init,
                subdivide,
                interval,
                pixels,
                merge,
                clear,

                image_buf: None,
                stage0: None,
                stage1: None,
            }
        }
    }

    /// # Safety
    /// It's doing GPU stuff, who knows?
    pub unsafe fn do_render(
        &mut self,
        image_size: u32,
        session: &piet_gpu_hal::Session,
    ) {
        let image_buf = if let Some(s) = self.image_buf.as_mut() {
            s.resize_to_fit(image_size, session);
            s
        } else {
            self.image_buf = Some(ImageBuffers::new(image_size, session));
            self.image_buf.as_ref().unwrap()
        };

        ///////////////////////////////////////////////////////////////////////
        // Stage 0: evaluation of 64x64 tiles using interval arithmetic

        // 8-fold subdivision at each stage, i.e. a 2D tile will be split into
        // 64 subtiles.  This must be kept in sync with `prelude.metal`!
        const SPLIT_RATIO: u32 = 8;
        const STAGE0_TILE_SIZE: u32 = SPLIT_RATIO.pow(2);

        let tile_count = (image_size / STAGE0_TILE_SIZE).pow(2);
        println!("Got tile count {:?}", tile_count);
        let stage0_cfg = RenderConfig {
            tile_size: STAGE0_TILE_SIZE,
            image_size,
            tile_count,
            choice_buf_size: ((self.config.choice_count + 15) / 16)
                .try_into()
                .unwrap(),
            var_index_x: usize::from(self.config.vars["X"])
                .try_into()
                .unwrap_or(u32::MAX),
            var_index_y: usize::from(self.config.vars["Y"])
                .try_into()
                .unwrap_or(u32::MAX),
            var_index_z: u32::MAX,
        };
        self.config_buf
            .write(std::slice::from_ref(&stage0_cfg))
            .unwrap();

        // Build Stage0 renderer, or resize an existing renderer to fit
        let stage0 = if let Some(s) = self.stage0.as_mut() {
            s.resize_to_fit(stage0_cfg.tile_count, session);
            s
        } else {
            self.stage0 = Some(IntervalRenderBuffers::new(
                stage0_cfg.tile_count,
                self.config.choice_count.try_into().unwrap(),
                session,
            ));
            self.stage0.as_ref().unwrap()
        };

        let clear_descriptor_set = session
            .create_simple_descriptor_set(
                &self.merge,
                &[
                    &self.config_buf,
                    &image_buf.image_64x64,
                    &image_buf.image_8x8,
                    &image_buf.image_1x1,
                ],
            )
            .unwrap();
        let init_descriptor_set = session
            .create_simple_descriptor_set(
                &self.init,
                &[
                    &self.config_buf,
                    &stage0.tiles,
                    &stage0.choices,
                    &stage0.out,
                ],
            )
            .unwrap();
        let stage0_descriptor_set = session
            .create_simple_descriptor_set(
                &self.interval,
                &[
                    &self.config_buf,
                    &stage0.tiles,
                    &stage0.choices,
                    &image_buf.image_64x64,
                    &stage0.out,
                ],
            )
            .unwrap();

        let query_pool = session.create_query_pool(2).unwrap();

        let mut cmd_buf = session.cmd_buf().unwrap();
        cmd_buf.begin();
        cmd_buf.reset_query_pool(&query_pool);
        {
            let mut pass = cmd_buf.begin_compute_pass(
                &ComputePassDescriptor::timer(&query_pool, 0, 1),
            );
            pass.dispatch(
                &self.clear,
                &clear_descriptor_set,
                ((image_size / 8).pow(2), 1, 1),
                (64, 1, 1),
            );
            pass.dispatch(
                &self.init,
                &init_descriptor_set,
                (((tile_count + 7) / 8), 1, 1),
                (8, 1, 1),
            );
            pass.dispatch(
                &self.interval,
                &stage0_descriptor_set,
                (((tile_count + 7) / 8), 1, 1),
                (8, 1, 1),
            );
            pass.end();
        }

        cmd_buf.finish_timestamps(&query_pool);
        cmd_buf.host_barrier();
        cmd_buf.finish();

        let submitted = session.run_cmd_buf(cmd_buf, &[], &[]).unwrap();
        submitted.wait().unwrap();
        let timestamps = session.fetch_query_pool(&query_pool);
        println!("stage 0: {:?}", timestamps);

        let active_tile_count =
            stage0.out.map_read(0..4).unwrap().cast_slice::<u32>()[0];
        println!(
            "active tiles: {} / {}",
            active_tile_count, stage0_cfg.tile_count
        );

        ///////////////////////////////////////////////////////////////////////
        // Stage 1: evaluation of 8x8 tiles using interval arithmetic
        let stage1_cfg = RenderConfig {
            tile_size: stage0_cfg.tile_size / SPLIT_RATIO,
            tile_count: active_tile_count * SPLIT_RATIO.pow(2),
            ..stage0_cfg
        };
        self.config_buf
            .write(std::slice::from_ref(&stage1_cfg))
            .unwrap();

        // Build stage1 renderer, or resize an existing renderer to fit
        let stage1 = if let Some(s) = self.stage1.as_mut() {
            s.resize_to_fit(stage1_cfg.tile_count, session);
            s
        } else {
            self.stage1 = Some(IntervalRenderBuffers::new(
                stage1_cfg.tile_count,
                self.config.choice_count.try_into().unwrap(),
                session,
            ));
            self.stage1.as_ref().unwrap()
        };

        let subdiv_descriptor_set = session
            .create_simple_descriptor_set(
                &self.init,
                &[
                    &self.config_buf,
                    &stage0.out,
                    &stage0.choices,
                    &stage1.tiles,
                    &stage1.choices,
                    &stage1.out,
                ],
            )
            .unwrap();
        let stage1_descriptor_set = session
            .create_simple_descriptor_set(
                &self.interval,
                &[
                    &self.config_buf,
                    &stage1.tiles,
                    &stage1.choices,
                    &image_buf.image_8x8,
                    &stage1.out,
                ],
            )
            .unwrap();

        let query_pool = session.create_query_pool(2).unwrap();
        let mut cmd_buf = session.cmd_buf().unwrap();
        cmd_buf.begin();
        cmd_buf.reset_query_pool(&query_pool);
        {
            let mut pass = cmd_buf.begin_compute_pass(
                &ComputePassDescriptor::timer(&query_pool, 0, 1),
            );
            pass.dispatch(
                &self.subdivide,
                &subdiv_descriptor_set,
                (active_tile_count, 1, 1),
                (SPLIT_RATIO.pow(2), 1, 1),
            );
            pass.dispatch(
                &self.interval,
                &stage1_descriptor_set,
                (active_tile_count, 1, 1),
                (SPLIT_RATIO.pow(2), 1, 1),
            );
            pass.end();
        }
        cmd_buf.finish_timestamps(&query_pool);
        cmd_buf.host_barrier();
        cmd_buf.finish();

        let submitted = session.run_cmd_buf(cmd_buf, &[], &[]).unwrap();
        submitted.wait().unwrap();

        let timestamps = session.fetch_query_pool(&query_pool);
        println!("stage 1: {:?}", timestamps);

        let active_subtile_count =
            stage1.out.map_read(0..4).unwrap().cast_slice::<u32>()[0];
        println!(
            "active tiles: {} / {}",
            active_subtile_count, stage1_cfg.tile_count
        );

        ///////////////////////////////////////////////////////////////////////
        // Stage 2: Per-pixel evaluation of remaining subtiles
        //
        // This step reuses the stage 1 configuration, but ignores its
        // tile_count in favor of the active_tiles member of stage1.out
        let stage2_descriptor_set = session
            .create_simple_descriptor_set(
                &self.pixels,
                &[
                    &self.config_buf,
                    &stage1.out,
                    &stage1.choices,
                    &image_buf.image_1x1,
                ],
            )
            .unwrap();
        let merge_descriptor_set = session
            .create_simple_descriptor_set(
                &self.merge,
                &[
                    &self.config_buf,
                    &image_buf.image_64x64,
                    &image_buf.image_8x8,
                    &image_buf.image_1x1,
                    &image_buf.final_image,
                ],
            )
            .unwrap();

        let query_pool = session.create_query_pool(2).unwrap();
        let mut cmd_buf = session.cmd_buf().unwrap();
        cmd_buf.begin();
        cmd_buf.reset_query_pool(&query_pool);
        {
            let mut pass = cmd_buf.begin_compute_pass(
                &ComputePassDescriptor::timer(&query_pool, 0, 1),
            );
            pass.dispatch(
                &self.pixels,
                &stage2_descriptor_set,
                (active_subtile_count, 1, 1),
                (SPLIT_RATIO.pow(2), 1, 1),
            );
            pass.dispatch(
                &self.merge,
                &merge_descriptor_set,
                ((image_size / 8).pow(2), 1, 1),
                (64, 1, 1),
            );
            pass.end();
        }
        cmd_buf.finish_timestamps(&query_pool);
        cmd_buf.host_barrier();
        cmd_buf.finish();

        let submitted = session.run_cmd_buf(cmd_buf, &[], &[]).unwrap();
        submitted.wait().unwrap();
        let timestamps = session.fetch_query_pool(&query_pool);
        println!("stage 2: {:?}", timestamps);
    }

    /// Reads the final image from the render buffer
    pub unsafe fn load_image(&self) -> Vec<[u8; 4]> {
        let mut out: Vec<[u8; 4]> = vec![];
        self.image_buf
            .as_ref()
            .unwrap()
            .final_image
            .read(&mut out)
            .unwrap();
        out
    }
}
