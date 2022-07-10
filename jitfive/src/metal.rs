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
            Mode::Pixel => "const device uint8_t*",
            Mode::Interval => "device uint8_t*",
        }
    }
}

impl Function {
    fn declaration(&self, mode: Mode) -> String {
        let mut out = String::new();
        out += &formatdoc!(
            "
            inline {} t_shape_{}(
                {} vars, {} choices",
            if self.root { mode.local_type() } else { "void" },
            self.index,
            mode.vars_type(),
            mode.choice_type(),
        );
        let mut first = true;
        for i in &self.inputs {
            if first {
                out += ",\n    ";
            } else {
                out += ", ";
            }
            first = false;
            out += &format!("const {} v{}", mode.local_type(), usize::from(*i));
        }
        let mut first = true;
        for i in &self.outputs {
            if first {
                out += ",\n    ";
            } else {
                out += ", ";
            }
            first = false;
            out +=
                &format!("thread {}& v{}", mode.local_type(), usize::from(*i));
        }
        out += "\n)";
        out
    }
    /// Generates text to call a function
    fn call(&self) -> String {
        let mut out = String::new();
        out += &format!("t_shape_{}(vars, choices", self.index);

        for i in &self.inputs {
            out += &format!(", v{}", usize::from(*i));
        }
        for i in &self.outputs {
            out += &format!(", v{}", usize::from(*i));
        }
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
                let mut switch = formatdoc!(
                    "switch (choices[{}]) {{
                        case LHS: v{out} = v{}; break;
                        case RHS: v{out} = v{}; break;
                        default: ",
                    usize::from(*choice),
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
                        let b = usize::from(*lhs);
                        let choice = usize::from(*choice);
                        match self {
                            Self::Max { .. } => formatdoc!(
                                "
                            if (v{a}[0] > v{b}[1]) {{
                                choices[{choice}] = LHS;
                                v{out} = v{a};
                            }} else if (v{b}[0] > v{a}[1]) {{
                                choices[{choice}] = RHS;
                                v{out} = v{b};
                            }} else {{
                                v{out} = i_max(v{a}, v{b});
                            }}
                            "
                            ),
                            Self::Min { .. } => formatdoc!(
                                "
                            if (v{a}[1] < v{b}[0]) {{
                                choices[{choice}] = LHS;
                                v{out} = v{a};
                            }} else if (v{b}[1] < v{a}[0]) {{
                                choices[{choice}] = RHS;
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
            #define CHOICE_COUNT {}
            ",
            self.config().var_count,
            self.config().choice_count,
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
                if cond.len() > 1 {
                    let mut first = true;
                    for c in cond {
                        if first {
                            first = false;
                        } else {
                            out += " || ";
                        }
                        out += &format!(
                            "(choices[{}] & {})",
                            usize::from(c.0),
                            c.1.to_metal()
                        );
                    }
                } else {
                    out += &format!(
                        "choices[{}] & {}",
                        usize::from(cond[0].0),
                        cond[0].1.to_metal()
                    );
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
    config: piet_gpu_hal::Buffer,
    tiles: piet_gpu_hal::Buffer,
    choices: piet_gpu_hal::Buffer,
    image: piet_gpu_hal::Buffer,
    out: piet_gpu_hal::Buffer, // RenderOut
}

impl IntervalRenderBuffers {
    fn new(config: &RenderConfig, session: &piet_gpu_hal::Session) -> Self {
        let tiles = session
            .create_buffer(
                (config.tile_count as usize * std::mem::size_of::<u32>())
                    .try_into()
                    .unwrap(),
                BufferUsage::STORAGE | BufferUsage::MAP_READ,
            )
            .unwrap();
        let choices = session
            .create_buffer(
                (config.tile_count * config.choice_count)
                    .try_into()
                    .unwrap(),
                BufferUsage::STORAGE,
            )
            .unwrap();

        assert_eq!(config.image_size % config.tile_size, 0);
        let image = session
            .create_buffer(
                (config.image_size / config.tile_size)
                    .pow(2)
                    .try_into()
                    .unwrap(),
                BufferUsage::STORAGE,
            )
            .unwrap();

        let out = session
            .create_buffer(
                ((config.tile_count as usize + 1) * std::mem::size_of::<u32>())
                    .try_into()
                    .unwrap(),
                BufferUsage::STORAGE | BufferUsage::MAP_READ,
            )
            .unwrap();

        let config = session
            .create_buffer_init(
                std::slice::from_ref(config),
                BufferUsage::STORAGE,
            )
            .unwrap();

        Self {
            config,
            tiles,
            choices,
            image,
            out,
        }
    }
}

pub struct Render {
    config: Config,

    /// Compute pipelines to initialize before the first interval evaluation
    init: piet_gpu_hal::Pipeline,

    /// Compute pipeline to perform interval evaluation
    interval: piet_gpu_hal::Pipeline,

    /// Compute pipelines to subdivide the results of interval evaluation
    subdivide: piet_gpu_hal::Pipeline,

    /// Compute pipelines to render individual pixels
    pixels: piet_gpu_hal::Pipeline,
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

    /// Subdivision factor between interval evaluation stages (1D).
    ///
    /// For example, if this is 8, each 2D tile will be split into 8x8 = 64
    /// subtiles during subdivision.
    pub split_ratio: u32,

    /// Index of the X variable in `vars`, or `u32::MAX` if not present
    pub var_index_x: u32,

    /// Index of the Y variable in `vars`, or `u32::MAX` if not present
    pub var_index_y: u32,

    /// Index of the Z variable in `vars`, or `u32::MAX` if not present
    pub var_index_z: u32,

    /// Number of choices in this program
    pub choice_count: u32,
}

impl Render {
    pub fn new(prog: &Program, session: &piet_gpu_hal::Session) -> Self {
        let shader_f = prog.to_metal(Mode::Pixel);
        let shader_i = prog.to_metal(Mode::Interval);

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

            Self {
                config: prog.config().clone(),
                init,
                subdivide,
                interval,
                pixels,
            }
        }
    }

    /// # Safety
    /// It's doing GPU stuff, who knows?
    pub unsafe fn render(
        &mut self,
        image_size: u32,
        session: &piet_gpu_hal::Session,
    ) -> Vec<[u8; 4]> {
        ///////////////////////////////////////////////////////////////////////
        // Stage 0: evaluation of 64x64 tiles using interval arithmetic
        const STAGE0_TILE_SIZE: u32 = 64;
        // 8-fold subdivision at each stage
        const SPLIT_RATIO: u32 = 8;

        let tile_count = (image_size / STAGE0_TILE_SIZE).pow(2);
        println!("Got tile count {:?}", tile_count);
        let stage0_cfg = RenderConfig {
            tile_size: STAGE0_TILE_SIZE,
            image_size,
            tile_count,
            split_ratio: SPLIT_RATIO,
            choice_count: self.config.choice_count.try_into().unwrap(),
            var_index_x: usize::from(self.config.vars["X"])
                .try_into()
                .unwrap_or(u32::MAX),
            var_index_y: usize::from(self.config.vars["Y"])
                .try_into()
                .unwrap_or(u32::MAX),
            var_index_z: u32::MAX,
        };
        let stage0 = IntervalRenderBuffers::new(&stage0_cfg, session);

        let init_descriptor_set = session
            .create_simple_descriptor_set(
                &self.init,
                &[&stage0.config, &stage0.tiles, &stage0.choices],
            )
            .unwrap();
        let stage0_descriptor_set = session
            .create_simple_descriptor_set(
                &self.interval,
                &[
                    &stage0.config,
                    &stage0.tiles,
                    &stage0.choices,
                    &stage0.image,
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

        let mut out: Vec<u32> = vec![];
        stage0.out.read(&mut out).unwrap();
        println!("Stage 0 out: {:x?}", out);
        stage0.tiles.read(&mut out).unwrap();
        println!("Stage 0 tiles: {:x?}", out);

        ///////////////////////////////////////////////////////////////////////
        // Stage 1: evaluation of 8x8 tiles using interval arithmetic
        let stage1_cfg = RenderConfig {
            tile_size: stage0_cfg.tile_size / SPLIT_RATIO,
            tile_count: active_tile_count * SPLIT_RATIO.pow(2),
            ..stage0_cfg
        };
        println!("{:#?}", stage1_cfg);
        let stage1 = IntervalRenderBuffers::new(&stage1_cfg, session);

        let subdiv_descriptor_set = session
            .create_simple_descriptor_set(
                &self.init,
                &[
                    &stage0.config,
                    &stage0.out,
                    &stage0.choices,
                    &stage1.tiles,
                    &stage1.choices,
                ],
            )
            .unwrap();
        let stage1_descriptor_set = session
            .create_simple_descriptor_set(
                &self.interval,
                &[
                    &stage1.config,
                    &stage1.tiles,
                    &stage1.choices,
                    &stage1.image,
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

        let mut out: Vec<u32> = vec![];
        stage1.out.read(&mut out).unwrap();
        println!("Stage 1 out: {:x?}", out);
        stage1.tiles.read(&mut out).unwrap();
        println!("Stage 1 tiles: {:x?}", out);

        ///////////////////////////////////////////////////////////////////////
        // Stage 2: Per-pixel evaluation of remaining subtiles
        //
        // This step reuses the stage 1 configuration, but ignores its
        // tile_count in favor of the active_tiles member of stage1.out
        let image = session
            .create_buffer(
                image_size.pow(2).try_into().unwrap(),
                BufferUsage::STORAGE | BufferUsage::MAP_READ,
            )
            .unwrap();
        let stage2_descriptor_set = session
            .create_simple_descriptor_set(
                &self.pixels,
                &[&stage1.config, &stage1.out, &stage1.choices, &image],
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
            pass.end();
        }
        cmd_buf.finish_timestamps(&query_pool);
        cmd_buf.host_barrier();
        cmd_buf.finish();

        let submitted = session.run_cmd_buf(cmd_buf, &[], &[]).unwrap();
        submitted.wait().unwrap();
        let timestamps = session.fetch_query_pool(&query_pool);
        println!("stage 2: {:?}", timestamps);

        let mut out: Vec<u8> = vec![];
        image.read(&mut out).unwrap();

        out.into_iter()
            .map(|i| match i {
                0 => [0, 0, 0, 0xFF],
                1 => [0xFF, 0xFF, 0xFF, 0xFF],
                2 => [0x88, 0x88, 0x88, 0xFF],
                _ => [0xFF, 0, 0, 0xFF],
            })
            .collect()
    }
}
