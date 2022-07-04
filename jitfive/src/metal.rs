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
    Float,
    Interval,
}

impl Mode {
    fn vars_type(&self) -> &str {
        match self {
            Mode::Float => "const device float*",
            Mode::Interval => "const device float2*",
        }
    }
    fn local_type(&self) -> &str {
        match self {
            Mode::Float => "float",
            Mode::Interval => "float2",
        }
    }
    fn choice_type(&self) -> &str {
        match self {
            Mode::Float => "const device uint8_t*",
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
        Some(match self {
            Self::Var { var, .. } => {
                format!("v{} = t_var(vars[{}]);", out, usize::from(*var))
            }
            Self::Const { value, .. } => {
                format!("v{} = t_const({});", out, value)
            }
            Self::Mul { lhs, rhs, .. } | Self::Add { lhs, rhs, .. } => {
                format!(
                    "v{} = t_{}(v{}, v{});",
                    out,
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
                    "switch (choices[{0}]) {{
                        case LHS: v{1} = v{2}; break;
                        case RHS: v{1} = v{3}; break;
                        default: ",
                    usize::from(*choice),
                    out,
                    usize::from(*lhs),
                    usize::from(*rhs),
                );
                switch += &match mode {
                    Mode::Float => format!(
                        "v{} = t_{}(v{}, v{}); break;",
                        out,
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
                                v{out} = t_max(v{a}, v{b});
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
                                v{out} = t_min(v{a}, v{b});
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
                format!("v{} = t_{}(v{});", out, self.name(), usize::from(*reg))
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
            #define VAR_COUNT {}
            #define CHOICE_COUNT {}
            {}
            ",
            self.config().var_count,
            self.config().choice_count,
            match mode {
                Mode::Float => METAL_PRELUDE_FLOAT,
                Mode::Interval => METAL_PRELUDE_INTERVAL,
            }
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
        }}",
            mode.local_type(),
            mode.vars_type(),
            mode.choice_type(),
            functions.get(&vec![]).unwrap().index
        );
        out += match mode {
            Mode::Float => METAL_KERNEL_FLOAT,
            Mode::Interval => METAL_KERNEL_INTERVAL,
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

const METAL_PRELUDE_FLOAT: &str = r#"
// Prelude
#include <metal_stdlib>

#define RHS 1
#define LHS 2

inline float t_mul(const float a, const float b) {
    return a * b;
}
inline float t_add(const float a, const float b) {
    return a + b;
}

inline float t_min(const float a, const float b) {
    return metal::fmin(a, b);
}
inline float t_max(const float a, const float b) {
    return metal::fmax(a, b);
}
inline float t_neg(const float a) {
    return -a;
}
inline float t_sqrt(const float a) {
    return metal::sqrt(a);
}
inline float t_const(const float a) {
    return a;
}
inline float t_var(const float a) {
    return a;
}
"#;

const METAL_PRELUDE_INTERVAL: &str = r#"
// Interval prelude
#include <metal_stdlib>

#define RHS 1
#define LHS 2

inline float2 t_mul(const float2 a, const float2 b) {
    if (a[0] < 0.0f) {
        if (a[1] > 0.0f) {
            if (b[0] < 0.0f) {
                if (b[1] > 0.0f) { // M * M
                    return float2(metal::fmin(a[0] * b[1], a[1] * b[0]),
                                  metal::fmax(a[0] * b[0], a[1] * b[1]));
                } else { // M * N
                    return float2(a[1] * b[0], a[0] * b[0]);
                }
            } else {
                if (b[1] > 0.0f) { // M * P
                    return float2(a[0] * b[1], a[1] * b[1]);
                } else { // M * Z
                    return float2(0.0f, 0.0f);
                }
            }
        } else {
            if (b[0] < 0.0f) {
                if (b[1] > 0.0f) { // N * M
                    return float2(a[0] * b[1], a[0] * b[0]);
                } else { // N * N
                    return float2(a[1] * b[1], a[0] * b[0]);
                }
            } else {
                if (b[1] > 0.0f) { // N * P
                    return float2(a[0] * b[1], a[1] * b[0]);
                } else { // N * Z
                    return float2(0.0f, 0.0f);
                }
            }
        }
    } else {
        if (a[1] > 0.0f) {
            if (b[0] < 0.0f) {
                if (b[1] > 0.0f) { // P * M
                    return float2(a[1] * b[0], a[1] * b[1]);
                } else {// P * N
                    return float2(a[1] * b[0], a[0] * b[1]);
                }
            } else {
                if (b[1] > 0.0f) { // P * P
                    return float2(a[0] * b[0], a[1] * b[1]);
                } else {// P * Z
                    return float2(0.0f, 0.0f);
                }
            }
        } else { // Z * ?
            return float2(0.0f, 0.0f);
        }
    }
}
inline float2 t_add(const float2 a, const float2 b) {
    return a + b;
}
inline float2 t_min(const float2 a, const float2 b) {
    return metal::fmin(a, b);
}
inline float2 t_max(const float2 a, const float2 b) {
    return metal::fmax(a, b);
}
inline float2 t_neg(const float2 a) {
    return float2(-a[1], -a[0]);
}
inline float2 t_sqrt(const float2 a) {
    if (a[1] < 0.0) {
        return float2(-1e8, 1e8); // XXX
    } else if (a[0] <= 0.0) {
        return float2(0.0, metal::sqrt(a[1]));
    } else {
        return float2(metal::sqrt(a[0]), metal::sqrt(a[1]));
    }
}
inline float2 t_const(const float a) {
    return float2(a, a);
}
inline float2 t_var(const float2 a) {
    return a;
}
"#;

const METAL_KERNEL_FLOAT: &str = r#"
kernel void main0(const device float* vars [[buffer(0)]],
                  const device uint8_t* choices [[buffer(1)]],
                  device float* result [[buffer(2)]],
                  uint index [[thread_position_in_grid]])
{
    result[index] = t_eval(&vars[index * VAR_COUNT],
                           &choices[index * CHOICE_COUNT]);
}
"#;
const METAL_KERNEL_INTERVAL: &str = r#"
kernel void main0(const device float2* vars [[buffer(0)]],
                  device uint8_t* choices [[buffer(1)]],
                  device float2* result [[buffer(2)]],
                  uint index [[thread_position_in_grid]])
{
    result[index] = t_eval(&vars[index * VAR_COUNT],
                           &choices[index * CHOICE_COUNT]);
}
"#;

// TODO:
/*
#define VC const device float* vars, const device uint8_t* choices
#define IF inline float
#define IV inline void
#define CF const float
#define TF thread float&
*/

////////////////////////////////////////////////////////////////////////////////

use piet_gpu_hal::{BindType, BufferUsage, ComputePassDescriptor, ShaderCode};

pub struct Render {
    config: Config,

    // Working memory
    choice_buf: piet_gpu_hal::Buffer,
    var_buf: piet_gpu_hal::Buffer,
    out_buf: piet_gpu_hal::Buffer,

    interval: piet_gpu_hal::Pipeline,
    pixels: piet_gpu_hal::Pipeline,
}

/// The configuration block passed to compute
///
/// Note: this should be kept in sync with the version in `METAL_PRELUDE`
/// above
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, Zeroable, Pod)]
pub struct RenderConfig {
    pub size_pixels: u32,
    // TODO
}

impl Render {
    pub unsafe fn new(prog: &Program, session: &piet_gpu_hal::Session) -> Self {
        let out_buf = session
            .create_buffer(8, BufferUsage::STORAGE | BufferUsage::MAP_READ)
            .unwrap();
        let var_buf = session.create_buffer(8, BufferUsage::STORAGE).unwrap();
        let choice_buf =
            session.create_buffer(8, BufferUsage::STORAGE).unwrap();

        let shader_f = prog.to_metal(Mode::Float);
        println!("{}", shader_f);
        let pixels = session
            .create_compute_pipeline(
                ShaderCode::Msl(&shader_f),
                &[
                    BindType::BufReadOnly,
                    BindType::BufReadOnly, // choices
                    BindType::Buffer,      // out
                ],
            )
            .unwrap();

        let shader_i = prog.to_metal(Mode::Interval);
        println!("{}", shader_i);
        let interval = session
            .create_compute_pipeline(
                ShaderCode::Msl(&shader_i),
                &[
                    BindType::BufReadOnly,
                    BindType::Buffer, // choices
                    BindType::Buffer, // out
                ],
            )
            .unwrap();
        Self {
            config: prog.config().clone(),
            choice_buf,
            var_buf,
            out_buf,
            interval,
            pixels,
        }
    }
    pub unsafe fn render(
        &mut self,
        size: usize,
        session: &piet_gpu_hal::Session,
    ) -> Vec<bool> {
        assert_eq!(size % 64, 0, "Size must be a multiple of 64");
        let thread_count = size * size;

        // Initialize variables
        let mut vars: Vec<f32> = vec![];
        for x in 0..size {
            let x = 1.0 - ((x as f32) / (size - 1) as f32) * 2.0;
            for y in 0..size {
                let y = ((y as f32) / (size - 1) as f32) * 2.0 - 1.0;
                vars.push(x);
                vars.push(y);
            }
        }
        // TODO: initialize choices and vars with a compute shader
        if vars.len() * std::mem::size_of::<f32>()
            > self.var_buf.size().try_into().unwrap()
        {
            self.var_buf = session
                .create_buffer_init(
                    &vars,
                    BufferUsage::MAP_WRITE | BufferUsage::STORAGE,
                )
                .unwrap();
        } else {
            self.var_buf.write(&vars).unwrap();
        }
        let choices = std::iter::repeat(0b11)
            .take(thread_count * self.config.choice_count)
            .collect::<Vec<u8>>();
        if choices.len() * std::mem::size_of::<u8>()
            > self.choice_buf.size().try_into().unwrap()
        {
            self.choice_buf = session
                .create_buffer_init(
                    &choices,
                    BufferUsage::MAP_WRITE | BufferUsage::STORAGE,
                )
                .unwrap();
        } else {
            self.choice_buf.write(&choices).unwrap();
        }

        // Resize out buffer to fit one value per thread
        let out_buf_size =
            u64::try_from(thread_count * std::mem::size_of::<f32>()).unwrap();
        if out_buf_size > self.out_buf.size() {
            self.out_buf = session
                .create_buffer(
                    out_buf_size,
                    BufferUsage::STORAGE | BufferUsage::MAP_READ,
                )
                .unwrap();
        }

        let descriptor_set = session
            .create_simple_descriptor_set(
                &self.pixels,
                &[&self.var_buf, &self.choice_buf, &self.out_buf],
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
                &descriptor_set,
                (u32::try_from(thread_count / 64).unwrap(), 1, 1),
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

        let mut dst: Vec<f32> = vec![];
        self.out_buf.read(&mut dst).unwrap();
        println!("{:?}", timestamps);

        dst.into_iter().map(|i| i < 0.0).collect()
    }
}
