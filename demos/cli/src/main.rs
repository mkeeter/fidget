use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use env_logger::Env;
use log::info;

use fidget::context::Context;

/// Simple test program
#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    cmd: Command,

    /// Input file
    #[clap(short, long)]
    input: PathBuf,
}

#[derive(Subcommand)]
enum Command {
    Render2d {
        #[clap(flatten)]
        settings: ImageSettings,

        /// Use brute-force (pixel-by-pixel) evaluation
        #[clap(short, long)]
        brute: bool,

        /// Render as a color-gradient SDF
        #[clap(long)]
        sdf: bool,
    },

    Render3d {
        #[clap(flatten)]
        settings: ImageSettings,

        /// Render in color
        #[clap(long)]
        color: bool,

        /// Render using an isometric perspective
        #[clap(long)]
        isometric: bool,
    },
    Mesh {
        #[clap(flatten)]
        settings: MeshSettings,
    },
}

#[derive(ValueEnum, Clone)]
enum EvalMode {
    Vm,

    #[cfg(feature = "jit")]
    Jit,
}

#[derive(Parser)]
struct ImageSettings {
    /// Name of a `.png` file to write
    #[clap(short, long)]
    out: Option<PathBuf>,

    /// Evaluator flavor
    #[clap(short, long, value_enum, default_value_t = EvalMode::Vm)]
    eval: EvalMode,

    /// Perform per-pixel evaluation using WGPU
    ///
    /// (interval evaluation still uses the mode specified by `--eval`)
    #[clap(long)]
    wgpu: bool,

    /// Number of threads to use
    #[clap(short, long)]
    threads: Option<NonZeroUsize>,

    /// Number of times to render (for benchmarking)
    #[clap(short = 'N', default_value_t = 1)]
    n: usize,

    /// Image size
    #[clap(short, long, default_value_t = 128)]
    size: u32,
}

#[derive(Parser)]
struct MeshSettings {
    /// Octree depth
    #[clap(short, long)]
    depth: u8,

    /// Name of a `.stl` file to write
    #[clap(short, long)]
    out: Option<PathBuf>,

    /// Evaluator flavor
    #[clap(short, long, value_enum, default_value_t = EvalMode::Vm)]
    eval: EvalMode,

    /// Number of threads to use
    #[clap(short, long, default_value_t = NonZeroUsize::new(8).unwrap())]
    threads: NonZeroUsize,

    /// Number of times to render (for benchmarking)
    #[clap(short = 'N', default_value_t = 1)]
    n: usize,
}

////////////////////////////////////////////////////////////////////////////////
fn run3d<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &ImageSettings,
    isometric: bool,
    mode_color: bool,
) -> Vec<u8> {
    let mut mat = nalgebra::Transform3::identity();
    if !isometric {
        *mat.matrix_mut().get_mut((3, 2)).unwrap() = 0.3;
    }
    let pool: Option<rayon::ThreadPool>;
    let threads = match settings.threads {
        Some(n) if n.get() == 1 => None,
        Some(n) => {
            pool = Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n.get())
                    .build()
                    .unwrap(),
            );
            pool.as_ref().map(fidget::render::ThreadPool::Custom)
        }
        None => Some(fidget::render::ThreadPool::Global),
    };
    let cfg = fidget::render::VoxelRenderConfig {
        image_size: fidget::render::VoxelSize::from(settings.size),
        tile_sizes: F::tile_sizes_3d(),
        threads,
        ..Default::default()
    };
    let shape = shape.apply_transform(mat.into());

    let mut depth = vec![];
    let mut color = vec![];
    for _ in 0..settings.n {
        (depth, color) = cfg.run(shape.clone());
    }

    let out = if mode_color {
        depth
            .into_iter()
            .zip(color)
            .flat_map(|(d, p)| {
                if d > 0 {
                    [p[0], p[1], p[2], 255]
                } else {
                    [0, 0, 0, 0]
                }
            })
            .collect()
    } else {
        let z_max = depth.iter().max().cloned().unwrap_or(1);
        depth
            .into_iter()
            .flat_map(|p| {
                if p > 0 {
                    let z = (p * 255 / z_max) as u8;
                    [z, z, z, 255]
                } else {
                    [0, 0, 0, 0]
                }
            })
            .collect()
    };

    out
}

////////////////////////////////////////////////////////////////////////////////

fn run2d<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &ImageSettings,
    brute: bool,
    sdf: bool,
) -> Vec<u8> {
    if brute {
        let tape = shape.float_slice_tape(Default::default());
        let mut eval = fidget::shape::Shape::<F>::new_float_slice_eval();
        let mut out: Vec<bool> = vec![];
        for _ in 0..settings.n {
            let mut xs = vec![];
            let mut ys = vec![];
            let div = (settings.size - 1) as f64;
            for i in 0..settings.size {
                let y = -(-1.0 + 2.0 * (i as f64) / div);
                for j in 0..settings.size {
                    let x = -1.0 + 2.0 * (j as f64) / div;
                    xs.push(x as f32);
                    ys.push(y as f32);
                }
            }
            let zs = vec![0.0; xs.len()];
            let values = eval.eval(&tape, &xs, &ys, &zs).unwrap();
            out = values.iter().map(|v| *v <= 0.0).collect();
        }
        // Convert from Vec<bool> to an image
        out.into_iter()
            .map(|b| if b { [u8::MAX; 4] } else { [0, 0, 0, 255] })
            .flat_map(|i| i.into_iter())
            .collect()
    } else {
        let pool: Option<rayon::ThreadPool>;
        let threads = match settings.threads {
            Some(n) if n.get() == 1 => None,
            Some(n) => {
                pool = Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(n.get())
                        .build()
                        .unwrap(),
                );
                pool.as_ref().map(fidget::render::ThreadPool::Custom)
            }
            None => Some(fidget::render::ThreadPool::Global),
        };
        let cfg = fidget::render::ImageRenderConfig {
            image_size: fidget::render::ImageSize::from(settings.size),
            tile_sizes: F::tile_sizes_2d(),
            threads,
            ..Default::default()
        };
        if sdf {
            let mut image = vec![];
            for _ in 0..settings.n {
                image =
                    cfg.run::<_, fidget::render::SdfRenderMode>(shape.clone());
            }
            image
                .into_iter()
                .flat_map(|a| [a[0], a[1], a[2], 255].into_iter())
                .collect()
        } else {
            let mut image = vec![];
            for _ in 0..settings.n {
                image = cfg
                    .run::<_, fidget::render::DebugRenderMode>(shape.clone());
            }
            image
                .into_iter()
                .flat_map(|p| p.as_debug_color().into_iter())
                .collect()
        }
    }
}

fn run_wgpu<F: fidget::eval::MathFunction + fidget::shape::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &ImageSettings,
    brute: bool,
    sdf: bool,
) -> Vec<u8> {
    if brute {
        run_wgpu_brute(shape, settings)
    } else {
        run_wgpu_smart(shape, settings)
    }
}

const WGPU_SHADER_BASE: &str = r#"
fn nan_f32() -> f32 {
    return bitcast<f32>(0x7FC00000);
}

fn compare_f32(lhs: f32, rhs: f32) -> f32 {
    if (lhs < rhs) {
        return -1.0;
    } else if (lhs > rhs) {
        return 1.0;
    } else if (lhs == rhs) {
        return 0.0;
    } else {
        return nan_f32();
    }
}

fn and_f32(lhs: f32, rhs: f32) -> f32 {
    if lhs == 0.0 {
        return rhs;
    } else {
        return lhs;
    }
}

fn or_f32(lhs: f32, rhs: f32) -> f32 {
    if lhs != 0.0 {
        return rhs;
    } else {
        return lhs;
    }
}

fn read_imm_f32(i: ptr<function, u32>) -> f32 {
    let imm = bitcast<f32>(tape[*i]);
    *i = *i + 1;
    return imm;
}

fn run_tape(start: u32, inputs: vec4<f32>) -> f32 {
    var i: u32 = start;
    var reg: array<f32, 256>;
    while (true) {
        let op = unpack4xU8(tape[i]);
        i = i + 1;
        switch op[0] {
            case OP_Output: {
                // XXX we're not actually writing to an output slot here
                let imm = tape[i];
                i = i + 1; 
                return reg[op[1]];
            }
            case OP_Input: {
                let imm = tape[i];
                i = i + 1;
                reg[op[1]] = inputs[imm];
            }
            case OP_CopyReg:    { reg[op[1]] = reg[op[2]]; }
            case OP_CopyImm:    { reg[op[1]] = read_imm_f32(&i); }
            case OP_NegReg:     { reg[op[1]] = -reg[op[2]]; }
            case OP_AbsReg:     { reg[op[1]] = abs(reg[op[2]]); }
            case OP_RecipReg:   { reg[op[1]] = 1.0 / reg[op[2]]; }
            case OP_SqrtReg:    { reg[op[1]] = sqrt(reg[op[2]]); }
            case OP_SquareReg: {
                let v = reg[op[2]];
                reg[op[1]] = v * v;
            }
            case OP_FloorReg:   { reg[op[1]] = floor(reg[op[2]]); }
            case OP_CeilReg:    { reg[op[1]] = ceil(reg[op[2]]); }
            case OP_RoundReg:   { reg[op[1]] = round(reg[op[2]]); }
            case OP_SinReg:     { reg[op[1]] = sin(reg[op[2]]); }
            case OP_CosReg:     { reg[op[1]] = cos(reg[op[2]]); }
            case OP_TanReg:     { reg[op[1]] = tan(reg[op[2]]); }
            case OP_AsinReg:    { reg[op[1]] = asin(reg[op[2]]); }
            case OP_AcosReg:    { reg[op[1]] = acos(reg[op[2]]); }
            case OP_AtanReg:    { reg[op[1]] = atan(reg[op[2]]); }
            case OP_ExpReg:     { reg[op[1]] = exp(reg[op[2]]); }
            case OP_LnReg:      { reg[op[1]] = log(reg[op[2]]); }
            case OP_NotReg:     { reg[op[1]] = f32(reg[op[2]] == 0.0); }
            case OP_AddRegImm:  { reg[op[1]] = reg[op[2]] + read_imm_f32(&i); }
            case OP_MulRegImm:  { reg[op[1]] = reg[op[2]] * read_imm_f32(&i); }
            case OP_DivRegImm:  { reg[op[1]] = reg[op[2]] / read_imm_f32(&i); }
            case OP_SubRegImm:  { reg[op[1]] = reg[op[2]] - read_imm_f32(&i); }
            case OP_ModRegImm:  { reg[op[1]] = reg[op[2]] % read_imm_f32(&i); }
            case OP_AtanRegImm: { reg[op[1]] = atan2(reg[op[2]], read_imm_f32(&i)); }
            case OP_CompareRegImm:  { reg[op[1]] = compare_f32(reg[op[2]], read_imm_f32(&i)); }

            case OP_DivImmReg:      { reg[op[1]] = read_imm_f32(&i) / reg[op[2]]; }
            case OP_SubImmReg:      { reg[op[1]] = read_imm_f32(&i) - reg[op[2]]; }
            case OP_ModImmReg:      { reg[op[1]] = read_imm_f32(&i) % reg[op[2]]; }
            case OP_AtanImmReg:     { reg[op[1]] = atan2(read_imm_f32(&i), reg[op[2]]); }
            case OP_CompareImmReg:  { reg[op[1]] = compare_f32(read_imm_f32(&i), reg[op[2]]); }

            case OP_MinRegImm:  { reg[op[1]] = min(reg[op[2]], read_imm_f32(&i)); }
            case OP_MaxRegImm:  { reg[op[1]] = max(reg[op[2]], read_imm_f32(&i)); }
            case OP_AndRegImm:  { reg[op[1]] = and_f32(reg[op[2]], read_imm_f32(&i)); }
            case OP_OrRegImm:   { reg[op[1]] = or_f32(reg[op[2]], read_imm_f32(&i)); }

            case OP_AddRegReg:      { reg[op[1]] = reg[op[2]] + reg[op[3]]; }
            case OP_MulRegReg:      { reg[op[1]] = reg[op[2]] * reg[op[3]]; }
            case OP_DivRegReg:      { reg[op[1]] = reg[op[2]] / reg[op[3]]; }
            case OP_SubRegReg:      { reg[op[1]] = reg[op[2]] - reg[op[3]]; }
            case OP_CompareRegReg:  { reg[op[1]] = reg[op[2]] - reg[op[3]]; }
            case OP_ModRegReg:      { reg[op[1]] = reg[op[2]] % reg[op[3]]; }

            case OP_MinRegReg:      { reg[op[1]] = min(reg[op[2]], reg[op[3]]); }
            case OP_MaxRegReg:      { reg[op[1]] = max(reg[op[2]], reg[op[3]]); }
            case OP_AndRegReg:      { reg[op[1]] = and_f32(reg[op[2]], reg[op[3]]); }
            case OP_OrRegReg:       { reg[op[1]] = or_f32(reg[op[2]], reg[op[3]]); }
            default: {
                break;
            }
        }
    }
    return nan_f32(); // unknown opcode
}
"#;

fn run_wgpu_brute<
    F: fidget::eval::MathFunction + fidget::shape::RenderHints,
>(
    shape: fidget::shape::Shape<F>,
    settings: &ImageSettings,
) -> Vec<u8> {
    let bytecode = shape.inner().to_bytecode();
    assert_eq!(fidget::bytecode::VERSION, 1, "unexpected bytecode version");
    assert_eq!(bytecode.mem_count, 0, "can't use Load / Store yet");

    use zerocopy::{FromBytes, IntoBytes};

    // Initialize wgpu
    let instance = wgpu::Instance::default();
    let (device, queue) = pollster::block_on(async {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find an appropriate adapter");
        adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create device")
    });

    let mut shader_code = String::new();
    for (op, i) in fidget::bytecode::iter_ops() {
        shader_code += &format!("const OP_{op}: u32 = {i};\n");
    }

    // Compute shader code
    shader_code += WGPU_SHADER_BASE;
    shader_code += r#"
@group(0) @binding(0) var<storage, read> vars: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tape: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx < arrayLength(&vars)) {
        result[idx] = run_tape(0u, vars[idx]);
    }
}
    "#;

    // Compile the shader
    let shader_module =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

    let pixel_count = (settings.size * settings.size) as usize;

    // Create buffers for the input and output data
    let vars_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vars"),
        size: (pixel_count * std::mem::size_of::<f32>() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let tape_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tape"),
        size: (bytecode.len() * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result"),
        size: (pixel_count * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: (pixel_count * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Create bind group layout and bind group
    let bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vars_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: tape_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: result_buffer.as_entire_binding(),
            },
        ],
    });

    // Create the compute pipeline
    let pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let compute_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    // Send over our image pixels
    // (TODO: generate this in the shader instead?)
    let mut image = vec![];
    for _i in 0..settings.n {
        let mut pixels = Vec::with_capacity(pixel_count);
        for x in 0..settings.size {
            let x = (x as f32) / (settings.size - 1) as f32 * 2.0 - 1.0;
            for y in 0..settings.size {
                let y = (y as f32) / (settings.size - 1) as f32 * 2.0 - 1.0;
                pixels.push([x, y, 0.0, 0.0]);
            }
        }
        queue.write_buffer(&vars_buffer, 0, pixels.as_bytes());
        queue.write_buffer(&tape_buffer, 0, bytecode.as_bytes());

        // Create a command encoder and dispatch the compute work
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None,
            });

        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                (pixel_count as u32 + 63) / 64,
                1,
                1,
            );
        }

        // Copy from the STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
        encoder.copy_buffer_to_buffer(
            &result_buffer,
            0,
            &out_buffer,
            0,
            (pixel_count * std::mem::size_of::<f32>()) as u64,
        );

        // Submit the commands and wait for the GPU to complete
        queue.submit(Some(encoder.finish()));

        // Map result buffer and read back the data
        let buffer_slice = out_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        let result = <[f32]>::ref_from_bytes(&data).unwrap();

        image = result
            .iter()
            .flat_map(|a| {
                if *a < 0.0 {
                    [255; 4].into_iter()
                } else {
                    [0, 0, 0, 255].into_iter()
                }
            })
            .collect();

        if pixel_count < 128 {
            println!("Result: {:?}", result);
        }

        // Clean up
        drop(data);
        out_buffer.unmap();
    }

    image
}

fn run_wgpu_smart<
    F: fidget::eval::MathFunction + fidget::shape::RenderHints,
>(
    shape: fidget::shape::Shape<F>,
    settings: &ImageSettings,
) -> Vec<u8> {
    let bytecode = shape.inner().to_bytecode();
    assert_eq!(fidget::bytecode::VERSION, 1, "unexpected bytecode version");
    assert_eq!(bytecode.mem_count, 0, "can't use Load / Store yet");

    use zerocopy::{FromBytes, Immutable, IntoBytes};

    // Initialize wgpu
    let instance = wgpu::Instance::default();
    let (device, queue) = pollster::block_on(async {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find an appropriate adapter");
        adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create device")
    });

    let mut shader_code = String::new();
    for (op, i) in fidget::bytecode::iter_ops() {
        shader_code += &format!("const OP_{op}: u32 = {i};\n");
    }

    // Square tiles
    #[derive(Debug, IntoBytes, Immutable)]
    #[repr(C)]
    struct Tile {
        corner: [u32; 2],
        start: u32,
        _padding: u32,
    }

    #[derive(Debug, IntoBytes, Immutable)]
    #[repr(C)]
    struct Config {
        window_size: u32,
        tile_size: u32,
    }

    // Compute shader code
    shader_code += WGPU_SHADER_BASE;
    shader_code += r#"
struct Tile {
    corner: vec2<u32>,
    start: u32,
    _padding: u32,
}

struct Window {
    size: u32,
    padding: vec3<u32>,
}

struct Config {
    window_size: u32,
    tile_size: u32,
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> tiles: array<Tile>;
@group(0) @binding(2) var<storage, read> tape: array<u32>;
@group(0) @binding(3) var<storage, read_write> result: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let pos_idx = id.xy;
    let tile_idx = id.z;
    if (tile_idx < arrayLength(&tiles)) {
        let tile = tiles[tile_idx];
        if (pos_idx.x < config.tile_size && pos_idx.y < config.tile_size) {
            // Absolute pixel position
            let pos_pixels = tile.corner + pos_idx;
            // Relative pixel position (±1)
            let pos_frac = 2.0 * vec2<f32>(pos_pixels - config.window_size / 2)
                / f32(config.window_size);

            let v = vec4<f32>(pos_frac, 0.0, 0.0);
            var p = 0u;
            if (run_tape(tile.start, v) < 0.0) {
                p = 0xFFFFFFFFu;
            } else {
                p = 0xFF000000u;
            };

            // Write to absolute position in the image
            result[pos_pixels.x + pos_pixels.y * config.window_size] = 0xFFFFFFFFu;
        }
    }
}
    "#;

    // Compile the shader
    let shader_module =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

    let pixel_count = settings.size.pow(2) as usize;

    const TILE_SIZE: u32 = 32;
    let tile_count = (settings.size / TILE_SIZE).pow(2);

    // Create buffers for the input and output data
    let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("config"),
        size: std::mem::size_of::<Config>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let tiles_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tiles"),
        size: (tile_count as usize * std::mem::size_of::<Tile>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false, // XXX?
    });
    let tape_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tape"),
        size: (bytecode.len() * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result"),
        size: (pixel_count * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: (pixel_count * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Create bind group layout and bind group
    let bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: config_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: tiles_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: tape_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: result_buffer.as_entire_binding(),
            },
        ],
    });

    // Create the compute pipeline
    let pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let compute_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    // Send over our image pixels
    // (TODO: generate this in the shader instead?)
    let mut image = vec![];
    for _i in 0..settings.n {
        let mut tiles = Vec::with_capacity(tile_count as usize);
        for x in (0..settings.size).step_by(TILE_SIZE as usize) {
            for y in (0..settings.size).step_by(TILE_SIZE as usize) {
                tiles.push(Tile {
                    corner: [x, y],
                    start: 0,
                    _padding: 0,
                });
            }
        }
        let config = Config {
            window_size: settings.size,
            tile_size: TILE_SIZE,
        };
        queue.write_buffer(&config_buf, 0, config.as_bytes());
        queue.write_buffer(&tiles_buf, 0, tiles.as_bytes());
        queue.write_buffer(&tape_buffer, 0, bytecode.as_bytes());

        // Create a command encoder and dispatch the compute work
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None,
            });

        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                TILE_SIZE,
                TILE_SIZE,
                tile_count.div_ceil(64),
            );
        }

        // Copy from the STORAGE | COPY_SRC -> COPY_DST | MAP_READ buffer
        encoder.copy_buffer_to_buffer(
            &result_buffer,
            0,
            &out_buffer,
            0,
            (pixel_count * std::mem::size_of::<f32>()) as u64,
        );

        // Submit the commands and wait for the GPU to complete
        queue.submit(Some(encoder.finish()));

        println!("{tiles:?}");
        // Map result buffer and read back the data
        let buffer_slice = out_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        let result = <[u32]>::ref_from_bytes(&data).unwrap();

        image = result.iter().flat_map(|a| a.to_le_bytes()).collect();

        if pixel_count < 128 {
            println!("Result: {:?}", result);
        }

        // Clean up
        drop(data);
        out_buffer.unmap();
    }

    image
}

////////////////////////////////////////////////////////////////////////////////

fn run_mesh<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &MeshSettings,
) -> fidget::mesh::Mesh {
    let mut mesh = fidget::mesh::Mesh::new();

    for _ in 0..settings.n {
        let settings = fidget::mesh::Settings {
            threads: settings.threads.into(),
            depth: settings.depth,
            ..Default::default()
        };
        let octree = fidget::mesh::Octree::build(&shape, settings);
        mesh = octree.walk_dual(settings);
    }
    mesh
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .init();

    let now = Instant::now();
    let args = Args::parse();
    let mut file = std::fs::File::open(&args.input)?;
    let (ctx, root) = Context::from_text(&mut file)?;
    info!("Loaded file in {:?}", now.elapsed());

    match args.cmd {
        Command::Render2d {
            settings,
            brute,
            sdf,
        } => {
            let start = Instant::now();
            let buffer = if settings.wgpu {
                match settings.eval {
                    #[cfg(feature = "jit")]
                    EvalMode::Jit => {
                        let shape = fidget::jit::JitShape::new(&ctx, root)?;
                        info!("Built shape in {:?}", start.elapsed());
                        run_wgpu(shape, &settings, brute, sdf)
                    }
                    EvalMode::Vm => {
                        let shape = fidget::vm::VmShape::new(&ctx, root)?;
                        info!("Built shape in {:?}", start.elapsed());
                        run_wgpu(shape, &settings, brute, sdf)
                    }
                }
            } else {
                match settings.eval {
                    #[cfg(feature = "jit")]
                    EvalMode::Jit => {
                        let shape = fidget::jit::JitShape::new(&ctx, root)?;
                        info!("Built shape in {:?}", start.elapsed());
                        run2d(shape, &settings, brute, sdf)
                    }
                    EvalMode::Vm => {
                        let shape = fidget::vm::VmShape::new(&ctx, root)?;
                        info!("Built shape in {:?}", start.elapsed());
                        run2d(shape, &settings, brute, sdf)
                    }
                }
            };

            info!(
                "Rendered {}x at {:?} ms/frame",
                settings.n,
                start.elapsed().as_micros() as f64
                    / 1000.0
                    / (settings.n as f64)
            );
            if let Some(out) = settings.out {
                image::save_buffer(
                    out,
                    &buffer,
                    settings.size,
                    settings.size,
                    image::ColorType::Rgba8,
                )?;
            }
        }
        Command::Render3d {
            settings,
            color,
            isometric,
        } => {
            let start = Instant::now();
            let buffer = match settings.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?}", start.elapsed());
                    run3d(shape, &settings, isometric, color)
                }
                EvalMode::Vm => {
                    let shape = fidget::vm::VmShape::new(&ctx, root)?;
                    info!("Built shape in {:?}", start.elapsed());
                    run3d(shape, &settings, isometric, color)
                }
            };
            info!(
                "Rendered {}x at {:?} ms/frame",
                settings.n,
                start.elapsed().as_micros() as f64
                    / 1000.0
                    / (settings.n as f64)
            );

            if let Some(out) = settings.out {
                info!("Writing image to {out:?}");
                image::save_buffer(
                    out,
                    &buffer,
                    settings.size,
                    settings.size,
                    image::ColorType::Rgba8,
                )?;
            }
        }
        Command::Mesh { settings } => {
            let start = Instant::now();
            let mesh = match settings.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?}", start.elapsed());
                    run_mesh(shape, &settings)
                }
                EvalMode::Vm => {
                    let shape = fidget::vm::VmShape::new(&ctx, root)?;
                    info!("Built shape in {:?}", start.elapsed());
                    run_mesh(shape, &settings)
                }
            };
            info!(
                "Rendered {}x at {:?} ms/iter",
                settings.n,
                start.elapsed().as_micros() as f64
                    / 1000.0
                    / (settings.n as f64)
            );
            if let Some(out) = settings.out {
                info!("Writing STL to {out:?}");
                mesh.write_stl(&mut std::fs::File::create(out)?)?;
            }
        }
    }

    Ok(())
}
