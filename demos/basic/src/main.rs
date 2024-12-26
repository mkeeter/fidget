use std::num::NonZero;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use env_logger::Env;
use log::info;

/// Simple test program
#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Main action
    #[clap(subcommand)]
    action: ActionCommand,

    /// Input file
    #[clap(short, long)]
    input: Option<PathBuf>,

    /// Input file
    #[clap(short = 's', long, value_enum, default_value_t = HardcodedShape::SphereAsm)]
    hardcoded_shape: HardcodedShape,

    /// Evaluator flavor
    #[clap(short, long, value_enum, default_value_t = EvalMode::Jit)]
    eval: EvalMode,

    /// Number of threads to use
    #[clap(long, default_value_t = 0)]
    num_threads: usize,

    /// Number of times to render (for benchmarking)
    #[clap(long, default_value_t = 1)]
    num_repeats: usize,
}

#[derive(ValueEnum, Clone)]
enum EvalMode {
    #[cfg(feature = "jit")]
    Jit,
    Vm,
}

#[derive(ValueEnum, Clone, Debug)]
enum HardcodedShape {
    SphereAsm,
    SphereTree,
}

#[derive(Subcommand)]
enum ActionCommand {
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

#[derive(Parser)]
struct ImageSettings {
    /// Image size
    #[clap(short, long, default_value_t = 1024)]
    size: u32,

    /// Name of a `.png` file to write
    #[clap(short, long)]
    output: Option<PathBuf>,
}

#[derive(Parser)]
struct MeshSettings {
    /// Octree depth
    #[clap(short, long, default_value_t = 6)]
    depth: u8,

    /// Name of a `.stl` file to write
    #[clap(short, long)]
    output: Option<PathBuf>,
}

////////////////////////////////////////////////////////////////////////////////

fn run_3d<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &ImageSettings,
    isometric: bool,
    color_mode: bool,
    num_repeats: usize,
    num_threads: usize,
) -> Vec<u8> {
    let mut mat = nalgebra::Transform3::identity();
    if !isometric {
        *mat.matrix_mut().get_mut((3, 2)).unwrap() = 0.3;
    }
    let pool: Option<rayon::ThreadPool>;
    let threads = match num_threads {
        0 => Some(fidget::render::ThreadPool::Global),
        1 => None,
        nn => {
            pool = Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(nn)
                    .build()
                    .unwrap(),
            );
            pool.as_ref().map(fidget::render::ThreadPool::Custom)
        }
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
    for _ in 0..num_repeats {
        (depth, color) = cfg.run(shape.clone());
    }

    let out = if color_mode {
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

fn run_2d<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &ImageSettings,
    brute: bool,
    sdf: bool,
    num_repeats: usize,
    num_threads: usize,
) -> Vec<u8> {
    if brute {
        let tape = shape.float_slice_tape(Default::default());
        let mut eval = fidget::shape::Shape::<F>::new_float_slice_eval();
        let mut out: Vec<bool> = vec![];
        for _ in 0..num_repeats {
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
        let threads = match num_threads {
            0 => Some(fidget::render::ThreadPool::Global),
            1 => None,
            nn => {
                pool = Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(nn)
                        .build()
                        .unwrap(),
                );
                pool.as_ref().map(fidget::render::ThreadPool::Custom)
            }
        };
        let cfg = fidget::render::ImageRenderConfig {
            image_size: fidget::render::ImageSize::from(settings.size),
            tile_sizes: F::tile_sizes_2d(),
            threads,
            ..Default::default()
        };
        if sdf {
            let mut image = vec![];
            for _ in 0..num_repeats {
                image =
                    cfg.run::<_, fidget::render::SdfRenderMode>(shape.clone());
            }
            image
                .into_iter()
                .flat_map(|a| [a[0], a[1], a[2], 255].into_iter())
                .collect()
        } else {
            let mut image = vec![];
            for _ in 0..num_repeats {
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

////////////////////////////////////////////////////////////////////////////////

fn run_mesh<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &MeshSettings,
    num_repeats: usize,
    num_threads: usize,
) -> fidget::mesh::Mesh {
    use fidget::mesh::ThreadCount;

    let mut mesh = fidget::mesh::Mesh::new();

    let threads = match num_threads {
        0 => ThreadCount::Many(NonZero::new(8).unwrap()),
        1 => ThreadCount::One,
        nn => ThreadCount::Many(NonZero::new(nn).unwrap()),
    };
    for _ in 0..num_repeats {
        let settings = fidget::mesh::Settings {
            threads,
            depth: settings.depth,
            ..Default::default()
        };
        let octree = fidget::mesh::Octree::build(&shape, settings);
        mesh = octree.walk_dual(settings);
    }

    mesh
}

fn run_action(
    ctx: fidget::context::Context,
    root: fidget::context::Node,
    args: Args,
) -> Result<()> {
    let mut top = Instant::now();
    match args.action {
        ActionCommand::Render3d {
            settings,
            color,
            isometric,
        } => {
            let buffer = match args.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (JIT)", top.elapsed());
                    top = Instant::now();
                    run_3d(
                        shape,
                        &settings,
                        isometric,
                        color,
                        args.num_repeats,
                        args.num_threads,
                    )
                }
                EvalMode::Vm => {
                    let shape = fidget::vm::VmShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (VM)", top.elapsed());
                    top = Instant::now();
                    run_3d(
                        shape,
                        &settings,
                        isometric,
                        color,
                        args.num_repeats,
                        args.num_threads,
                    )
                }
            };
            info!(
                "Rendered {}x at {:?} ms/frame",
                args.num_repeats,
                top.elapsed().as_micros() as f64
                    / 1000.0
                    / (args.num_repeats as f64)
            );
            if let Some(path) = settings.output {
                info!("Writing PNG to {:?}", &path);
                image::save_buffer(
                    &path,
                    &buffer,
                    settings.size,
                    settings.size,
                    image::ColorType::Rgba8,
                )?;
            }
        }
        ActionCommand::Render2d {
            settings,
            brute,
            sdf,
        } => {
            let buffer = match args.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (JIT)", top.elapsed());
                    top = Instant::now();
                    run_2d(
                        shape,
                        &settings,
                        brute,
                        sdf,
                        args.num_repeats,
                        args.num_threads,
                    )
                }
                EvalMode::Vm => {
                    let shape = fidget::vm::VmShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (VM)", top.elapsed());
                    top = Instant::now();
                    run_2d(
                        shape,
                        &settings,
                        brute,
                        sdf,
                        args.num_repeats,
                        args.num_threads,
                    )
                }
            };
            info!(
                "Rendered {}x at {:?} ms/frame",
                args.num_repeats,
                top.elapsed().as_micros() as f64
                    / 1000.0
                    / (args.num_repeats as f64)
            );
            if let Some(path) = settings.output {
                info!("Writing PNG to {path:?}");
                image::save_buffer(
                    path,
                    &buffer,
                    settings.size,
                    settings.size,
                    image::ColorType::Rgba8,
                )?;
            }
        }
        ActionCommand::Mesh { settings } => {
            let mesh = match args.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (JIT)", top.elapsed());
                    top = Instant::now();
                    run_mesh(
                        shape,
                        &settings,
                        args.num_repeats,
                        args.num_threads,
                    )
                }
                EvalMode::Vm => {
                    let shape = fidget::vm::VmShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (VM)", top.elapsed());
                    top = Instant::now();
                    run_mesh(
                        shape,
                        &settings,
                        args.num_repeats,
                        args.num_threads,
                    )
                }
            };
            info!(
                "Rendered {}x at {:?} ms/iter",
                args.num_repeats,
                top.elapsed().as_micros() as f64
                    / 1000.0
                    / (args.num_repeats as f64)
            );
            info!(
                "Mesh has {} vertices {} triangles",
                mesh.vertices.len(),
                mesh.triangles.len()
            );
            if let Some(path) = settings.output {
                info!("Writing STL to {:?}", &path);
                let mut handle = std::fs::File::create(&path)?;
                mesh.write_stl(&mut handle)?;
            }
        }
    }

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////

fn main() -> Result<()> {
    use fidget::context::Context;

    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .init();

    let sphere_text = "
# This is a comment!
0x600000b90000 var-x
0x600000b900a0 square 0x600000b90000
0x600000b90050 var-y
0x600000b900f0 square 0x600000b90050
0x600000b90300 var-z
0x600000b90350 square 0x600000b90300
0x600000b90140 add 0x600000b900a0 0x600000b900f0
0x600000b90150 add 0x600000b90140 0x600000b90350
0x600000b90190 sqrt 0x600000b90150
0x600000b901e0 const 1
0x600000b90200 sub 0x600000b90190 0x600000b901e0
";

    let args = Args::parse();

    let (ctx, root) = match &args.input {
        None => {
            let top = Instant::now();
            let ret = match args.hardcoded_shape {
                HardcodedShape::SphereAsm => {
                    Context::from_text(&mut sphere_text.as_bytes()).unwrap()
                }
                HardcodedShape::SphereTree => {
                    use fidget::context::Tree;
                    let mut tree = Tree::x().square();
                    tree += Tree::y().square();
                    tree += Tree::z().square();
                    tree = tree.sqrt() - 1;
                    let mut ctx = Context::new();
                    let root = ctx.import(&tree);
                    (ctx, root)
                }
            };
            info!(
                "Created hardcoded model {:?} in {:?}",
                args.hardcoded_shape,
                top.elapsed()
            );
            ret
        }
        Some(path) => {
            info!("Loading model from {:?}", path);
            let top = Instant::now();
            let mut handle = std::fs::File::open(&path).unwrap();
            let ret = Context::from_text(&mut handle).unwrap();
            info!("Loaded model in {:?}", top.elapsed());
            ret
        }
    };

    info!("Context has {} items", ctx.len());
    info!("Root is {:?}", root);

    run_action(ctx, root, args)
}
