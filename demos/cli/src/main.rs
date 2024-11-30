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
            let buffer = match settings.eval {
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
