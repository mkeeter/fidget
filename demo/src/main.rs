use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use env_logger::Env;
use log::info;

use fidget::context::{Context, Node};

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
    #[clap(short, long, default_value_t = 8)]
    threads: usize,

    /// Number of times to render (for benchmarking)
    #[clap(short = 'N', default_value_t = 1)]
    n: usize,

    /// Image size
    #[clap(short, long, default_value_t = 128)]
    size: u32,
}

#[derive(Parser)]
struct MeshSettings {
    /// Minimum octree depth
    #[clap(short, long)]
    depth: u8,

    /// Name of a `.stl` file to write
    #[clap(short, long)]
    out: Option<PathBuf>,

    /// Evaluator flavor
    #[clap(short, long, value_enum, default_value_t = EvalMode::Vm)]
    eval: EvalMode,

    /// Number of threads to use
    #[clap(short, long, default_value_t = 8)]
    threads: u8,

    /// Number of times to render (for benchmarking)
    #[clap(short = 'N', default_value_t = 1)]
    n: usize,
}

////////////////////////////////////////////////////////////////////////////////
fn run3d<I: fidget::eval::Family>(
    ctx: &Context,
    node: Node,
    settings: &ImageSettings,
    isometric: bool,
    mode_color: bool,
) -> (Vec<u8>, std::time::Instant) {
    let start = Instant::now();
    let tape = ctx.get_tape(node).unwrap();
    info!("Built tape in {:?}", start.elapsed());

    let mut mat = nalgebra::Transform3::identity();
    if !isometric {
        *mat.matrix_mut().get_mut((3, 2)).unwrap() = 0.3;
    }
    let cfg = fidget::render::RenderConfig {
        image_size: settings.size as usize,
        tile_sizes: I::tile_sizes_3d().to_vec(),
        threads: settings.threads,

        mat,
    };

    let start = Instant::now();
    let mut depth = vec![];
    let mut color = vec![];
    for _ in 0..settings.n {
        (depth, color) = fidget::render::render3d::<I>(tape.clone(), &cfg);
    }

    let out = if mode_color {
        depth
            .into_iter()
            .zip(color.into_iter())
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

    (out, start)
}

////////////////////////////////////////////////////////////////////////////////

fn run2d<I: fidget::eval::Family>(
    ctx: &Context,
    node: Node,
    settings: &ImageSettings,
    brute: bool,
    sdf: bool,
) -> (Vec<u8>, std::time::Instant) {
    let start = Instant::now();
    let tape = ctx.get_tape::<I>(node).unwrap();
    info!("Built tape in {:?}", start.elapsed());

    if brute {
        let eval = tape.new_float_slice_evaluator();
        let mut out: Vec<bool> = vec![];
        let start = Instant::now();
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
            let values = eval.eval(&xs, &ys, &zs, &[]).unwrap();
            out = values.into_iter().map(|v| v <= 0.0).collect();
        }
        // Convert from Vec<bool> to an image
        let out = out
            .into_iter()
            .map(|b| if b { [u8::MAX; 4] } else { [0, 0, 0, 255] })
            .flat_map(|i| i.into_iter())
            .collect();
        (out, start)
    } else {
        let cfg = fidget::render::RenderConfig {
            image_size: settings.size as usize,
            tile_sizes: I::tile_sizes_2d().to_vec(),
            threads: settings.threads,

            mat: nalgebra::Transform2::identity(),
        };
        let start = Instant::now();
        let out = if sdf {
            let mut image = vec![];
            for _ in 0..settings.n {
                image = fidget::render::render2d(
                    tape.clone(),
                    &cfg,
                    &fidget::render::SdfRenderMode,
                );
            }
            image
                .into_iter()
                .flat_map(|a| [a[0], a[1], a[2], 255].into_iter())
                .collect()
        } else {
            let mut image = vec![];
            for _ in 0..settings.n {
                image = fidget::render::render2d(
                    tape.clone(),
                    &cfg,
                    &fidget::render::DebugRenderMode,
                );
            }
            image
                .into_iter()
                .flat_map(|p| p.as_debug_color().into_iter())
                .collect()
        };
        (out, start)
    }
}

////////////////////////////////////////////////////////////////////////////////

fn run_mesh<I: fidget::eval::Family>(
    ctx: &Context,
    node: Node,
    settings: &MeshSettings,
) -> (fidget::mesh::Mesh, std::time::Instant) {
    let start = Instant::now();
    let tape = ctx.get_tape::<I>(node).unwrap();
    info!("Built tape in {:?}", start.elapsed());

    let start = Instant::now();
    let mut mesh = fidget::mesh::Mesh::new();

    for _ in 0..settings.n {
        let octree = fidget::mesh::Octree::build(
            &tape,
            fidget::mesh::Settings {
                threads: settings.threads,
                min_depth: settings.depth,
                max_depth: settings.depth,
            },
        );
        mesh = octree.walk_dual();
    }
    (mesh, start)
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
            let (buffer, start) = match settings.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => run2d::<fidget::jit::Eval>(
                    &ctx, root, &settings, brute, sdf,
                ),
                EvalMode::Vm => {
                    run2d::<fidget::vm::Eval>(&ctx, root, &settings, brute, sdf)
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
                    settings.size as u32,
                    settings.size as u32,
                    image::ColorType::Rgba8,
                )?;
            }
        }
        Command::Render3d {
            settings,
            color,
            isometric,
        } => {
            let (buffer, start) = match settings.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => run3d::<fidget::jit::Eval>(
                    &ctx, root, &settings, isometric, color,
                ),
                EvalMode::Vm => run3d::<fidget::vm::Eval>(
                    &ctx, root, &settings, isometric, color,
                ),
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
                    settings.size as u32,
                    settings.size as u32,
                    image::ColorType::Rgba8,
                )?;
            }
        }
        Command::Mesh { settings } => {
            let (mesh, start) = match settings.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    run_mesh::<fidget::jit::Eval>(&ctx, root, &settings)
                }
                EvalMode::Vm => {
                    run_mesh::<fidget::vm::Eval>(&ctx, root, &settings)
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
