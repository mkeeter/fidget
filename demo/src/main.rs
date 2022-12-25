use std::time::Instant;

use clap::Parser;
use env_logger::Env;
use log::info;

use fidget::{
    context::{Context, Node},
    eval::Eval,
};

/// Simple test program
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Render `.dot` files representing compilation
    #[clap(short, long)]
    dot: bool,

    /// Name of a `.png` file to write
    #[clap(short, long)]
    image: Option<String>,

    /// Use the interpreter
    #[clap(long, requires = "image")]
    interpreter: bool,

    /// Render using the `dynvm`-compiled function
    #[clap(short, long, requires = "image", conflicts_with = "interpreter")]
    jit: bool,

    /// Use brute-force (pixel-by-pixel) evaluation
    #[clap(short, long)]
    brute: bool,

    #[clap(short, long, requires = "image", default_value = "8")]
    threads: usize,

    /// Render in color
    #[clap(short, long, requires = "threedee")]
    color: bool,

    /// Render in 3D
    #[clap(long, requires = "image", conflicts_with = "brute")]
    threedee: bool,

    /// Render in 3D
    #[clap(long, requires = "image", conflicts_with = "threedee")]
    sdf: bool,

    #[clap(short = 'N', default_value = "1", requires = "image")]
    n: usize,

    /// Image size
    #[clap(short, long, requires = "image", default_value = "128")]
    size: u32,

    /// Image size
    #[clap(long, requires = "threedee")]
    isometric: bool,

    /// Name of the model file to load
    filename: String,
}

////////////////////////////////////////////////////////////////////////////////
fn run3d<I: fidget::eval::Family>(
    ctx: &Context,
    node: Node,
    args: &Args,
) -> (Vec<u8>, std::time::Instant) {
    let start = Instant::now();
    let tape = ctx.get_tape(node).unwrap();
    info!("Built tape in {:?}", start.elapsed());

    let mut mat = nalgebra::Transform3::identity();
    if !args.isometric {
        *mat.matrix_mut().get_mut((3, 2)).unwrap() = 0.3;
    }
    let cfg = fidget::render::RenderConfig {
        image_size: args.size as usize,
        tile_sizes: I::tile_sizes_3d().to_vec(),
        threads: args.threads,

        mat,
    };

    let start = Instant::now();
    let mut depth = vec![];
    let mut color = vec![];
    for _ in 0..args.n {
        (depth, color) = fidget::render::render3d::<I>(tape.clone(), &cfg);
    }

    let out = if args.color {
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

fn run<I: fidget::eval::Family>(
    ctx: &Context,
    node: Node,
    args: &Args,
) -> (Vec<u8>, std::time::Instant) {
    let start = Instant::now();
    let tape = ctx.get_tape(node).unwrap();
    info!("Built tape in {:?}", start.elapsed());

    if args.brute {
        let mut eval = I::new_float_slice_evaluator(tape);
        let mut out: Vec<bool> = vec![];
        let start = Instant::now();
        for _ in 0..args.n {
            let mut xs = vec![];
            let mut ys = vec![];
            let div = (args.size - 1) as f64;
            for i in 0..args.size {
                let y = -(-1.0 + 2.0 * (i as f64) / div);
                for j in 0..args.size {
                    let x = -1.0 + 2.0 * (j as f64) / div;
                    xs.push(x as f32);
                    ys.push(y as f32);
                }
            }
            let zs = vec![0.0; xs.len()];
            let mut values = vec![0.0; xs.len()];
            eval.eval_s(&xs, &ys, &zs, &[], &mut values).unwrap();
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
            image_size: args.size as usize,
            tile_sizes: I::tile_sizes_2d().to_vec(),
            threads: args.threads,

            mat: nalgebra::Transform2::identity(),
        };
        let start = Instant::now();
        let out = if args.sdf {
            let mut image = vec![];
            for _ in 0..args.n {
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
            for _ in 0..args.n {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .init();

    let now = Instant::now();
    let args = Args::parse();
    let mut file = std::fs::File::open(&args.filename)?;
    let (ctx, root) = Context::from_text(&mut file)?;
    info!("Loaded file in {:?}", now.elapsed());

    if let Some(img) = &args.image {
        let (buffer, start): (Vec<u8>, _) = if args.interpreter {
            if args.threedee {
                run3d::<fidget::vm::Eval>(&ctx, root, &args)
            } else {
                run::<fidget::vm::Eval>(&ctx, root, &args)
            }
        } else if args.jit {
            if args.threedee {
                run3d::<fidget::jit::Eval>(&ctx, root, &args)
            } else {
                run::<fidget::jit::Eval>(&ctx, root, &args)
            }
        } else {
            let start = Instant::now();
            let scale = args.size;
            let mut out = Vec::with_capacity((scale * scale) as usize);
            for _ in 0..args.n {
                out.clear();
                let div = (scale - 1) as f64;
                for i in 0..scale {
                    let y = -(-1.0 + 2.0 * (i as f64) / div);
                    for j in 0..scale {
                        let x = -1.0 + 2.0 * (j as f64) / div;
                        let v = ctx.eval_xyz(root, x, y, 0.0)? as f32;
                        out.extend(if v <= 0.0 {
                            [u8::MAX; 4]
                        } else {
                            [0, 0, 0, 255]
                        });
                    }
                }
            }
            (out, start)
        };
        info!(
            "Rendered {}x at {:?} ms/frame",
            args.n,
            start.elapsed().as_micros() as f64 / 1000.0 / (args.n as f64)
        );

        image::save_buffer(
            img,
            &buffer,
            args.size as u32,
            args.size as u32,
            image::ColorType::Rgba8,
        )?;
    }
    Ok(())
}
