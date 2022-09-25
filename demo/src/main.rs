use std::time::Instant;

use clap::Parser;
use env_logger::Env;
use log::info;

use fidget::context::{Context, Node};

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

    /// Render using the `dynasm`-compiled function
    #[clap(short, long, requires = "image", conflicts_with = "interpreter")]
    jit: bool,

    /// Use brute-force (pixel-by-pixel) evaluation
    #[clap(short, long)]
    brute: bool,

    #[clap(short = 'N', default_value = "1", requires = "image")]
    n: usize,

    /// Image size
    #[clap(short, long, requires = "image", default_value = "128")]
    size: u32,

    /// Name of the model file to load
    filename: String,
}

fn run<I>(
    ctx: &Context,
    node: Node,
    brute: bool,
    size: u32,
    n: usize,
) -> (Vec<u8>, std::time::Instant)
where
    for<'s> I: fidget::eval::EvalFamily<'s>,
{
    let start = Instant::now();
    let tape = ctx.get_tape(node, I::REG_LIMIT);
    info!("Built tape in {:?}", start.elapsed());

    if brute {
        let mut eval = tape.get_float_evaluator();
        use fidget::eval::FloatSliceEval;
        let mut out: Vec<bool> = vec![];
        let start = Instant::now();
        for _ in 0..n {
            let mut xs = vec![];
            let mut ys = vec![];
            let div = (size - 1) as f64;
            for i in 0..size {
                let y = -(-1.0 + 2.0 * (i as f64) / div);
                for j in 0..size {
                    let x = -1.0 + 2.0 * (j as f64) / div;
                    xs.push(x as f32);
                    ys.push(y as f32);
                }
            }
            let zs = vec![0.0; xs.len()];
            let mut values = vec![0.0; xs.len()];
            eval.eval_s(&xs, &ys, &zs, &mut values);
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
            image_size: size as usize,
            tile_size: 256,
            subtile_size: 64,
            threads: 8,
            interval_subdiv: 3,

            dx: 0.0,
            dy: 0.0,
            scale: 1.0,
        };
        let start = Instant::now();
        let mut image = vec![];
        for _ in 0..n {
            image = fidget::render::render::<I, fidget::render::DebugRenderMode>(
                tape.clone(),
                &cfg,
            );
        }
        let out = image
            .into_iter()
            .flat_map(|p| p.as_debug_color().into_iter())
            .collect();
        (out, start)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .init();

    let now = Instant::now();
    let args = Args::parse();
    let mut file = std::fs::File::open(args.filename)?;
    let (ctx, root) = Context::from_text(&mut file)?;
    info!("Loaded file in {:?}", now.elapsed());

    if let Some(img) = args.image {
        let (buffer, start): (Vec<u8>, _) = if args.interpreter {
            run::<fidget::eval::AsmFamily>(
                &ctx, root, args.brute, args.size, args.n,
            )
        } else if args.jit {
            run::<fidget::asm::dynasm::JitEvalFamily>(
                &ctx, root, args.brute, args.size, args.n,
            )
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
