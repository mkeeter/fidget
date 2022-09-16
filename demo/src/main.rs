use std::time::Instant;

use clap::Parser;
use env_logger::Env;
use jitfive::context::Context;
use log::info;

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
    asm: bool,

    /// Use brute-force (pixel-by-pixel) evaluation
    #[clap(short, long, requires = "asm")]
    brute: bool,

    #[clap(short = 'N', default_value = "1", requires = "image")]
    n: usize,

    /// Image size
    #[clap(short, long, requires = "image", default_value = "128")]
    size: u32,

    /// Name of the model file to load
    filename: String,
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
            let scale = args.size;
            let start = Instant::now();
            let tape = ctx.get_tape(root, u8::MAX);
            info!("Built tape in {:?}", start.elapsed());

            let mut eval = tape.get_float_evaluator();
            let mut out = vec![];
            let start = Instant::now();
            for _ in 0..args.n {
                out.clear();
                let div = (scale - 1) as f64;
                for i in 0..scale {
                    let y = -(-1.0 + 2.0 * (i as f64) / div);
                    for j in 0..scale {
                        let x = -1.0 + 2.0 * (j as f64) / div;
                        let v = eval.eval(x as f32, y as f32, 0.0);
                        out.push(v <= 0.0);
                    }
                }
            }

            // Convert from Vec<bool> to an image
            let out = out
                .into_iter()
                .map(|b| if b { [u8::MAX; 4] } else { [0, 0, 0, 255] })
                .flat_map(|i| i.into_iter())
                .collect();
            (out, start)
        } else if args.asm {
            if args.brute {
                let scale = args.size;

                let start = Instant::now();
                let tape =
                    ctx.get_tape(root, jitfive::asm::dynasm::REGISTER_LIMIT);
                info!("Got tape in {:?}", start.elapsed());

                let start = Instant::now();
                let jit = jitfive::asm::dynasm::build_vec_fn(&tape);
                let eval = jit.get_evaluator();
                info!("Built JIT function in {:?}", start.elapsed());

                let mut out = vec![];
                let start = Instant::now();
                for _ in 0..args.n {
                    out.clear();
                    let div = (scale - 1) as f64;
                    for i in 0..scale {
                        let y = -(-1.0 + 2.0 * (i as f64) / div);
                        for j in 0..(scale / 4) {
                            let mut x = [0.0; 4];
                            for i in 0u32..4 {
                                x[i as usize] = (-1.0
                                    + 2.0 * ((j * 4 + i) as f64) / div)
                                    as f32;
                            }
                            let v = eval.eval(x, [y as f32; 4], [0.0; 4]);
                            out.extend(v.into_iter().map(|v| v <= 0.0));
                        }
                    }
                }

                // Convert from Vec<bool> to an image
                let out = out
                    .into_iter()
                    .map(|b| if b { [u8::MAX; 4] } else { [0, 0, 0, 255] })
                    .flat_map(|i| i.into_iter())
                    .collect();
                (out, start)
            } else {
                let start = Instant::now();
                let tape =
                    ctx.get_tape(root, jitfive::asm::dynasm::REGISTER_LIMIT);
                info!("Got tape in {:?}", start.elapsed());

                let cfg = jitfive::render::RenderConfig {
                    image_size: args.size as usize,
                    tile_size: 256,
                    subtile_size: 64,
                    threads: 8,
                    interval_subdiv: 3,
                };
                let start = Instant::now();
                let mut image = vec![];
                for _ in 0..args.n {
                    image = jitfive::render::render(&tape, &cfg);
                }
                let out = image
                    .into_iter()
                    .flat_map(|p| p.as_color().into_iter())
                    .collect();
                (out, start)
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
