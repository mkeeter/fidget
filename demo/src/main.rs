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
            let mut out = Vec::with_capacity((scale * scale) as usize);
            let start = Instant::now();
            let scheduled = jitfive::scheduled::schedule(&ctx, root);
            info!("Scheduled in {:?}", start.elapsed());

            let start = Instant::now();
            let tape = jitfive::backend::tape64::Tape::new(&scheduled);
            info!("Built tape in {:?}", start.elapsed());

            let mut eval = tape.get_evaluator();

            let start = Instant::now();
            let div = (scale - 1) as f64;
            for i in 0..scale {
                let y = -(-1.0 + 2.0 * (i as f64) / div);
                for j in 0..scale {
                    let x = -1.0 + 2.0 * (j as f64) / div;
                    let v = eval.f(x as f32, y as f32, 0.0);
                    out.push(v <= 0.0);
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
                let mut out = Vec::with_capacity((scale * scale) as usize);

                let start = Instant::now();
                let scheduled = jitfive::scheduled::schedule(&ctx, root);
                let tape = jitfive::backend::tape64::Tape::new_with_reg_limit(
                    &scheduled,
                    jitfive::backend::dynasm::REGISTER_LIMIT,
                );
                let jit = jitfive::backend::dynasm::build_vec_fn_64(&tape);
                let eval = jit.get_evaluator();
                info!("Built JIT function in {:?}", start.elapsed());
                info!("{:x?}", eval.v([0.0; 4], [0.0; 4], [0.0; 4]));

                let start = Instant::now();
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
                        let v = eval.v(x, [y as f32; 4], [0.0; 4]);
                        out.extend(v.into_iter().map(|v| v <= 0.0));
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
                let scheduled = jitfive::scheduled::schedule(&ctx, root);
                let tape = jitfive::backend::tape64::Tape::new(&scheduled);
                let image = jitfive::render::render(args.size as usize, tape);
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
            let div = (scale - 1) as f64;
            for i in 0..scale {
                let y = -(-1.0 + 2.0 * (i as f64) / div);
                for j in 0..scale {
                    let x = -1.0 + 2.0 * (j as f64) / div;
                    let v = ctx.eval_xyz(root, x, y, 0.0)? as f32;
                    out.push(v <= 0.0);
                }
            }

            // Convert from Vec<bool> to an image
            let out = out
                .into_iter()
                .map(|b| if b { [u8::MAX; 4] } else { [0, 0, 0, 255] })
                .flat_map(|i| i.into_iter())
                .collect();
            (out, start)
        };
        info!("Finished rendering in {:?}", start.elapsed());

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
