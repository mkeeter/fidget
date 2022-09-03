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
            let scheduled = jitfive::scheduled::schedule(&ctx, root);
            let tape = jitfive::backend::tape32::Tape::new(&scheduled);
            let mut eval = tape.get_evaluator();

            let start = Instant::now();
            let div = (scale - 1) as f64;
            for i in 0..scale {
                let y = -(-1.0 + 2.0 * (i as f64) / div);
                for j in 0..scale {
                    let x = -1.0 + 2.0 * (j as f64) / div;
                    let v = eval.f(x as f32, y as f32);
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
            let scale = args.size;
            let mut out = Vec::with_capacity((scale * scale) as usize);

            let start = Instant::now();
            let scheduled = jitfive::scheduled::schedule(&ctx, root);
            let tape = jitfive::backend::tape32::Tape::new_with_reg_limit(
                &scheduled,
                jitfive::backend::dynasm::REGISTER_LIMIT,
            );
            let jit = jitfive::backend::dynasm::build_float_fn(&tape);
            let eval = jit.get_evaluator();
            info!("Built JIT function in {:?}", start.elapsed());

            let start = Instant::now();
            let i_jit = jitfive::backend::dynasm::build_interval_fn(&tape);
            let i_eval = i_jit.get_evaluator();
            info!("Built interval JIT function in {:?}", start.elapsed());
            let start = Instant::now();
            println!("{:?}", i_eval.i([-0.5, 0.0], [-0.5, 0.0]));
            info!("Calculated in {:?}", start.elapsed());

            let start = Instant::now();
            let div = (scale - 1) as f64;
            for i in 0..scale {
                let y = -(-1.0 + 2.0 * (i as f64) / div);
                for j in 0..scale {
                    let x = -1.0 + 2.0 * (j as f64) / div;
                    let v = eval.f(x as f32, y as f32);
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
