use std::time::Instant;

use clap::Parser;
use env_logger::Env;
use jitfive::{compiler::Compiler, context::Context};
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

    /// Render using the JIT-compiled function
    #[clap(short, long, requires = "image", conflicts_with = "gpu")]
    jit: bool,

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
    let (ctx, node) = Context::from_text(&mut file)?;
    info!("Loaded file in {:?}", now.elapsed());

    let now = Instant::now();
    let compiler = Compiler::new(&ctx, node);
    info!("Built Compiler in {:?}", now.elapsed());

    if args.dot {
        std::fs::write("stage0.dot", compiler.stage0_dot())?;
        std::fs::write("stage1.dot", compiler.stage1_dot())?;
    }

    let llvm_ctx = inkwell::context::Context::create();
    let jit_fn = jitfive::backend::llvm::to_jit_fn(&compiler, &llvm_ctx)?;

    /*
    if let Some(m) = args.metal {
        std::fs::write(m, prog.to_metal(jitfive::metal::Mode::Interval))?
    }
    */
    if let Some(img) = args.image {
        let buffer: Vec<[u8; 4]> = /*if args.gpu {
            let now = Instant::now();
            let out = gpu::render(&prog, args.size);
            println!("Done with GPU render in {:?}", now.elapsed());
            out
        } else*/ {
            let out = if args.jit {
                let choices = vec![-1i32; (compiler.num_choices + 15) / 16];
                // Copied from `Context::render_2d`
                let now = Instant::now();
                let scale = args.size;
                let mut out = Vec::with_capacity((scale * scale) as usize);
                let div = (scale - 1) as f32;
                for i in 0..scale {
                    let y = -(-1.0 + 2.0 * (i as f32) / div);
                    for j in 0..scale {
                        let x = -1.0 + 2.0 * (j as f32) / div;
                        let v = unsafe { jit_fn.call(x, y, choices.as_ptr()) };
                        out.push(v <= 0.0);
                    }
                }
                info!("Finished rendering in {:?}", now.elapsed());
                out
            } else {
                ctx.render_2d(node, args.size)?
            };
            out.into_iter()
                .map(|b| if b { [u8::MAX; 4] } else { [0, 0, 0, 255] })
                .collect()
        };

        // Flatten into a single array
        let buffer: Vec<u8> =
            buffer.into_iter().flat_map(|i| i.into_iter()).collect();

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

////////////////////////////////////////////////////////////////////////////////

/*
mod gpu {
    use super::*;
    use piet_gpu_hal::{Instance, InstanceFlags, Session};

    pub fn render(prog: &Program, size: u32) -> Vec<[u8; 4]> {
        let (instance, _) =
            Instance::new(None, InstanceFlags::empty()).unwrap();

        unsafe {
            let device = instance.device(None).unwrap();
            let session = Session::new(device);
            let mut metal = Render::new(prog, &session);
            for _i in 0..20 {
                metal.do_render(size, &session);
            }
            metal.load_image()
        }
    }
}
*/
