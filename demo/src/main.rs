use std::time::Instant;

use clap::Parser;
use jitfive::{
    compiler::Compiler, context::Context, metal::Render, program::Program,
};

/// Simple test program
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Name of a `.dot` file to write
    #[clap(short, long)]
    dot: Option<String>,

    /// Name of a `.png` file to write
    #[clap(short, long)]
    image: Option<String>,

    /// Render using the GPU
    #[clap(short, long, requires = "image")]
    gpu: bool,

    /// Name of a `.metal` file to write
    #[clap(short, long)]
    metal: Option<String>,

    /// Image size
    #[clap(short, long, requires = "image", default_value = "128")]
    size: usize,

    /// Name of the model file to load
    filename: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let now = Instant::now();
    let args = Args::parse();
    let mut file = std::fs::File::open(args.filename)?;
    let (ctx, node) = Context::from_text(&mut file)?;
    println!("Loaded file in {:?}", now.elapsed());

    let now = Instant::now();
    let compiler = Compiler::new(&ctx, node);
    println!("Build Compiler in {:?}", now.elapsed());

    let now = Instant::now();
    let prog = Program::from_compiler(&compiler);
    println!("Built Program in {:?}", now.elapsed());

    if let Some(dot) = args.dot {
        let mut out = std::fs::File::create(dot)?;
        compiler.write_dot_grouped(&mut out)?;
    }

    if let Some(m) = args.metal {
        std::fs::write(m, prog.to_metal(jitfive::metal::Mode::Float))?
    }
    if let Some(img) = args.image {
        let out = if args.gpu {
            gpu::render(&prog, args.size)
        } else {
            ctx.render_2d(node, args.size)?
        };
        let buffer: Vec<u8> = out
            .into_iter()
            .map(|b| if b { u8::MAX } else { 0 })
            .collect();
        image::save_buffer(
            img,
            &buffer,
            args.size as u32,
            args.size as u32,
            image::ColorType::L8,
        )?;
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////

mod gpu {
    use super::*;
    use piet_gpu_hal::{Instance, InstanceFlags, Session};

    pub fn render(prog: &Program, size: usize) -> Vec<bool> {
        let (instance, _) =
            Instance::new(None, InstanceFlags::empty()).unwrap();

        unsafe {
            let device = instance.device(None).unwrap();
            let session = Session::new(device);
            let mut metal = Render::new(prog, &session);
            metal.render(size, &session)
        }
    }
}
