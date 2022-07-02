use clap::Parser;
use jitfive::{compiler::Compiler, context::Context, program::Program};

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

    /// Name of a `.metal` file to write
    #[clap(short, long)]
    metal: Option<String>,

    /// Image size
    #[clap(short, long, requires = "image", default_value = "100")]
    size: usize,

    /// Name of the model file to load
    filename: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut file = std::fs::File::open(args.filename)?;
    let (ctx, node) = Context::from_text(&mut file)?;

    let compiler = Compiler::new(&ctx, node);
    let prog = Program::from_compiler(&compiler);

    if let Some(dot) = args.dot {
        let mut out = std::fs::File::create(dot)?;
        compiler.write_dot_grouped(&mut out)?;
    }
    if let Some(metal) = args.metal {
        let mut out = std::fs::File::create(metal)?;
        prog.write_metal(&mut out)?;
    }
    if let Some(img) = args.image {
        let out = ctx.render_2d(node, args.size)?;
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
