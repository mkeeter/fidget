use clap::Parser;

/// Simple test program
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Name of the model file to load
    filename: String,
}

use jitfive::Context;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let (ctx, node) =
        Context::from_text(&mut std::fs::File::open(args.filename)?);
    ctx.to_dot(&mut std::fs::File::create("out.dot")?, node)?;
    Ok(())
}
