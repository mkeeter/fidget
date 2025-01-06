mod options;
mod prepare;
mod process;

use anyhow::Result;
use env_logger::Env;

////////////////////////////////////////////////////////////////////////////////

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .init();

    let args = options::parse_options();

    let (ctx, root) = prepare::prepare_tape(&args);

    process::run_action(ctx, root, &args)
}
