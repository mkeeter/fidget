mod options;
mod process;

use anyhow::Result;
use env_logger::Env;
use log::info;
use std::time::Instant;

////////////////////////////////////////////////////////////////////////////////

fn main() -> Result<()> {
    use fidget::context::Context;

    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .init();

    let sphere_text = "
# This is a comment!
0x600000b90000 var-x
0x600000b900a0 square 0x600000b90000
0x600000b90050 var-y
0x600000b900f0 square 0x600000b90050
0x600000b90300 var-z
0x600000b90350 square 0x600000b90300
0x600000b90140 add 0x600000b900a0 0x600000b900f0
0x600000b90150 add 0x600000b90140 0x600000b90350
0x600000b90190 sqrt 0x600000b90150
0x600000b901e0 const 1
0x600000b90200 sub 0x600000b90190 0x600000b901e0
";

    let args = options::parse_options();

    use options::HardcodedShape;
    let (ctx, root) = match &args.input {
        None => {
            let top = Instant::now();
            let ret = match args.hardcoded_shape {
                HardcodedShape::SphereAsm => {
                    Context::from_text(&mut sphere_text.as_bytes()).unwrap()
                }
                HardcodedShape::SphereTree => {
                    use fidget::context::Tree;
                    let mut tree = Tree::x().square();
                    tree += Tree::y().square();
                    tree += Tree::z().square();
                    tree = tree.sqrt() - 1;
                    let mut ctx = Context::new();
                    let root = ctx.import(&tree);
                    (ctx, root)
                }
            };
            info!(
                "Created hardcoded model {:?} in {:?}",
                args.hardcoded_shape,
                top.elapsed()
            );
            ret
        }
        Some(path) => {
            info!("Loading model from {:?}", path);
            let top = Instant::now();
            let mut handle = std::fs::File::open(&path).unwrap();
            let ret = Context::from_text(&mut handle).unwrap();
            info!("Loaded model in {:?}", top.elapsed());
            ret
        }
    };

    info!("Context has {} items", ctx.len());
    info!("Root is {:?}", root);

    process::run_action(ctx, root, args)
}
