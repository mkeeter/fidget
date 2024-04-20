async function run() {
  const fidget = await import("./pkg")!;
  onmessage = function (e: any) {
    const foo = fidget.eval_script("x + y");
    const out = fidget.render_region(foo, 512, 0, 4);
    console.log(out.length);
  };
}
run();
