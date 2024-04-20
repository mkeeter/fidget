async function run() {
  var fidget: any = null;
  onmessage = function (e: any) {
    if (e.data.fidget) {
      fidget = e.data.fidget;
    } else {
      const out = fidget.render_region(e.data.tree, 512, 0, 4);
      console.log(out);
    }
  };
}
run();
