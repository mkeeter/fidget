import { WorkerRequest, RequestKind } from "./message"

async function run() {
  const fidget = await import("./pkg")!;
  onmessage = function (e: any) {
    console.log("onmessage");
    let req = e.data as WorkerRequest;
    console.log(req);
    switch (req.kind) {
        case RequestKind.Script:
          console.log("got script", req.script);
    }
    const foo = fidget.eval_script("x + y");
    const out = fidget.render_region(foo, 512, 0, 4);
    console.log(out.length);
  };
}
run();
