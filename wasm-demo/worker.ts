import {
  ImageResponse,
  RequestKind,
  ScriptRequest,
  StartRequest,
  StartedResponse,
  WorkerRequest,
} from "./message";

var fidget: any = null;

class Worker {
  /// Index of this worker, between 0 and workers_per_side
  index: number;

  /// Total number of workers per image side
  workers_per_side: number;

  constructor(req: StartRequest) {
    this.index = req.index;
    this.workers_per_side = req.workers_per_side;
  }

  render(s: ScriptRequest) {
    const tree = fidget.eval_script(s.script);
    const out = fidget.render_region(tree, 512, 0, 4);
    postMessage(new ImageResponse(out));
  }
}

async function run() {
  fidget = await import("./pkg")!;
  let worker: null | Worker = null;
  onmessage = function (e: any) {
    let req = e.data as WorkerRequest;
    switch (req.kind) {
      case RequestKind.Start: {
        console.log("STARTING");
        let r = req as StartRequest;
        worker = new Worker(req as StartRequest);
        break;
      }
      case RequestKind.Script: {
        worker!.render(req as ScriptRequest);
        break;
      }
      default:
        console.error(`unknown worker request ${req}`);
    }
  };
  postMessage(new StartedResponse());
}
run();
