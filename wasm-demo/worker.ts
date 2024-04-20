import {
  ImageResponse,
  RequestKind,
  ScriptRequest,
  StartRequest,
  StartedResponse,
  WorkerRequest,
} from "./message";

import {
RENDER_SIZE,
WORKERS_PER_SIDE,
WORKER_COUNT,
} from "./constants";

var fidget: any = null;

class Worker {
  /// Index of this worker, between 0 and workers_per_side
  index: number;

  constructor(req: StartRequest) {
    this.index = req.index;
  }

  render(s: ScriptRequest) {
    const tree = fidget.eval_script(s.script);
    const out = fidget.render_region(
      tree,
      RENDER_SIZE,
      this.index,
      WORKERS_PER_SIDE,
    );
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
