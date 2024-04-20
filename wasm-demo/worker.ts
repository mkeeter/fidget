import {
  ImageResponse,
  RequestKind,
  ShapeRequest,
  StartRequest,
  StartedResponse,
  WorkerRequest,
} from "./message";

import { RENDER_SIZE, WORKERS_PER_SIDE, WORKER_COUNT } from "./constants";

var fidget: any = null;

class Worker {
  /// Index of this worker, between 0 and workers_per_side
  index: number;

  constructor(req: StartRequest) {
    this.index = req.index;
  }

  render(s: ShapeRequest) {
    const shape = fidget.deserialize_tape(s.tape);
    const out = fidget.render_region(
      shape,
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
      case RequestKind.Shape: {
        worker!.render(req as ShapeRequest);
        break;
      }
      default:
        console.error(`unknown worker request ${req}`);
    }
  };
  postMessage(new StartedResponse());
}
run();
