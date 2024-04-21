import {
  ImageResponse,
  RequestKind,
  ScriptRequest,
  ScriptResponse,
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

  run(s: ScriptRequest) {
    let shape = null;
    let result = "Ok(..)";
    try {
      shape = fidget.eval_script(s.script);
    } catch (error) {
      // Do some string formatting to make errors cleaner
      result = error
        .toString()
        .replace("Rhai evaluation error: ", "Rhai evaluation error:\n")
        .replace(" (line ", "\n(line ")
        .replace(" (expecting ", "\n(expecting ");
    }

    let tape = null;
    if (shape) {
      tape = fidget.serialize_into_tape(shape);
    }
    postMessage(new ScriptResponse(result, tape));
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
      case RequestKind.Script: {
        worker!.run(req as ScriptRequest);
        break;
      }
      default:
        console.error(`unknown worker request ${req}`);
    }
  };
  postMessage(new StartedResponse());
}
run();
