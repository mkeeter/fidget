import {
  ImageResponse,
  RenderMode,
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
    let out: Uint8Array;
    switch (s.mode) {
      case RenderMode.Bitmap: {
        out = fidget.render_region_2d(
          shape,
          RENDER_SIZE,
          this.index,
          WORKERS_PER_SIDE,
        );
        break;
      }
      case RenderMode.Heightmap: {
        out = fidget.render_region_heightmap(
          shape,
          RENDER_SIZE,
          this.index,
          WORKERS_PER_SIDE,
        );
        break;
      }
      case RenderMode.Normals: {
        out = fidget.render_region_normals(
          shape,
          RENDER_SIZE,
          this.index,
          WORKERS_PER_SIDE,
        );
        break;
      }
    }
    postMessage(new ImageResponse(out), { transfer: [out.buffer] });
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
      postMessage(new ScriptResponse(result, tape), {
        transfer: [tape.buffer],
      });
    } else {
      postMessage(new ScriptResponse(result, tape));
    }
  }
}

async function run() {
  fidget = await import("../../crate/pkg")!;
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
