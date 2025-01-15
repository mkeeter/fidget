import {
  ImageResponse,
  RenderMode,
  RequestKind,
  CancelledResponse,
  StartRequest,
  ScriptRequest,
  ScriptResponse,
  RenderRequest,
  StartedResponse,
  WorkerRequest,
} from "./message";

import { RENDER_SIZE } from "./constants";

import * as fidget from "../../crate/pkg/fidget_wasm_demo";

class Worker {
  render(s: RenderRequest) {
    const shape = fidget.deserialize_tape(s.tape);
    const cancel = fidget.JsCancelToken.from_ptr(s.cancel_token_ptr);
    let out: Uint8Array;
    const size = Math.round(RENDER_SIZE / Math.pow(2, s.depth));
    try {
      switch (s.mode) {
        case RenderMode.Bitmap: {
          const camera = fidget.JsCamera2.deserialize(s.camera);
          out = fidget.render_2d(shape, size, camera, cancel);
          break;
        }
        case RenderMode.Heightmap: {
          const camera = fidget.JsCamera3.deserialize(s.camera);
          out = fidget.render_heightmap(shape, size, camera, cancel);
          break;
        }
        case RenderMode.Normals: {
          const camera = fidget.JsCamera3.deserialize(s.camera);
          out = fidget.render_normals(shape, size, camera, cancel);
          break;
        }
      }
      postMessage(new ImageResponse(out, s.depth), { transfer: [out.buffer] });
    } catch (e) {
      postMessage(new CancelledResponse());
    }
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
  let worker = new Worker();
  onmessage = function (e: any) {
    let req = e.data as WorkerRequest;
    switch (req.kind) {
      case RequestKind.Start: {
        fidget.initSync((req as StartRequest).init);
        postMessage(new StartedResponse());
        break;
      }
      case RequestKind.Shape: {
        worker!.render(req as RenderRequest);
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
}
run();
