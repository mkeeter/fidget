import * as fidget from "../../crate/pkg/fidget_wasm_demo";
import { Camera } from "./camera";

export class ScriptRequest {
  kind: 'script';
  script: string;

  constructor(script: string) {
    this.script = script;
    this.kind = 'script';
  }
}

type RenderMode = 'bitmap' | 'heightmap' | 'normals';
export class RenderRequest {
  kind: 'shape';
  tape: Uint8Array;
  camera: Uint8Array;
  depth: number;
  mode: RenderMode;
  cancel_token_ptr: number; // pointer

  constructor(
    tape: Uint8Array,
    camera: Camera,
    depth: number,
    mode: RenderMode,
    cancel_token_ptr: number,
  ) {
    this.tape = tape;
    this.kind = 'shape';
    this.camera = camera.camera.serialize_view();
    this.depth = depth;
    this.mode = mode;
    this.cancel_token_ptr = cancel_token_ptr;
  }
}

export class StartRequest {
  kind: 'start';
  init: object;

  constructor(init: object) {
    this.init = init;
    this.kind = 'start';
  }
}

export type WorkerRequest = StartRequest | ScriptRequest | RenderRequest;

////////////////////////////////////////////////////////////////////////////////

export class StartedResponse {
  kind: 'started';

  constructor() {
    this.kind = 'started';
  }
}

export class ScriptResponse {
  kind: 'script';
  output: string;
  tape: Uint8Array | null;

  constructor(output: string, tape: Uint8Array | null) {
    this.output = output;
    this.tape = tape;
    this.kind = 'script';
  }
}

export class ImageResponse {
  kind: 'image';
  data: Uint8Array;
  depth: number;

  constructor(data: Uint8Array, depth: number) {
    this.data = data;
    this.depth = depth;
    this.kind = 'image';
  }
}

export class CancelledResponse {
  kind: 'cancelled';

  constructor() {
    this.kind = 'cancelled';
  }
}

export type WorkerResponse =
  | StartedResponse
  | ScriptResponse
  | ImageResponse
  | CancelledResponse;
