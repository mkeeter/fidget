import * as fidget from "../../crate/pkg/fidget_wasm_demo";

export enum RequestKind {
  Script,
  Start,
  Shape,
}

export class ScriptRequest {
  kind: RequestKind.Script;
  script: string;

  constructor(script: string) {
    this.script = script;
    this.kind = RequestKind.Script;
  }
}

export enum RenderMode {
  Bitmap,
  Heightmap,
  Normals,
}

export class ShapeRequest {
  kind: RequestKind.Shape;
  tape: Uint8Array;
  camera: Uint8Array;
  mode: RenderMode;

  constructor(
    tape: Uint8Array,
    camera2: fidget.JsCamera2,
    camera3: fidget.JsCamera3,
    mode: RenderMode,
  ) {
    this.tape = tape;
    this.kind = RequestKind.Shape;
    switch (mode) {
      case RenderMode.Bitmap: {
        this.camera = camera2.serialize();
        break;
      }
      case RenderMode.Heightmap:
      case RenderMode.Normals: {
        this.camera = camera3.serialize();
        break;
      }
    }
    this.mode = mode;
  }
}

export type WorkerRequest = ScriptRequest | ShapeRequest;

////////////////////////////////////////////////////////////////////////////////

export enum ResponseKind {
  Started,
  Script,
  Image,
}

export class StartedResponse {
  kind: ResponseKind.Started;

  constructor() {
    this.kind = ResponseKind.Started;
  }
}

export class ScriptResponse {
  kind: ResponseKind.Script;
  output: string;
  tape: Uint8Array | null;

  constructor(output: string, tape: Uint8Array | null) {
    this.output = output;
    this.tape = tape;
    this.kind = ResponseKind.Script;
  }
}

export class ImageResponse {
  kind: ResponseKind.Image;
  data: Uint8Array;

  constructor(data: Uint8Array) {
    this.data = data;
    this.kind = ResponseKind.Image;
  }
}

export type WorkerResponse = StartedResponse | ScriptResponse | ImageResponse;
