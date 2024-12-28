import * as fidget from "../../crate/pkg/fidget_wasm_demo";
import { Camera } from "./camera";

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
  depth: number;
  mode: RenderMode;

  constructor(
    tape: Uint8Array,
    camera: Camera,
    depth: number,
    mode: RenderMode,
  ) {
    this.tape = tape;
    this.kind = RequestKind.Shape;
    this.camera = camera.camera.serialize();
    this.depth = depth;
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
  depth: number;

  constructor(data: Uint8Array, depth: number) {
    this.data = data;
    this.depth = depth;
    this.kind = ResponseKind.Image;
  }
}

export type WorkerResponse = StartedResponse | ScriptResponse | ImageResponse;
