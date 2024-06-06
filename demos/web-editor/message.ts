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

export class StartRequest {
  kind: RequestKind.Start;
  index: number;

  constructor(index: number) {
    this.index = index;
    this.kind = RequestKind.Start;
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
  mode: RenderMode;

  constructor(tape: Uint8Array, mode: RenderMode) {
    this.tape = tape;
    this.kind = RequestKind.Shape;
    this.mode = mode;
  }
}

export type WorkerRequest = ScriptRequest | ShapeRequest | StartRequest;

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
