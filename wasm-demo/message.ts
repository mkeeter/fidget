export enum RequestKind {
  Start,
  Shape,
}

export class StartRequest {
  kind: RequestKind.Start;
  index: number;

  constructor(index: number) {
    this.index = index;
    this.kind = RequestKind.Start;
  }
}

export class ShapeRequest {
  kind: RequestKind.Shape;
  tape: Uint8Array;

  constructor(tape: Uint8Array) {
    this.tape = tape;
    this.kind = RequestKind.Shape;
  }
}

export type WorkerRequest = ShapeRequest | StartRequest;

////////////////////////////////////////////////////////////////////////////////

export enum ResponseKind {
  Started,
  Image,
}

export class StartedResponse {
  kind: ResponseKind.Started;

  constructor() {
    this.kind = ResponseKind.Started;
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

export type WorkerResponse = StartedResponse | ImageResponse;
