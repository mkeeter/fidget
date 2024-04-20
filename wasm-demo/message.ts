export enum RequestKind {
  Start,
  Script,
}

export class StartRequest {
  kind: RequestKind.Start;
  index: number;

  constructor(index: number) {
    this.index = index;
    this.kind = RequestKind.Start;
  }
}

export class ScriptRequest {
  kind: RequestKind.Script;
  script: string;

  constructor(s: string) {
    this.script = s;
    this.kind = RequestKind.Script;
  }
}

export type WorkerRequest = ScriptRequest | StartRequest;

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
