enum RequestKind {
  Start,
  Script,
}

class StartRequest {
  kind: RequestKind.Start;
  index: number;
  workers_per_side: number;

  constructor(index: number, workers_per_side: number) {
    this.index = index;
    this.workers_per_side = workers_per_side;
    this.kind = RequestKind.Start;
  }
}

class ScriptRequest {
  kind: RequestKind.Script;
  script: string;

  constructor(s: string) {
    this.script = s;
    this.kind = RequestKind.Script;
  }
}

type WorkerRequest = ScriptRequest | StartRequest;

////////////////////////////////////////////////////////////////////////////////

enum ResponseKind {
  Started,
  Image,
}

class StartedResponse {
  kind: ResponseKind.Started;

  constructor() {
    this.kind = ResponseKind.Started;
  }
}

class ImageResponse {
  kind: ResponseKind.Image;
  data: Uint8Array;

  constructor(data: Uint8Array) {
    this.data = data;
    this.kind = ResponseKind.Image;
  }
}

type WorkerResponse = StartedResponse | ImageResponse;

export {
  ImageResponse,
  RequestKind,
  ResponseKind,
  ScriptRequest,
  StartRequest,
  StartedResponse,
  WorkerRequest,
  WorkerResponse,
};
