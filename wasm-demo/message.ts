enum RequestKind {
  Script,
}
class ScriptRequest {
    kind: RequestKind.Script;
    script: string;

    constructor(s: string) {
      this.script = s;
      this.kind = RequestKind.Script;
    }
}

type WorkerRequest =
  | ScriptRequest;

export { ScriptRequest, WorkerRequest, RequestKind };
