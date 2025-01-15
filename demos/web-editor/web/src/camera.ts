import * as fidget from "../../crate/pkg/fidget_wasm_demo";

export enum CameraKind {
  TwoD,
  ThreeD,
}

export type Camera2D = {
  kind: CameraKind.TwoD;
  camera: fidget.JsCamera2;
  translateHandle: fidget.JsTranslateHandle2 | null;
};

export type Camera3D = {
  kind: CameraKind.ThreeD;
  camera: fidget.JsCamera3;
  translateHandle: fidget.JsTranslateHandle3 | null;
  rotateHandle: fidget.JsRotateHandle | null;
};

export type Camera = Camera2D | Camera3D;
