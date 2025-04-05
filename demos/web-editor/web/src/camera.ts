import * as fidget from "../../crate/pkg/fidget_wasm_demo";

export enum CameraKind {
  TwoD,
  ThreeD,
}

export type Camera2D = {
  kind: CameraKind.TwoD;
  camera: fidget.JsCanvas2;
};

export type Camera3D = {
  kind: CameraKind.ThreeD;
  camera: fidget.JsCanvas3;
};

export type Camera = Camera2D | Camera3D;
