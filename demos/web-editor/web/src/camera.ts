import * as fidget from "../../crate/pkg/fidget_wasm_demo";

export type CameraKind = '2d' | '3d';
export type Camera2D = {
  kind: '2d';
  camera: fidget.JsCanvas2;
};

export type Camera3D = {
  kind: '3d';
  camera: fidget.JsCanvas3;
};

export type Camera = Camera2D | Camera3D;
