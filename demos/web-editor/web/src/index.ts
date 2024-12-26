import { basicSetup } from "codemirror";
import { EditorView, ViewPlugin, keymap, lineNumbers } from "@codemirror/view";
import {
  foldGutter,
  HighlightStyle,
  syntaxHighlighting,
} from "@codemirror/language";
import { EditorState } from "@codemirror/state";
import { defaultKeymap } from "@codemirror/commands";
import { tags } from "@lezer/highlight";

import {
  RenderMode,
  ResponseKind,
  ScriptRequest,
  ScriptResponse,
  ShapeRequest,
  WorkerRequest,
  WorkerResponse,
} from "./message";

import { rhai } from "./rhai";

import { RENDER_SIZE } from "./constants";
import * as fidget from "../../crate/pkg/fidget_wasm_demo";

import GYROID_SCRIPT from "../../../../models/gyroid-sphere.rhai";

async function setup() {
  await fidget.default();
  (window as any).fidget = fidget; // for easy of poking
  const app = new App();
}

class App {
  editor: Editor;
  output: Output;
  scene: Scene;
  worker: Worker;

  tape: Uint8Array | null; // current tape to render
  rerender: boolean; // should we rerender?
  rendering: boolean; // are we currently rendering?

  start_time: number;

  constructor() {
    let scene = new Scene();

    let requestRedraw = this.requestRedraw.bind(this);
    scene.canvas.addEventListener("wheel", (event) => {
      scene.zoomAbout(event);
      event.preventDefault();
      requestRedraw();
    });
    scene.canvas.addEventListener("contextmenu", (event) => {
      event.preventDefault();
    });
    scene.canvas.addEventListener("mousedown", (event) => {
      if (event.button === 0) {
        scene.beginTranslate(event);
      } else if (event.button === 2) {
        scene.beginRotate(event);
      }
    });
    window.addEventListener("mouseup", (event) => {
      scene.endDrag();
    });
    window.addEventListener("mousemove", (event) => {
      if (scene.drag(event)) {
        requestRedraw();
      }
    });

    this.scene = scene;

    // Hot-patch the gyroid script to be eval (instead of exec) flavored
    const re = /draw\((.*)\);/;
    const script = GYROID_SCRIPT.split(/\r?\n/)
      .map((line: string) => {
        let m = line.match(re);
        if (m) {
          return m[1];
        } else {
          return line;
        }
      })
      .join("\n")
      .trim();

    this.editor = new Editor(
      script,
      document.getElementById("editor-outer"),
      this.onScriptChanged.bind(this),
    );
    this.output = new Output(document.getElementById("output-outer"));
    this.worker = new Worker(new URL("./worker.ts", import.meta.url));
    this.worker.onmessage = (m) => {
      this.onWorkerMessage(m.data as WorkerResponse);
    };

    this.tape = null;
    this.rerender = false;

    // Also re-render if the mode changes
    const select = document.getElementById("mode");
    select.addEventListener("change", this.onModeChanged.bind(this), false);
  }

  requestRedraw() {
    if (this.rendering) {
      this.rerender = true;
    } else {
      this.beginRender(this.tape);
    }
  }

  onModeChanged() {
    const text = this.editor.view.state.doc.toString();
    this.scene.resetCameras();
    this.onScriptChanged(text);
  }

  onScriptChanged(text: string) {
    document.getElementById("status").textContent = "Evaluating...";
    this.worker.postMessage(new ScriptRequest(text));
  }

  getMode() {
    const e = document.getElementById("mode") as HTMLSelectElement;
    switch (e.value) {
      case "bitmap": {
        return RenderMode.Bitmap;
      }
      case "normals": {
        return RenderMode.Normals;
      }
      case "heightmap": {
        return RenderMode.Heightmap;
      }
      default: {
        return null;
      }
    }
  }

  beginRender(tape: Uint8Array) {
    document.getElementById("status").textContent = "Rendering...";
    this.start_time = performance.now();
    const mode = this.getMode();
    this.worker.postMessage(
      new ShapeRequest(tape, this.scene.camera2, this.scene.camera3, mode),
    );
    this.rerender = false;
    this.rendering = true;
  }

  onWorkerMessage(req: WorkerResponse) {
    switch (req.kind) {
      case ResponseKind.Started: {
        // Once the worker has started, do an initial render
        const text = this.editor.view.state.doc.toString();
        this.onScriptChanged(text);
        break;
      }
      case ResponseKind.Image: {
        this.scene.setTextureRegion(req.data);
        requestAnimationFrame((event) => this.scene.draw());

        const endTime = performance.now();
        document.getElementById("status").textContent =
          `Rendered in ${(endTime - this.start_time).toFixed(2)} ms`;
        this.rendering = false;

        // Immediately start rendering again if pending
        if (this.rerender) {
          this.beginRender(this.tape);
        }
        break;
      }
      case ResponseKind.Script: {
        let r = req as ScriptResponse;
        this.output.setText(r.output);
        if (r.tape) {
          this.tape = r.tape;
          this.beginRender(r.tape);
        } else {
          document.getElementById("status").textContent = "";
        }
        break;
      }

      default: {
        console.error(`unknown worker req ${req}`);
      }
    }
  }
}

class Editor {
  view: EditorView;
  timeout: number | null;

  constructor(
    initial_script: string,
    div: Element,
    cb: (text: string) => void,
  ) {
    this.timeout = null;

    const rhaiHighlight = HighlightStyle.define([
      { tag: tags.definitionKeyword, color: "#C62828" },
      { tag: tags.controlKeyword, color: "#6A1B9A" },
      { tag: tags.function(tags.name), color: "#0277BD" },
      { tag: tags.number, color: "#2E7D32" },
      { tag: tags.string, color: "#AD1457", fontStyle: "italic" },
    ]);

    this.view = new EditorView({
      doc: initial_script,
      extensions: [
        basicSetup,
        keymap.of(defaultKeymap),
        rhai(),
        syntaxHighlighting(rhaiHighlight),
        EditorView.updateListener.of((v) => {
          if (v.docChanged) {
            if (this.timeout) {
              window.clearTimeout(this.timeout);
            }
            const text = v.state.doc.toString();
            this.timeout = window.setTimeout(() => cb(text), 250);
          }
        }),
      ],
      parent: div,
    });
    div.children[0].id = "editor";
  }
}

class Output {
  view: EditorView;
  constructor(div: Element) {
    this.view = new EditorView({
      doc: "",
      extensions: [
        // Match basicSetup, but without any line numbers
        lineNumbers({ formatNumber: () => "" }),
        foldGutter(),
        EditorView.editable.of(false),
      ],
      parent: div,
    });
    div.children[0].id = "output";
  }
  setText(t: string) {
    this.view.dispatch({
      changes: { from: 0, to: this.view.state.doc.length, insert: t },
    });
  }
}

// WebGL wrangling is based on https://github.com/mdn/dom-examples (CC0)
class Buffers {
  pos: WebGLBuffer;
  constructor(gl: WebGLRenderingContext) {
    // Create a buffer for the square's positions.
    this.pos = gl.createBuffer();

    // Select the positionBuffer as the one to apply buffer
    // operations to from here out.
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pos);

    // Now create an array of positions for the square.
    const positions = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

    // Now pass the list of positions into WebGL to build the
    // shape. We do this by creating a Float32Array from the
    // JavaScript array, then use it to fill the current buffer.
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
  }
}

class ProgramInfo {
  program: WebGLProgram;
  vertexPositionAttrib: number;
  uSampler: WebGLUniformLocation;

  constructor(gl: WebGLRenderingContext) {
    // Vertex shader program
    const vsSource = `
      attribute vec4 aVertexPosition;
      varying highp vec2 vTextureCoord;
      void main() {
        gl_Position = aVertexPosition;
        vTextureCoord = (aVertexPosition.xy + 1.0) / 2.0;
      }
    `;

    const fsSource = `
      varying highp vec2 vTextureCoord;
      uniform sampler2D uSampler;
      void main() {
        gl_FragColor = texture2D(uSampler, vTextureCoord);
      }
    `;

    // Initialize a shader program; this is where all the lighting
    // for the vertices and so forth is established.
    this.program = initShaderProgram(gl, vsSource, fsSource);
    this.vertexPositionAttrib = gl.getAttribLocation(
      this.program,
      "aVertexPosition",
    );
    this.uSampler = gl.getUniformLocation(this.program, "uSampler");
  }
}

export enum CameraKind {
  TwoD,
  ThreeD,
}

class Scene {
  canvas: HTMLCanvasElement;
  gl: WebGLRenderingContext;
  programInfo: ProgramInfo;
  buffers: Buffers;
  texture: WebGLTexture;

  camera2: fidget.JsCamera2;
  translateHandle2: fidget.JsTranslateHandle2 | null;

  camera3: fidget.JsCamera3;
  translateHandle3: fidget.JsTranslateHandle3 | null;
  rotateHandle3: fidget.JsRotateHandle | null;

  constructor() {
    this.canvas = document.querySelector<HTMLCanvasElement>("#glcanvas");
    this.camera2 = new fidget.JsCamera2();
    this.camera3 = new fidget.JsCamera3();
    this.rotateHandle3 = null;
    this.translateHandle3 = null;
    this.translateHandle2 = null;

    this.gl = this.canvas.getContext("webgl");
    if (this.gl === null) {
      alert(
        "Unable to initialize WebGL. Your browser or machine may not support it.",
      );
      return;
    }
    this.buffers = new Buffers(this.gl);
    this.programInfo = new ProgramInfo(this.gl);
    this.texture = this.gl.createTexture();

    // Bind an initial texture of the correct size
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      RENDER_SIZE,
      RENDER_SIZE,
      0, // border
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      new Uint8Array(RENDER_SIZE * RENDER_SIZE * 4),
    );
    this.gl.generateMipmap(this.gl.TEXTURE_2D);
  }

  resetCameras() {
    this.camera2 = new fidget.JsCamera2();
    this.camera3 = new fidget.JsCamera3();
    this.rotateHandle3 = null;
    this.translateHandle3 = null;
    this.translateHandle2 = null;
  }

  zoomAbout(event: WheelEvent) {
    let [x, y] = this.screenToWorld(event);
    this.camera2.zoom_about(Math.pow(2, event.deltaY / 100.0), x, y);
    this.camera3.zoom_about(Math.pow(2, event.deltaY / 100.0), x, y);
  }

  screenToWorld(event: MouseEvent): readonly [number, number] {
    let rect = this.canvas.getBoundingClientRect();
    let x = ((event.clientX - rect.left) / RENDER_SIZE - 0.5) * 2.0;
    let y = ((event.clientY - rect.top) / RENDER_SIZE - 0.5) * 2.0;
    return [x, y];
  }

  beginTranslate(event: MouseEvent) {
    let [x, y] = this.screenToWorld(event);
    this.translateHandle2 = this.camera2.begin_translate(x, y);
    this.translateHandle3 = this.camera3.begin_translate(x, y);
  }

  beginRotate(event: MouseEvent) {
    let [x, y] = this.screenToWorld(event);
    this.rotateHandle3 = this.camera3.begin_rotate(x, y);
  }

  drag(event: MouseEvent): boolean {
    let [x, y] = this.screenToWorld(event);
    let changed = false;
    if (this.translateHandle2) {
      changed = this.camera2.translate(this.translateHandle2, x, y) || changed;
    }
    if (this.rotateHandle3) {
      changed = this.camera3.rotate(this.rotateHandle3, x, y) || changed;
    }
    if (this.translateHandle3) {
      changed = this.camera3.translate(this.translateHandle3, x, y) || changed;
    }

    return changed;
  }

  endDrag() {
    this.translateHandle2 = null;
    this.rotateHandle3 = null;
    this.translateHandle3 = null;
  }

  setTextureRegion(data: Uint8Array) {
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      RENDER_SIZE,
      RENDER_SIZE,
      0, // border
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      data,
    );
  }

  draw() {
    this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    this.setPositionAttribute();

    // Tell WebGL to use our program when drawing
    this.gl.useProgram(this.programInfo.program);

    // Tell WebGL we want to affect texture unit 0
    this.gl.activeTexture(this.gl.TEXTURE0);

    // Bind the texture to texture unit 0
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
    this.gl.uniform1i(this.programInfo.uSampler, 0);

    const offset = 0;
    const vertexCount = 4;
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, offset, vertexCount);
  }

  setPositionAttribute() {
    const numComponents = 2; // pull out 2 values per iteration
    const type = this.gl.FLOAT; // the data in the buffer is 32bit floats
    const normalize = false; // don't normalize
    const stride = 0; // how many bytes to get from one set of values to the next
    // 0 = use type and numComponents above
    const offset = 0; // how many bytes inside the buffer to start from
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.pos);
    this.gl.vertexAttribPointer(
      this.programInfo.vertexPositionAttrib,
      numComponents,
      type,
      normalize,
      stride,
      offset,
    );
    this.gl.enableVertexAttribArray(this.programInfo.vertexPositionAttrib);
  }
}

// Initialize a shader program, so WebGL knows how to draw our data
function initShaderProgram(
  gl: WebGLRenderingContext,
  vsSource: string,
  fsSource: string,
) {
  const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
  const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);

  // Create the shader program
  const shaderProgram = gl.createProgram();
  gl.attachShader(shaderProgram, vertexShader);
  gl.attachShader(shaderProgram, fragmentShader);
  gl.linkProgram(shaderProgram);

  // If creating the shader program failed, alert
  if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    alert(
      `Unable to initialize the shader program: ${gl.getProgramInfoLog(
        shaderProgram,
      )}`,
    );
    return null;
  }

  return shaderProgram;
}

//
// creates a shader of the given type, uploads the source and
// compiles it.
//
function loadShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);

  // Send the source to the shader object
  gl.shaderSource(shader, source);

  // Compile the shader program
  gl.compileShader(shader);

  // See if it compiled successfully
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    alert(
      `An error occurred compiling the shaders: ${gl.getShaderInfoLog(shader)}`,
    );
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

////////////////////////////////////////////////////////////////////////////////

setup();
