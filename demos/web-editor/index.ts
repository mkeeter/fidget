import { basicSetup } from "codemirror";
import { EditorView, ViewPlugin, keymap, lineNumbers } from "@codemirror/view";
import { foldGutter } from "@codemirror/language";
import { EditorState } from "@codemirror/state";
import { defaultKeymap } from "@codemirror/commands";

import {
  RenderMode,
  ResponseKind,
  ScriptRequest,
  ScriptResponse,
  ShapeRequest,
  StartRequest,
  WorkerRequest,
  WorkerResponse,
} from "./message";

import { RENDER_SIZE, WORKERS_PER_SIDE, WORKER_COUNT } from "./constants";

import INITIAL_SCRIPT from "../../models/gyroid-sphere.rhai";

var fidget: any = null;

async function setup() {
  fidget = await import("./pkg")!;
  const app = new App();
}

class App {
  editor: Editor;
  output: Output;
  scene: Scene;
  workers: Array<Worker>;
  workers_started: number;
  workers_done: number;
  start_time: number;

  constructor() {
    this.scene = new Scene();
    this.editor = new Editor(
      INITIAL_SCRIPT,
      document.getElementById("editor-outer"),
      this.onScriptChanged.bind(this),
    );
    this.output = new Output(document.getElementById("output-outer"));
    this.workers = [];
    this.workers_started = 0;
    this.workers_done = 0;

    for (let i = 0; i < WORKER_COUNT; ++i) {
      const worker = new Worker(new URL("./worker.ts", import.meta.url));
      worker.onmessage = (m) => {
        this.onWorkerMessage(i, m.data as WorkerResponse);
      };
      this.workers.push(worker);
    }

    // Also re-render if the mode changes
    const select = document.getElementById("mode");
    select.addEventListener("change", this.onModeChanged.bind(this), false);
  }

  onModeChanged() {
    const text = this.editor.view.state.doc.toString();
    this.onScriptChanged(text);
  }

  onScriptChanged(text: string) {
    this.workers[0].postMessage(new ScriptRequest(text));
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

  onWorkerMessage(i: number, req: WorkerResponse) {
    switch (req.kind) {
      case ResponseKind.Image: {
        const region_size = RENDER_SIZE / WORKERS_PER_SIDE;
        const x = Math.trunc(i / WORKERS_PER_SIDE) * region_size;
        const y = (WORKERS_PER_SIDE - (i % WORKERS_PER_SIDE) - 1) * region_size;
        this.scene.setTextureRegion(x, y, region_size, req.data);
        this.scene.draw();

        this.workers_done += 1;
        if (this.workers_done == WORKER_COUNT) {
          const endTime = performance.now();
          document.getElementById("status").textContent =
            `Rendered in ${endTime - this.start_time} ms`;
        }
        break;
      }
      case ResponseKind.Started: {
        this.workers[i].postMessage(new StartRequest(i));
        this.workers_started += 1;
        if (this.workers_started == WORKER_COUNT) {
          this.onScriptChanged(INITIAL_SCRIPT);
        }
        break;
      }
      case ResponseKind.Script: {
        let r = req as ScriptResponse;
        this.output.setText(r.output);
        if (r.tape) {
          this.start_time = performance.now();
          this.workers_done = 0;
          const mode = this.getMode();
          this.workers.forEach((w) => {
            w.postMessage(new ShapeRequest(r.tape, mode));
          });
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
    this.view = new EditorView({
      doc: initial_script,
      extensions: [
        basicSetup,
        keymap.of(defaultKeymap),
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

class Scene {
  gl: WebGLRenderingContext;
  programInfo: ProgramInfo;
  buffers: Buffers;
  texture: WebGLTexture;

  constructor() {
    const canvas = document.querySelector<HTMLCanvasElement>("#glcanvas");
    this.gl = canvas.getContext("webgl");
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

  setTextureRegion(x: number, y: number, size: number, data: Uint8Array) {
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
    this.gl.texSubImage2D(
      this.gl.TEXTURE_2D,
      0,
      x,
      y,
      size,
      size,
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
