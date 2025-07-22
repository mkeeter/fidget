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
  ScriptRequest,
  ScriptResponse,
  StartRequest,
  RenderRequest,
  WorkerRequest,
  WorkerResponse,
} from "./message";
import { Camera, CameraKind } from "./camera";
import { rhai } from "./rhai";
import { RENDER_SIZE, MAX_DEPTH } from "./constants";

import * as fidget from "../../crate/pkg/fidget_wasm_demo";

import GYROID_SCRIPT from "../../../../models/gyroid-sphere.rhai";

async function setup() {
  await fidget.default();
  await fidget.initThreadPool(navigator.hardwareConcurrency);

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
  startDepth: number; // starting render depth
  currentDepth: number; // current render depth
  cancel: fidget.JsCancelToken | null;

  startTime: number;

  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url));
    this.worker.onmessage = (m) => {
      this.onWorkerMessage(m.data as WorkerResponse);
    };
    this.worker.postMessage(
      new StartRequest({
        module: fidget.get_module(),
        memory: fidget.get_memory(),
      }),
    );
    // everything else is handled in this.init() once the worker starts
  }

  init() {
    let scene = new Scene();

    let requestRedraw = this.requestRedraw.bind(this);
    scene.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      if (scene.zoomAbout(event)) {
        requestRedraw();
      }
    });
    scene.canvas.addEventListener("contextmenu", (event) => {
      event.preventDefault();
    });
    scene.canvas.addEventListener(
      "pointerdown",
      (event) => {
        event.preventDefault();
        scene.beginDrag(event);
      },
      { passive: false },
    );
    window.addEventListener(
      "pointerup",
      (event) => {
        event.preventDefault();
        scene.endDrag();
      },
      { passive: false },
    );
    window.addEventListener(
      "pointermove",
      (event) => {
        event.preventDefault();
        if (scene.drag(event)) {
          requestRedraw();
        }
      },
      { passive: false },
    );
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

    this.tape = null;
    this.rerender = false;
    this.rendering = false;
    this.startDepth = 0;
    this.currentDepth = MAX_DEPTH;
    this.cancel = null;

    // Also re-render if the mode changes
    const select = document.getElementById("mode");
    select.addEventListener("change", this.onModeChanged.bind(this), false);
  }

  requestRedraw() {
    if (this.rendering) {
      if (this.cancel && this.currentDepth != this.startDepth) {
        this.cancel.cancel();
        this.cancel = null;
      }
      this.rerender = true;
    } else {
      this.currentDepth = this.startDepth;
      this.beginRender(this.tape);
    }
  }

  onModeChanged() {
    switch (this.getMode()) {
      case "heightmap":
      case "normals":
      case "shaded": {
        this.scene.resetCamera("3d");
        break;
      }
      case "bitmap": {
        this.scene.resetCamera("2d");
        break;
      }
    }
    this.requestRedraw();
  }

  onScriptChanged(text: string) {
    document.getElementById("status").textContent = "Evaluating...";
    this.worker.postMessage(new ScriptRequest(text));
  }

  getMode() {
    const e = document.getElementById("mode") as HTMLSelectElement;
    switch (e.value) {
      case "bitmap":
      case "normals":
      case "heightmap":
      case "shaded": {
        return e.value;
      }
      default: {
        return null;
      }
    }
  }

  beginRender(tape: Uint8Array) {
    document.getElementById("status").textContent = "Rendering...";
    this.startTime = performance.now();
    const mode = this.getMode();
    this.cancel = new fidget.JsCancelToken();
    this.worker.postMessage(
      new RenderRequest(
        tape,
        this.scene.camera,
        this.currentDepth,
        mode,
        this.cancel.get_ptr(),
      ),
    );
    this.rerender = false;
    this.rendering = true;
  }

  onWorkerMessage(req: WorkerResponse) {
    switch (req.kind) {
      case "started": {
        // Initialize the rest of the app, and do an initial render
        this.init();
        const text = this.editor.view.state.doc.toString();
        this.onScriptChanged(text);
        break;
      }
      case "image": {
        this.scene.setTextureRegion(req.data, req.depth);
        this.scene.draw(req.depth);

        const endTime = performance.now();
        const dt = endTime - this.startTime;
        if (this.currentDepth == 0) {
          document.getElementById("status").textContent =
            `Rendered in ${dt.toFixed(2)} ms`;
        }
        this.rendering = false;

        // If this is our initial render resolution, adjust our max depth to hit
        // a target framerate.
        if (this.currentDepth == this.startDepth) {
          if (dt > 50 && this.startDepth != MAX_DEPTH) {
            this.startDepth += 1;
          } else if (dt < 10 && this.startDepth > 0) {
            this.startDepth -= 1;
          }
        }
        if (this.rerender) {
          this.currentDepth = this.startDepth;
          this.beginRender(this.tape);
        } else if (this.currentDepth > 0) {
          // Render again at the next resolution
          this.currentDepth -= 1;
          this.beginRender(this.tape);
        }
        break;
      }
      case "cancelled": {
        // Cancellation always implies a rerender
        this.currentDepth = this.startDepth;
        this.beginRender(this.tape);
        break;
      }
      case "script": {
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
      { tag: tags.name },
      { tag: tags.number, color: "#2E7D32" },
      { tag: tags.string, color: "#AD1457", fontStyle: "italic" },
      { tag: tags.definitionKeyword, color: "#C62828" },
      { tag: tags.comment, color: "#bbbbbb" },
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
        vTextureCoord = vec2((aVertexPosition.xy + 1.0) / 2.0);
        vTextureCoord.y = 1.0 - vTextureCoord.y;
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

class ShadedProgramInfo {
  program: WebGLProgram;
  vertexPositionAttrib: number;
  uGeometrySampler: WebGLUniformLocation;
  uMaxDepth: WebGLUniformLocation;

  constructor(gl: WebGLRenderingContext) {
    // Vertex shader program (same as regular)
    const vsSource = `
      attribute vec4 aVertexPosition;
      varying highp vec2 vTextureCoord;
      void main() {
        gl_Position = aVertexPosition;
        vTextureCoord = vec2((aVertexPosition.xy + 1.0) / 2.0);
        vTextureCoord.y = 1.0 - vTextureCoord.y;
      }
    `;

    // Fragment shader for shaded rendering
    const fsSource = `
      precision highp float;
      varying highp vec2 vTextureCoord;
      uniform sampler2D uGeometrySampler;
      uniform float uMaxDepth;

      struct Light {
        vec3 position;
        float intensity;
      };

      void main() {
        vec4 geometryData = texture2D(uGeometrySampler, vTextureCoord);
        
        // Extract depth and normal from geometry texture
        float depth = geometryData.r;
        vec3 normal = geometryData.gba;
        
        // If depth is 0, this pixel is transparent
        if (depth == 0.0) {
          discard;
        }
        
        // Calculate 3D position from texture coordinates and depth
        vec3 p = vec3(
          (vTextureCoord.xy - 0.5) * 2.0,
          2.0 * (depth - 0.5)
        );
        
        vec3 n = normalize(normal);
        
        // Define lights (matching the Rust implementation)
        Light lights[3];
        lights[0] = Light(vec3(5.0, -5.0, 10.0), 0.5);
        lights[1] = Light(vec3(-5.0, 0.0, 10.0), 0.15);
        lights[2] = Light(vec3(0.0, -5.0, 10.0), 0.15);
        
        // Calculate lighting
        float accum = 0.2; // ambient light
        for (int i = 0; i < 3; i++) {
          vec3 lightDir = normalize(lights[i].position - p);
          accum += max(dot(lightDir, n), 0.0) * lights[i].intensity;
        }
        accum = clamp(accum, 0.0, 1.0);
        
        gl_FragColor = vec4(accum, accum, accum, 1.0);
      }
    `;

    this.program = initShaderProgram(gl, vsSource, fsSource);
    this.vertexPositionAttrib = gl.getAttribLocation(
      this.program,
      "aVertexPosition",
    );
    this.uGeometrySampler = gl.getUniformLocation(
      this.program,
      "uGeometrySampler",
    );
    this.uMaxDepth = gl.getUniformLocation(this.program, "uMaxDepth");
  }
}

class Scene {
  canvas: HTMLCanvasElement;
  gl: WebGLRenderingContext;
  basicProgram: ProgramInfo;
  shadedProgram: ShadedProgramInfo;
  buffers: Buffers;
  rgbaTextures: WebGLTexture[];
  depthNormalTextures: WebGLTexture[];
  camera: Camera;
  currentMode: string;

  constructor() {
    this.canvas = document.querySelector<HTMLCanvasElement>("#glcanvas");
    this.camera = {
      kind: "3d",
      camera: new fidget.JsCanvas3(this.canvas.width, this.canvas.height),
    };

    this.gl = this.canvas.getContext("webgl");
    if (this.gl === null) {
      alert(
        "Unable to initialize WebGL. Your browser or machine may not support it.",
      );
      return;
    }

    this.buffers = new Buffers(this.gl);
    this.basicProgram = new ProgramInfo(this.gl);
    this.shadedProgram = new ShadedProgramInfo(this.gl);
    this.currentMode = "normals";

    // RGBA textures for bitmap/heightmap/normals modes
    this.rgbaTextures = [];
    // Float textures for depth+normal data (shaded mode)
    this.depthNormalTextures = [];

    for (var depth = 0; depth <= MAX_DEPTH; ++depth) {
      const size = Math.round(RENDER_SIZE / Math.pow(2, depth));

      // RGBA texture
      const texture = this.gl.createTexture();
      this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_WRAP_S,
        this.gl.CLAMP_TO_EDGE,
      );
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_WRAP_T,
        this.gl.CLAMP_TO_EDGE,
      );
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_MIN_FILTER,
        this.gl.NEAREST,
      );
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_MAG_FILTER,
        this.gl.NEAREST,
      );
      this.gl.texImage2D(
        this.gl.TEXTURE_2D,
        0,
        this.gl.RGBA,
        size,
        size,
        0, // border
        this.gl.RGBA,
        this.gl.UNSIGNED_BYTE,
        new Uint8Array(size * size * 4),
      );
      this.gl.generateMipmap(this.gl.TEXTURE_2D);
      this.rgbaTextures.push(texture);

      // Depth/normal texture for shaded mode (RGBA32F equivalent using RGBA + FLOAT)
      const depthNormalTexture = this.gl.createTexture();
      this.gl.bindTexture(this.gl.TEXTURE_2D, depthNormalTexture);
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_WRAP_S,
        this.gl.CLAMP_TO_EDGE,
      );
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_WRAP_T,
        this.gl.CLAMP_TO_EDGE,
      );
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_MIN_FILTER,
        this.gl.NEAREST,
      );
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_MAG_FILTER,
        this.gl.NEAREST,
      );
      this.gl.texImage2D(
        this.gl.TEXTURE_2D,
        0,
        this.gl.RGBA,
        size,
        size,
        0, // border
        this.gl.RGBA,
        this.gl.FLOAT,
        new Float32Array(size * size * 4),
      );
      this.depthNormalTextures.push(depthNormalTexture);
    }
  }

  resetCamera(kind: CameraKind) {
    let width = this.canvas.width;
    let height = this.canvas.height;
    switch (kind) {
      case "2d": {
        this.camera = {
          kind,
          camera: new fidget.JsCanvas2(width, height),
        };
        break;
      }
      case "3d": {
        this.camera = {
          kind,
          camera: new fidget.JsCanvas3(width, height),
        };
        break;
      }
    }
  }

  zoomAbout(event: WheelEvent) {
    let [x, y] = this.screenPosition(event);
    return this.camera.camera.zoom_about(event.deltaY, x, y);
  }

  screenPosition(event: MouseEvent): readonly [number, number] {
    let rect = this.canvas.getBoundingClientRect();
    let x = event.clientX - rect.left;
    let y = event.clientY - rect.top;
    return [x, y];
  }

  beginDrag(event: MouseEvent) {
    const [x, y] = this.screenPosition(event);
    const button = event.button === 0;
    if (this.camera.kind === "3d") {
      this.camera.camera.begin_drag(x, y, button);
    } else {
      this.camera.camera.begin_drag(x, y);
    }
  }

  drag(event: MouseEvent): boolean {
    const [x, y] = this.screenPosition(event);
    const out = this.camera.camera.drag(x, y);
    return out;
  }

  endDrag() {
    this.camera.camera.end_drag();
  }

  setTextureRegion(data: Uint8Array, depth: number) {
    const mode = (document.getElementById("mode") as HTMLSelectElement).value;
    const size = Math.round(RENDER_SIZE / Math.pow(2, depth));

    if (mode === "shaded") {
      // Handle depth+normal data for shaded rendering
      const size = Math.round(RENDER_SIZE / Math.pow(2, depth));
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.depthNormalTextures[depth]);
      const floatData = new Float32Array(data.buffer);
      this.gl.texImage2D(
        this.gl.TEXTURE_2D,
        0,
        this.gl.RGBA,
        size,
        size,
        0, // border
        this.gl.RGBA,
        this.gl.FLOAT,
        floatData,
      );
    } else {
      // Handle RGBA data
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.rgbaTextures[depth]);
      this.gl.texImage2D(
        this.gl.TEXTURE_2D,
        0,
        this.gl.RGBA,
        size,
        size,
        0, // border
        this.gl.RGBA,
        this.gl.UNSIGNED_BYTE,
        data,
      );
    }
    this.currentMode = mode;
  }

  draw(depth: number) {
    this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    this.setPositionAttribute();

    if (this.currentMode === "shaded") {
      this.drawShaded(depth);
    } else {
      this.drawBasic(depth);
    }
  }

  drawBasic(depth: number) {
    this.gl.useProgram(this.basicProgram.program);

    // Tell WebGL we want to affect texture unit 0
    this.gl.activeTexture(this.gl.TEXTURE0);

    // Bind the texture to texture unit 0
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.rgbaTextures[depth]);
    this.gl.uniform1i(this.basicProgram.uSampler, 0);

    const offset = 0;
    const vertexCount = 4;
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, offset, vertexCount);
  }

  drawShaded(depth: number) {
    this.gl.useProgram(this.shadedProgram.program);

    // Tell WebGL we want to affect texture unit 0
    this.gl.activeTexture(this.gl.TEXTURE0);

    // Bind the depth+normal texture to texture unit 0
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.depthNormalTextures[depth]);
    this.gl.uniform1i(this.shadedProgram.uGeometrySampler, 0);

    // Set max depth for normalization
    const size = Math.round(RENDER_SIZE / Math.pow(2, depth));
    this.gl.uniform1f(this.shadedProgram.uMaxDepth, size);

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

    // Use the appropriate program's vertex attribute based on current mode
    const vertexAttrib =
      this.currentMode === "shaded"
        ? this.shadedProgram.vertexPositionAttrib
        : this.basicProgram.vertexPositionAttrib;

    this.gl.vertexAttribPointer(
      vertexAttrib,
      numComponents,
      type,
      normalize,
      stride,
      offset,
    );
    this.gl.enableVertexAttribArray(vertexAttrib);
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
