import { basicSetup } from "codemirror";
import { EditorView, ViewPlugin, keymap, lineNumbers } from "@codemirror/view";
import { foldGutter } from "@codemirror/language";
import { EditorState } from "@codemirror/state";
import { defaultKeymap } from "@codemirror/commands";

import { ScriptRequest, WorkerRequest } from "./message"

const RENDER_SIZE = 512;
var fidget: any = null;

async function setup() {
  fidget = await import("./pkg")!;
  const app = new App("y + x*x");

  console.log("booted");
}

class App {
  editor: Editor;
  output: Output;
  scene: Scene;
  worker: Worker;

  constructor(initial_script: string) {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url));
    this.scene = new Scene();
    this.editor = new Editor(
      initial_script,
      document.getElementById("editor-outer"),
      this.onScriptChanged.bind(this),
    );
    this.output = new Output(document.getElementById("output-outer"));
    this.onScriptChanged(initial_script);
    this.worker.onmessage = this.onWorkerMessage.bind(this);
  }

  onScriptChanged(text: string) {
    console.log(this.worker);
    let shape = null;
    let result = null;
    try {
      const shapeTree = fidget.eval_script(text);
      result = "Ok(..)";
      const startTime = performance.now();

      this.worker.postMessage(new ScriptRequest(text));
      shape = fidget.render(shapeTree, RENDER_SIZE);
      const endTime = performance.now();
      document.getElementById("status").textContent =
        `Rendered in ${endTime - startTime} ms`;
    } catch (error) {
      // Do some string formatting to make errors cleaner
      result = error
        .toString()
        .replace("Rhai evaluation error: ", "Rhai evaluation error:\n")
        .replace(" (line ", "\n(line ")
        .replace(" (expecting ", "\n(expecting ");
    }
    this.output.setText(result);
    if (shape) {
      this.scene.setTexture(shape);
      this.scene.draw();
    }
  }

  onWorkerMessage(msg: any) {
    console.log(msg);
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
          console.log("HI THERE");
          if (v.docChanged) {
            if (this.timeout) {
              window.clearTimeout(this.timeout);
            }
            const text = v.state.doc.toString();
            this.timeout = window.setTimeout(() => cb(text), 500);
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
  }

  setTexture(data: Uint8Array) {
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);

    const level = 0;
    const internalFormat = this.gl.RGBA;
    const width = RENDER_SIZE;
    const height = RENDER_SIZE;
    const border = 0;
    const srcFormat = this.gl.RGBA;
    const srcType = this.gl.UNSIGNED_BYTE;
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      level,
      internalFormat,
      width,
      height,
      border,
      srcFormat,
      srcType,
      data,
    );

    this.gl.generateMipmap(this.gl.TEXTURE_2D);
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
