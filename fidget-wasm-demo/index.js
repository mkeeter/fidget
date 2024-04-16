import {basicSetup} from "codemirror"
import {EditorView, ViewPlugin, keymap, lineNumbers} from "@codemirror/view"
import {foldGutter} from "@codemirror/language"
import {EditorState} from "@codemirror/state"
import {defaultKeymap} from "@codemirror/commands"

async function setup() {
    const fidget = await import('./pkg')
      .catch(console.error);

    function setScript(text) {
        try {
            let shape = fidget.eval_script(text);
            var v = "Ok(..)";
            console.log(fidget.render(shape));
        } catch (error) {
            var v = error.toString();
            // Do some string formatting to make errors cleaner
            v = v.replace("Rhai error: ", "Rhai error:\n")
                .replace(" (line ", "\n(line ")
                .replace(" (expecting ", "\n(expecting ");
        }
        output.dispatch(
            {changes: {from: 0, to: output.state.doc.length, insert: v}});
    }

    var timeout = null;
    let myView = new EditorView({
        doc: "hello",
        extensions: [
            basicSetup,
            keymap.of(defaultKeymap),
            EditorView.updateListener.of(v => {
                if (v.docChanged) {
                    if (timeout) {
                        clearTimeout(timeout);
                    }
                    let text = v.state.doc.toString();
                    timeout = setTimeout(() => setScript(text), 500);
                }
            })
        ],
        parent: document.getElementById("editor-outer"),
    })
    document.getElementById("editor-outer").children[0].id = "editor" 

    let output = new EditorView({
        doc: "",
        extensions: [
            // Match basicSetup, but without any line numbers
            lineNumbers({"formatNumber": () => ""}),
            foldGutter(),
            EditorView.editable.of(false)
        ],
        parent: document.getElementById("output-outer"),
    })
    document.getElementById("output-outer").children[0].id = "output"
    console.log("booted");
}

function initBuffers(gl) {
    const positionBuffer = initPositionBuffer(gl);

    return {
        position: positionBuffer,
    };
}

function initPositionBuffer(gl) {
    // Create a buffer for the square's positions.
    const positionBuffer = gl.createBuffer();

    // Select the positionBuffer as the one to apply buffer
    // operations to from here out.
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

    // Now create an array of positions for the square.
    const positions = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

    // Now pass the list of positions into WebGL to build the
    // shape. We do this by creating a Float32Array from the
    // JavaScript array, then use it to fill the current buffer.
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    return positionBuffer;
}

function glInit() {
  const canvas = document.querySelector("#glcanvas");
  // Initialize the GL context
  const gl = canvas.getContext("webgl");

  // Only continue if WebGL is available and working
  if (gl === null) {
    alert(
      "Unable to initialize WebGL. Your browser or machine may not support it."
    );
    return;
  }

  // Set clear color to black, fully opaque
  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  // Clear the color buffer with specified clear color
  gl.clear(gl.COLOR_BUFFER_BIT);

  // Vertex shader program
  const vsSource = `
    attribute vec4 aVertexPosition;
    void main() {
      gl_Position = aVertexPosition;
    }
`;

  const fsSource = `
    void main() {
      gl_FragColor = vec4(gl_FragCoord.x / 100.0, gl_FragCoord.y / 100.0, 1.0, 1.0);
    }
  `;

  // Initialize a shader program; this is where all the lighting
  // for the vertices and so forth is established.
  const shaderProgram = initShaderProgram(gl, vsSource, fsSource);

  // Collect all the info needed to use the shader program.
  // Look up which attribute our shader program is using
  // for aVertexPosition and look up uniform locations.
  const programInfo = {
    program: shaderProgram,
    attribLocations: {
      vertexPosition: gl.getAttribLocation(shaderProgram, "aVertexPosition"),
    },
  };

  // Here's where we call the routine that builds all the
  // objects we'll be drawing.
  const buffers = initBuffers(gl);

  // Draw the scene
  drawScene(gl, programInfo, buffers);
}

function drawScene(gl, programInfo, buffers) {
  gl.clearColor(0.0, 0.0, 0.0, 1.0); // Clear to black, fully opaque
  gl.clearDepth(1.0); // Clear everything
  gl.enable(gl.DEPTH_TEST); // Enable depth testing
  gl.depthFunc(gl.LEQUAL); // Near things obscure far things

  // Clear the canvas before we start drawing on it.

  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // Tell WebGL how to pull out the positions from the position
  // buffer into the vertexPosition attribute.
  setPositionAttribute(gl, buffers, programInfo);

  // Tell WebGL to use our program when drawing
  gl.useProgram(programInfo.program);

  {
    const offset = 0;
    const vertexCount = 4;
    gl.drawArrays(gl.TRIANGLE_STRIP, offset, vertexCount);
  }
}

// Tell WebGL how to pull out the positions from the position
// buffer into the vertexPosition attribute.
function setPositionAttribute(gl, buffers, programInfo) {
  const numComponents = 2; // pull out 2 values per iteration
  const type = gl.FLOAT; // the data in the buffer is 32bit floats
  const normalize = false; // don't normalize
  const stride = 0; // how many bytes to get from one set of values to the next
  // 0 = use type and numComponents above
  const offset = 0; // how many bytes inside the buffer to start from
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
  gl.vertexAttribPointer(
    programInfo.attribLocations.vertexPosition,
    numComponents,
    type,
    normalize,
    stride,
    offset
  );
  gl.enableVertexAttribArray(programInfo.attribLocations.vertexPosition);
}

//
// Initialize a shader program, so WebGL knows how to draw our data
//
function initShaderProgram(gl, vsSource, fsSource) {
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
        shaderProgram
      )}`
    );
    return null;
  }

  return shaderProgram;
}

//
// creates a shader of the given type, uploads the source and
// compiles it.
//
function loadShader(gl, type, source) {
  const shader = gl.createShader(type);

  // Send the source to the shader object

  gl.shaderSource(shader, source);

  // Compile the shader program

  gl.compileShader(shader);

  // See if it compiled successfully

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    alert(
      `An error occurred compiling the shaders: ${gl.getShaderInfoLog(shader)}`
    );
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

setup();
glInit();
