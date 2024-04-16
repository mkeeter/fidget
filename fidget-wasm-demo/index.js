import {basicSetup} from "codemirror"
import {EditorView, ViewPlugin, keymap} from "@codemirror/view"
import {defaultKeymap} from "@codemirror/commands"

async function run() {
    const fidget = await import('./pkg')
      .catch(console.error);

    function setScript(text) {
        try {
            let v = fidget.eval_script(text);
            console.log(v);
        } catch (error) {
            console.log(error);
        }
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
        parent: document.getElementById("editor"),
    })
    console.log("booted");
}
run();
