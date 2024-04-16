import {EditorView, ViewPlugin, keymap} from "@codemirror/view"
import {defaultKeymap} from "@codemirror/commands"

async function run() {
    const fidget = await import('./pkg')
      .catch(console.error);
    var timeout = null;

    function setScript(text) {
        try {
            let v = fidget.eval_script(text);
            console.log(v);
        } catch (error) {
            console.log(error);
        }
    }

    let myView = new EditorView({
        doc: "hello",
        extensions: [
            keymap.of(defaultKeymap),
            EditorView.updateListener.of(v => {
                if (v.docChanged) {
                    if (timeout) {
                        clearTimeout(timeout);
                        timeout = null;
                    }
                    let text = v.state.doc.toString();
                    timeout = setTimeout(() => setScript(text), 500);
                }
            })
        ],
        parent: document.body
    })
    console.log("booted");
}
run();
