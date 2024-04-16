import {basicSetup} from "codemirror"
import {EditorView, ViewPlugin, keymap, lineNumbers} from "@codemirror/view"
import {foldGutter} from "@codemirror/language"
import {EditorState} from "@codemirror/state"
import {defaultKeymap} from "@codemirror/commands"

async function run() {
    const fidget = await import('./pkg')
      .catch(console.error);

    function setScript(text) {
        try {
            console.log("evaluating script", text);
            fidget.eval_script(text);
            var v = "Ok(..)";
        } catch (error) {
            var v = error.toString();
        }
        console.log(output.state.doc.length);
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
run();
