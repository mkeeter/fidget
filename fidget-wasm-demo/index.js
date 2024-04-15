import {EditorView, keymap} from "@codemirror/view"
import {defaultKeymap} from "@codemirror/commands"

let myView = new EditorView({
  doc: "hello",
  extensions: [keymap.of(defaultKeymap)],
  parent: document.body
})

async function run() {
    let fidget = await import('./pkg')
      .catch(console.error);
    window.fidget = fidget;

    console.log("booted");
}
run();
