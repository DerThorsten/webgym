// The Code section's editor (CodeMirror 6, python) + opening / saving files of this
// machine (File System Access API where there is one: Save writes back to the opened
// file; else a file input / a download).
import { basicSetup, EditorView } from "codemirror";
import { indentWithTab } from "@codemirror/commands";
import { Prec } from "@codemirror/state";
import { keymap } from "@codemirror/view";
import { python } from "@codemirror/lang-python";
import { oneDark } from "@codemirror/theme-one-dark";

/** an editor in `parent`; onChange(text) after edits, onRun() on Ctrl/Cmd+Enter;
 * returns {get, set} */
export function createEditor(parent, onChange, onRun) {
  const view = new EditorView({
    parent,
    extensions: [
      Prec.highest(keymap.of([{ key: "Mod-Enter", run: () => (onRun(), true) }])),
      basicSetup,
      keymap.of([indentWithTab]),
      python(),
      oneDark,
      EditorView.updateListener.of((u) => u.docChanged && onChange(view.state.doc.toString())),
    ],
  });
  return {
    get: () => view.state.doc.toString(),
    set: (text) => view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } }),
  };
}

const TYPES = [{ description: "Python", accept: { "text/x-python": [".py"] } }];
const handles = {}; // kind -> the file handle opened / saved to (Save writes there again)

/** a .py file of this machine: {name, text}, or null (cancelled) */
export async function openFile(kind) {
  if (window.showOpenFilePicker) {
    try {
      const [handle] = await showOpenFilePicker({ types: TYPES });
      handles[kind] = handle;
      const file = await handle.getFile();
      return { name: file.name, text: await file.text() };
    } catch (e) {
      if (e.name === "AbortError") return null;
      throw e;
    }
  }
  // (Firefox, Safari: a file input)
  const input = Object.assign(document.createElement("input"), { type: "file", accept: ".py" });
  const file = await new Promise((resolve) => {
    input.onchange = () => resolve(input.files[0] ?? null);
    input.oncancel = () => resolve(null);
    input.click();
  });
  return file && { name: file.name, text: await file.text() };
}

/** save `text`: to the file opened / saved before, else ask where (or download) */
export async function saveFile(kind, name, text) {
  if (window.showSaveFilePicker) {
    try {
      handles[kind] ??= await showSaveFilePicker({ suggestedName: name, types: TYPES });
      const writable = await handles[kind].createWritable();
      await writable.write(text);
      await writable.close();
      return handles[kind].name;
    } catch (e) {
      if (e.name === "AbortError") return null;
      throw e;
    }
  }
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([text], { type: "text/x-python" })),
    download: name,
  });
  a.click();
  URL.revokeObjectURL(a.href);
  return name;
}

/** forget the file of `kind` (another env / algorithm is shown) */
export function forgetFile(kind) {
  delete handles[kind];
}
