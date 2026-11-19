// The Code section's editor (Monaco, python; loaded when the panel is first shown) +
// opening / saving files of this machine (File System Access API where there is one:
// Save writes back to the opened file; else a file input / a download).

/** Monaco with python only (not all its languages): the core, the python colors, its worker */
async function loadMonaco() {
  const [monaco, , { default: EditorWorker }] = await Promise.all([
    import("monaco-editor/esm/vs/editor/editor.api"),
    import("monaco-editor/esm/vs/basic-languages/python/python.contribution"),
    import("monaco-editor/esm/vs/editor/editor.worker?worker"),
  ]);
  self.MonacoEnvironment = { getWorker: () => new EditorWorker() };
  return monaco;
}

/** an editor in `parent`; onChange(text) after edits, onRun() on Ctrl/Cmd+Enter;
 * returns {get, set} (they work before Monaco is loaded: the text waits) */
export function createEditor(parent, onChange, onRun) {
  let text = "";
  let view = null;
  let quiet = false; // (set() itself is no edit)
  const observer = new ResizeObserver(async () => {
    if (view || !parent.clientWidth) return;
    observer.disconnect();
    const monaco = await loadMonaco();
    view = monaco.editor.create(parent, {
      value: text, language: "python", theme: "vs-dark", automaticLayout: true,
      fontSize: 12, minimap: { enabled: false }, scrollBeyondLastLine: false, tabSize: 4,
    });
    view.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, onRun);
    view.onDidChangeModelContent(() => quiet || onChange(view.getValue()));
  });
  observer.observe(parent);
  const editor = {
    get: () => (view ? view.getValue() : text),
    set: (next) => {
      text = next;
      if (!view) return;
      quiet = true;
      view.setValue(next);
      quiet = false;
    },
  };
  window.codeText = editor.get; // (the page test reads it: Monaco only draws the visible lines)
  return editor;
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
