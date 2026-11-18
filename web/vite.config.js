// vite: bundles main.js + the module worker; public/ (pyjs assets, python archives,
// config.json; see build_page.py) is copied as it is
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const WEBGYM = fileURLToPath(new URL("../webgym", import.meta.url));
const BUILD_PAGE = fileURLToPath(new URL("./build_page.py", import.meta.url));

/** dev server: a change of the python sources (webgym/) re-packs them
 * (build_page.py --sources) and reloads the page */
function pythonSources() {
  return {
    name: "webgym-python-sources",
    configureServer(server) {
      server.watcher.add(WEBGYM);
      let timer = null;
      server.watcher.on("change", (file) => {
        if (!file.startsWith(WEBGYM) || !file.endsWith(".py")) return;
        clearTimeout(timer); // (several saves at once: one rebuild)
        timer = setTimeout(() => {
          execFile("python", [BUILD_PAGE, "--sources"], (error, stdout, stderr) => {
            if (error) return server.config.logger.error(`webgym sources: ${stderr || error.message}`);
            server.config.logger.info(`webgym sources re-packed: ${stdout.trim()}`, { timestamp: true });
            server.ws.send({ type: "full-reload" });
          });
        }, 300);
      });
    },
  };
}

export default defineConfig({
  base: "./",
  worker: { format: "es" },
  plugins: [pythonSources()],
  // the app, its help page + the worker benchmark page (bench.html)
  build: { rollupOptions: { input: { main: "index.html", help: "help.html", bench: "bench.html" } } },
  // pyjs finds its assets via the assetsUrl option, not via the dep optimizer
  optimizeDeps: { exclude: ["@derthorsten/pyjs"] },
  server: { headers: { "Cache-Control": "no-store" } },
});
