// vite: bundles main.js + the module worker; public/ (pyjs assets, python archives,
// config.json; see build_page.py) is copied as it is
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  worker: { format: "es" },
  // the app + the worker benchmark page (bench.html)
  build: { rollupOptions: { input: { main: "index.html", bench: "bench.html" } } },
  // pyjs finds its assets via the assetsUrl option, not via the dep optimizer
  optimizeDeps: { exclude: ["@derthorsten/pyjs"] },
  server: { headers: { "Cache-Control": "no-store" } },
});
