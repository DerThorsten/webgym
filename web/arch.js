// The pyjs architecture of this page: ?arch=wasm32 / wasm64, else wasm64 where the
// browser has 64-bit memory (memory64: Chrome, Firefox), wasm32 elsewhere (Safari).
// A module with just a 64-bit memory validates where memory64 is supported (header,
// memory section: 1 memory, flags 0x04 = 64-bit, minimum 0).
const MEMORY64 = WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 5, 3, 1, 4, 0]));

/** "wasm32" / "wasm64" for the page at `pageUrl` */
export function pageArch(pageUrl) {
  const asked = new URL(pageUrl).searchParams.get("arch");
  return asked === "wasm32" || asked === "wasm64" ? asked : MEMORY64 ? "wasm64" : "wasm32";
}
