// The network view: the watched policy's MLP (App.network()): layers as columns of
// neurons colored by their current activation (orange +, blue -), connections by their
// weight (orange +, blue -, opacity = size), input / output names at the sides.
// NEAT's networks (any shape, App.network() with "graph": true): neurons in columns by
// depth; neurons / connections not seen before glow green for a while.

const POS = [245, 166, 66]; // orange
const NEG = [79, 163, 255]; // blue
const rgba = (v, alpha) => {
  const [r, g, b] = v >= 0 ? POS : NEG;
  return `rgba(${r},${g},${b},${alpha})`;
};
const MAX_NEURONS = 32; // larger layers: the strongest activations are drawn

/** draw `net` ({sizes, weights, activations, inputs, outputs, argmax}) into `canvas` */
export function drawNetwork(canvas, net) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!net) return;
  if (net.graph) return drawGraph(canvas, ctx, w, h, net);
  const labelW = 105; // room for the input names + values (left)
  const outW = 105; // room for the output names + values (right)
  const layers = net.activations.map((a) => {
    // the neurons drawn (the strongest ones of a large layer, in their order)
    let idx = a.map((_, i) => i);
    if (idx.length > MAX_NEURONS) idx = idx.sort((i, j) => Math.abs(a[j]) - Math.abs(a[i])).slice(0, MAX_NEURONS).sort((i, j) => i - j);
    return idx;
  });
  const x = (l) => labelW + (l / (layers.length - 1)) * (w - labelW - outW);
  const y = (l, k) => ((k + 0.5) / layers[l].length) * (h - 8) + 4;
  const maxW = Math.max(1e-6, ...net.weights.flat(2).map(Math.abs));

  // connections (weights[l][i][j]: neuron i of layer l -> neuron j of layer l + 1)
  for (let l = 0; l < net.weights.length; l++) {
    layers[l].forEach((i, a) =>
      layers[l + 1].forEach((j, b) => {
        const v = net.weights[l][i][j];
        ctx.strokeStyle = rgba(v, Math.min(0.9, Math.abs(v) / maxW) * 0.8);
        ctx.lineWidth = 0.5 + (1.5 * Math.abs(v)) / maxW;
        ctx.beginPath();
        ctx.moveTo(x(l), y(l, a));
        ctx.lineTo(x(l + 1), y(l + 1, b));
        ctx.stroke();
      }),
    );
  }
  // neurons: color = activation (inputs scaled to their largest value)
  const last = layers.length - 1;
  const best = net.argmax ? net.activations[last].indexOf(Math.max(...net.activations[last])) : -1;
  layers.forEach((idx, l) => {
    const act = net.activations[l];
    const scale = l === 0 ? Math.max(1e-6, ...act.map(Math.abs)) : 1;
    const r = Math.max(2.5, Math.min(7, (h - 8) / idx.length / 2.6));
    idx.forEach((i, k) => {
      const v = act[i] / scale;
      ctx.fillStyle = rgba(v, 0.25 + 0.75 * Math.min(1, Math.abs(v)));
      ctx.strokeStyle = l === last && i === best ? "#fff" : "#666";
      ctx.lineWidth = l === last && i === best ? 2 : 1;
      ctx.beginPath();
      ctx.arc(x(l), y(l, k), r, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    });
  });
  // names
  ctx.font = "10px system-ui";
  ctx.fillStyle = getComputedStyle(canvas).color;
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  // what the network sees: the input names with their current values
  layers[0].forEach((i, k) =>
    ctx.fillText(`${net.inputs[i] ?? `in ${i}`} ${net.activations[0][i].toFixed(2)}`, x(0) - 9, y(0, k)),
  );
  ctx.textAlign = "left";
  layers[last].forEach((i, k) => {
    const value = net.activations[last][i].toFixed(2);
    ctx.fillText(`${net.outputs[i] ?? `out ${i}`} ${value}`, x(last) + 9, y(last, k));
  });
}

// NEAT's networks (a graph of any shape): when each neuron / connection was first seen
// (neurons by their NEAT id), to highlight new structure for a while
const NEW_MS = 3000;
const NEW = "120,230,120"; // (green: new)
const seen = new Map(); // key -> time first drawn
let seenFor = null; // (the inputs it was for: another env starts over)

/** a graph ({nodes: [{key, column, value}], edges: [[from, to, weight]], inputs,
 * outputs}): columns by depth, inputs left, outputs right; new parts glow */
function drawGraph(canvas, ctx, w, h, net) {
  const now = performance.now();
  const fresh = seenFor !== net.inputs.join() || seen.size === 0; // (all old at first)
  seenFor = net.inputs.join();
  const keys = [...net.nodes.map((n) => n.key), ...net.edges.map(([a, b]) => `${net.nodes[a].key}>${net.nodes[b].key}`)];
  for (const k of keys) if (!seen.has(k)) seen.set(k, fresh ? -NEW_MS : now);
  const glow = (k) => Math.max(0, 1 - (now - seen.get(k)) / NEW_MS); // 1 new .. 0 old

  // names: inputs left, outputs right, with their values (their room: the longest)
  ctx.font = "10px system-ui";
  const label = (n) => {
    const k = Number(n.key.slice(1));
    const name = n.key[0] === "i" ? (net.inputs[k] ?? `in ${k}`) : (net.outputs[k] ?? `out ${k}`);
    return `${name} ${n.value.toFixed(2)}`;
  };
  const room = (side) => 14 + Math.max(0, ...net.nodes.filter((n) => n.key[0] === side).map((n) => ctx.measureText(label(n)).width));
  const labelW = room("i");
  const outW = room("o");
  const columns = Math.max(1, ...net.nodes.map((n) => n.column));
  const count = new Map(); // nodes per column, and each node's place in its column
  const place = net.nodes.map((n) => {
    const k = count.get(n.column) ?? 0;
    count.set(n.column, k + 1);
    return k;
  });
  const px = net.nodes.map((n) => labelW + (n.column / columns) * (w - labelW - outW));
  const py = net.nodes.map((n, i) => ((place[i] + 0.5) / count.get(n.column)) * (h - 8) + 4);
  const maxW = Math.max(1e-6, ...net.edges.map(([, , v]) => Math.abs(v)));

  // connections (curves: they may skip columns)
  net.edges.forEach(([a, b, v]) => {
    const g = glow(`${net.nodes[a].key}>${net.nodes[b].key}`);
    const dx = (px[b] - px[a]) / 2;
    ctx.strokeStyle = g > 0 ? `rgba(${NEW},${0.3 + 0.6 * g})` :rgba(v, Math.min(0.9, Math.abs(v) / maxW) * 0.8);
    ctx.lineWidth = 0.5 + (1.5 * Math.abs(v)) / maxW + 2 * g;
    ctx.beginPath();
    ctx.moveTo(px[a], py[a]);
    ctx.bezierCurveTo(px[a] + dx, py[a], px[b] - dx, py[b], px[b], py[b]);
    ctx.stroke();
  });
  // neurons: color = value (inputs scaled to their largest value); new ones glow
  const inputs = net.nodes.filter((n) => n.column === 0).map((n) => Math.abs(n.value));
  const scale = Math.max(1e-6, ...inputs);
  const r = Math.max(2.5, Math.min(7, (h - 8) / Math.max(...count.values()) / 2.6));
  net.nodes.forEach((n, i) => {
    const v = n.column === 0 ? n.value / scale : n.value;
    const g = glow(n.key);
    if (g > 0) {
      ctx.fillStyle = `rgba(${NEW},${0.6 * g})`;
      ctx.beginPath();
      ctx.arc(px[i], py[i], r + 6 * g + 2, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.fillStyle = rgba(v, 0.25 + 0.75 * Math.min(1, Math.abs(v)));
    ctx.strokeStyle = "#666";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(px[i], py[i], r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  });
  ctx.fillStyle = getComputedStyle(canvas).color;
  ctx.textBaseline = "middle";
  net.nodes.forEach((n, i) => {
    if (n.key[0] === "h") return;
    ctx.textAlign = n.key[0] === "i" ? "right" : "left";
    ctx.fillText(label(n), px[i] + (n.key[0] === "i" ? -9 : 9), py[i]);
  });
}
