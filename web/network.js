// The network view: the watched policy's MLP (App.network()): layers as columns of
// neurons colored by their current activation (orange +, blue -), connections by their
// weight (orange +, blue -, opacity = size), input / output names at the sides.

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
