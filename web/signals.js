// The Signals tab: what the watched agent sees (observations) and does (actions), and its
// reward, over the last seconds: one low strip per value, stacked, time to the right.

const KEEP = 5; // s kept (the most the window shows)
let seconds = 2.5; // the window shown (short: the signals move fast, details stay visible)

/** how many seconds the strips show (at most KEEP) */
export function setSignalSeconds(s) {
  seconds = Math.min(KEEP, Math.max(0.1, Number(s) || 2.5));
}
const ROW = 24; // px per strip
const HEAD = 18; // px per group heading
const LABEL = 120; // px for the names on the left
const VALUE = 46; // px for the current values on the right

let buffer = { episode: null, names: "", dt: 1 / 60, obs: [], actions: [], rows: [] };

/** add the rows of a display state's `trace` (the steps since the last state) */
export function addSignals(trace) {
  if (!trace) return;
  const names = [...trace.obs, "|", ...trace.actions].join(",");
  if (trace.episode !== buffer.episode || names !== buffer.names) {
    // (a new episode or another world: start over)
    buffer = { episode: trace.episode, names, dt: trace.dt, obs: trace.obs, actions: trace.actions, rows: [] };
  }
  buffer.rows.push(...trace.rows);
  const keep = Math.round(KEEP / buffer.dt);
  if (buffer.rows.length > keep) buffer.rows.splice(0, buffer.rows.length - keep);
}

/** draw the buffer into `canvas` (its height follows the number of strips) */
export function drawSignals(canvas) {
  const { obs, actions, dt } = buffer;
  const keep = Math.max(2, Math.round(seconds / dt));
  const rows = buffer.rows.slice(-keep); // (the window: the last `seconds`)
  const groups = [
    ["Observations", obs.map((name, k) => ({ name, k }))],
    ["Actions", actions.map((name, k) => ({ name, k: obs.length + k, range: [-1, 1] }))],
    ["Reward", [{ name: "reward", k: obs.length + actions.length }]],
  ];
  const strips = obs.length + actions.length + 1;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = groups.length * HEAD + strips * ROW + 4;
  canvas.style.height = `${h}px`;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px system-ui";
  // the strips: between the names and the values, as wide as the panel; the line always
  // spans the strip (an episode's first moments: fewer rows than the window, stretched)
  const x0 = LABEL, x1 = w - VALUE;
  const x = (i) => x0 + (i / Math.max(1, rows.length - 1)) * (x1 - x0);
  let top = 0;
  for (const [title, items] of groups) {
    ctx.fillStyle = "#ccc";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(title, 2, top + HEAD / 2 + 2);
    top += HEAD;
    for (const { name, k, range } of items) {
      const values = rows.map((r) => r[k]);
      // the y range: the window's own (actions: always -1..1)
      let [lo, hi] = range || [Math.min(0, ...values), Math.max(0, ...values)];
      if (hi - lo < 1e-6) [lo, hi] = [lo - 1, hi + 1];
      const y = (v) => top + ROW - 3 - ((v - lo) / (hi - lo)) * (ROW - 6);
      ctx.fillStyle = "#1e1e1e";
      ctx.fillRect(x0, top + 1, x1 - x0, ROW - 2);
      ctx.strokeStyle = "#3a3a3a"; // the zero line
      ctx.beginPath();
      ctx.moveTo(x0, y(0));
      ctx.lineTo(x1, y(0));
      ctx.stroke();
      ctx.strokeStyle = title === "Actions" ? "#ff8c42" : title === "Reward" ? "#7bd88f" : "#4fa3ff";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      values.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
      ctx.stroke();
      ctx.lineWidth = 1;
      ctx.fillStyle = "#aaa";
      ctx.textAlign = "left";
      ctx.fillText(name, 10, top + ROW / 2, LABEL - 14);
      ctx.textAlign = "right";
      if (values.length) ctx.fillText(values[values.length - 1].toFixed(2), w - 2, top + ROW / 2);
      top += ROW;
    }
  }
}
