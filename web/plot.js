// A small line plot on a 2D canvas: several series over the training iterations,
// y axis from 0 to the max, legend at the top left.

/** draw `series` ([{label, color, values}]) into `canvas` */
export function plot(canvas, series) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const style = getComputedStyle(canvas);
  const pad = { left: 34, right: 6, top: 18, bottom: 16 };
  const n = Math.max(2, ...series.map((s) => s.values.length));
  const ymax = Math.max(1, ...series.flatMap((s) => s.values));
  const x = (i) => pad.left + (i / (n - 1)) * (w - pad.left - pad.right);
  const y = (v) => h - pad.bottom - (v / ymax) * (h - pad.top - pad.bottom);

  // axes + labels
  ctx.strokeStyle = style.getPropertyValue("--grid") || "#444";
  ctx.fillStyle = style.color;
  ctx.font = "10px system-ui";
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, h - pad.bottom);
  ctx.lineTo(w - pad.right, h - pad.bottom);
  ctx.stroke();
  ctx.textAlign = "right";
  ctx.fillText(String(Math.round(ymax)), pad.left - 4, pad.top + 4);
  ctx.fillText("0", pad.left - 4, h - pad.bottom);
  ctx.textAlign = "center";
  ctx.fillText(`iteration ${n > 2 ? n : ""}`, (w + pad.left) / 2, h - 3);

  // series + legend
  let legendX = pad.left + 4;
  for (const s of series) {
    ctx.strokeStyle = ctx.fillStyle = s.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    s.values.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.stroke();
    ctx.textAlign = "left";
    ctx.fillText(s.label, legendX, 11);
    legendX += ctx.measureText(s.label).width + 12;
  }
}
