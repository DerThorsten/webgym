// The tours: a few steps, each a frame around one element of the page (the rest dimmed)
// and a caption with Next / Skip; a step with `tab` opens that tab first, a step whose
// element isn't there (e.g. no simulation workers on a phone) is left out. The quick
// tour shows once (first visit, remembered in this browser); the "Tour" / "Detailed tour"
// links start them again.

const DONE = "webgym.tour";

/** the control (a widget of the app) whose label contains `text` */
const control = (text) => () =>
  [...document.querySelectorAll(".control")].find((c) => c.querySelector("label")?.textContent.includes(text));
/** the k-th button of the top bar (Train / Stop, New run) */
const barButton = (k) => () => document.querySelectorAll("#controls-bar button")[k];

const TOURS = {
  quick: [
    { target: "#controls-environment select", tab: "world",
      text: "<b>Pick a world</b> in the World tab: what it is about is written below it; some have settings (e.g. how many balls)." },
    { target: barButton(0),
      text: "<b>Press Train.</b> The agents learn right here in your browser, live; the numbers next to it show how they are doing." },
    { target: "#stage", text: "<b>The world.</b> Drag to pan, the wheel zooms, drag a body to push it. The progress plot is below." },
    { target: "#tabs", text: "<b>More in the panel:</b> how it trains, the policy, the log." },
    { target: "#tab-policy", tab: "policy",
      text: "<b>The policy:</b> once there is one (trained or restored), its neural network shows here, live: what the agent sees on the left, its actions on the right. Below it the checkpoints: save runs, restore them, or try a pretrained one." },
    { target: "#code-toggle",
      text: "<b>The python code</b> of the world and of the algorithm: edit it, press Apply, it runs. That was the quick tour: the <i>Detailed tour</i> (top bar) explains every control.",
      next: "detailed" },
  ],
  detailed: [
    // the top bar
    { target: barButton(0), text: "<b>Train / Stop.</b> Starts learning; Stop keeps the run, <i>Continue training</i> goes on with it. Training runs in a worker of its own: the page stays responsive." },
    { target: barButton(1), text: "<b>New run</b> throws the run away (network, history, autosave) and starts training from scratch." },
    { target: "#chips", text: "<b>The live numbers:</b> the world (with its settings), training and its iteration, the population's mean and best return, the episode you watch and its return so far." },
    { target: control("Play yourself"), text: "<b>Play yourself:</b> you steer with the keyboard instead of the policy (the keys show next to it); training goes on in the background." },
    { target: control("Auto reset"), text: "<b>Auto reset:</b> a new episode when the watched one ends. Off: it goes on after a fall, see whether the agent (or you) recover; R starts a new one.<br>Fun with it off: tease a trained agent with the mouse (knock the walker over, nudge the lander off course, push the car back) and watch how it copes." },
    // the World tab
    { target: "#controls-environment select", tab: "world", text: "<b>The world.</b> Each one is a python file: a box2d world with agents in it, what they see, what they do, what earns reward." },
    { target: "#controls-environment .info", tab: "world", text: "<b>What it is about:</b> the task, the observations, the actions, the reward and the keys, from the world's own python code." },
    { target: control("Show policies"), tab: "world", text: "<b>Show policies:</b> how many policies you watch at once: the current one (solid) and as ghosts the best candidates of the last iteration, all from the same start." },
    { target: control("Replay speed"), tab: "world", text: "<b>Replay speed:</b> slow motion or fast forward for watching (training always runs as fast as it can)." },
    { target: control("Camera follows"), tab: "world", text: "<b>Camera follows:</b> the world moves the camera with its agent (the walkers, the car, the cart). Off: pan and zoom yourself." },
    { target: control("Decoration"), tab: "world", text: "<b>Decoration:</b> looks only, e.g. the lander's exhaust and flags; never part of training." },
    // the Training tab
    { target: control("Algorithm"), tab: "training", text: "<b>The algorithm.</b> CEM and ES try many variations of the policy and keep the best (fast, robust); PPO and DQN learn from every step with pytorch (loaded in the background)." },
    { target: "#controls-training .group", tab: "training", text: "<b>Its parameters</b>, each with an ⓘ; the network's size (layers, hidden) starts a new run." },
    { target: control("Population"), tab: "training", text: "<b>Population:</b> how many agents learn at once, in one world (they don't touch each other)." },
    { target: "#helpers-control", tab: "training", text: "<b>Simulation workers:</b> more workers, each its own python on another CPU core, run a share of the population: CEM and ES train several times faster." },
    // Policy, Log
    { target: "#tab-policy", tab: "policy", text: "<b>Policy:</b> the network, live (neurons colored by activation, connections by weight); the checkpoints of this world + algorithm, kept in your browser, and pretrained runs." },
    { target: "#tab-log", tab: "log", text: "<b>Log:</b> the detailed state, the packages loaded (e.g. pytorch) and every message." },
    // the middle, the right
    { target: "#stage", tab: "world", text: "<b>The world</b>, simulated live: drag the background to pan, the wheel zooms, drag a body to push it around (while watching)." },
    { target: "#plot-strip", text: "<b>Training progress:</b> per iteration the mean and the best return (PPO / DQN: their losses next to it). Click its title to hide it." },
    { target: "#code-toggle", text: "<b>Code:</b> the world's and the algorithm's python source. Edit it and Apply: it runs at once, and it is kept in this browser (Revert brings the original back). Open / Save use files of your machine." },
    { target: "#help-link", text: "<b>Help:</b> all of this in one page, with pictures, and things to try (\"Having fun with agents\")." },
  ],
};

/** has the quick tour been shown in this browser? */
export function tourDone() {
  try {
    return localStorage.getItem(DONE) === "done";
  } catch {
    return true; // (no storage: don't show it on every visit)
  }
}

/** run a tour ("quick" / "detailed") from its first step */
export function startTour(kind = "quick") {
  document.getElementById("tour-frame")?.remove();
  document.getElementById("tour-card")?.remove();
  const frame = Object.assign(document.createElement("div"), { id: "tour-frame" });
  const card = Object.assign(document.createElement("div"), { id: "tour-card" });
  document.body.append(frame, card);
  const steps = TOURS[kind];
  let step = 0;

  const end = () => {
    frame.remove();
    card.remove();
    removeEventListener("resize", place);
    try {
      localStorage.setItem(DONE, "done");
    } catch {
      // (not remembered)
    }
  };
  const target = () => {
    const t = steps[step].target;
    const el = typeof t === "function" ? t() : document.querySelector(t);
    return el && el.getBoundingClientRect().width ? el : null; // (not there / hidden: none)
  };
  /** the frame around the step's element, the caption below it (or above, or beside a
   * tall element) */
  function place() {
    const r = target().getBoundingClientRect();
    const pad = 4;
    Object.assign(frame.style, {
      left: `${r.left - pad}px`, top: `${r.top - pad}px`, width: `${r.width + 2 * pad}px`, height: `${r.height + 2 * pad}px`,
    });
    const below = r.top + r.height + pad + 12;
    const above = r.top - pad - 12 - card.offsetHeight;
    let top, left;
    if (below + card.offsetHeight < innerHeight) [top, left] = [below, r.left];
    else if (above > 8) [top, left] = [above, r.left];
    else [top, left] = [r.top + 16, r.left + r.width + pad + 12]; // (a tall element: beside it)
    card.style.top = `${Math.min(Math.max(8, top), innerHeight - card.offsetHeight - 8)}px`;
    card.style.left = `${Math.min(Math.max(8, left), innerWidth - card.offsetWidth - 8)}px`;
  }
  function show() {
    const s = steps[step];
    if (s.tab) document.querySelector(`#tabs button[data-tab="${s.tab}"]`)?.click();
    if (!target()) return step < steps.length - 1 ? (step++, show()) : end(); // (left out)
    const last = step === steps.length - 1;
    card.innerHTML =
      `<div class="step">${kind === "detailed" ? "Detailed tour · " : ""}${step + 1} / ${steps.length}</div>` +
      `<div>${s.text}</div><div class="buttons">${last ? "" : '<button class="skip">Skip</button>'}` +
      (s.next ? '<button class="more">Detailed tour</button>' : "") +
      `<button class="primary next">${last ? "Done" : "Next"}</button></div>`;
    card.querySelector(".next").onclick = () => (last ? end() : (step++, show()));
    card.querySelector(".skip")?.addEventListener("click", end);
    card.querySelector(".more")?.addEventListener("click", () => startTour(s.next));
    place();
  }
  addEventListener("resize", place);
  show();
}
