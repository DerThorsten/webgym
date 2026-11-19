// The tours: a few steps, each a frame around one element of the page (the rest dimmed)
// and a caption with Next / Skip; a step with `tab` opens that tab first, a step whose
// element isn't there (e.g. no simulation workers on a phone) is left out. The first
// visit shows a welcome card (remembered in this browser) that offers them; the "Tour" /
// "Detailed tour" links start them again.

const DONE = "webgym.tour";

/** the control (a widget of the app) whose label contains `text` */
const control = (text) => () =>
  [...document.querySelectorAll(".control")].find((c) => c.querySelector("label")?.textContent.includes(text));
/** the k-th button above the watching panel (Train / Stop, New run) */
const barButton = (k) => () => document.querySelectorAll("#controls-bar button")[k];

const TOURS = {
  quick: [
    { target: "#controls-environment select", tab: "world",
      text: "<b>Pick a world</b> in the World tab: what it is about is written below it; some have settings (e.g. how many objects)." },
    { target: barButton(0),
      text: "<b>Press Train.</b> The agents learn right here in your browser, live; the numbers in the top bar show how they are doing." },
    { target: "#stage", text: "<b>The world.</b> Drag to pan, the wheel (or two fingers) zooms, drag a body to push it. The progress plot is below." },
    { target: "#tabs", text: "<b>More in the panel:</b> how it trains, toys, the policy, the log." },
    { target: "#tab-policy", tab: "policy",
      text: "<b>The policy:</b> once there is one (trained or restored), its neural network shows here, live: what the agent sees on the left, its actions on the right. Below it the checkpoints: save runs, restore them, or try a pretrained one." },
    { target: "#tab-toys", tab: "toys",
      text: "<b>Toys:</b> once it walks (or flies), tease it: a click into the world drops a box, a ball or a plank on it, sets off an explosion, or puts a ramp, a step or a seesaw in its way; or change the weather: rain, wind, moon gravity. Opening the tab turns auto reset off and shows one policy: see how it copes. R clears the world." },
    { target: "#code-toggle",
      text: "<b>The python code</b> of the world and of the algorithm: edit it, press Apply, it runs. That was the quick tour: the <i>Detailed tour</i> (top bar) explains every control.",
      next: "detailed" },
  ],
  detailed: [
    // below the tabs: training's buttons, auto reset
    { target: barButton(0), text: "<b>Train / Stop.</b> Starts learning; Stop keeps the run, <i>Continue training</i> goes on with it. Training runs in a worker of its own: the page stays responsive." },
    { target: barButton(1), text: "<b>New run</b> throws the run away (network, history, autosave) and starts training from scratch." },
    { target: "#chips", text: "<b>The live numbers:</b> the world (with its settings), training and its iteration, the population's mean and best return, the episode you watch and its return so far." },
    { target: control("Auto reset"), text: "<b>Auto reset:</b> a new episode when the watched one ends. Off: it goes on after a fall, see whether the agent recovers; R starts a new one.<br>Fun with it off: tease a trained agent with the mouse (knock the walker over, nudge the lander off course, push the car back) and watch how it copes." },
    // the World tab
    { target: "#controls-environment select", tab: "world", text: "<b>The world.</b> Each one is a python file: a box2d world with agents in it, what they see, what they do, what earns reward." },
    { target: "#controls-environment .info", tab: "world", text: "<b>What it is about:</b> the task, the observations, the actions and the reward, from the world's own python code." },
    { target: control("Show policies"), text: "<b>Show policies:</b> how many policies you watch at once: the current one (solid) and as ghosts the best candidates of the last iteration, all from the same start." },
    { target: control("Replay speed"), text: "<b>Replay speed:</b> slow motion or fast forward for watching (training always runs as fast as it can)." },
    { target: control("Camera follows"), text: "<b>Camera follows:</b> the world moves the camera with its agent (the walkers, the car, the cart). Off: pan and zoom yourself." },
    { target: control("Decoration"), text: "<b>Decoration:</b> looks only, e.g. the lander's exhaust and flags; never part of training." },
    // the Training tab
    { target: control("Algorithm"), tab: "training", text: "<b>The algorithm.</b> CEM and ES try many variations of the policy and keep the best (fast, robust); PPO and DQN learn from every step with pytorch (loaded in the background)." },
    { target: "#controls-training .group", tab: "training", text: "<b>Its parameters</b>, each with an ⓘ; the network's size (layers, hidden) starts a new run." },
    { target: control("Population"), tab: "training", text: "<b>Population:</b> how many agents learn at once, in one world (they don't touch each other)." },
    { target: "#helpers-control", tab: "training", text: "<b>Simulation workers:</b> more workers, each its own python on another CPU core, run a share of the population: CEM and ES train several times faster." },
    // Policy, Log
    { target: "#controls-toys .tools", tab: "toys", text: "<b>Toys:</b> pick a tool in the folds, then click into empty space: drop a box, a ball or a plank, set off an explosion, put a ramp, a step or a seesaw on the ground. They are real bodies: the agent has to deal with them. A click on a body always grabs it; click the tool again to put it away. Opening this tab turns auto reset off and shows one policy; R (reset) clears them." },
    { target: "#controls-toys details", tab: "toys", text: "<b>The folds</b> (click a title): each with its tools and their settings: size, weight and bounce of what you drop, the explosion, the ramps and steps, and the weather: rain, wind and gravity (try the moon: 0.17)." },
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

/** the first visit: what this is, in a few lines, then: a tour, the help page or skip */
export function welcome() {
  const back = Object.assign(document.createElement("div"), { id: "welcome-back" });
  const card = Object.assign(document.createElement("div"), { id: "welcome" });
  // (a phone: the same sizes as the phone layout of index.html)
  const phone = matchMedia("(max-width: 700px), (max-height: 500px)").matches;
  card.innerHTML = `
    <h2>Welcome to webgym</h2>
    ${phone ? `<p class="phone-note">📱 webgym is made for the desktop (a big screen, a mouse,
      several CPU cores). On a phone it may still work, just smaller and slower.</p>` : ""}
    <p><b>Reinforcement learning, live in your browser.</b> Agents (a cart balancing a pole,
      a lunar lander, walkers, a ragdoll, ...) learn by trial and error in a physics world.
      You watch them learn as it happens.</p>
    <p>Everything runs right here, on your machine: python, numpy, the box2d physics and
      pytorch, compiled to WebAssembly by <a href="https://emscripten-forge.org" target="_blank" rel="noopener">emscripten-forge</a>.
      No server, nothing to install.</p>
    <ul>
      <li>Press <b>Train</b> and see a whole population learn; the plot below the world
        shows the progress.</li>
      <li>Look inside: the agent's <b>neural network</b>, live (Policy tab): what it sees,
        how its neurons fire and its weights, as it acts.</li>
      <li>Or restore a <b>pretrained</b> run (Policy tab) and tease it in the <b>Toys</b>
        tab: ramps, explosions, rain.</li>
      <li>Every world and algorithm is plain python: open <b>Code</b>, change it, run it.</li>
    </ul>
    <div class="buttons">
      <button class="skip">Skip</button>
      <button class="to-help">Help page</button>
      <button class="detailed">Detailed tour</button>
      <button class="primary quick">Quick tour</button>
    </div>`;
  document.body.append(back, card);
  const close = () => {
    back.remove();
    card.remove();
    try {
      localStorage.setItem(DONE, "done");
    } catch {
      // (not remembered)
    }
  };
  card.querySelector(".skip").onclick = close;
  back.onclick = close;
  card.querySelector(".to-help").onclick = () => (close(), open("help.html", "_blank"));
  card.querySelector(".quick").onclick = () => (close(), startTour("quick"));
  card.querySelector(".detailed").onclick = () => (close(), startTour("detailed"));
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
    if (!drawerOpen) app.classList.remove("drawer-open"); // (as it was)
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
  // (a phone in landscape: the panel is a drawer, opened for the steps in it)
  const app = document.getElementById("app");
  const drawerOpen = app.classList.contains("drawer-open");
  function show() {
    const s = steps[step];
    if (s.tab) document.querySelector(`#tabs button[data-tab="${s.tab}"]`)?.click();
    app.classList.add("drawer-open");
    if (!drawerOpen && !target()?.closest("#side")) app.classList.remove("drawer-open");
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
