import { useCallback, useEffect, useRef, useState } from "react";
import { duration, STEPS, useLog, useSceneState } from "./api";
import { NotifyToggle, Toasts, useStepToasts } from "./Toasts";
import type { HistoryRecord, State, Step } from "./api";
import { progress } from "./parse";
import type { Progress } from "./parse";
import { Viewer } from "./Viewer";
import { Images } from "./Images";
import { Stats } from "./Stats";
import { Logs } from "./Logs";
import { Renders } from "./Renders";

type Tab = "3d" | "images" | "renders" | "stats" | "logs";
const TABS: [Tab, string][] = [["3d", "3D"], ["images", "Images"], ["renders", "Renders"], ["stats", "Stats"], ["logs", "Logs"]];

export function App() {
  const { state, offline } = useSceneState();
  const [tab, setTab] = useState<Tab>("3d");
  const [logPick, setLogPick] = useState<HistoryRecord | null>(null);
  const [focus, setFocus] = useState<string | null>(null); // image name to highlight in 3D
  const current = state?.current ?? null;
  const live = useLog(current?.log ?? null, true);
  const describe = useCallback((record: HistoryRecord) => (state ? stepMetric(record.step, state)[0] : ""), [state]);
  const { toasts, dismiss } = useStepToasts(state, describe);
  const totals = state
    ? { images: state.images.count, frames: state.images.panos || state.images.count, iterations: Number(current?.iterations) || undefined }
    : null;
  const now = state && totals ? progressOf(state, live.lines, live.partial, totals) : null;
  const left = useRemaining(state, now);

  useEffect(() => {
    if (!state) return;
    const step = state.current?.step;
    const done = now && now.done !== null && now.total ? ` ${Math.round((100 * now.done) / now.total)}%` : "";
    document.title = step ? `${step}${done} · ${state.name}` : `${state.name} · gausser`;
  }, [state, now]);

  if (!state) {
    return <div className="empty">{offline ? "Cannot load this scene. Run `gausser monitor` in its folder." : "Loading…"}</div>;
  }
  return (
    <div className="app">
      <header className="header">
        <a className="back" href="./" title="All scenes">scenes /</a>
        <h1 className="scene-name">{state.name}</h1>
        <span className="scene-root" title={state.root}>{state.root.replace(/^\/Users\/[^/]+/, "~")}</span>
        <RunLine state={state} offline={offline} />
        <NotifyToggle />
      </header>
      <section className="rail">
        {STEPS.map((step) => (
          <StepCard
            key={step} step={step} state={state} progress={current?.step === step ? now : null} left={left}
            onClick={() => {
              setLogPick(lastRun(state, step) ?? null);
              setTab("logs");
            }}
          />
        ))}
      </section>
      <nav className="tabs">
        {TABS.map(([id, label]) => (
          <button key={id} className={`tab ${tab === id ? "active" : ""}`} onClick={() => setTab(id)}>{label}</button>
        ))}
      </nav>
      <main className="content">
        {tab === "3d" && <Viewer state={state} focus={focus} />}
        {tab === "images" && (
          <Images state={state} onShow={(name) => { setFocus(name); setTab("3d"); }} />
        )}
        {tab === "renders" && <Renders state={state} />}
        {tab === "stats" && <Stats state={state} />}
        {tab === "logs" && <Logs state={state} pick={logPick} onPick={setLogPick} live={live} />}
      </main>
      <Toasts toasts={toasts} dismiss={dismiss} />
    </div>
  );
}

function progressOf(state: State, lines: string[], partial: string, totals: { images: number; frames: number; iterations?: number }): Progress | null {
  const current = state.current;
  if (!current) return null;
  if (!current.log) {
    // started by a gausser without logs: count what its staging folder holds
    const count = current.live.count;
    return { phase: "no log", done: count ?? null, total: count === undefined ? null : totals.images };
  }
  return progress(current.step, lines, partial, totals);
}

export function lastRun(state: State, step: Step): HistoryRecord | undefined {
  return [...state.history].reverse().find((record) => record.step === step);
}

function RunLine({ state, offline }: { state: State; offline: boolean }) {
  if (offline) return <span className="run-line offline">monitor stopped, showing the last state</span>;
  const holder = state.holder;
  if (!holder) return <span className="run-line">idle</span>;
  return (
    <span className="run-line running">
      <code>$ gausser {holder.argv.join(" ")}</code>
      <span className="cursor" aria-hidden>▌</span>
      <span className="elapsed">{duration(state.now - holder.start)}</span>
    </span>
  );
}

function StepCard({ step, state, progress, left, onClick }: {
  step: Step; state: State; progress: Progress | null; left: number | null; onClick: () => void;
}) {
  const run = lastRun(state, step);
  const [status, label] = stepStatus(step, state, run, progress !== null);
  const [metric, sub] = stepMetric(step, state);
  const fraction = progress ? (progress.total ? Math.min(1, (progress.done ?? 0) / progress.total) : null) : status === "idle" ? 0 : 1;
  const elapsed = run && (run.end ?? state.now) - run.start;
  return (
    <button className={`station ${status}`} onClick={onClick} title={`${step}: show the log`} style={{ "--step": `var(--${step})` } as React.CSSProperties}>
      <span className="station-name">{step}</span>
      <span className={`segment ${fraction === null ? "indeterminate" : ""}`}>
        <span style={{ width: `${(fraction ?? 0.3) * 100}%` }} />
      </span>
      <span className="station-metric">{progress ? progressText(progress) : metric}</span>
      <span className="station-sub">
        {progress
          ? `${progress.phase} · ${duration(state.now - (state.current?.start ?? state.now))}${left !== null ? ` · ~${duration(left)} left` : ""}`
          : [status === "done" || status === "idle" ? sub : label, elapsed !== undefined && duration(elapsed)].filter(Boolean).join(" · ")}
      </span>
    </button>
  );
}

const WINDOW = 300; // seconds of progress the rate is measured over

/**
 * Seconds left in the running phase: the rate over the last five minutes of progress seen by this page.
 * Until the page has seen 20 s of progress, a step whose count starts with the step (mask, train) uses its
 * whole run so far. A solve's phases each count their own thing, so the estimate is for the phase only.
 */
function useRemaining(state: State | null, progress: Progress | null): number | null {
  const samples = useRef<{ key: string; points: [number, number][] }>({ key: "", points: [] });
  const current = state?.current;
  if (!state || !current || !progress || progress.done === null || !progress.total) return null;
  const key = `${current.step}-${current.start}-${progress.phase}-${progress.total}`;
  if (samples.current.key !== key) samples.current = { key, points: [] };
  const points = samples.current.points;
  const last = points[points.length - 1];
  if (!last || last[0] !== state.now) points.push([state.now, progress.done]);
  while (points.length > 2 && state.now - points[0][0] > WINDOW) points.shift();

  const remaining = progress.total - progress.done;
  const [t0, d0] = points[0];
  let rate = 0;
  if (state.now - t0 >= 20 && progress.done > d0) rate = (progress.done - d0) / (state.now - t0);
  else if ((current.step === "mask" || current.step === "train") && progress.done > 0) rate = progress.done / (state.now - current.start);
  return rate > 0 ? remaining / rate : null;
}

function progressText(progress: Progress): string {
  if (progress.done === null || !progress.total) return "running";
  return `${progress.done.toLocaleString()} / ${progress.total.toLocaleString()}`;
}

export function stepStatus(step: Step, state: State, run: HistoryRecord | undefined, running: boolean): [string, string] {
  if (running) return ["running", "running"];
  if (run && run.ok === false && run.end) {
    // a failure counts only until the step succeeds again
    return ["failed", run.error === "interrupted" ? "interrupted" : "failed"];
  }
  const { images, masks, model, exports, trained, lods, lod_current, dense, meshes, mesh_current } = state;
  switch (step) {
    case "frames":
      return images.count ? ["done", "done"] : ["idle", "no images"];
    case "mask":
      if (masks.count) return masks.done ? ["done", "done"] : ["stale", "incomplete"];
      return masks.exclude ? ["done", "exclude only"] : ["idle", "none"];
    case "solve":
      if (!model) return ["idle", "not solved"];
      return model.current ? ["done", "done"] : ["stale", "out of date"];
    case "train":
      if (!exports.length) return ["idle", "not trained"];
      return trained ? ["done", "done"] : ["stale", "out of date"];
    case "lod":
      if (!lods.length) return ["idle", "none"];
      return lod_current ? ["done", "done"] : ["stale", "out of date"];
    case "dense":
      if (!dense) return ["idle", "none"];
      return dense.current ? ["done", "done"] : ["stale", "out of date"];
    case "mesh":
      if (!meshes.length) return ["idle", "none"];
      return mesh_current ? ["done", "done"] : ["stale", "out of date"];
  }
}

export function stepMetric(step: Step, state: State): [string, string] {
  const { images, masks, model, exports, lods, dense, meshes } = state;
  const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
  switch (step) {
    case "frames":
      if (images.panos) return [plural(images.panos, "panorama"), `${images.count.toLocaleString()} views`];
      if (!images.count) return [images.loose ? plural(images.loose, "loose photo") : "no images", "ffmpeg"];
      return [plural(images.count, "image"), images.size ? `${images.size[0]}×${images.size[1]}` : "ffmpeg"];
    case "mask":
      if (masks.count) return [`${masks.count.toLocaleString()} / ${images.count.toLocaleString()}`, "sam3"];
      return [masks.exclude ? plural(masks.exclude, "exclusion mask") : "not masked", "sam3"];
    case "solve":
      if (!model) return ["not solved", "colmap"];
      return [`${model.registered.toLocaleString()} / ${images.count.toLocaleString()} placed`, model.no_masks ? "colmap, no masks" : "colmap"];
    case "train": {
      const newest = exports[exports.length - 1];
      return newest ? [plural(exports.length, "export"), newest.run.split("/").slice(1).join("/")] : ["not trained", "brush"];
    }
    case "lod":
      return [lods.length ? plural(lods.length, "set") : "none", "splat-transform"];
    case "dense":
      return [dense ? plural(dense.points, "point") : "none", "openmvs"];
    case "mesh":
      return [meshes.length ? plural(meshes.length, "mesh") : "none", "openmvs"];
  }
}
