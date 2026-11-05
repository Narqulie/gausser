import { useState } from "react";
import { duration, STEPS, useLog, useSceneState } from "./api";
import type { HistoryRecord, State, Step } from "./api";
import { progress } from "./parse";
import type { Progress } from "./parse";
import { Viewer } from "./Viewer";
import { Images } from "./Images";
import { Stats } from "./Stats";
import { Logs } from "./Logs";

type Tab = "3d" | "images" | "stats" | "logs";
const TABS: [Tab, string][] = [["3d", "3D"], ["images", "Images"], ["stats", "Stats"], ["logs", "Logs"]];

export function App() {
  const { state, offline } = useSceneState();
  const [tab, setTab] = useState<Tab>("3d");
  const [logPick, setLogPick] = useState<HistoryRecord | null>(null);
  const [focus, setFocus] = useState<string | null>(null); // image name to highlight in 3D
  const current = state?.current ?? null;
  const live = useLog(current?.log ?? null, true);

  if (!state) {
    return <div className="empty">{offline ? "The monitor is not reachable." : "Loading…"}</div>;
  }
  const totals = { images: state.images.count, frames: state.images.panos || state.images.count };
  const now = progressOf(state, live.lines, live.partial, totals);

  return (
    <div className="app">
      <header className="header">
        <span className="brand">gausser</span>
        <span className="scene-name">{state.name}</span>
        <span className="scene-root" title={state.root}>{state.root}</span>
        <RunBadge state={state} offline={offline} />
      </header>
      <section className="pipeline">
        {STEPS.map((step) => (
          <StepCard
            key={step} step={step} state={state} progress={current?.step === step ? now : null}
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
        {tab === "stats" && <Stats state={state} />}
        {tab === "logs" && <Logs state={state} pick={logPick} onPick={setLogPick} live={live} />}
      </main>
    </div>
  );
}

function progressOf(state: State, lines: string[], partial: string, totals: { images: number; frames: number }): Progress | null {
  const current = state.current;
  if (!current) return null;
  if (!current.log) {
    // started by a gausser without logs: count what its staging folder holds
    const count = current.live.count;
    return { phase: "no log", done: count ?? null, total: count === undefined ? null : totals.images };
  }
  return progress(current.step, lines, partial, totals);
}

function lastRun(state: State, step: Step): HistoryRecord | undefined {
  return [...state.history].reverse().find((record) => record.step === step);
}

function RunBadge({ state, offline }: { state: State; offline: boolean }) {
  if (offline) return <span className="badge offline"><span className="dot" />monitor offline</span>;
  const holder = state.holder;
  if (!holder) return <span className="badge"><span className="dot" />idle</span>;
  return (
    <span className="badge running">
      <span className="dot" />
      <code>gausser {holder.argv.join(" ")}</code>
      <span>{duration(state.now - holder.start)}</span>
    </span>
  );
}

function StepCard({ step, state, progress, onClick }: { step: Step; state: State; progress: Progress | null; onClick: () => void }) {
  const run = lastRun(state, step);
  const [status, label] = stepStatus(step, state, run, progress !== null);
  const [metric, sub] = stepMetric(step, state);
  const fraction = progress && progress.total ? Math.min(1, (progress.done ?? 0) / progress.total) : null;
  const elapsed = run && (run.end ?? state.now) - run.start;
  return (
    <div className={`step ${progress ? "running" : ""}`} onClick={onClick} title="Show the log">
      <div className="step-head">
        <span className={`step-name step-color-${step}`}>{step}</span>
        <span className={`chip ${status}`}>{label}</span>
      </div>
      <div className="step-metric">{metric}</div>
      <div className="step-sub">
        {progress
          ? `${progress.phase}${progress.done !== null && progress.total ? ` · ${progress.done}/${progress.total}` : ""} · ${duration(state.now - (state.current?.start ?? state.now))}`
          : `${sub}${elapsed !== undefined ? ` · last ${duration(elapsed)}` : ""}`}
      </div>
      {progress && (
        <div className={`bar ${fraction === null ? "indeterminate" : ""}`}>
          <div style={{ width: `${(fraction ?? 0) * 100}%` }} />
        </div>
      )}
    </div>
  );
}

function stepStatus(step: Step, state: State, run: HistoryRecord | undefined, running: boolean): [string, string] {
  if (running) return ["running", "running"];
  if (run && run.ok === false && run.end) {
    // a failure counts only until the step succeeds again
    return ["failed", run.error === "interrupted" ? "interrupted" : "failed"];
  }
  const { images, masks, model, exports, trained, lods, lod_current } = state;
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
  }
}

function stepMetric(step: Step, state: State): [string, string] {
  const { images, masks, model, exports, lods } = state;
  switch (step) {
    case "frames":
      return images.panos
        ? [`${images.panos} panoramas`, `${images.count} views (${images.views} each)`]
        : [`${images.count} images`, images.size ? `${images.size[0]}×${images.size[1]}` : images.loose ? `${images.loose} loose photos` : "—"];
    case "mask":
      return [
        masks.count ? `${masks.count}/${images.count} masks` : "—",
        masks.exclude ? `${masks.exclude} exclude masks` : "SAM3",
      ];
    case "solve":
      return model
        ? [`${model.registered}/${images.count} registered`, model.no_masks ? "solved without masks" : "COLMAP"]
        : ["—", "COLMAP"];
    case "train": {
      const newest = exports[exports.length - 1];
      return newest
        ? [`${exports.length} export${exports.length > 1 ? "s" : ""}`, newest.run.split("/").slice(1).join("/")]
        : ["—", "Brush / Msplat"];
    }
    case "lod":
      return [lods.length ? `${lods.length} LOD set${lods.length > 1 ? "s" : ""}` : "—", "splat-transform"];
  }
}
