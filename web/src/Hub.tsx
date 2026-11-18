import { useEffect, useRef, useState } from "react";
import { ago, clock, duration, get, prettyPath, sceneHref, STEPS } from "./api";
import type { HistoryRecord, Scenes, SceneSummary, State } from "./api";
import { stepMetric, stepStatus } from "./App";

const RUNS = 30; // steps shown before "show all"

/** Every scene gausser has worked on, and every step it ran. */
export function Hub() {
  const [data, setData] = useState<Scenes | null>(null);
  const [offline, setOffline] = useState(false);
  const [filter, setFilter] = useState("");
  const [allRuns, setAllRuns] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const next = await get<Scenes>("scenes");
        if (!stop) {
          setData(next);
          setOffline(false);
        }
      } catch {
        if (!stop) setOffline(true);
      }
      if (!stop) setTimeout(tick, 4000);
    };
    tick();
    document.title = "scenes · gausser";
    const key = (event: KeyboardEvent) => {
      if (event.key === "/" && document.activeElement !== input.current) {
        event.preventDefault();
        input.current?.focus();
      }
    };
    addEventListener("keydown", key);
    return () => {
      stop = true;
      removeEventListener("keydown", key);
    };
  }, []);

  if (!data) return <div className="empty">{offline ? "The monitor is not running. Start it with `gausser monitor`." : "Loading…"}</div>;
  const query = filter.trim().toLowerCase();
  const scenes = data.scenes.filter((s) => !query || `${s.name} ${s.root}`.toLowerCase().includes(query));
  const running = scenes.filter((s) => !s.missing && s.holder);
  const missing = scenes.filter((s) => s.missing);
  const groups = new Map<string, SceneSummary[]>(); // in the order of each folder's newest scene
  for (const scene of scenes) {
    if (scene.missing || scene.holder) continue;
    const group = scene.group ?? scene.root.slice(0, scene.root.lastIndexOf("/"));
    groups.set(group, [...(groups.get(group) ?? []), scene]);
  }
  const names = new Map(data.scenes.map((s) => [s.root, s.name]));
  const shown = new Set(scenes.map((s) => s.root));
  const runs = data.runs.filter((run) => shown.has(run.scene));
  const total = data.scenes.length;
  const gone = data.scenes.filter((s) => s.missing).length;
  return (
    <div className="hub">
      <header className="header">
        <h1 className="scene-name">scenes</h1>
        <span className="scene-root">
          {data.scenes.filter((s) => s.holder).length} running · {total - gone} scenes · {gone} missing
        </span>
        <input
          ref={input} className="filter" type="search" placeholder="filter  /" value={filter}
          onChange={(event) => setFilter(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              setFilter("");
              event.currentTarget.blur();
            }
          }}
        />
      </header>
      {total === 0 && <p className="label hub-note">Scenes appear here once a gausser step runs in them or `gausser monitor` opens them.</p>}
      {total > 0 && scenes.length === 0 && <p className="label hub-note">No scene matches “{filter}”.</p>}
      {running.length > 0 && (
        <section className="scene-group">
          <h2 className="group-head live">running</h2>
          {running.map((scene) => <SceneRow key={scene.root} scene={scene} now={data.now} />)}
        </section>
      )}
      {[...groups].map(([group, list]) => (
        <section key={group} className="scene-group">
          <h2 className="group-head" title={group}>{prettyPath(group)} <span className="count">{list.length}</span></h2>
          {list.map((scene) => <SceneRow key={scene.root} scene={scene} now={data.now} />)}
        </section>
      ))}
      {missing.length > 0 && (
        <details className="scene-group missing-folders" open={query ? true : undefined}>
          <summary className="group-head">{missing.length} missing folders <span className="count">drive unplugged, moved or deleted</span></summary>
          {missing.map((scene) => <SceneRow key={scene.root} scene={scene} now={data.now} />)}
        </details>
      )}
      <section className="all-runs">
        <h2>Steps</h2>
        <table>
          <tbody>
            {(allRuns ? runs : runs.slice(0, RUNS)).map((run) => (
              <RunRow key={`${run.scene}-${run.step}-${run.start}`} run={run} name={names.get(run.scene) ?? run.scene} now={data.now} />
            ))}
          </tbody>
        </table>
        {!allRuns && runs.length > RUNS && (
          <button className="more" onClick={() => setAllRuns(true)}>show all {runs.length} steps</button>
        )}
      </section>
    </div>
  );
}

function SceneRow({ scene, now }: { scene: SceneSummary; now: number }) {
  const full = scene.history && scene.images ? (scene as State) : null;
  const running = full?.holder;
  return (
    <a className={`scene-row ${scene.missing ? "missing" : ""}`} href={scene.missing ? undefined : sceneHref(scene.root)} title={scene.root}>
      <span className="scene-row-name">{scene.name}</span>
      <span className="mini-rail">
        {STEPS.map((step) => {
          const status = full ? stepStatus(step, full, full.history.filter((r) => r.step === step).pop(), full.current?.step === step)[0] : "idle";
          return <span key={step} className={`mini ${status}`} style={{ "--step": `var(--${step})` } as React.CSSProperties} title={full ? `${step}: ${stepMetric(step, full)[0]}` : step} />;
        })}
      </span>
      <span className="scene-row-info">
        {scene.missing ? prettyPath(scene.root) : scene.error ? scene.error : running ? <RunningLine state={full} argv={running.argv} /> : full ? summaryLine(full) : ""}
      </span>
      <span className="scene-row-when">{running ? duration(now - running.start) : ago(now - scene.last)}</span>
    </a>
  );
}

/** The running step and the command's own arguments, without the scene folder the row already names. */
function RunningLine({ state, argv }: { state: State; argv: string[] }) {
  const args: string[] = [];
  for (let i = 1; i < argv.length; i++) {
    if (argv[i] === "-s" || argv[i] === "--scene") i++;
    else if (!argv[i].startsWith("--scene=") && argv[i] !== "--monitor") args.push(argv[i]);
  }
  const step = state.current?.step ?? argv[0];
  return (
    <span className="live" title={`gausser ${argv.join(" ")}`}>
      {argv[0] !== step && `${argv[0]} → `}<span className={`step-${step}`}>{step}</span> {args.join(" ")}
    </span>
  );
}

function summaryLine(state: State): string {
  const parts = [stepMetric("frames", state)[0]];
  if (state.model) parts.push(stepMetric("solve", state)[0]);
  if (state.exports.length) parts.push(stepMetric("train", state)[0]);
  return parts.join(", ");
}

function RunRow({ run, name, now }: { run: HistoryRecord & { scene: string }; name: string; now: number }) {
  const status = run.end === undefined ? "running" : run.ok ? "ok" : run.error === "interrupted" ? "interrupted" : "failed";
  return (
    <tr onClick={() => (location.href = sceneHref(run.scene))}>
      <td className="label">{clock(run.start)}</td>
      <td>{name}</td>
      <td className={`step-${run.step}`}>{run.step}</td>
      <td className={status === "ok" ? "ok" : status === "running" ? "live" : "bad"}>{status}</td>
      <td>{duration((run.end ?? now) - run.start)}</td>
      <td className="label">{run.run ? String(run.run) : run.error && status === "failed" ? run.error : ""}</td>
    </tr>
  );
}
