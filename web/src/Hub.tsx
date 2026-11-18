import { useEffect, useRef, useState } from "react";
import { ago, clock, duration, get, prettyPath, sceneHref, STEPS } from "./api";
import type { HistoryRecord, Scenes, SceneSummary, State } from "./api";
import { stepMetric, stepStatus } from "./App";

const RUNS = 30; // steps shown before "show all"

/** What a scene is doing: a running command, a last step that failed or was killed, a trained splat, or neither. */
type Kind = "running" | "failed" | "idle" | "done" | "missing";
const KINDS: [Kind, string][] = [["running", "running"], ["failed", "failed"], ["idle", "unfinished"], ["done", "trained"], ["missing", "missing"]];
const RANK: Record<Kind, number> = { running: 0, failed: 1, idle: 2, done: 3, missing: 4 };
type Sort = "status" | "name" | "last";

interface Status {
  kind: Kind;
  word: string; // running, failed, interrupted, killed, trained, unfinished, missing
  why: string; // the bullet's tooltip
}

function sceneStatus(scene: SceneSummary): Status {
  if (scene.missing) return { kind: "missing", word: "missing", why: "the folder is gone: drive unplugged, moved or deleted" };
  if (scene.error) return { kind: "failed", word: "unreadable", why: scene.error };
  if (scene.holder) return { kind: "running", word: "running", why: `gausser ${scene.holder.argv.join(" ")}` };
  const last = [...(scene.history ?? [])].sort((a, b) => a.start - b.start).pop();
  if (last && last.end === undefined) {
    return { kind: "failed", word: "killed", why: `${last.step} has no end record: its process was killed or the Mac shut down` };
  }
  if (last && last.ok === false) {
    const word = last.error === "interrupted" ? "interrupted" : "failed";
    return { kind: "failed", word, why: `${last.step} ${word}${last.error && word === "failed" ? `: ${last.error}` : ""}` };
  }
  if (scene.trained) return { kind: "done", word: "trained", why: "the newest splat is trained from the current model" };
  return { kind: "idle", word: "unfinished", why: last ? `last step: ${last.step}, no current splat yet` : "no step has run yet" };
}

function compare(sort: Sort, flip: boolean) {
  return (a: SceneSummary, b: SceneSummary) => {
    const order = sort === "name" ? a.name.localeCompare(b.name)
      : sort === "last" ? b.last - a.last
      : RANK[sceneStatus(a).kind] - RANK[sceneStatus(b).kind] || b.last - a.last;
    return flip ? -order : order;
  };
}

/** Every scene gausser has worked on, and every step it ran. */
export function Hub() {
  const [data, setData] = useState<Scenes | null>(null);
  const [offline, setOffline] = useState(false);
  const [filter, setFilter] = useState("");
  const [allRuns, setAllRuns] = useState(false);
  const [only, setOnly] = useState<Kind | null>(null);
  const [sort, setSort] = useState<Sort>("last");
  const [flip, setFlip] = useState(false);
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
  const matching = data.scenes.filter((s) => !query || `${s.name} ${s.root}`.toLowerCase().includes(query));
  const counts = new Map<Kind, number>();
  for (const scene of matching) counts.set(sceneStatus(scene).kind, (counts.get(sceneStatus(scene).kind) ?? 0) + 1);
  const scenes = matching.filter((s) => !only || sceneStatus(s).kind === only).sort(compare(sort, flip));
  const running = scenes.filter((s) => !s.missing && s.holder);
  const missing = scenes.filter((s) => s.missing);
  const groups = new Map<string, SceneSummary[]>(); // in the order of each folder's first scene
  for (const scene of scenes) {
    if (scene.missing || scene.holder) continue;
    const group = scene.group ?? scene.root.slice(0, scene.root.lastIndexOf("/"));
    groups.set(group, [...(groups.get(group) ?? []), scene]);
  }
  const names = new Map(data.scenes.map((s) => [s.root, s.name]));
  const held = new Map(data.scenes.flatMap((s) => (s.holder ? [[s.root, s.holder.start] as const] : [])));
  const shown = new Set(scenes.map((s) => s.root));
  const runs = data.runs.filter((run) => shown.has(run.scene));
  const total = data.scenes.length;
  const sortBy = (key: Sort) => {
    if (key === sort) setFlip(!flip);
    else {
      setSort(key);
      setFlip(false);
    }
  };
  const head = (key: Sort, label: string) => (
    <button className={`col ${sort === key ? "active" : ""}`} onClick={() => sortBy(key)}>
      {label}{sort === key ? (flip ? " ▴" : " ▾") : ""}
    </button>
  );
  return (
    <div className="hub">
      <div className="hub-top">
      <header className="header">
        <h1 className="scene-name">scenes</h1>
        <span className="chips">
          {KINDS.map(([kind, label]) => (
            <button
              key={kind} className={`chip ${only === kind ? "active" : ""}`} disabled={!counts.get(kind)}
              onClick={() => setOnly(only === kind ? null : kind)} title={only === kind ? "show all" : `only ${label}`}
            >
              <span className={`dot ${kind}`} />{label} <span className="count">{counts.get(kind) ?? 0}</span>
            </button>
          ))}
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
      <div className="columns">
        {head("status", "status")}
        {head("name", "name")}
        <span className="col">steps</span>
        <span className="col">progress</span>
        {head("last", "last activity")}
      </div>
      </div>
      {total === 0 && <p className="label hub-note">Scenes appear here once a gausser step runs in them or `gausser monitor` opens them.</p>}
      {total > 0 && scenes.length === 0 && <p className="label hub-note">No scene matches{query ? ` “${filter}”` : ""}{only ? ` among the ${only === "idle" ? "unfinished" : only} ones` : ""}.</p>}
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
        <details className="scene-group missing-folders" open={query || only ? true : undefined}>
          <summary className="group-head">{missing.length} missing folders <span className="count">drive unplugged, moved or deleted</span></summary>
          {missing.map((scene) => <SceneRow key={scene.root} scene={scene} now={data.now} />)}
        </details>
      )}
      <section className="all-runs">
        <h2>Steps</h2>
        <table>
          <tbody>
            {(allRuns ? runs : runs.slice(0, RUNS)).map((run) => (
              <RunRow key={`${run.scene}-${run.step}-${run.start}`} run={run} name={names.get(run.scene) ?? run.scene} held={held.get(run.scene)} now={data.now} />
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
  const status = sceneStatus(scene);
  return (
    <a className={`scene-row ${scene.missing ? "missing" : ""}`} href={scene.missing ? undefined : sceneHref(scene.root)} title={scene.root}>
      <span className={`scene-status ${status.kind}`} title={status.why}><span className={`dot ${status.kind}`} />{status.word}</span>
      <span className="scene-row-name">{scene.name}</span>
      <span className="mini-rail">
        {STEPS.map((step) => {
          const status = full ? stepStatus(step, full, full.history.filter((r) => r.step === step).pop(), full.current?.step === step)[0] : "idle";
          return <span key={step} className={`mini ${status}`} style={{ "--step": `var(--${step})` } as React.CSSProperties} title={full ? `${step}: ${stepMetric(step, full)[0]}` : step} />;
        })}
      </span>
      <span className="scene-row-info">
        {scene.missing ? prettyPath(scene.root)
          : running ? <RunningLine state={full} argv={running.argv} />
          : status.kind === "failed" ? <span className="bad">{status.why}</span>
          : full ? summaryLine(full) : ""}
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

function RunRow({ run, name, held, now }: { run: HistoryRecord & { scene: string }; name: string; held?: number; now: number }) {
  const open = run.end === undefined;
  const status = open ? (held !== undefined && run.start >= held ? "running" : "killed")
    : run.ok ? "ok" : run.error === "interrupted" ? "interrupted" : "failed";
  return (
    <tr onClick={() => (location.href = sceneHref(run.scene))}>
      <td className="label">{clock(run.start)}</td>
      <td>{name}</td>
      <td className={`step-${run.step}`}>{run.step}</td>
      <td className={status === "ok" ? "ok" : status === "running" ? "live" : "bad"}>{status}</td>
      <td>{status === "killed" ? "" : duration((run.end ?? now) - run.start)}</td>
      <td className="label">{run.run ? String(run.run) : run.error && status === "failed" ? run.error : ""}</td>
    </tr>
  );
}
