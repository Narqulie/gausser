import { useEffect, useState } from "react";
import { ago, clock, duration, get, sceneHref, STEPS } from "./api";
import type { HistoryRecord, Scenes, SceneSummary, State } from "./api";
import { stepMetric, stepStatus } from "./App";

/** Every scene gausser has worked on, and every step it ran. */
export function Hub() {
  const [data, setData] = useState<Scenes | null>(null);
  const [offline, setOffline] = useState(false);
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
    return () => {
      stop = true;
    };
  }, []);

  if (!data) return <div className="empty">{offline ? "The monitor is not running. Start it with `gausser monitor`." : "Loading…"}</div>;
  const names = new Map(data.scenes.map((s) => [s.root, s.name]));
  return (
    <div className="hub">
      <header className="header">
        <h1 className="scene-name">scenes</h1>
        <span className="scene-root">{data.scenes.length} folders gausser has worked on</span>
      </header>
      <section className="scene-list">
        {data.scenes.length === 0 && <p className="label">Scenes appear here once a gausser step runs in them or `gausser monitor` opens them.</p>}
        {data.scenes.map((scene) => <SceneRow key={scene.root} scene={scene} now={data.now} />)}
      </section>
      <section className="all-runs">
        <h2>Steps</h2>
        <table>
          <tbody>
            {data.runs.map((run) => (
              <RunRow key={`${run.scene}-${run.step}-${run.start}`} run={run} name={names.get(run.scene) ?? run.scene} now={data.now} />
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function SceneRow({ scene, now }: { scene: SceneSummary; now: number }) {
  const full = scene.history && scene.images ? (scene as State) : null;
  const running = full?.holder;
  return (
    <a className={`scene-row ${scene.missing ? "missing" : ""}`} href={scene.missing ? undefined : sceneHref(scene.root)}>
      <span className="scene-row-name">{scene.name}</span>
      <span className="mini-rail">
        {STEPS.map((step) => {
          const status = full ? stepStatus(step, full, full.history.filter((r) => r.step === step).pop(), full.current?.step === step)[0] : "idle";
          return <span key={step} className={`mini ${status}`} style={{ "--step": `var(--${step})` } as React.CSSProperties} title={full ? `${step}: ${stepMetric(step, full)[0]}` : step} />;
        })}
      </span>
      <span className="scene-row-info">
        {scene.missing ? "folder not found (drive unplugged?)" : scene.error ? scene.error : running ? <span className="live">$ gausser {running.argv.join(" ")}</span> : full ? summaryLine(full) : ""}
      </span>
      <span className="scene-row-path" title={scene.root}>{scene.root.replace(/^\/Users\/[^/]+/, "~")}</span>
      <span className="scene-row-when">{running ? duration(now - running.start) : ago(now - scene.last)}</span>
    </a>
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
