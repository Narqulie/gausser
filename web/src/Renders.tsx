import { useMemo, useState } from "react";
import { thumbUrl, useFetched } from "./api";
import type { Evaluation, State } from "./api";

type Pair = Evaluation["images"][number];

/** Brush's renders of the held-out photos (`train --eval-split N`), next to the photos. */
export function Renders({ state }: { state: State }) {
  const live = state.current?.step === "train" ? state.current.live.staging : undefined;
  const runs = useMemo(() => {
    const list: { path: string; label: string }[] = [];
    if (live) list.push({ path: live, label: "training now" });
    for (const run of [...new Set(state.exports.map((e) => e.run))].reverse()) list.push({ path: run, label: run.split("/").slice(1).join("/") });
    return list;
  }, [live, state.exports]);
  const [runPick, setRunPick] = useState<string | null>(null);
  const run = runs.find((r) => r.path === runPick) ?? runs[0];
  const evals = useFetched<Evaluation[]>(run ? "evals" : null, { run: run?.path ?? "" }, run?.path === live ? Math.floor(state.now / 15) : run?.path);
  const [stepPick, setStepPick] = useState<number | null>(null);
  const [sort, setSort] = useState<"psnr" | "name">("psnr");
  const [open, setOpen] = useState<Pair | null>(null);

  if (!run) return <div className="empty">Nothing trained yet.</div>;
  const evaluation = evals?.find((e) => e.step === stepPick) ?? evals?.[evals.length - 1];
  const pairs = [...(evaluation?.images ?? [])].sort((a, b) => (sort === "name" ? a.name.localeCompare(b.name) : (a.psnr ?? 0) - (b.psnr ?? 0)));
  const aspect = state.images.size ? state.images.size[0] / state.images.size[1] : 1.5;
  const scored = pairs.filter((p) => p.psnr !== null);
  const mean = scored.length ? scored.reduce((sum, p) => sum + p.psnr!, 0) / scored.length : null;

  return (
    <div className="images">
      <div className="toolbar">
        <label>run
          <select value={run.path} onChange={(e) => setRunPick(e.target.value)}>
            {runs.map((r) => <option key={r.path} value={r.path}>{r.label}</option>)}
          </select>
        </label>
        {evals && evals.length > 0 && (
          <label>step
            <select value={evaluation?.step} onChange={(e) => setStepPick(Number(e.target.value))}>
              {evals.map((e) => <option key={e.step} value={e.step}>{e.step.toLocaleString()}</option>)}
            </select>
          </label>
        )}
        <label>sort
          <select value={sort} onChange={(e) => setSort(e.target.value as "psnr" | "name")}>
            <option value="psnr">worst first</option>
            <option value="name">name</option>
          </select>
        </label>
        {mean !== null && <span className="count">{pairs.length} held-out photos, mean PSNR {mean.toFixed(2)} dB <span className="label">(whole image)</span></span>}
      </div>
      {!evals ? (
        <div className="empty">Loading…</div>
      ) : evals.length === 0 ? (
        <div className="empty">
          This run saved no renders. Train with `gausser train --eval-split 8` to hold out every 8th photo; Brush then renders each one every 1000 steps.
        </div>
      ) : (
        <div className="grid-scroll">
          <div
            className="grid"
            style={{ gridTemplateColumns: "repeat(auto-fill, minmax(420px, 1fr))", gap: "14px 10px", "--aspect": aspect } as React.CSSProperties}
          >
            {pairs.map((pair) => (
              <div key={pair.name} className="tile" onClick={() => setOpen(pair)}>
                <div className="pair">
                  {pair.photo && <img loading="lazy" src={thumbUrl(pair.photo, 384, 0)} alt="" />}
                  <img loading="lazy" src={thumbUrl(pair.render, 384, evaluation!.mtime)} alt="" />
                </div>
                <div className="tile-info">
                  <span className="name">{pair.name}</span>
                  <span>{pair.psnr !== null ? `${pair.psnr.toFixed(2)} dB` : "no photo"}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      {open && evaluation && <Swipe pair={open} version={evaluation.mtime} step={evaluation.step} onClose={() => setOpen(null)} />}
    </div>
  );
}

/** Photo under the render; the cursor moves the edge between them. */
function Swipe({ pair, version, step, onClose }: { pair: Pair; version: number; step: number; onClose: () => void }) {
  const [edge, setEdge] = useState(50);
  return (
    <div className="lightbox" onClick={onClose}>
      <div
        className="swipe" onClick={(e) => e.stopPropagation()}
        onPointerMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          setEdge(Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100)));
        }}
      >
        {pair.photo && <img src={thumbUrl(pair.photo, 1024, 0)} alt="photo" />}
        <img className="swipe-top" src={thumbUrl(pair.render, 1024, version)} alt="render" style={{ clipPath: `inset(0 0 0 ${edge}%)` }} />
        <span className="swipe-edge" style={{ left: `${edge}%` }} />
      </div>
      <span className="label">
        left: photo · right: Brush at step {step.toLocaleString()} · {pair.name}{pair.psnr !== null ? ` · ${pair.psnr.toFixed(2)} dB` : ""}
      </span>
    </div>
  );
}
