import { useMemo } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { clock, duration, useFetched, useLog } from "./api";
import type { FramesMeta, HistoryRecord, ImageInfo, Model, State, Step } from "./api";
import { maskScores, solveStats, trainStats } from "./parse";

const COLORS: Record<Step, string> = { frames: "#fede5d", mask: "#b381c5", solve: "#36f9f6", train: "#ff7edb", lod: "#72f1b8" };
const AXIS = { stroke: "#6f6385", fontSize: 11 };
const TOOLTIP = {
  contentStyle: { background: "#120e1c", border: "1px solid #495495", borderRadius: 6, fontFamily: "inherit" },
  labelStyle: { color: "#a99bbd" },
  itemStyle: { color: "#f4eefa" },
  cursor: { fill: "rgba(255,126,219,.08)" },
};
const GRID = <CartesianGrid stroke="#3b2f55" strokeDasharray="3 4" vertical={false} />;

function lastOf(state: State, step: Step): HistoryRecord | null {
  return [...state.history].reverse().find((r) => r.step === step && r.log) ?? null;
}

export function Stats({ state }: { state: State }) {
  const images = useFetched<ImageInfo[]>("images", {}, [state.images.count, state.masks.count, state.masks.exclude]);
  const model = useFetched<Model>(state.model ? "model" : null, { path: state.model?.path ?? "" }, state.model?.mtime);
  const frames = useFetched<FramesMeta | null>("frames", {}, state.images.count);
  const following = (step: Step) => state.current?.step === step;
  const trainRun = lastOf(state, "train");
  const maskRun = lastOf(state, "mask");
  const solveRun = lastOf(state, "solve");
  const framesRun = lastOf(state, "frames");
  const trainLog = useLog(trainRun?.log ?? null, following("train"));
  const maskLog = useLog(maskRun?.log ?? null, following("mask"));
  const solveLog = useLog(solveRun?.log ?? null, following("solve"));
  const framesLog = useLog(framesRun?.log ?? null, false);

  const train = useMemo(() => trainStats(trainLog.lines), [trainLog.lines]);
  const scores = useMemo(() => maskScores(maskLog.lines), [maskLog.lines]);
  const solve = useMemo(() => solveStats(solveLog.lines), [solveLog.lines]);
  const extracted = useMemo(() => {
    for (const line of framesLog.lines) {
      const match = line.match(/Extracted (\d+) frames/);
      if (match) return Number(match[1]);
    }
    return null;
  }, [framesLog.lines]);
  const cameras = useMemo(() => new Map(model?.cameras.map((c) => [c.name, c]) ?? []), [model]);

  const timings = state.history.filter((r) => r.end !== undefined || following(r.step)).slice(-40).map((r) => ({
    label: `${r.step} · ${clock(r.start)}`,
    step: r.step,
    minutes: ((r.end ?? state.now) - r.start) / 60,
    ok: r.ok !== false,
  }));
  const totals = (Object.keys(COLORS) as Step[]).map((step) => {
    const runs = state.history.filter((r) => r.step === step && r.end !== undefined);
    return { step, runs: runs.length, seconds: runs.reduce((sum, r) => sum + (r.end! - r.start), 0) };
  }).filter((t) => t.runs);
  const sharpness = (images ?? []).map((i, index) => ({
    index, name: i.name, sharpness: i.sharpness, registered: !model || cameras.has(i.name),
    mask: i.masks !== undefined ? i.masks * 100 : i["masks-exclude"] !== undefined ? i["masks-exclude"] * 100 : null,
  }));
  const seen = (images ?? []).map((i, index) => ({ index, name: i.name, points: cameras.get(i.name)?.points ?? 0 }));

  return (
    <div className="stats">
      <section className="panel wide">
        <h3>Step timings</h3>
        <p className="note">
          {totals.length
            ? totals.map((t) => `${t.step} ${duration(t.seconds)}${t.runs > 1 ? ` (${t.runs} runs)` : ""}`).join(" · ")
            : "No step has been recorded in .gausser/history.jsonl yet."}
        </p>
        {timings.length > 0 && (
          <ResponsiveContainer width="100%" height={Math.max(120, timings.length * 22 + 40)}>
            <BarChart data={timings} layout="vertical" margin={{ left: 20, right: 20 }}>
              {GRID}
              <XAxis type="number" {...AXIS} unit=" min" />
              <YAxis type="category" dataKey="label" width={210} {...AXIS} />
              <Tooltip {...TOOLTIP} formatter={(v) => duration(Number(v) * 60)} />
              <Bar dataKey="minutes" radius={[0, 4, 4, 0]}>
                {timings.map((t, i) => <Cell key={i} fill={COLORS[t.step]} fillOpacity={t.ok ? 0.85 : 0.35} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </section>

      {frames && (
        <section className="panel">
          <h3>Sharp Frames selection</h3>
          <p className="note">
            {frames.total_selected} frames chosen{extracted ? ` of ${extracted} extracted` : ""} · {frames.selection_method} · min buffer {frames.min_buffer}
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart margin={{ right: 20 }}>
              {GRID}
              <XAxis type="number" dataKey="original_index" name="source frame" {...AXIS} />
              <YAxis type="number" dataKey="sharpness_score" name="sharpness" {...AXIS} />
              <Tooltip {...TOOLTIP} />
              <Scatter data={frames.selected_items} fill="#fede5d" />
            </ScatterChart>
          </ResponsiveContainer>
        </section>
      )}

      {sharpness.length > 0 && (
        <section className="panel">
          <h3>Image sharpness</h3>
          <p className="note">Variance of the Laplacian at 512 px{model ? "; red = not registered by COLMAP" : ""}</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={sharpness} margin={{ right: 20 }}>
              {GRID}
              <XAxis dataKey="index" {...AXIS} />
              <YAxis {...AXIS} />
              <Tooltip {...TOOLTIP} labelFormatter={(_, p) => p?.[0]?.payload?.name ?? ""} />
              <Bar dataKey="sharpness">
                {sharpness.map((s) => <Cell key={s.index} fill={s.registered ? "#36f9f6" : "#fe4450"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>
      )}

      {model && (
        <section className="panel">
          <h3>COLMAP: 3D points seen per image</h3>
          <p className="note">
            {model.cameras.length}/{state.images.count} registered · {model.stats.points.toLocaleString()} points ·
            reprojection {model.stats.mean_error.toFixed(2)} px{solve.pairs ? ` · ${solve.pairs} matching` : ""}
            {solve.features.length ? ` · ${Math.round(solve.features.reduce((a, b) => a + b, 0) / solve.features.length)} features/image` : ""}
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={seen} margin={{ right: 20 }}>
              {GRID}
              <XAxis dataKey="index" {...AXIS} />
              <YAxis {...AXIS} />
              <Tooltip {...TOOLTIP} labelFormatter={(_, p) => p?.[0]?.payload?.name ?? ""} />
              <Bar dataKey="points" fill="#36f9f6" />
            </BarChart>
          </ResponsiveContainer>
        </section>
      )}

      {model && (
        <section className="panel">
          <h3>COLMAP: track lengths</h3>
          <p className="note">Images that see each 3D point (mean {model.stats.mean_track.toFixed(1)}; the last bar is 30 or more)</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={model.stats.track_lengths.map(([length, count]) => ({ length, count }))} margin={{ right: 20 }}>
              {GRID}
              <XAxis dataKey="length" {...AXIS} />
              <YAxis {...AXIS} />
              <Tooltip {...TOOLTIP} />
              <Bar dataKey="count" fill="#b381c5" />
            </BarChart>
          </ResponsiveContainer>
        </section>
      )}

      {scores.length > 0 && (
        <section className="panel">
          <h3>SAM3 {maskRun?.exclude ? "exclusion" : "scores"}</h3>
          <p className="note">
            {maskRun?.prompt ? `prompt "${String(maskRun.prompt)}" · ` : ""}
            {scores.filter((s) => s.score !== null).length}/{scores.length} hits{maskRun?.exclude ? " · share of each image excluded" : ""}
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={scores.map((s, index) => ({ ...s, index, value: s.score ?? 0 }))} margin={{ right: 20 }}>
              {GRID}
              <XAxis dataKey="index" {...AXIS} />
              <YAxis {...AXIS} domain={[0, 1]} />
              <Tooltip {...TOOLTIP} labelFormatter={(_, p) => p?.[0]?.payload?.name ?? ""} formatter={(_, __, p) => p.payload.note} />
              <Bar dataKey="value">
                {scores.map((s, i) => <Cell key={i} fill={s.score === null ? "#fe4450" : "#b381c5"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>
      )}

      {sharpness.some((s) => s.mask !== null) && (
        <section className="panel">
          <h3>Mask coverage</h3>
          <p className="note">Share of each image the trainer uses (white in the mask)</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={sharpness} margin={{ right: 20 }}>
              {GRID}
              <XAxis dataKey="index" {...AXIS} />
              <YAxis {...AXIS} unit="%" domain={[0, 100]} />
              <Tooltip {...TOOLTIP} labelFormatter={(_, p) => p?.[0]?.payload?.name ?? ""} formatter={(v) => `${Number(v).toFixed(1)}%`} />
              <Bar dataKey="mask" fill="#72f1b8" />
            </BarChart>
          </ResponsiveContainer>
        </section>
      )}

      {train.refine.length > 0 && (
        <section className="panel">
          <h3>Training: splat count</h3>
          <p className="note">
            {String(trainRun?.trainer ?? "brush")}
            {train.views ? ` · ${train.views.train} training views, ${train.views.eval} eval` : ""}
            {rate(train.refine)}
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={train.refine} margin={{ right: 20 }}>
              {GRID}
              <XAxis dataKey="iteration" type="number" {...AXIS} />
              <YAxis {...AXIS} tickFormatter={(v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${Math.round(v / 1e3)}k` : String(v))} />
              <Tooltip {...TOOLTIP} formatter={(v) => Number(v).toLocaleString()} />
              <Line dataKey="splats" stroke="#ff7edb" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </section>
      )}

      {train.evals.length > 0 && (
        <section className="panel">
          <h3>Training: eval PSNR / SSIM</h3>
          <p className="note">On the held-out images (`gausser train --eval-split N`)</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={train.evals} margin={{ right: 20 }}>
              {GRID}
              <XAxis dataKey="iteration" type="number" {...AXIS} />
              <YAxis yAxisId="psnr" {...AXIS} domain={["auto", "auto"]} />
              <YAxis yAxisId="ssim" orientation="right" {...AXIS} domain={[0, 1]} />
              <Tooltip {...TOOLTIP} />
              <Line yAxisId="psnr" dataKey="psnr" stroke="#36f9f6" strokeWidth={2} isAnimationActive={false} />
              <Line yAxisId="ssim" dataKey="ssim" stroke="#fede5d" strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </section>
      )}
    </div>
  );
}

function rate(refine: { iteration: number; seconds: number }[]): string {
  const timed = refine.filter((r) => r.seconds > 0);
  if (timed.length < 2) return "";
  const a = timed[0];
  const b = timed[timed.length - 1];
  return ` · ${((b.iteration - a.iteration) / (b.seconds - a.seconds)).toFixed(1)} steps/s`;
}
