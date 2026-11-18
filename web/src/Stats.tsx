import { useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { clock, duration, useFetched, useLog, useLogs } from "./api";
import type { FramesMeta, HistoryRecord, ImageInfo, Model, State, Step } from "./api";
import { maskScores, solveStats, trainStats } from "./parse";
import type { TrainStats } from "./parse";

const RUN_COLORS = ["#ff7edb", "#36f9f6", "#fede5d", "#72f1b8", "#ff8b39"];

function runLabel(run: HistoryRecord): string {
  return run.run ? String(run.run).split("/").slice(1).join("/") : `${run.step} ${clock(run.start)}`;
}

function runDetail(run: HistoryRecord, lines: string[] | undefined, live: boolean, now: number): string {
  const stats: TrainStats | null = lines ? trainStats(lines) : null;
  const parts = [String(run.trainer ?? "brush"), `${Number(run.iterations ?? 0).toLocaleString()} steps`];
  parts.push(live ? "running" : run.ok ? duration((run.end ?? now) - run.start) : String(run.error ?? "unfinished"));
  const splats = stats?.refine[stats.refine.length - 1]?.splats;
  if (splats) parts.push(`${splats.toLocaleString()} splats`);
  if (stats && rate(stats.refine)) parts.push(rate(stats.refine));
  const psnr = stats?.evals[stats.evals.length - 1]?.psnr;
  if (psnr) parts.push(`PSNR ${psnr.toFixed(2)}`);
  return parts.join(", ");
}

const COLORS: Record<Step, string> = { frames: "#fede5d", mask: "#ff8b39", solve: "#36f9f6", train: "#ff7edb", lod: "#72f1b8", dense: "#2de2e6", mesh: "#f97e72", map: "#6c9bff" };
const AXIS = { stroke: "#614d85", fontSize: 11, tickLine: false };
const TOOLTIP = {
  contentStyle: { background: "#171520", border: "1px solid #34294f", borderRadius: 3, fontFamily: "inherit" },
  labelStyle: { color: "#848bbd" },
  itemStyle: { color: "#f4eefa" },
  cursor: { fill: "rgba(255,126,219,.08)" },
};
const GRID = <CartesianGrid stroke="#2e2445" vertical={false} />;

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
  const trainRuns = useMemo(() => [...state.history].reverse().filter((r) => r.step === "train" && r.log), [state.history]);
  const [picks, setPicks] = useState<string[] | null>(null);
  const picked = picks ?? trainRuns.slice(0, 1).map((r) => r.log);
  const trained = useLogs(picked, following("train") ? trainRun?.log ?? null : null);
  const series = picked.map((log, i) => {
    const run = trainRuns.find((r) => r.log === log);
    return { log, color: RUN_COLORS[i], label: run ? runLabel(run) : log, stats: trainStats(trained[log] ?? []) };
  });
  const maskLog = useLog(maskRun?.log ?? null, following("mask"));
  const solveLog = useLog(solveRun?.log ?? null, following("solve"));

  const scores = useMemo(() => maskScores(maskLog.lines), [maskLog.lines]);
  const solve = useMemo(() => solveStats(solveLog.lines), [solveLog.lines]);
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
      <section className="chart wide">
        <h2>Time per step</h2>
        <p>
          {totals.length
            ? totals.map((t) => `${t.step} ${duration(t.seconds)}${t.runs > 1 ? ` (${t.runs} runs)` : ""}`).join(" · ")
            : "Steps appear here as they finish."}
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
        <section className="chart">
          <h2>Frames chosen from the video</h2>
          <p>
            {frames.total_selected} of {frames.analysed} frames ({frames.method}{frames.per_level.length > 1 ? `, level ${frames.level} of ${frames.per_level.length - 1}` : ""}). Sharpness by time in the video.
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart margin={{ right: 20 }}>
              {GRID}
              <XAxis type="number" dataKey="time" name="seconds" {...AXIS} />
              <YAxis type="number" dataKey="sharpness" name="sharpness" {...AXIS} />
              <Tooltip {...TOOLTIP} />
              <Scatter data={frames.items} fill="#fede5d" />
            </ScatterChart>
          </ResponsiveContainer>
        </section>
      )}

      {sharpness.length > 0 && (
        <section className="chart">
          <h2>Sharpness per image</h2>
          <p>Laplacian variance at 512 px.{model ? " Red: COLMAP could not place the image." : ""}</p>
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
        <section className="chart">
          <h2>Points each image sees</h2>
          <p>
            {model.cameras.length} of {state.images.count} placed, {model.stats.points.toLocaleString()} points,{" "}
            {model.stats.mean_error.toFixed(2)} px reprojection error
            {solve.features.length ? `, ${Math.round(solve.features.reduce((a, b) => a + b, 0) / solve.features.length)} features per image` : ""}
            {solve.pairs ? `, ${solve.pairs} matching` : ""}
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
        <section className="chart">
          <h2>Track lengths</h2>
          <p>How many images see each point: {model.stats.mean_track.toFixed(1)} on average. The last bar is 30 or more.</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={model.stats.track_lengths.map(([length, count]) => ({ length, count }))} margin={{ right: 20 }}>
              {GRID}
              <XAxis dataKey="length" {...AXIS} />
              <YAxis {...AXIS} />
              <Tooltip {...TOOLTIP} />
              <Bar dataKey="count" fill="#36f9f6" />
            </BarChart>
          </ResponsiveContainer>
        </section>
      )}

      {scores.length > 0 && (
        <section className="chart">
          <h2>{maskRun?.exclude ? "Share of each image excluded" : "SAM3 score per image"}</h2>
          <p>
            {maskRun?.prompt ? `"${String(maskRun.prompt)}": ` : ""}
            found in {scores.filter((s) => s.score !== null).length} of {scores.length} images
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={scores.map((s, index) => ({ ...s, index, value: s.score ?? 0 }))} margin={{ right: 20 }}>
              {GRID}
              <XAxis dataKey="index" {...AXIS} />
              <YAxis {...AXIS} domain={[0, 1]} />
              <Tooltip {...TOOLTIP} labelFormatter={(_, p) => p?.[0]?.payload?.name ?? ""} formatter={(_, __, p) => p.payload.note} />
              <Bar dataKey="value">
                {scores.map((s, i) => <Cell key={i} fill={s.score === null ? "#fe4450" : "#ff8b39"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>
      )}

      {sharpness.some((s) => s.mask !== null) && (
        <section className="chart">
          <h2>Mask coverage</h2>
          <p>Share of each image the trainer learns from.</p>
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

      {trainRuns.length > 0 && (
        <section className="chart wide">
          <h2>Training runs</h2>
          <p>Pick up to five runs to compare.</p>
          <div className="run-picks">
            {trainRuns.map((run, i) => {
              const on = picked.includes(run.log);
              const color = RUN_COLORS[picked.indexOf(run.log)] ?? "var(--dim)";
              return (
                <label key={run.log} className={on ? "on" : ""} style={{ "--run": color } as React.CSSProperties}>
                  <input type="checkbox" checked={on} disabled={!on && picked.length >= RUN_COLORS.length}
                    onChange={() => setPicks(on ? picked.filter((l) => l !== run.log) : [...picked, run.log])} />
                  <span className="swatch" />
                  {runLabel(run)}
                  <span className="label">{runDetail(run, trained[run.log], i === 0 && following("train"), state.now)}</span>
                </label>
              );
            })}
          </div>
        </section>
      )}

      {series.some((r) => r.stats.refine.length) && (
        <section className="chart">
          <h2>Splats during training</h2>
          <p>Splat count after each refinement (every 200 steps).</p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart margin={{ right: 20 }}>
              {GRID}
              <XAxis dataKey="iteration" type="number" allowDuplicatedCategory={false} {...AXIS} domain={[0, "dataMax"]} />
              <YAxis {...AXIS} tickFormatter={(v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${Math.round(v / 1e3)}k` : String(v))} />
              <Tooltip {...TOOLTIP} formatter={(v) => Number(v).toLocaleString()} />
              {series.map((r) => (
                <Line key={r.log} data={r.stats.refine} dataKey="splats" name={r.label} stroke={r.color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </section>
      )}

      {series.some((r) => r.stats.evals.length) && (
        <section className="chart">
          <h2>PSNR on held-out images</h2>
          <p>Brush's eval every 1000 steps (`gausser train --eval-split N`). Dashed: SSIM, right axis.</p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart margin={{ right: 20 }}>
              {GRID}
              <XAxis dataKey="iteration" type="number" allowDuplicatedCategory={false} {...AXIS} domain={[0, "dataMax"]} />
              <YAxis yAxisId="psnr" {...AXIS} domain={["auto", "auto"]} unit=" dB" />
              <YAxis yAxisId="ssim" orientation="right" {...AXIS} domain={[0, 1]} />
              <Tooltip {...TOOLTIP} />
              {series.map((r) => (
                <Line key={r.log} yAxisId="psnr" data={r.stats.evals} dataKey="psnr" name={`${r.label} PSNR`} stroke={r.color} strokeWidth={1.5} isAnimationActive={false} />
              ))}
              {series.map((r) => (
                <Line key={`${r.log}-ssim`} yAxisId="ssim" data={r.stats.evals} dataKey="ssim" name={`${r.label} SSIM`} stroke={r.color} strokeDasharray="4 3" strokeWidth={1} dot={false} isAnimationActive={false} />
              ))}
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
  return `${((b.iteration - a.iteration) / (b.seconds - a.seconds)).toFixed(1)} steps/s`;
}
