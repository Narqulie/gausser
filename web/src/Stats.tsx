import { useMemo } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { clock, duration, useFetched, useLog } from "./api";
import type { FramesMeta, HistoryRecord, ImageInfo, Model, State, Step } from "./api";
import { maskScores, solveStats, trainStats } from "./parse";

const COLORS: Record<Step, string> = { frames: "#fede5d", mask: "#ff8b39", solve: "#36f9f6", train: "#ff7edb", lod: "#72f1b8" };
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
          <h2>Frames sharp-frames chose</h2>
          <p>
            {frames.total_selected} of {extracted ?? "?"} extracted frames, {frames.selection_method}, at least {frames.min_buffer} apart. Sharpness by position in the video.
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

      {train.refine.length > 0 && (
        <section className="chart">
          <h2>Splats during training</h2>
          <p>
            {String(trainRun?.trainer ?? "brush")}
            {train.views ? `, ${train.views.train} training views${train.views.eval ? `, ${train.views.eval} held out` : ""}` : ""}
            {rate(train.refine)}
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={train.refine} margin={{ right: 20 }}>
              {GRID}
              <XAxis dataKey="iteration" type="number" {...AXIS} />
              <YAxis {...AXIS} tickFormatter={(v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${Math.round(v / 1e3)}k` : String(v))} />
              <Tooltip {...TOOLTIP} formatter={(v) => Number(v).toLocaleString()} />
              <Line dataKey="splats" stroke="#ff7edb" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </section>
      )}

      {train.evals.length > 0 && (
        <section className="chart">
          <h2>PSNR and SSIM on held-out images</h2>
          <p>From `gausser train --eval-split N`, every 1000 steps.</p>
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
  return `, ${((b.iteration - a.iteration) / (b.seconds - a.seconds)).toFixed(1)} steps/s`;
}
