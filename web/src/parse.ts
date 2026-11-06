// Progress and statistics read from step logs (.gausser/logs, cleaned by the monitor).
import type { Step } from "./api";

export interface Progress {
  phase: string;
  done: number | null;
  total: number | null;
}

const last = <T,>(items: T[]) => items[items.length - 1];

function lastMatch(lines: string[], pattern: RegExp, from = 0): RegExpMatchArray | null {
  for (let i = lines.length - 1; i >= from; i--) {
    const match = lines[i].match(pattern);
    if (match) return match;
  }
  return null;
}

const num = (text: string) => Number(text.replace(/,/g, ""));

/** Where a running step is, from its log so far. */
export function progress(
  step: Step, lines: string[], partial: string, totals: { images: number; frames: number; iterations?: number },
): Progress {
  const all = partial ? [...lines, partial] : lines;
  const commands = all.map((line, i) => [i, line] as const).filter(([, line]) => line.startsWith("$ "));
  const command = last(commands);
  const since = command ? command[0] : 0;
  const tool = command ? command[1].split(" ").slice(1, 3).map((part) => part.split("/").pop()).join(" ") : "";

  if (step === "frames") {
    const pano = lastMatch(all, /^\[(\d+)\/(\d+)\] 360° frame/, since);
    if (pano) return { phase: "rendering 360° views", done: num(pano[1]), total: num(pano[2]) };
    const latest = lastMatch(all, /^(analysed (\d+)\/(\d+) frames|Extracting|Selecting)[^.]*/);
    if (latest?.[2]) return { phase: "analysing", done: num(latest[2]), total: num(latest[3]) };
    return { phase: latest ? latest[0].toLowerCase() : "starting", done: null, total: null };
  }
  if (step === "mask") {
    const match = lastMatch(all, /^\[(\d+)\/(\d+)\]/);
    return match ? { phase: "SAM3", done: num(match[1]), total: num(match[2]) } : { phase: "loading SAM3", done: null, total: null };
  }
  if (step === "solve") {
    if (tool.endsWith("feature_extractor")) {
      const match = lastMatch(all, /Processed file \[(\d+)\/(\d+)\]/, since);
      return { phase: "features", done: match ? num(match[1]) : 0, total: match ? num(match[2]) : totals.images };
    }
    if (tool.endsWith("_matcher")) {
      const block = lastMatch(all, /Processing block \[(\d+)\/(\d+), (\d+)\/(\d+)\]/, since);
      if (block) {
        const [a, b, c, d] = block.slice(1, 5).map(num);
        return { phase: "matching", done: (a - 1) * d + c, total: b * d };
      }
      const image = lastMatch(all, /Matching image \[(\d+)\/(\d+)\]/, since);
      return { phase: "matching", done: image ? num(image[1]) : null, total: image ? num(image[2]) : null };
    }
    if (tool.endsWith("mapper")) {
      const match = lastMatch(all, /num_reg_(?:frames|images)=(\d+)/, since);
      return { phase: "mapping", done: match ? num(match[1]) : 0, total: totals.frames };
    }
    return { phase: tool.split(" ").pop() || "starting", done: null, total: null };
  }
  if (step === "train") {
    const brush = lastMatch(all, /(\d+)\/(\d+)\s+Steps/);
    if (brush) return { phase: "training", done: num(brush[1]), total: num(brush[2]) };
    const msplat = lastMatch(all, /^([\d,]+)\/([\d,]+)\s+[\d,]+ splats/);
    if (msplat) return { phase: "training", done: num(msplat[1]), total: num(msplat[2]) };
    // Without a terminal Brush draws no progress bar; its refine and eval lines still count steps.
    const step = lastMatch(all, /(?:Refine|Eval) iter (\d+)/);
    if (step && totals.iterations) return { phase: "training", done: num(step[1]), total: totals.iterations };
    return { phase: "loading", done: null, total: null };
  }
  return { phase: tool.split(" ").pop() || "starting", done: null, total: null };
}

export interface TrainStats {
  refine: { iteration: number; splats: number; seconds: number }[];
  evals: { iteration: number; psnr: number; ssim: number }[];
  views: { train: number; eval: number } | null;
}

/** Splat growth and eval scores from a Brush (or Msplat) log. */
export function trainStats(lines: string[]): TrainStats {
  const refine: TrainStats["refine"] = [];
  const evals: TrainStats["evals"] = [];
  let views: TrainStats["views"] = null;
  let first: number | null = null;
  for (const line of lines) {
    const time = line.match(/^\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)/);
    const seconds = time ? Date.parse(time[1]) / 1000 : null;
    if (seconds !== null && first === null) first = seconds;
    let match = line.match(/Refine iter (\d+), (\d+) splats/);
    if (match) refine.push({ iteration: num(match[1]), splats: num(match[2]), seconds: seconds !== null && first !== null ? seconds - first : 0 });
    match = line.match(/^([\d,]+)\/[\d,]+\s+([\d,]+) splats/);
    if (match) refine.push({ iteration: num(match[1]), splats: num(match[2]), seconds: 0 });
    match = line.match(/Eval iter (\d+): PSNR ([\d.]+), ssim ([\d.]+)/i);
    if (match) evals.push({ iteration: num(match[1]), psnr: Number(match[2]), ssim: Number(match[3]) });
    match = line.match(/Loaded dataset with (\d+) training, (\d+) eval views/);
    if (match) views = { train: num(match[1]), eval: num(match[2]) };
  }
  return { refine, evals, views };
}

/** SAM3 score per image (null: no hit) from a mask log. */
export function maskScores(lines: string[]): { name: string; score: number | null; note: string }[] {
  const scores = [];
  for (const line of lines) {
    const match = line.match(/^\[\d+\/\d+\] (\S+)\s+(.*)$/);
    if (!match) continue;
    const value = Number(match[2]);
    const excluded = match[2].match(/([\d.]+)% excluded/);
    scores.push({
      name: match[1],
      score: Number.isFinite(value) ? value : excluded ? Number(excluded[1]) / 100 : null,
      note: match[2],
    });
  }
  return scores;
}

/** COLMAP numbers worth showing from a solve log. */
export function solveStats(lines: string[]): { features: number[]; pairs: string | null; registered: string | null } {
  const features = [];
  for (const line of lines) {
    const match = line.match(/Features:\s+(\d+)/);
    if (match) features.push(num(match[1]));
  }
  const registered = lastMatch(lines, /^Model: (.*)$/);
  const pairs = lastMatch(lines, /Generating (\w+) image pairs/);
  return { features, pairs: pairs ? pairs[1] : null, registered: registered ? registered[1] : null };
}
