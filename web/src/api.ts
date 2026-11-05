import { useEffect, useRef, useState } from "react";

export type Step = "frames" | "mask" | "solve" | "train" | "lod";
export const STEPS: Step[] = ["frames", "mask", "solve", "train", "lod"];

export interface HistoryRecord {
  step: Step;
  start: number;
  end?: number;
  ok?: boolean;
  error?: string;
  log: string;
  [info: string]: unknown;
}

export interface Export {
  path: string;
  run: string;
  size: number;
  mtime: number;
  iteration: number | null;
  marker: string | null;
}

export interface Live {
  staging?: string;
  exports?: Export[];
  model?: { path: string; registered: number; mtime: number };
  count?: number;
}

export interface State {
  name: string;
  root: string;
  now: number;
  holder: { pid: number; argv: string[]; start: number } | null;
  history: HistoryRecord[];
  current: (HistoryRecord & { live: Live }) | null;
  images: { count: number; size: [number, number] | null; loose: number; panos: number; views: number };
  masks: { count: number; exclude: number; done: boolean };
  model: { path: string; registered: number; mtime: number; current: boolean; no_masks: boolean } | null;
  exports: Export[];
  trained: boolean;
  lods: { path: string; mtime: number }[];
  lod_current: boolean;
}

export interface Camera {
  name: string;
  center: [number, number, number];
  rotation: number[]; // camera to world, row-major 3×3, OpenCV axes
  fx: number;
  fy: number;
  width: number;
  height: number;
  points: number;
}

export interface Model {
  up: [number, number, number];
  cameras: Camera[];
  points: { xyz: number[]; rgb: number[] };
  stats: { points: number; mean_error: number; mean_track: number; track_lengths: [number, number][] };
}

export interface ImageInfo {
  name: string;
  frame: string;
  view: number | null;
  mtime: number;
  mask: string | false;
  exclude: string | false;
  sharpness: number;
  masks?: number; // white share of the Brush mask
  "masks-exclude"?: number;
}

export interface FramesMeta {
  input_path: string;
  total_selected: number;
  selection_method: string;
  num_frames_requested: number;
  min_buffer: number;
  selected_items: { output_filename: string; original_index: number; sharpness_score: number }[];
}

export interface LogChunk {
  text: string;
  partial: string;
  offset: number;
  truncated: boolean;
}

export async function get<T>(route: string, params: Record<string, string | number> = {}): Promise<T> {
  const query = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]));
  const response = await fetch(`api/${route}?${query}`);
  if (!response.ok) throw new Error(`${route}: ${response.status}`);
  return response.json() as Promise<T>;
}

export const fileUrl = (path: string) => `api/file?path=${encodeURIComponent(path)}`;
export const thumbUrl = (path: string, size: number, version: number) =>
  `api/thumb?path=${encodeURIComponent(path)}&size=${size}&v=${Math.round(version)}`;

/** Poll the scene state; `null` while the monitor is unreachable. */
export function useSceneState(interval = 2000): { state: State | null; offline: boolean } {
  const [state, setState] = useState<State | null>(null);
  const [offline, setOffline] = useState(false);
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const next = await get<State>("state");
        if (!stop) {
          setState(next);
          setOffline(false);
        }
      } catch {
        if (!stop) setOffline(true);
      }
      if (!stop) setTimeout(tick, interval);
    };
    tick();
    return () => {
      stop = true;
    };
  }, [interval]);
  return { state, offline };
}

/** Fetch `route` again whenever `key` changes; keep the previous value while loading. */
export function useFetched<T>(route: string | null, params: Record<string, string | number>, key: unknown): T | null {
  const [value, setValue] = useState<T | null>(null);
  const keyText = JSON.stringify([route, params, key]);
  useEffect(() => {
    if (!route) {
      setValue(null);
      return;
    }
    let stop = false;
    get<T>(route, params).then((v) => !stop && setValue(v), () => {});
    return () => {
      stop = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keyText]);
  return value;
}

/** A step log, followed live while `follow` is set. */
export function useLog(name: string | null, follow: boolean): { lines: string[]; partial: string; truncated: boolean } {
  const [lines, setLines] = useState<string[]>([]);
  const [partial, setPartial] = useState("");
  const [truncated, setTruncated] = useState(false);
  const offset = useRef<number | null>(null);
  useEffect(() => {
    setLines([]);
    setPartial("");
    offset.current = null;
    if (!name) return;
    let stop = false;
    const tick = async () => {
      try {
        const params: Record<string, string | number> = { name };
        if (offset.current !== null) params.offset = offset.current;
        const chunk = await get<LogChunk>("log", params);
        if (stop) return;
        if (offset.current === null) setTruncated(chunk.truncated);
        offset.current = chunk.offset;
        const fresh = chunk.text.split("\n").filter((line, i, all) => line !== "" || i < all.length - 1);
        if (fresh.length) setLines((old) => [...old, ...fresh].slice(-20000));
        setPartial(chunk.partial);
      } catch {
        /* the next tick retries */
      }
      if (!stop && follow) setTimeout(tick, 1000);
    };
    tick();
    return () => {
      stop = true;
    };
  }, [name, follow]);
  return { lines, partial, truncated };
}

export function duration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

export function bytes(n: number): string {
  if (n < 1e6) return `${(n / 1e3).toFixed(0)} kB`;
  if (n < 1e9) return `${(n / 1e6).toFixed(1)} MB`;
  return `${(n / 1e9).toFixed(2)} GB`;
}

export function clock(epoch: number): string {
  return new Date(epoch * 1000).toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}
