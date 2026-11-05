import { useEffect, useRef } from "react";
import { clock, duration, useLog } from "./api";
import type { HistoryRecord, State } from "./api";

interface LiveLog {
  lines: string[];
  partial: string;
  truncated: boolean;
}

export function Logs({ state, pick, onPick, live }: {
  state: State; pick: HistoryRecord | null; onPick: (record: HistoryRecord) => void; live: LiveLog;
}) {
  const runs = [...state.history].reverse();
  const current = state.current;
  const chosen = pick ?? current ?? runs[0] ?? null;
  const isLive = !!current && !!chosen && chosen.log === current.log && chosen.start === current.start;
  const fetched = useLog(isLive ? null : chosen?.log ?? null, false);
  const log = isLive ? live : fetched;
  const box = useRef<HTMLPreElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    const el = box.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [log.lines.length, log.partial, chosen?.log]);

  return (
    <div className="logs">
      <div className="runs">
        {runs.length === 0 && !current && <div className="empty">Steps run with this gausser are listed here.</div>}
        {current && !current.log && (
          <div className="run active" style={{ "--step": `var(--${current.step})` } as React.CSSProperties}>
            <span className="what"><span className={`step-${current.step}`}>{current.step}</span> <span className="live">running</span></span>
            <span>{duration(state.now - current.start)}</span>
            <span className="when">started by an older gausser, so there is no log</span>
          </div>
        )}
        {runs.map((run) => {
          const running = current?.start === run.start && current.step === run.step;
          const status = running ? "running" : run.end === undefined ? "died" : run.ok ? "ok" : run.error === "interrupted" ? "interrupted" : "failed";
          return (
            <div
              key={`${run.step}-${run.start}`} className={`run ${chosen === run || (chosen?.start === run.start && chosen.step === run.step) ? "active" : ""}`}
              onClick={() => onPick(run)} style={{ "--step": `var(--${run.step})` } as React.CSSProperties}
            >
              <span className="what">
                <span className={`step-${run.step}`}>{run.step}</span>{" "}
                <span className={status === "ok" ? "ok" : status === "running" ? "live" : "bad"}>{status}</span>
              </span>
              <span>{duration((run.end ?? state.now) - run.start)}</span>
              <span className="when">{clock(run.start)}{detail(run)}</span>
              {run.error && run.error !== "interrupted" && <span className="error" title={run.error}>{run.error}</span>}
            </div>
          );
        })}
      </div>
      <pre
        className="log" ref={box}
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        }}
      >
        {!chosen && "Choose a step on the left."}
        {log.truncated && <div className="meta">Showing the last 4 MB; the whole log is .gausser/logs/{chosen?.log}</div>}
        {log.lines.map((line, i) => <Line key={i} text={line} />)}
        {log.partial && <div className="partial">{log.partial}</div>}
      </pre>
    </div>
  );
}

function detail(run: HistoryRecord): string {
  const parts = Object.entries(run)
    .filter(([key]) => !["step", "start", "end", "ok", "error", "log"].includes(key))
    .map(([key, value]) => `${key}=${typeof value === "string" ? value.split("/").pop() : String(value)}`);
  return parts.length ? ` · ${parts.join(" ")}` : "";
}

function Line({ text }: { text: string }) {
  let kind = "";
  if (text.startsWith("$ ")) kind = "cmd";
  else if (/^E\d{8}|error|failed|panic/i.test(text)) kind = "err";
  else if (/^W\d{8}|warn/i.test(text)) kind = "warn";
  else if (text.startsWith("gausser:")) kind = "meta";
  return <div className={kind}>{text || " "}</div>;
}
