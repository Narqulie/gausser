import { useEffect, useRef, useState } from "react";
import { duration } from "./api";
import type { HistoryRecord, State } from "./api";

interface Toast {
  id: string;
  step: string;
  title: string;
  body: string;
  ok: boolean;
}

/** A toast (and a desktop notification, when allowed) each time a step of this scene ends. */
export function useStepToasts(state: State | null, describe: (record: HistoryRecord) => string) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const seen = useRef<Set<string> | null>(null);

  useEffect(() => {
    if (!state) return;
    const ended = state.history.filter((r) => r.end !== undefined);
    const key = (r: HistoryRecord) => `${r.step}-${r.start}`;
    if (seen.current === null) {
      seen.current = new Set(ended.map(key)); // only steps that end while the page is open
      return;
    }
    for (const record of ended) {
      if (seen.current.has(key(record))) continue;
      seen.current.add(key(record));
      const took = duration(record.end! - record.start);
      const toast: Toast = record.ok
        ? { id: key(record), step: record.step, ok: true, title: `${record.step} finished in ${took}`, body: describe(record) }
        : record.error === "interrupted"
          ? { id: key(record), step: record.step, ok: false, title: `${record.step} was interrupted after ${took}`, body: "" }
          : { id: key(record), step: record.step, ok: false, title: `${record.step} failed after ${took}`, body: record.error ?? "" };
      setToasts((old) => [...old, toast]);
      setTimeout(() => setToasts((old) => old.filter((t) => t.id !== toast.id)), toast.ok ? 8000 : 20000);
      if (document.hidden && "Notification" in window && Notification.permission === "granted") {
        new Notification(`${state.name}: ${toast.title}`, { body: toast.body, tag: toast.id });
      }
    }
  }, [state, describe]);

  const dismiss = (id: string) => setToasts((old) => old.filter((t) => t.id !== id));
  return { toasts, dismiss };
}

export function Toasts({ toasts, dismiss }: { toasts: Toast[]; dismiss: (id: string) => void }) {
  return (
    <div className="toasts" role="status">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.ok ? "" : "failed"}`} style={{ "--step": `var(--${t.step})` } as React.CSSProperties} onClick={() => dismiss(t.id)}>
          <span className="toast-title">{t.title}</span>
          {t.body && <span className="toast-body">{t.body}</span>}
        </div>
      ))}
    </div>
  );
}

/** Ask once for desktop notifications; they appear only while the tab is in the background. */
export function NotifyToggle() {
  const supported = "Notification" in window;
  const [permission, setPermission] = useState(supported ? Notification.permission : "denied");
  if (!supported || permission === "denied") return null;
  if (permission === "granted") return <span className="label" title="Desktop notifications when a step ends while this tab is hidden">notifications on</span>;
  return (
    <button className="link-button" onClick={() => Notification.requestPermission().then(setPermission)}>
      notify me when a step ends
    </button>
  );
}
