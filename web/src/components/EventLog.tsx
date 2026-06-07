import type { RunState } from "../lib/runState";
import { C } from "../theme";

export default function EventLog({ state }: { state: RunState }) {
  const lines = [...state.log].reverse();
  return (
    <div className="panel p-3">
      <div className="label mb-2" style={{ color: C.dim }}>
        event log
      </div>
      <div className="font-mono text-[11px] space-y-1 max-h-[180px] overflow-auto">
        {lines.length === 0 && !state.ended && (
          <div style={{ color: C.faint }}>— no fault events —</div>
        )}
        {lines.map((l, i) => (
          <div key={i} className="flex gap-2">
            <span style={{ color: C.faint }}>{String(l.step).padStart(4, "0")}</span>
            <span style={{ color: l.kind === "onset" ? C.alarm : C.dim }}>
              {l.kind === "onset" ? "▲" : "·"} {l.text}
            </span>
          </div>
        ))}
        {state.ended && (
          <div style={{ color: C.faint }}>
            {String(state.lastStep).padStart(4, "0")} · recording ended
          </div>
        )}
      </div>
    </div>
  );
}
