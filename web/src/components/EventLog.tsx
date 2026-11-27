import type { RunState } from "../lib/runState";
import { C } from "../theme";

export default function EventLog({ state }: { state: RunState }) {
  const lines = [...state.log].reverse();
  return (
    <div className="section p-5">
      <span className="label">Event log</span>
      <div className="mono mt-3 space-y-1.5" style={{ fontSize: 11.5, maxHeight: 168, overflow: "auto" }}>
        {lines.length === 0 && !state.ended && (
          <div style={{ color: C.faint }}>no events recorded</div>
        )}
        {lines.map((l, i) => (
          <div key={i} className="flex gap-3">
            <span style={{ color: C.faint }}>{String(l.step).padStart(4, "0")}</span>
            <span style={{ color: l.kind === "onset" ? C.accent : C.muted }}>{l.text}</span>
          </div>
        ))}
        {state.ended && (
          <div className="flex gap-3">
            <span style={{ color: C.faint }}>{String(state.lastStep).padStart(4, "0")}</span>
            <span style={{ color: C.faint }}>recording ended</span>
          </div>
        )}
      </div>
    </div>
  );
}
