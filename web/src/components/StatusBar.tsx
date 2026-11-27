import type { RunState } from "../lib/runState";
import { C } from "../theme";

export default function StatusBar({
  runId, mode, state, speed,
}: { runId: string; mode: "LIVE" | "REPORT"; state: RunState; speed?: number }) {
  const onset = state.status === "onset";
  const cells: { label: string; value: string; color?: string }[] = [
    { label: "Run", value: runId },
    { label: "Mode", value: mode === "LIVE" ? "Live" : "Report" },
    { label: "Step", value: String(state.lastStep).padStart(4, "0") },
    { label: "Status", value: onset ? "Onset" : "Nominal", color: onset ? C.accent : C.ink },
    ...(mode === "LIVE" ? [{ label: "Replay", value: `${speed ?? 24}/s` }] : []),
    { label: "Recording", value: state.ended ? "Ended" : mode === "LIVE" ? "Active" : "Archive" },
  ];
  return (
    <div className="section flex items-stretch flex-wrap">
      {cells.map((c, i) => (
        <div
          key={c.label}
          className="px-5 py-3 flex flex-col gap-1"
          style={{ borderLeft: i === 0 ? "none" : `1px solid ${C.rule}` }}
        >
          <span className="label">{c.label}</span>
          <span className="mono" style={{ fontSize: 13, color: c.color || C.ink }}>{c.value}</span>
        </div>
      ))}
    </div>
  );
}
