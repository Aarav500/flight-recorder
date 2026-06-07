import type { RunState } from "../lib/runState";
import { C } from "../theme";

function Field({ k, v, color, mono }: { k: string; v: string; color?: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="label">{k}</span>
      <span className={mono ? "font-mono text-sm" : "text-sm"} style={{ color: color || C.text }}>
        {v}
      </span>
    </div>
  );
}

export default function StatusBar({
  runId, mode, state, speed,
}: { runId: string; mode: "LIVE" | "REPORT"; state: RunState; speed?: number }) {
  const onset = state.status === "onset";
  return (
    <div className="panel px-4 py-2.5 flex items-center gap-x-6 gap-y-2 flex-wrap text-[12px]">
      <Field k="RUN" v={runId} color={C.text} />
      <Field k="MODE" v={mode} color={mode === "LIVE" ? C.phos : C.cyan} />
      <Field k="STEP" v={String(state.lastStep).padStart(4, "0")} color={C.text} mono />
      <div className="flex items-center gap-2">
        <span className="lamp" style={{ color: onset ? C.alarm : C.phos }} />
        <span
          className={`font-display tracking-[0.2em] ${onset ? "glow-alarm animate-blink" : "glow-phos"}`}
          style={{ color: onset ? C.alarm : C.phos }}
        >
          {onset ? "ONSET" : "NOMINAL"}
        </span>
      </div>
      {mode === "LIVE" && <Field k="RATE" v={`${speed ?? 20}/s`} color={C.dim} mono />}
      <div className="ml-auto flex items-center gap-2">
        <span
          className={`lamp ${state.ended ? "" : "animate-blink"}`}
          style={{ color: state.ended ? C.faint : C.amber }}
        />
        <span className="label">{state.ended ? "REC ENDED" : mode === "LIVE" ? "RECORDING" : "ARCHIVE"}</span>
      </div>
    </div>
  );
}
