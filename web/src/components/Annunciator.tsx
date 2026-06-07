import type { RunState } from "../lib/runState";
import { C } from "../theme";

export default function Annunciator({ state }: { state: RunState }) {
  const onset = state.status === "onset";

  if (!onset) {
    return (
      <div className="panel px-5 py-3 flex items-center gap-4 rise">
        <span className="lamp" style={{ color: C.phos }} />
        <span className="font-display tracking-[0.2em] text-sm glow-phos" style={{ color: C.phos }}>
          NO FAULT
        </span>
        <span className="label">rollout geometry within contraction tube · detector armed</span>
        <span className="ml-auto label" style={{ color: C.faint }}>
          watching kl accel · entropy collapse · advantage drift
        </span>
      </div>
    );
  }

  const lead = state.lead;
  return (
    <div
      className="px-5 py-3.5 rounded relative overflow-hidden animate-alarmpulse rise"
      style={{ background: "linear-gradient(180deg,#1b060a,#100406)", border: `1px solid ${C.alarm}` }}
    >
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ background: "repeating-linear-gradient(45deg, rgba(255,59,71,.06) 0 11px, transparent 11px 22px)" }}
      />
      <div className="relative flex items-center gap-4 flex-wrap">
        <span className="lamp animate-blink" style={{ color: C.alarm }} />
        <span className="font-display tracking-[0.22em] text-lg glow-alarm" style={{ color: C.alarm }}>
          ▲ ONSET DETECTED
        </span>
        <span className="font-mono text-sm" style={{ color: C.text }}>
          step <b style={{ color: C.alarm }}>{state.onsetStep}</b>
        </span>
        {lead != null && (
          <span
            className="ml-auto font-mono text-[13px] px-3 py-1 rounded"
            style={{
              color: lead >= 0 ? C.phos : C.alarm,
              border: `1px solid ${lead >= 0 ? C.phos : C.alarm}`,
              boxShadow: lead >= 0 ? "0 0 18px -6px #4fe0a8" : undefined,
            }}
          >
            {lead >= 0
              ? `▸ FIRED ${lead} STEPS BEFORE ORACLE-GAP TURN`
              : `LAGGED ${-lead} STEPS`}
          </span>
        )}
      </div>
      {state.onsetExplanation && (
        <div className="relative font-mono text-[11px] mt-2" style={{ color: C.dim }}>
          {state.onsetExplanation}
        </div>
      )}
    </div>
  );
}
