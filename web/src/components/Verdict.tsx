import { Stat } from "./bits";
import { verdictExplanation } from "../copy";
import type { RunState } from "../lib/runState";
import { C } from "../theme";

export default function Verdict({ state }: { state: RunState }) {
  const lead = state.lead;
  const led = lead != null && lead > 0;
  const onset = state.status === "onset";
  return (
    <div className="section p-6 rise">
      <div className="flex items-start gap-x-12 gap-y-6 flex-wrap">
        <div>
          <span className="label">Lead time</span>
          <div className="flex items-baseline gap-2 mt-1.5">
            <span
              className="mono"
              style={{ fontSize: 46, fontWeight: 500, lineHeight: 1, color: led ? C.accent : C.ink }}
            >
              {lead != null ? `${lead > 0 ? "+" : ""}${lead}` : "—"}
            </span>
            <span className="label" style={{ marginBottom: 5 }}>steps</span>
          </div>
          <span className="label mt-2.5 inline-block" style={{ fontWeight: 500, color: C.faint }}>
            geometry onset → oracle-gap turn
          </span>
        </div>
        <div className="flex gap-x-10 gap-y-4 flex-wrap pt-0.5">
          <Stat label="Onset step" value={state.onsetStep != null ? String(state.onsetStep) : "—"}
                size="lg" color={led ? C.accent : C.ink} />
          <Stat label="Oracle-gap turn" value={state.oracleTurn != null ? String(state.oracleTurn) : "—"} size="lg" />
          <Stat label="Verdict" value={onset ? "Hack onset" : "Clean"}
                size="lg" color={onset ? C.accent : C.ink} />
        </div>
      </div>
      <p className="prose mt-6 max-w-[780px]">{verdictExplanation(lead)}</p>
    </div>
  );
}
