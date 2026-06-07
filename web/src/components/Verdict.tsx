import { Stat } from "./bits";
import type { RunState } from "../lib/runState";
import { C } from "../theme";

export default function Verdict({ state }: { state: RunState }) {
  const lead = state.lead;
  const led = lead != null && lead > 0;
  const onset = state.status === "onset";
  return (
    <div className="panel p-4 rise" style={{ borderColor: onset ? C.alarm : C.line }}>
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-x-10 gap-y-3 flex-wrap">
          <Stat k="onset step" v={state.onsetStep != null ? String(state.onsetStep) : "—"}
                color={C.alarm} big />
          <Stat k="oracle-gap turn" v={state.oracleTurn != null ? String(state.oracleTurn) : "—"}
                color={C.cyan} big />
          <Stat k="lead" v={lead != null ? `${lead >= 0 ? "+" : ""}${lead}` : "—"}
                color={led ? C.phos : C.dim} big />
          <Stat k="verdict" v={onset ? "HACK ONSET" : "CLEAN"}
                color={onset ? C.alarm : C.phos} big />
        </div>
        {led && (
          <div
            className="font-display tracking-[0.16em] text-sm px-4 py-2 rounded glow-phos"
            style={{ color: C.phos, border: `1px solid ${C.phos}` }}
          >
            GEOMETRY LED THE ORACLE GAP BY {lead} STEPS
          </div>
        )}
      </div>
    </div>
  );
}
