import Instrument from "./Instrument";
import type { RunState } from "../lib/runState";
import { C } from "../theme";

export default function DetectorPanel({ state }: { state: RunState }) {
  const onset = state.status === "onset";
  return (
    <div className="space-y-3">
      <div className="label" style={{ color: C.dim }}>
        contraction-tube detector
      </div>
      <Instrument label="tube distance (escape)" color={C.phos} steps={state.steps}
                  ys={state.tube} onsetStep={state.onsetStep} height={84} />
      <Instrument label="cusum statistic" color={C.amber} steps={state.steps}
                  ys={state.cusum} onsetStep={state.onsetStep} height={84} />
      <div className="panel px-3 py-2.5 flex items-start gap-2.5">
        <span className="lamp mt-1" style={{ color: onset ? C.alarm : C.phos }} />
        <div className="font-mono text-[11px] leading-relaxed" style={{ color: C.dim }}>
          rule: <span style={{ color: C.text }}>tube-exit ∧ cusum</span> sustained (2-of-N)
          {onset ? (
            <>
              {" "}→{" "}
              <span style={{ color: C.alarm }}>ONSET @ step {state.onsetStep}</span>
            </>
          ) : (
            <span style={{ color: C.faint }}> · armed, no fault</span>
          )}
        </div>
      </div>
    </div>
  );
}
