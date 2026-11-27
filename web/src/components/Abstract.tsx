import { abstract } from "../copy";
import type { RunState } from "../lib/runState";
import { last } from "../theme";

export default function Abstract({ state }: { state: RunState }) {
  const text = abstract({
    steps: state.lastStep + 1,
    onset: state.onsetStep,
    turn: state.oracleTurn,
    lead: state.lead,
    trainLast: last(state.train),
    oracleLast: last(state.oracle),
  });
  return (
    <div className="section p-6 rise">
      <span className="label">Abstract</span>
      <p className="prose mt-2.5 max-w-[860px]">{text}</p>
    </div>
  );
}
