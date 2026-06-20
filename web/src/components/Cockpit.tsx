import type { ReactNode } from "react";
import StatusBar from "./StatusBar";
import Annunciator from "./Annunciator";
import ScissorsChart from "./ScissorsChart";
import InstrumentGrid from "./InstrumentGrid";
import DetectorPanel from "./DetectorPanel";
import EventLog from "./EventLog";
import CrossoverPanel from "./CrossoverPanel";
import type { RunState } from "../lib/runState";
import { C } from "../theme";

export default function Cockpit({
  runId, mode, state, speed, top, showAnnunciator = true, demo = false,
}: {
  runId: string;
  mode: "LIVE" | "REPORT";
  state: RunState;
  speed?: number;
  top?: ReactNode;
  showAnnunciator?: boolean;
  demo?: boolean;
}) {
  return (
    <div className="space-y-5">
      <StatusBar runId={runId} mode={mode} state={state} speed={speed} />
      {showAnnunciator && <Annunciator state={state} />}
      {top}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 items-start">
        <div className="lg:col-span-2 space-y-5">
          <ScissorsChart state={state} intro={mode === "LIVE"} />
          <InstrumentGrid state={state} />
        </div>
        <div className="space-y-5">
          <DetectorPanel state={state} />
          {demo && <CrossoverPanel />}
          <EventLog state={state} />
        </div>
      </div>
      {demo && <TakeawayBanner />}
    </div>
  );
}

/** Fixed takeaway — the structural finding, shown on the demo screen. */
function TakeawayBanner() {
  return (
    <div className="section flex items-stretch overflow-hidden">
      <span style={{ width: 3, background: C.accent }} />
      <div className="px-5 py-3.5">
        <span style={{ fontSize: 14.5, color: C.ink, lineHeight: 1.5 }}>
          <strong>Geometry detects convergence, not hacking.</strong>{" "}
          <span style={{ color: C.ink2 }}>
            This is structural (Proposition 1), not a tuning bug: a hack and a benign jump onto a
            higher-reward mode are the same geometric event; the only differentiator is the held-out
            oracle, which an oracle-blind detector cannot see.
          </span>
        </span>
      </div>
    </div>
  );
}
