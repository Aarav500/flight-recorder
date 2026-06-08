import type { ReactNode } from "react";
import StatusBar from "./StatusBar";
import Annunciator from "./Annunciator";
import ScissorsChart from "./ScissorsChart";
import InstrumentGrid from "./InstrumentGrid";
import DetectorPanel from "./DetectorPanel";
import EventLog from "./EventLog";
import type { RunState } from "../lib/runState";

export default function Cockpit({
  runId, mode, state, speed, top, showAnnunciator = true,
}: {
  runId: string;
  mode: "LIVE" | "REPORT";
  state: RunState;
  speed?: number;
  top?: ReactNode;
  showAnnunciator?: boolean;
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
          <EventLog state={state} />
        </div>
      </div>
    </div>
  );
}
