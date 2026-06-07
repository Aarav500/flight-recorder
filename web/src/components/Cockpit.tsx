import type { ReactNode } from "react";
import StatusBar from "./StatusBar";
import Annunciator from "./Annunciator";
import ScissorsChart from "./ScissorsChart";
import InstrumentGrid from "./InstrumentGrid";
import DetectorPanel from "./DetectorPanel";
import EventLog from "./EventLog";
import type { RunState } from "../lib/runState";

export default function Cockpit({
  runId, mode, state, speed, top,
}: {
  runId: string;
  mode: "LIVE" | "REPORT";
  state: RunState;
  speed?: number;
  top?: ReactNode;
}) {
  return (
    <div className="space-y-4">
      <StatusBar runId={runId} mode={mode} state={state} speed={speed} />
      <Annunciator state={state} />
      {top}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 space-y-4">
          <ScissorsChart state={state} />
          <InstrumentGrid state={state} />
        </div>
        <div className="space-y-4">
          <DetectorPanel state={state} />
          <EventLog state={state} />
        </div>
      </div>
    </div>
  );
}
