import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import Frame from "../components/Frame";
import Cockpit from "../components/Cockpit";
import { liveSocket, type RunEvent } from "../api";
import { reduceAll, emptyRun, type RunState } from "../lib/runState";
import { C } from "../theme";

export default function LiveDashboard() {
  const { id = "" } = useParams();
  const [speed, setSpeed] = useState(24);
  const [state, setState] = useState<RunState>(emptyRun());
  const eventsRef = useRef<RunEvent[]>([]);
  const dirty = useRef(false);

  useEffect(() => {
    eventsRef.current = [];
    dirty.current = false;
    setState(emptyRun());
    const ws = liveSocket(id, speed, (e) => {
      eventsRef.current.push(e);
      dirty.current = true;
    });
    let raf = 0;
    const tick = () => {
      if (dirty.current) {
        dirty.current = false;
        setState(reduceAll(eventsRef.current));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      ws.close();
    };
  }, [id, speed]);

  return (
    <Frame right={<SpeedCtl speed={speed} setSpeed={setSpeed} />}>
      <Cockpit runId={id} mode="LIVE" state={state} speed={speed} />
    </Frame>
  );
}

function SpeedCtl({ speed, setSpeed }: { speed: number; setSpeed: (n: number) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="label">replay rate</span>
      {[12, 24, 60, 240].map((s) => (
        <button
          key={s}
          onClick={() => setSpeed(s)}
          className="font-mono text-[12px] px-2 py-1 rounded"
          style={{ color: s === speed ? C.phos : C.dim, border: `1px solid ${s === speed ? C.phos : C.line}` }}
        >
          {s}/s
        </button>
      ))}
    </div>
  );
}
