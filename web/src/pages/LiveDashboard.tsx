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
    <div className="flex items-center gap-3">
      <span className="label">Replay</span>
      <div className="flex items-center" style={{ border: `1px solid ${C.rule}` }}>
        {[12, 24, 60, 240].map((s, i) => (
          <button
            key={s}
            onClick={() => setSpeed(s)}
            className="mono px-2.5 py-1"
            style={{
              fontSize: 12,
              color: s === speed ? C.ink : C.faint,
              background: s === speed ? "#f1efe8" : "transparent",
              borderLeft: i === 0 ? "none" : `1px solid ${C.rule}`,
            }}
          >
            {s}/s
          </button>
        ))}
      </div>
    </div>
  );
}
