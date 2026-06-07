import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import Frame from "../components/Frame";
import Cockpit from "../components/Cockpit";
import Verdict from "../components/Verdict";
import { getRun } from "../api";
import { reduceAll, emptyRun, type RunState } from "../lib/runState";
import { C } from "../theme";

export default function ReportViewer() {
  const { id = "" } = useParams();
  const [state, setState] = useState<RunState>(emptyRun());
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getRun(id)
      .then((ev) => {
        const s = reduceAll(ev);
        s.ended = true;
        setState(s);
      })
      .catch((e) => setErr(String(e)));
  }, [id]);

  return (
    <Frame
      right={
        <Link to={`/live/${id}`} className="label" style={{ color: C.phos }}>
          ▶ open live
        </Link>
      }
    >
      {err ? (
        <div className="panel p-4 font-mono text-[12px]" style={{ color: C.alarm }}>
          run not found in archive
        </div>
      ) : (
        <Cockpit runId={id} mode="REPORT" state={state} top={<Verdict state={state} />} />
      )}
    </Frame>
  );
}
