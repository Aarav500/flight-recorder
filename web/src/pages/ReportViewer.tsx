import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import Frame from "../components/Frame";
import Cockpit from "../components/Cockpit";
import Abstract from "../components/Abstract";
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
        <Link to={`/live/${id}`} className="label" style={{ color: C.ink }}>
          Open live →
        </Link>
      }
    >
      {err ? (
        <div className="section p-5" style={{ fontSize: 13, color: C.ink2 }}>
          Run not found in archive.
        </div>
      ) : (
        <Cockpit
          runId={id}
          mode="REPORT"
          state={state}
          showAnnunciator={false}
          top={
            <>
              <Abstract state={state} />
              <Verdict state={state} />
            </>
          }
        />
      )}
    </Frame>
  );
}
