import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Frame from "../components/Frame";
import { listRuns, type RunSummary } from "../api";
import { C } from "../theme";

const COLS = "minmax(0,1.4fr) 90px 70px 80px 110px 90px 120px";

export default function RunList() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    listRuns().then(setRuns).catch((e) => setErr(String(e)));
  }, []);

  return (
    <Frame>
      <p className="prose max-w-[800px] mb-7">
        Flight Recorder records the geometry of GRPO training runs and flags the onset of
        reward hacking from rollout statistics alone — before a held-out oracle reveals it.
        Each row below is a recorded run. Open one to read the detector's finding.
      </p>

      {err && (
        <div className="section p-4 mb-5" style={{ fontSize: 13, color: C.ink2 }}>
          Backend unreachable. Start it with{" "}
          <span className="mono" style={{ color: C.accent }}>flr serve --runs runs</span>.
        </div>
      )}

      {runs && runs.length === 0 && <Empty />}

      {runs && runs.length > 0 && (
        <div className="section">
          <div
            className="grid items-center px-5 py-2.5 rule-b"
            style={{ gridTemplateColumns: COLS, columnGap: 16 }}
          >
            {["Run", "Status", "Steps", "Onset", "Oracle turn", "Lead", ""].map((h) => (
              <span key={h} className="label">{h}</span>
            ))}
          </div>
          {runs.map((r, i) => (
            <Row key={r.id} r={r} first={i === 0} />
          ))}
        </div>
      )}
    </Frame>
  );
}

function Row({ r, first }: { r: RunSummary; first: boolean }) {
  const onset = r.status === "onset";
  const led = r.lead != null && r.lead > 0;
  return (
    <div
      className={`grid items-center px-5 py-3.5 ${first ? "" : "rule-t"}`}
      style={{ gridTemplateColumns: COLS, columnGap: 16 }}
    >
      <Link to={`/report/${r.id}`} className="mono truncate" style={{ fontSize: 13, color: C.ink }}>
        {r.id}
      </Link>
      <span className="label" style={{ color: onset ? C.accent : C.muted, fontWeight: 600 }}>
        {onset ? "Onset" : "Clean"}
      </span>
      <span className="mono" style={{ fontSize: 12.5, color: C.ink2 }}>{r.steps}</span>
      <span className="mono" style={{ fontSize: 12.5, color: onset ? C.accent : C.faint }}>
        {r.onset_step != null ? r.onset_step : "—"}
      </span>
      <span className="mono" style={{ fontSize: 12.5, color: C.ink2 }}>
        {r.oracle_turn != null ? r.oracle_turn : "—"}
      </span>
      <span className="mono" style={{ fontSize: 12.5, color: led ? C.accent : C.ink2 }}>
        {r.lead != null ? `${r.lead > 0 ? "+" : ""}${r.lead}` : "—"}
      </span>
      <span className="flex gap-4 justify-end">
        <Link to={`/report/${r.id}`} className="label" style={{ color: C.ink }}>Report</Link>
        <Link to={`/live/${r.id}`} className="label" style={{ color: C.muted }}>Live</Link>
      </span>
    </div>
  );
}

function Empty() {
  return (
    <div className="section p-6">
      <h2 style={{ fontSize: 15, fontWeight: 600, color: C.ink }}>No runs on record</h2>
      <p className="mt-2 mb-3" style={{ fontSize: 13, color: C.muted, maxWidth: 620 }}>
        Generate a synthetic run, then point the server at its directory:
      </p>
      <pre
        className="mono p-4"
        style={{ fontSize: 12, background: "#f4f2ec", border: `1px solid ${C.rule}`, color: C.ink2, overflow: "auto" }}
      >{`flr synth --seed 1 --steps 200 --tstar 100 --out runs/demo.jsonl
flr serve --runs runs`}</pre>
    </div>
  );
}
