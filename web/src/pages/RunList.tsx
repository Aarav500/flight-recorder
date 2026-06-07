import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Frame from "../components/Frame";
import { listRuns, type RunSummary } from "../api";
import { C } from "../theme";

export default function RunList() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    listRuns().then(setRuns).catch((e) => setErr(String(e)));
  }, []);

  return (
    <Frame>
      <div className="label mb-3" style={{ color: C.dim }}>
        recorded runs · black-box archive
      </div>
      {err && (
        <div className="panel p-4 mb-4 font-mono text-[12px]" style={{ color: C.amber }}>
          backend unreachable — start it with{" "}
          <span style={{ color: C.phos }}>flr serve --runs runs</span>
        </div>
      )}
      {runs && runs.length === 0 && <EmptyState />}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {runs?.map((r) => (
          <RunCard key={r.id} r={r} />
        ))}
      </div>
    </Frame>
  );
}

function RunCard({ r }: { r: RunSummary }) {
  const onset = r.status === "onset";
  return (
    <div className="panel p-4 rise flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="font-mono text-sm" style={{ color: C.text }}>
          {r.id}
        </span>
        <span className="flex items-center gap-2">
          <span className="lamp" style={{ color: onset ? C.alarm : C.phos }} />
          <span className="label" style={{ color: onset ? C.alarm : C.phos }}>
            {onset ? "ONSET" : "CLEAN"}
          </span>
        </span>
      </div>
      <div className="grid grid-cols-3 gap-2 font-mono text-[12px]">
        <KV k="steps" v={String(r.steps)} />
        <KV k="onset" v={r.onset_step != null ? String(r.onset_step) : "—"}
            color={onset ? C.alarm : undefined} />
        <KV k="lead" v={r.lead != null ? `${r.lead >= 0 ? "+" : ""}${r.lead}` : "—"}
            color={r.lead != null && r.lead > 0 ? C.phos : undefined} />
      </div>
      <div className="flex gap-2 mt-1">
        <Link to={`/live/${r.id}`}
          className="flex-1 text-center py-1.5 rounded font-display tracking-[0.16em] text-[12px]"
          style={{ color: C.phos, border: `1px solid ${C.line}` }}>
          ▶ LIVE
        </Link>
        <Link to={`/report/${r.id}`}
          className="flex-1 text-center py-1.5 rounded font-display tracking-[0.16em] text-[12px]"
          style={{ color: C.cyan, border: `1px solid ${C.line}` }}>
          ◷ REPORT
        </Link>
      </div>
    </div>
  );
}

function KV({ k, v, color }: { k: string; v: string; color?: string }) {
  return (
    <div className="flex flex-col">
      <span className="label">{k}</span>
      <span style={{ color: color || C.text }}>{v}</span>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="panel p-6 rise font-mono text-[12px] leading-relaxed" style={{ color: C.dim }}>
      <div className="font-display tracking-[0.18em] text-base mb-3" style={{ color: C.text }}>
        NO RUNS ON RECORD
      </div>
      generate one, then point the server at it:
      <pre className="mt-3 p-3 rounded overflow-auto"
        style={{ background: "#0a0f10", border: `1px solid ${C.line}`, color: C.phos }}>
{`flr synth --seed 1 --steps 200 --tstar 100 --out runs/demo.jsonl
flr serve --runs runs`}
      </pre>
    </div>
  );
}
