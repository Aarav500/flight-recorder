import { Link } from "react-router-dom";
import type { ReactNode } from "react";
import { TAGLINE } from "../copy";
import { C } from "../theme";

export default function Frame({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="min-h-full flex flex-col">
      <header className="rule-b">
        <div className="max-w-[1180px] mx-auto w-full px-6 py-5 flex items-end justify-between gap-4">
          <Link to="/" className="block">
            <div className="flex items-center gap-2.5">
              <span className="tick" style={{ height: 17, display: "inline-block" }} />
              <h1 style={{ fontSize: 19, fontWeight: 600, letterSpacing: "-0.012em", color: C.ink }}>
                Flight Recorder
              </h1>
            </div>
            <p className="mt-1.5" style={{ marginLeft: 19, color: C.muted, fontSize: 12.5 }}>
              {TAGLINE}
            </p>
          </Link>
          <div className="flex items-center gap-4 pb-0.5">{right}</div>
        </div>
      </header>
      <main className="max-w-[1180px] mx-auto w-full px-6 py-7 flex-1">{children}</main>
      <footer className="rule-t">
        <div className="max-w-[1180px] mx-auto w-full px-6 py-4 flex items-center justify-between">
          <span className="label" style={{ fontWeight: 500 }}>Oracle-blind detector · rollout geometry only</span>
          <span className="mono" style={{ fontSize: 11, color: C.faint }}>v0.1 · Apache-2.0</span>
        </div>
      </footer>
    </div>
  );
}
