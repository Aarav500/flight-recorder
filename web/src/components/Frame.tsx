import { Link } from "react-router-dom";
import type { ReactNode } from "react";

export default function Frame({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="min-h-full px-4 sm:px-6 lg:px-8 py-5 max-w-[1440px] mx-auto">
      <header className="flex items-center justify-between mb-5">
        <Link to="/" className="flex items-center gap-3">
          <span className="inline-block w-2.5 h-7 bg-phos rounded-[1px] animate-flicker"
                style={{ boxShadow: "0 0 14px #4fe0a8" }} />
          <div className="leading-none">
            <div className="font-display text-text tracking-[0.22em] text-lg glow-phos">
              FLIGHT&nbsp;RECORDER
            </div>
            <div className="label mt-1.5">reward-hacking onset · black box</div>
          </div>
        </Link>
        <div className="flex items-center gap-4">{right}</div>
      </header>
      {children}
      <footer className="label mt-8 flex items-center justify-between opacity-70">
        <span>oracle-blind detector · rollout geometry only</span>
        <span>v0.1 · apache-2.0</span>
      </footer>
    </div>
  );
}
