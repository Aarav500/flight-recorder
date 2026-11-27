"""Writes jobs_phi.txt. Seeds 0-9 repeat study L so the readout's no-effect-on-training claim can be
checked on the exploit and control arms. Confirmatory runs must use fresh seeds."""
lines = ["# THROWAWAY SPIKE. N=4 first: it is the only team size where true reward falls in study L."]
for n in (4, 2, 8):
    for s in range(10):
        for cond in ("exploit", "patched", "control"):
            lines.append(f"--n {n} --cond {cond} --seed {s} --params shared")
lines.append("# Does the proposal's own setting (separate networks, same init) exploit the proxy at all?")
for s in range(3):
    for cond in ("exploit", "patched"):
        lines.append(f"--n 4 --cond {cond} --seed {s} --params indep --common-init")
open("jobs_phi.txt", "w", newline="\n").write("\n".join(lines) + "\n")
print(len([l for l in lines if not l.startswith("#")]), "jobs")
