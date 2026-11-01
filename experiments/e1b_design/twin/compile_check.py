"""Does compile() reject anything ast.parse accepts among E1 completions that pass G_parse?"""
import ast, json, sys, warnings
from pathlib import Path
REPO = Path(r"<home>/OneDrive/Desktop/Fligh recorder")
sys.path.insert(0, str(REPO))
from flightrecorder.repro.reward_testhack import extract_code
from twin_gate import gate
warnings.simplefilter("ignore")
n = diff = 0; ex = []
for fp in sorted((REPO / "runs/e1/main").glob("judge_judge*_s2*_lr1e-6.json")):
    d = json.loads(fp.read_text())
    for c in d["completions"]:
        for i, t in enumerate(c["texts"]):
            if gate(t, d["task"]) == 1.0:
                n += 1
                try:
                    compile(extract_code(t), "submission.py", "exec")
                except Exception as e:
                    diff += 1; ex.append((fp.stem, c["step"], i, c["hidden"][i], repr(e)[:80]))
print("G_parse passes:", n, "of which compile() rejects:", diff)
for e in ex[:10]: print(e)
