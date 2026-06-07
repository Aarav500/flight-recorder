import json
from flightrecorder.cli import main


def test_synth_writes_artifact(tmp_path):
    out = tmp_path / "run.jsonl"
    rc = main(["synth", "--seed", "1", "--steps", "120", "--tstar", "60", "--out", str(out)])
    assert rc == 0 and out.exists()
    kinds = {json.loads(l)["kind"] for l in out.read_text().splitlines()}
    assert {"frame", "oracle", "detector"} <= kinds


def test_eval_prints_summary(tmp_path, capsys):
    rc = main(["eval", "--seeds", "4", "--steps", "150", "--tstar", "80"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mean_lead" in out and "fpr" in out
