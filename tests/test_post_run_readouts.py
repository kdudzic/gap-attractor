import csv
import json
import re
from pathlib import Path

from hydra import compose, initialize_config_dir

from gap_attractor.harness.runner import run_experiment
from gap_attractor.providers import MockProvider
from gap_attractor.providers.base import ProviderResponse


ROOT = Path(__file__).resolve().parents[1]


def cfg(tmp_path):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=[
            "provider=mock", "wikipedia=fixture", "experiment=time_capsule", "variant=a_hub",
            "experiment.step_limit=3", "experiment.readout_selection_count=2",
            "batch.start_pages=[Civic Atlas]", f"output.directory={tmp_path.as_posix()}",
        ])


class ReadoutProvider(MockProvider):
    def chat(self, messages, **kwargs):
        system = messages[0].get("content") or ""
        if '"selections"' not in system:
            return super().chat(messages, **kwargs)
        titles = re.findall(r"^\s*-\s+(.+)$", "\n".join(m.get("content") or "" for m in messages), flags=re.M)
        count = int(re.search(r"exactly (\d+) distinct", system).group(1))
        payload = {"selections": [{"title": title, "reason": f"keep {title}"} for title in titles[:count]],
                   "selection_principle": "first titles in the controlled test"}
        return ProviderResponse(json.dumps(payload), {"prompt_tokens": 11, "completion_tokens": 7, "cost": 0.001},
                                "stop", 0, None, [], response_id=f"readout_{self.calls}", response_model=self.model)


def use_provider(monkeypatch, provider):
    import gap_attractor.harness.runner as runner_module
    monkeypatch.setattr(runner_module, "make_provider", lambda _: provider)


def run_dir(dataset):
    return next(path for path in dataset.root.iterdir() if (path / "run.json").exists())


def test_paired_readouts_share_candidates_and_seed_and_are_logged(tmp_path, monkeypatch):
    use_provider(monkeypatch, ReadoutProvider())
    dataset, _ = run_experiment(cfg(tmp_path), ROOT)
    folder = run_dir(dataset)
    doc = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    readouts = doc["readouts"]
    assert [r["id"] for r in readouts] == ["neutral_collection", "time_capsule"]
    assert all(r["valid"] and len(r["selections"]) == 2 for r in readouts)
    assert readouts[0]["candidate_titles"] == readouts[1]["candidate_titles"]
    assert readouts[0]["paired_sampling_seed"] == readouts[1]["paired_sampling_seed"]
    assert {r["order"] for r in readouts} == {1, 2}
    assert (folder / "readouts.json").exists()
    with (dataset.root / "readout_log.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and {row["Config ID"] for row in rows} == {"CFG-TCAP-DS41-NEUTRAL", "CFG-TCAP-DS41-CAPSULE"}
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["readouts_total"] == summary["readouts_valid"] == 2
    assert summary["input_tokens"] == 22 and summary["output_tokens"] == 14
    assert summary["cost_usd"] == 0.002
    with (dataset.root / "run_log.csv").open(encoding="utf-8") as handle:
        nav = next(csv.DictReader(handle))
    assert "Paired post-run readouts valid 2/2" in nav["Notes"]


def test_readout_retries_invalid_json_without_changing_candidates(tmp_path, monkeypatch):
    class RetryReadoutProvider(ReadoutProvider):
        failed = False
        def chat(self, messages, **kwargs):
            if '"selections"' in (messages[0].get("content") or "") and not self.failed:
                self.failed = True
                return ProviderResponse('{"selections": []}', {}, "stop", 0, None, [])
            return super().chat(messages, **kwargs)

    use_provider(monkeypatch, RetryReadoutProvider())
    dataset, _ = run_experiment(cfg(tmp_path), ROOT)
    readouts = json.loads((run_dir(dataset) / "readouts.json").read_text(encoding="utf-8"))["readouts"]
    retried = next(r for r in readouts if len(r["raw_responses"]) == 2)
    assert retried["valid"] and retried["error"] is None
    assert readouts[0]["candidate_titles"] == readouts[1]["candidate_titles"]
