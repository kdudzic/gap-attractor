"""Resume from a perturbation page while preserving a completed prefix path.

The prefix is stored outside the continuation run and shown as assistant turns
before the continuation's own history. A matched reset condition can use the
built-in ``path_only`` memory with the same perturbation start page.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gap_attractor.harness.memory import Memory


class CheckpointPathMemory(Memory):
    """Prepend a recorded checkpoint path to this run's live path history."""

    name = "checkpoint_path"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.path = Path(str(cfg.experiment.checkpoint_history_file)).expanduser().resolve()
        self.document = json.loads(self.path.read_text(encoding="utf-8"))
        self.record_for_run = None
        self.recipient_start = None

    @classmethod
    def validate(cls, cfg) -> None:
        if str(cfg.experiment.get("action_format", "text")) != "text":
            raise ValueError("CheckpointPathMemory currently requires experiment.action_format=text")
        source = cfg.experiment.get("checkpoint_history_file")
        if not source:
            raise ValueError("CheckpointPathMemory requires experiment.checkpoint_history_file")
        path = Path(str(source)).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"checkpoint history file does not exist: {path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read checkpoint history file {path}: {error}") from error
        if document.get("schema_version") != 1 or not isinstance(document.get("mapping"), dict):
            raise ValueError("checkpoint history file must have schema_version=1 and a mapping object")
        for start, record in document["mapping"].items():
            items = record.get("items") if isinstance(record, dict) else None
            if not isinstance(items, list) or not items:
                raise ValueError(f"checkpoint history mapping for {start!r} needs at least one item")
            if any(not isinstance(item, dict) or "choice" not in item for item in items):
                raise ValueError(f"checkpoint history mapping for {start!r} has invalid items")

    def start(self, run) -> None:
        self.recipient_start = str(run.doc["start_page"])
        self.record_for_run = self.document["mapping"].get(self.recipient_start)
        if self.record_for_run is None:
            raise ValueError(f"no checkpoint history mapped for start page {self.recipient_start!r}")

    @staticmethod
    def _turn(item: dict) -> dict:
        return {
            "role": "assistant",
            "content": json.dumps({"reasoning": item.get("reasoning"), "chose": item["choice"]}),
        }

    def messages(self, rule: str, history: list[dict], observation: str) -> list[dict]:
        system = {"role": "system", "content": self.cfg.experiment.system_prompt + "\n\n" + rule}
        prefix = [self._turn(item) for item in self.record_for_run["items"]]
        live = [self._turn(item) for item in history]
        return [system] + prefix + live + [{"role": "user", "content": observation}]

    def settings(self) -> dict:
        items = self.record_for_run["items"]
        return {
            "control": "checkpoint_path",
            "history_file": str(self.path),
            "history_file_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "source_batch": self.document.get("source_batch"),
            "recipient_start_page": self.recipient_start,
            "checkpoint_turns": len(items),
            "source_run_ids": sorted({str(item.get("source_run_id") or "") for item in items}),
            "matching": "recorded prefix replies are prepended before this continuation's live path replies",
        }
