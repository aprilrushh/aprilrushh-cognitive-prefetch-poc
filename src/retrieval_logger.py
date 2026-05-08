"""
Retrieval Logger (Brain Bet Ledger v0.2 - 2M lens)
====================================================
모든 prefetch 의 모든 step 을 영구 logging.
PoC v2 의 모든 후속 실험의 foundation.
"""

from __future__ import annotations
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class RetrievalEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    layer_idx: int = -1
    token_position: int = -1
    seq_length: int = -1
    similarity_fn: str = "dot"
    separation_fn: str = "softmax"
    beta: float = 1.0
    alpha: float = 1.5
    top_k: int = 128
    sim_max: float = 0.0
    sim_mean: float = 0.0
    sim_std: float = 0.0
    top1_score: float = 0.0
    top1_top2_gap: float = 0.0
    sparsity: float = 0.0
    entropy: float = 0.0
    predicted_top_k_indices: list = field(default_factory=list)
    actual_top_k_indices: list = field(default_factory=list)
    hit_rate: Optional[float] = None
    recall_at_k: Optional[float] = None
    multi_update_iterations: int = 1
    multi_update_triggered: bool = False
    similarity_time_ms: float = 0.0
    separation_time_ms: float = 0.0
    projection_time_ms: float = 0.0
    total_time_ms: float = 0.0
    prefetch_bytes: int = 0
    prefetch_time_ms: float = 0.0
    prefetch_hit: Optional[bool] = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


class RetrievalLogger:
    def __init__(self, log_dir: str = "logs",
                 session_name: str = "default", fsync: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{session_name}_{timestamp}_{str(uuid.uuid4())[:6]}"
        self.log_path = self.log_dir / f"{self.session_id}.jsonl"
        self.fsync = fsync
        self.event_count = 0
        self._file = open(self.log_path, "a", encoding="utf-8")
        self._write_header()

    def _write_header(self):
        header = {
            "type": "session_start",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
        }
        self._file.write(json.dumps(header) + "\n")
        if self.fsync:
            self._file.flush()
            os.fsync(self._file.fileno())

    def log(self, event: RetrievalEvent):
        event.session_id = self.session_id
        event_dict = asdict(event)
        event_dict["type"] = "retrieval_event"
        self._file.write(json.dumps(event_dict) + "\n")
        self.event_count += 1
        if self.fsync and self.event_count % 10 == 0:
            self._file.flush()
            os.fsync(self._file.fileno())

    def log_dict(self, **kwargs):
        event = RetrievalEvent(**kwargs)
        self.log(event)

    def close(self):
        footer = {
            "type": "session_end",
            "session_id": self.session_id,
            "total_events": self.event_count,
            "timestamp": datetime.now().isoformat(),
        }
        self._file.write(json.dumps(footer) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    class Timer:
        def __init__(self):
            self.start_ns = 0
            self.elapsed_ms = 0.0

        def __enter__(self):
            self.start_ns = time.perf_counter_ns()
            return self

        def __exit__(self, *args):
            elapsed_ns = time.perf_counter_ns() - self.start_ns
            self.elapsed_ms = elapsed_ns / 1e6

    @staticmethod
    def timer():
        return RetrievalLogger.Timer()

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas needed")
        self._file.flush()
        events = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("type") == "retrieval_event":
                    events.append(obj)
        return pd.DataFrame(events)

    @classmethod
    def load(cls, log_path):
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas needed")
        events = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("type") == "retrieval_event":
                    events.append(obj)
        return pd.DataFrame(events)


if __name__ == "__main__":
    import tempfile
    print("=" * 60)
    print("Retrieval Logger smoke test")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as tmp:
        with RetrievalLogger(log_dir=tmp, session_name="smoke") as logger:
            for i in range(50):
                with logger.timer() as t:
                    time.sleep(0.0001)
                event = RetrievalEvent(
                    layer_idx=i % 80,
                    seq_length=4096 + i,
                    separation_fn="entmax",
                    beta=1.0 + (i % 5) * 0.5,
                    top1_score=0.85 + (i % 10) * 0.01,
                    sparsity=0.002,
                    total_time_ms=t.elapsed_ms,
                )
                logger.log(event)
            print(f"  Events logged: {logger.event_count}")
            df = logger.to_dataframe()
            print(f"  DataFrame shape: {df.shape}")
            print(f"  Columns: {len(df.columns)}")
    print("✓ Done")
