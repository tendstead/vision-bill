"""Test-owned source/photo plans and observable barriers shared by CPU stubs."""

import copy
import threading
from typing import Any


class Scenario:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.requests: list[dict[str, Any]] = []
        self.plans: dict[tuple[str, int], dict[str, Any]] = {}
        self.barriers: dict[str, threading.Event] = {}

    def configure(self, body: dict[str, Any]) -> None:
        plans = {}
        for plan in body.get("plans", []):
            key = (plan["source_sha256"], plan["photo_index"])
            if key in plans or plan["result"] not in ("success", "empty", "503", "timeout"):
                raise ValueError("invalid or duplicate source/photo plan")
            plans[key] = dict(plan)
        with self.lock:
            for event in self.barriers.values():
                event.set()
            self.plans = plans
            self.requests.clear()
            self.barriers = {name: threading.Event() for name in body.get("barriers", [])}

    def plan(self, digest: str, index: int) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.plans.get((digest, index), {"result": "success"}))

    def record(self, entry: dict[str, Any]) -> None:
        with self.lock:
            self.requests.append(entry)

    def wait(self, name: str, timeout: float) -> bool:
        with self.lock:
            event = self.barriers[name]
        return event.wait(timeout)

    def release(self, name: str) -> None:
        with self.lock:
            self.barriers[name].set()

    def audit(self) -> list[dict[str, Any]]:
        with self.lock:
            return copy.deepcopy(self.requests)
