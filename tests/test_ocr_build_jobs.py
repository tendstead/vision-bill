"""The ROCm builder sizes both native builds from its own Linux limits."""

import os
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "paddleocr-service/build/build_jobs.py"


def test_v2_budget_observes_affinity_parent_quota_and_remaining_memory(tmp_path, monkeypatch):
    resolve = runpy.run_path(str(SCRIPT))["build_jobs"]
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(6)))
    proc = tmp_path / "proc"
    groups = tmp_path / "groups"
    (proc / "self").mkdir(parents=True)
    leaf = groups / "builder" / "worker"
    leaf.mkdir(parents=True)
    (proc / "meminfo").write_text(f"MemAvailable: {24 * 1024**2} kB\n")
    (proc / "self/cgroup").write_text("0::/builder/worker\n")
    (leaf / "cpu.max").write_text("max 100000")
    (leaf.parent / "cpu.max").write_text("390000 100000")
    assert resolve("auto", proc, groups) == 3
    gib = 1024**3
    (groups / "memory.max").write_text(str(12 * gib))
    (groups / "memory.current").write_text(str(4 * gib))
    assert resolve("auto", proc, groups) == 2
    (groups / "memory.current").write_text(str(12 * gib))
    assert resolve("auto", proc, groups) == 1
    assert resolve("12", proc, groups) == 12


def test_v1_budget_and_unlimited_fallback(tmp_path, monkeypatch):
    resolve = runpy.run_path(str(SCRIPT))["build_jobs"]
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1, 2})
    proc = tmp_path / "proc"
    groups = tmp_path / "groups"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text(f"MemAvailable: {40 * 1024**2} kB\n")
    (proc / "self/cgroup").write_text("1:cpu,cpuacct:/\n2:memory:/\n")
    cpu = groups / "cpu,cpuacct"
    cpu.mkdir(parents=True)
    (cpu / "cpu.cfs_quota_us").write_text("-1")
    (cpu / "cpu.cfs_period_us").write_text("100000")
    assert resolve("auto", proc, groups) == 3
    (cpu / "cpu.cfs_quota_us").write_text("200000")
    assert resolve("auto", proc, groups) == 2
    memory = groups / "memory"
    memory.mkdir()
    (memory / "memory.limit_in_bytes").write_text(str(6 * 1024**3))
    (memory / "memory.usage_in_bytes").write_text(str(1024**3))
    assert resolve("auto", proc, groups) == 1
    for invalid in ("0", "-1", "2.5", "unknown"):
        with pytest.raises(ValueError):
            resolve(invalid, proc, groups)
