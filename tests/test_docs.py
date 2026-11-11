# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""Keep the documentation honest: run its examples and check its references against the code."""
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def python_blocks(md: Path) -> list[str]:
    return [b for b in BLOCK.findall(md.read_text(encoding="utf-8")) if not b.lstrip().startswith("# sketch")]


DOC_FILES = sorted(p for p in DOCS.glob("*.md") if python_blocks(p))


@pytest.mark.parametrize("md", DOC_FILES, ids=[p.name for p in DOC_FILES])
def test_doc_examples_run(md, tmp_path, monkeypatch, capsys):
    """Every non-sketch ```python block runs, in order, in a shared namespace, in an empty folder."""
    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "__docs__"}
    for i, code in enumerate(python_blocks(md), 1):
        try:
            exec(compile(code, f"{md.name}[block {i}]", "exec"), ns)
        except Exception as e:  # pragma: no cover - failure path
            pytest.fail(f"{md.name} block {i} failed: {type(e).__name__}: {e}\n---\n{code}")


def test_tools_reference_is_current():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_tools_reference.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def _has_path(obj, parts: list[str]) -> bool:
    """True if the dotted path exists starting at any nested dict."""
    if isinstance(obj, dict):
        cur, ok = obj, True
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok:
            return True
        return any(_has_path(v, parts) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_path(v, parts) for v in obj)
    return False


def test_metrics_reference_matches_real_output(tmp_path):
    from paralingo import SpeechProfileEngine

    sr = 16000
    t = np.arange(int(2.0 * sr)) / sr
    x = np.concatenate([np.zeros(4000), 0.3 * np.sin(2 * np.pi * 170 * t), np.zeros(8000),
                        0.3 * np.sin(2 * np.pi * 180 * t), np.zeros(4000)])
    eng = SpeechProfileEngine(tmp_path, agent_id="docs-test")
    text = ("Um, I think I was, uh, charged twice this month and I want I want to understand why, "
            "because the second charge looks wrong to me and I need it fixed today please?")
    eng.analyze_utterance("u", samples=x, sample_rate=sr, transcript=text, response_latency_s=0.7)
    utt = eng.get_profile("u")["sessions"][-1]["utterances"][-1]

    md = (DOCS / "profile-format.md").read_text(encoding="utf-8")
    section = md[md.index("## One utterance"):md.index("### `skills`")]
    names = set()
    for line in section.splitlines():
        if line.startswith("| `"):
            first_cell = line.split("|")[1]
            names.update(re.findall(r"`([a-z_]+(?:\.[a-z_]+)*)`", first_cell))
    assert len(names) > 40, names
    missing = sorted(n for n in names if not _has_path(utt, n.split(".")))
    assert not missing, f"documented but not produced: {missing}"


def test_mulaw_decoder_in_docs_matches_reference():
    audioop = pytest.importorskip("audioop")  # stdlib up to Python 3.12
    code = next(b for b in python_blocks(DOCS / "integration-guide.md") if "def mulaw_to_float" in b)
    ns: dict = {}
    exec(code.split("# e.g.")[0], ns)
    data = bytes(range(256))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ref = np.frombuffer(audioop.ulaw2lin(data, 2), dtype="<i2").astype(np.float64)
    assert np.array_equal(np.round(ns["mulaw_to_float"](data) * 32768.0), ref)
