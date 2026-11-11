"""Dev helper: ask ad-hoc questions.   python submission/ask.py "question" ["question 2" ...]  [--asof 2026-05-31]
Questions may carry an as_of date as  "question @@2026-05-31"."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import run_eval  # noqa: E402

snaps = run_eval.load_snapshots(ROOT / "snapshots")
ks = run_eval.load_submission(ROOT / "submission")
for s in snaps:
    ks.ingest(s)
width = 600
for q in sys.argv[1:]:
    as_of = None
    if "@@" in q:
        q, as_of = q.split("@@")
        q, as_of = q.strip(), as_of.strip()
    t = time.time()
    a = run_eval.to_answer(ks.answer(q, as_of))
    print("=" * 90)
    print("Q:", q, "| as_of:", as_of, f"| {time.time() - t:.2f}s")
    print("A:", a.text[:width].replace("\n", " / "), "| conf:", a.confident)
    print("S:", [s.url.replace("https://www.dupaco.com", "") for s in a.sources][:4])
