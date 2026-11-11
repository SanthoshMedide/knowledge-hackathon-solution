"""Dev helper: run the public eval and print every answer next to what was expected.

    python submission/review.py [--only dup-009,dup-016] [--misses]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import run_eval  # noqa: E402
from scoring import score_item  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--misses", action="store_true")
ap.add_argument("--eval", default=str(ROOT / "eval" / "public_eval.jsonl"))
ap.add_argument("--width", type=int, default=700)
args = ap.parse_args()

items = [json.loads(l) for l in Path(args.eval).read_text().splitlines() if l.strip()]
if args.only:
    want = set(args.only.split(","))
    items = [i for i in items if i["id"] in want]

snaps = run_eval.load_snapshots(ROOT / "snapshots")
ks = run_eval.load_submission(ROOT / "submission")
t0 = time.time()
for s in snaps:
    ks.ingest(s)
print(f"ingest {time.time() - t0:.1f}s")

tot = wsum = 0
for it in items:
    t = time.time()
    a = run_eval.to_answer(ks.answer(it["question"], it.get("as_of")))
    dt = time.time() - t
    r = score_item(it, a)
    tot += r["score"] * it.get("weight", 1)
    wsum += it.get("weight", 1)
    if args.misses and r["score"] >= 0.999:
        continue
    print("=" * 100)
    print(f"{it['id']} [{it['category']}] score={r['score']} corr={r['correctness']} src={r['source_hit']} {dt:.2f}s")
    print("Q:", it["question"], "| as_of:", it.get("as_of"))
    print("EXPECT:", it["expected_answer"][:200], "| must:", it["must_include"], "| not:", it["must_not_include"])
    print("MISSED:", r["missed"], "TRAP:", r["trap_hits"])
    print("A:", (a.text or "")[: args.width].replace("\n", " / "), "| conf:", a.confident)
    print("S:", [s.url.replace("https://www.dupaco.com", "") for s in a.sources][:5])
print("\nQA weighted:", round(tot / wsum, 4))
