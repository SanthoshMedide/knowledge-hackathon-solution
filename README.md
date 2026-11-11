# Knowledge system that knows what's true *now*

Hackathon submission: a rules + retrieval system (Python standard library only, no API keys) that reads dated
website snapshots and answers with the current, correctly-tiered value, a source, and an honest "I don't know".

* `submission/solution.py` - the `build()` entry point (`ingest`, `answer`, `conflicts`)
* `submission/NOTES.md` - approach, the written-down "what is current" rules, known limits
* `tools/` - dev helpers (`review.py`, `ask.py`); they expect the hackathon kit layout (`harness/`, `snapshots/`, `eval/`)

Run with the kit's scorer from the kit's top folder:

    python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl

The website snapshots are not included (shared for the event only).
