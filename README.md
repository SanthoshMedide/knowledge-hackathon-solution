# Knowledge system that knows what's true *now*

Hackathon submission: a rules + retrieval system (Python standard library only, no API keys) that reads dated
website snapshots and answers with the current, correctly-tiered value, a source, and an honest "I don't know".

* `submission/solution.py` - the `build()` entry point (`ingest`, `answer`, `conflicts`)
* `submission/NOTES.md` - approach, the written-down "what is current" rules, API keys (none), known limits
* `harness/`, `eval/` - the kit's scorer and public questions, unchanged, so the repo runs from a fresh clone
* `tools/` - dev helpers (`review.py`, `ask.py`)

The website snapshots are NOT in this repo (shared for the event only). To run, put the kit's `snapshots_zip/`
folder next to `harness/` (the scorer unpacks it on first run), then from the top folder:

    python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
