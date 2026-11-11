# Submission notes

**API keys needed: none.** Pure Python 3.9+ standard library, no network, no `requirements.txt` needed.
Ingest of all five snapshots takes about 5 s; typical `answer()` latency is well under 0.1 s.

Run it:

    python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
    python tools/review.py --misses      # dev helper: every answer next to what was expected
    python tools/ask.py "What is the 36-month personal loan APR?"   # ad-hoc questions ("... @@2026-05-31" for as_of)

## How it works

`ingest()` keeps **every** snapshot (history, not just the latest copy):

* pages are cut into prose chunks / table chunks and stored once in a de-duplicated store
  (a chunk remembers which snapshots it appears in) with a BM25 inverted index;
* every markdown **rate table** is parsed into rows: product family, tier, term, model-year range,
  dividend rate / APY / APR / points and the table's own "effective as of" date.

`answer(question, as_of)`:

1. `as_of` picks the newest snapshot taken on or before that date (also understands "as of May 15, 2026" in the text);
   before the first snapshot we say we don't know.
2. Routing, in order: link questions → how-to-join / eligibility / servicing (phone, update contact info) →
   company facts (branch count, employees, assets ...) → **rate questions** → "hours for branch X" guard → retrieval.
3. **Rate questions** use the structured tables, not prose. The product family is detected (share savings, certificates,
   IRAs, mortgages, auto, boat/RV, cards, business vs personal ...), then rows are selected by term, balance tier
   (nested "or less" caps pick the smallest cap that fits), model year, jumbo/Roth/Traditional, "highest/lowest".
   "Do any tiers pay 0.35%?" gets a yes/no. "How did X change?" gives the series across snapshots.
4. Everything else is BM25 retrieval with recency and structure features (see below), then the best sentences of the
   best chunk are returned with sources.

## Rules for "what is true now" (written down)

* **Newest "effective as of" date wins.** An official rate sheet (`/rates`, `/business-rates`) beats product pages,
  which beat landing pages / old blog posts; a non-official page only overrides the rate sheet if it is dated later.
  (This is why the 5.34 % auto rate on the `switch-and-earn` page, effective 1/1/2026, loses to 5.39 %.)
* **Newest dated page wins for prose facts.** Press releases repeat an "About Dupaco" paragraph that drifts over time
  (19 → 23 → 22 branch locations after the Hy-Vee closure); the most recently dated page is used. When several pages share
  a title (Forbes recognitions) the newest dated one is preferred.
* **Qualifiers are facts, not conflicts**: term, tier, model year, rate vs APY vs APR, personal vs business,
  purchase vs refinance, jumbo vs regular are matched exactly; a tier is never "averaged" with another.
* **Honesty**: if a qualifier the member asks for is not on the site (7/1 ARM, FHA, construction rate, a 60-month jumbo
  certificate, an ATV loan for 72 months, hours for a branch with no published hours) we answer
  `confident=False` with no numbers, instead of guessing.
* Relative-time questions ("last year") with several dated pages return each year's figure, labelled with the page year.

## Retrieval details

BM25 over chunks (title, heading, URL slug and text indexed) with: a page-type prior (core pages > news > blog > files),
a small recency bonus, answer-type boosts (time patterns for hours, `$` amounts for fee/cost questions, digits for
"how many", phone / address patterns), phrase (bigram) matches, short-page-title coverage, and a down-weight for
"Keep reading" teaser cards. Oversized PDF text blocks are split into ~700-character chunks.

## conflicts()

Rows from all rate tables are aligned across pages and crawls by (family, term, tier, flags, n-th occurrence, model-year
tier rank). A conflict is reported when (a) different pages in the same crawl disagree numerically
(e.g. 2.50 % on the rate sheet vs "up to 3.00 %" on a landing page) or (b) the official rate sheet itself changed between
downloads. The resolution is the value on the newest official sheet, with the reason. The "About Dupaco" boilerplate
(branch locations, employees, assets, counties served) is compared across press releases the same way.

## Known limits

* Retrieval answers are verbatim passages, not rewritten prose.
* Per-branch hours are not published on the site, so those questions are declined unless a page gives them.
* Product families are keyword-driven; a brand-new product name would fall back to retrieval.
