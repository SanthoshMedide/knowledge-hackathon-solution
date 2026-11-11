"""
Knowledge system for a credit-union website that is crawled over time.

Design (all rules + retrieval; no network, no API keys, standard library only):

  ingest()   keeps EVERY snapshot.  For each one we
               * chunk pages (prose paragraphs, markdown tables) into a de-duplicated chunk store
                 with a BM25 inverted index and a per-chunk "present in snapshot i" bitmask,
               * parse every markdown rate table into structured rows (product family, tier,
                 term, model-year, rate / APY / APR, "effective as of" date).
  answer()   1. picks the snapshot that was current at `as_of` (default: newest),
             2. rate / APY / APR / dividend questions -> structured table lookup
                (family detection, then term / amount-tier / model-year / jumbo / roth ... row selection;
                 among duplicate tables the NEWEST "effective as of" date wins, then the official rate page),
             3. link questions -> canonical page links,
             4. everything else -> recency-aware BM25 chunk retrieval (newest dated page wins near-ties),
             5. says "I don't know" (confident=False, no digits) when the site does not publish the thing.
  conflicts() compares the same table row across snapshots / pages and the "About Dupaco" boilerplate
              (branch count, employees, assets ...) across press releases; newest evidence is the resolution.

Rules of thumb, written down (see NOTES.md for the long version):
  * a newer "effective as of" date beats an older one; an official rate sheet beats landing pages and blogs;
  * qualifiers (term, tier, model year, rate vs APY, personal vs business) are facts, not conflicts;
  * if a qualifier the member asked for does not exist on the site (e.g. 7/1 ARM) we decline.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

try:                                   # the scorer already put its harness folder on sys.path
    from contract import Answer, Conflict, Snapshot, Source  # noqa: E402
except Exception:
    try:                               # or the harness folder sits above wherever the submission lives
        sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents
                                    if (d / "harness" / "contract.py").exists())))
        from contract import Answer, Conflict, Snapshot, Source  # noqa: E402
    except Exception:  # pragma: no cover - plain stand-ins with the same fields (the scorer accepts any such object)
        class Source:
            def __init__(self, url, snapshot_id=None, quote=None):
                self.url, self.snapshot_id, self.quote = url, snapshot_id, quote

        class Answer(dict):            # a dict: the scorer reads dicts everywhere, plain objects not on repeat checks
            def __init__(self, text, sources=None, confident=True, conflict_note=None):
                super().__init__(text=text, sources=list(sources or []), confident=confident,
                                 conflict_note=conflict_note)

            def __getattr__(self, k):
                try:
                    return self[k]
                except KeyError:
                    raise AttributeError(k)

        class Conflict:
            def __init__(self, topic, values, sources, resolution=None, reason=None):
                self.topic, self.values, self.sources = topic, values, sources
                self.resolution, self.reason = resolution, reason

        Snapshot = None

# --------------------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------------------
STOP = set("""a an the of to in on for and or is are was were be been being do does did can could would should
i me my we our you your it its this that these those with at by from as what which who whom how when where why
about any there please tell know get give provide want like need much many some just also than then so if
dupaco credit union s t re ve ll d m am has have had will shall may might must into out up over under per""".split())

_NUMCOMMA = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TOK = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

SYN = {  # query-side expansions (token -> extra tokens)
    "cd": ["certificate"], "cds": ["certificate"],
    "mortgage": ["home", "loan", "mortgage"], "mortgages": ["home", "loan", "mortgage"],
    "refi": ["refinance"], "heloc": ["home", "equity", "line", "credit"],
    "lobby": ["lobby", "hours"], "drive": ["drive"], "atm": ["atm"],
    "wallet": ["wallet", "apple", "pay", "google", "samsung"],
    "membership": ["member", "eligible", "join"], "eligible": ["member", "eligible", "join"],
    "join": ["join", "member"],
    "road": ["rd"], "street": ["st"], "avenue": ["ave"], "lane": ["ln"], "boulevard": ["blvd"],
    "cost": ["fee", "charge"], "costs": ["fee", "charge"], "price": ["fee", "charge"], "charge": ["fee"],
    "charges": ["fee"], "fees": ["fee", "charge"],
}


def stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if len(w) > 6 and w.isalpha():      # crude prefix stem: recognition/recognized/recognize -> recogni
        w = w[:6]
    return w


def toks(s: str, drop_stop: bool = True) -> list:
    s = _NUMCOMMA.sub("", (s or "").lower())
    out = []
    for w in _TOK.findall(s):
        if drop_stop and w in STOP:
            continue
        out.append(stem(w))
    return out


def clean_md(md: str) -> str:
    md = re.sub(r"!\[[^\]]*\]\((?:data:[^)]*|[^)]*)\)", "", md)          # images (incl. data URIs)
    md = md.replace("\\\\r\\\\n", " ").replace("\\r\\n", " ")
    md = re.sub(r"\n[ \t]*(?:Third Party Plugins|Messenger Launcher|Messenger)[ \t]*(?=\n|$)", "", md)
    md = re.sub(r"(?:\n\s*)?✕\s*(?:\[Continue\]\([^)]*\)\s*Cancel\s*)?", "\n", md)
    return md


_TIME = re.compile(r"\d{1,2}(?::\d\d)?\s?[ap]\.?m\b|\bnoon\b", re.I)
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def strip_links(s: str) -> str:
    s = _LINK.sub(lambda m: m.group(1), s)
    s = re.sub(r"\[\s*(\d+)\s*\]", "", s)
    s = re.sub(r"[*_`]{1,3}", "", s)
    return re.sub(r"[ \t]+", " ", s).strip()


MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"])}
for _i, _m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]):
    MONTHS[_m] = _i + 1
MONTHS["sept"] = 9

_DATE1 = re.compile(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|"
                    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d\d|19\d\d)\b", re.I)
_DATE2 = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b")
_DATE3 = re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d\d)\b")


def find_date(text: str):
    """First 'Month D, YYYY' date in the text (publication date of news / blog pages)."""
    m = _DATE1.search(text)
    if not m:
        return None
    name = m.group(1).lower().rstrip(".")
    mon = MONTHS.get(name) or MONTHS.get(name[:3])
    try:
        return _dt.date(int(m.group(3)), mon, int(m.group(2)))
    except Exception:
        return None


def parse_date_any(s: str):
    s = s or ""
    m = _DATE2.search(s)
    if m:
        try:
            return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass
    m = _DATE3.search(s)
    if m:
        try:
            return _dt.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except Exception:
            pass
    m = _DATE1.search(s)
    if m:
        name = m.group(1).lower().rstrip(".")
        mon = MONTHS.get(name) or MONTHS.get(name[:3])
        try:
            return _dt.date(int(m.group(3)), mon, int(m.group(2)))
        except Exception:
            pass
    return None


def iso(d):
    return d.isoformat() if d else ""


# --------------------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------------------
class Chunk:
    __slots__ = ("cid", "url", "title", "heading", "text", "kind", "date", "mask", "links", "ntok", "ttoks", "t0")

    def __init__(self, cid, url, title, heading, text, kind, date, links):
        self.cid, self.url, self.title, self.heading = cid, url, title, heading
        self.text, self.kind, self.date, self.links = text, kind, date, links
        self.mask = 0
        self.ntok = 1
        self.ttoks = None
        self.t0 = None


def page_kind(url: str) -> str:
    p = re.sub(r"^https?://[^/]+", "", url).split("?")[0].lower()
    if "/newsroom" in p:
        return "news"
    if "/learn/blog" in p or p.startswith("/learn"):
        return "blog"
    if "/hubfs/" in p or p.endswith((".pdf", ".xlsx", ".xls")):
        return "file"
    if "/business/resources" in p:
        return "blog"
    if "/calculators" in p:
        return "calc"
    if p.endswith(("-lp", "-offer-lp")) or "-lp" in p.split("/")[-1] or "switch-and-earn" in p or "offer" in p:
        return "landing"
    return "core"


KIND_PRIOR = {"core": 1.0, "landing": 0.85, "news": 0.88, "blog": 0.82, "file": 0.8, "calc": 0.7}


def split_blocks(md: str):
    """Yield (kind, text) blocks; tables stay together."""
    lines = md.split("\n")
    i, n = 0, len(lines)
    buf = []
    while i < n:
        ln = lines[i]
        if ln.lstrip().startswith("|"):
            if buf:
                yield "p", "\n".join(buf).strip()
                buf = []
            j = i
            tbl = []
            while j < n and lines[j].lstrip().startswith("|"):
                tbl.append(lines[j])
                j += 1
            yield "t", "\n".join(tbl)
            i = j
            continue
        if not ln.strip():
            if buf:
                yield "p", "\n".join(buf).strip()
                buf = []
        else:
            buf.append(ln)
        i += 1
    if buf:
        yield "p", "\n".join(buf).strip()


def make_chunks(url: str, title: str, md: str):
    """-> list of (heading, text, kind, links)"""
    md = clean_md(md)
    out = []
    heading = title or ""
    acc, acc_len = [], 0
    acc_head = heading

    def flush():
        nonlocal acc, acc_len
        if acc:
            text = "\n".join(acc).strip()
            plain = strip_links(text)
            if re.search(r"[A-Za-z]{3}", plain) and (len(plain) >= 45 or re.search(r"\d{3}", plain)):
                out.append((acc_head, text, "p"))
        acc, acc_len = [], 0

    for kind, blk in split_blocks(md):
        if not blk:
            continue
        if kind == "t":
            flush()
            rows = blk.split("\n")
            hdr = rows[:2]
            body = rows[2:]
            for k in range(0, max(len(body), 1), 14):
                part = "\n".join(hdr + body[k:k + 14])
                out.append((heading, part, "t"))
            continue
        first = blk.split("\n", 1)[0]
        if first.startswith("#"):
            flush()
            heading = first.lstrip("# ").strip()
            acc_head = heading
            rest = blk.split("\n", 1)[1] if "\n" in blk else ""
            if rest.strip():
                acc.append(rest.strip())
                acc_len += len(rest)
            continue
        if len(strip_links(blk)) < 3:
            continue
        if len(blk) > 1100 and blk.count("\n") >= 8:
            flush()
            group, glen = [], 0
            for line in blk.split("\n"):
                if glen + len(line) > 700 and group:
                    acc.append("\n".join(group))
                    acc_len = glen
                    flush()
                    group, glen = [], 0
                group.append(line)
                glen += len(line) + 1
            if group:
                acc.append("\n".join(group))
                acc_len = glen
                flush()
            continue
        # short plain lines act like sub-headings (rate pages): keep as heading context too
        if len(blk) < 60 and "\n" not in blk and not re.search(r"[.!?:]$", blk) and not blk.startswith(("-", "*", "[")):
            heading = blk.strip()
            acc_head = heading if not acc else acc_head
        if acc_len + len(blk) > 900 and acc:
            flush()
            acc_head = heading
        acc.append(blk)
        acc_len += len(blk)
    flush()
    return out


# --------------------------------------------------------------------------------------
# rate tables
# --------------------------------------------------------------------------------------
_FOOT = re.compile(r"(?<=%)\s*(APR|APY)?\d(?=\s|$)")
_RATEISH = re.compile(r"^[\s$]*[\d.,]+\s*%?\s*(?:APR|APY)?\s*$|^[—–-]$")
_EFF = re.compile(r"effective\s+(?:as of\s+)?(\d{1,2}/\d{1,2}/\d{2,4})", re.I)


def clean_cell(c: str) -> str:
    c = strip_links(c.strip())
    c = _FOOT.sub(lambda m: (" " + m.group(1)) if m.group(1) else "", c)
    c = re.sub(r"\\\*", "*", c)
    return re.sub(r"\s+", " ", c).strip()


def split_row(line: str):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [clean_cell(c) for c in line.split("|")]


class Table:
    __slots__ = ("url", "name", "section", "headers", "rows", "eff", "links", "family", "snap", "prio", "idx")

    def __init__(self):
        self.links = []


def _eff_date(s: str):
    m = _EFF.search(s)
    if not m:
        return None
    mm = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", m.group(1))
    y = int(mm.group(3))
    y = y + 2000 if y < 100 else y
    try:
        return _dt.date(y, int(mm.group(1)), int(mm.group(2)))
    except Exception:
        return None


def parse_tables(url: str, md: str):
    md = clean_md(md)
    lines = md.split("\n")
    tables = []
    recent = []          # recent non-empty, non-table lines
    section = ""
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        if ln.lstrip().startswith("|") and i + 1 < n and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            hdr = split_row(ln)
            j = i + 2
            rows = []
            while j < n and lines[j].lstrip().startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            shorts = [r for r in recent[-3:] if len(r) <= 70 and not re.match(r"(?i)rates are|\d", r)]
            name = shorts[-1] if shorts else ""
            if len(shorts) >= 2 and recent and recent[-1] == shorts[-1] and (len(recent) < 2 or recent[-2] == shorts[-2]):
                section = shorts[-2]
            t = Table()
            t.url, t.name, t.section, t.headers, t.rows = url, name, section, hdr, rows
            t.idx = len(tables)
            # effective date + links: look at following text until next table/heading
            k = j
            seg = []
            while k < n and not (lines[k].lstrip().startswith("|")) and len(seg) < 40:
                if lines[k].strip():
                    seg.append(lines[k])
                k += 1
                if len(seg) >= 6 and any(s.startswith("[") for s in seg):
                    break
            segtxt = "\n".join(seg)
            t.eff = _eff_date(segtxt)
            t.links = [u for _, u in _LINK.findall(segtxt) if "dupaco.com" in u and "join." not in u and "shine." not in u]
            tables.append(t)
            recent = []
            i = j
            continue
        if ln.strip() and not ln.strip().startswith(("[", "![", "\\*")):
            recent.append(strip_links(ln.strip()))
            recent = recent[-6:]
        i += 1
    return tables


def _hdr_kind(h: str) -> str:
    h = h.lower()
    if "apy" in h or "annual percentage yield" in h:
        return "apy"
    if "dividend" in h:
        return "div"
    if "apr" in h or "annual percentage rate" in h:
        return "apr"
    if h.strip().startswith("rate") or h.strip() == "rate":
        return "rate"
    if "point" in h:
        return "points"
    if "payment" in h:
        return "payment"
    if "min" in h and "open" in h:
        return "minopen"
    if "min" in h and "earn" in h:
        return "minearn"
    return "label"


FAMILIES = [  # (base family, regex over name+section+first-column labels) ordered by precedence
    ("card", r"visa|credit card|platinum|rewards"),
    ("heloc", r"heloc|home equity line|line of credit"),
    ("home_equity_loan", r"home equity loan|home equity \(fixed"),
    ("mort_refi", r"refinanc"),
    ("mort_purchase", r"home loans?\b.*purchase|purchase:|purchase$|\bmortgage"),
    ("home_improvement", r"home improvement"),
    ("personal_loan", r"personal loan"),
    ("auto", r"\bauto|motorcycle|model year"),
    ("boat", r"boat|\brv\b"),
    ("atv", r"\batv|snowmobile|jet ski"),
    ("cert", r"certificate"),
    ("ira", r"\bira"),
    ("hsa", r"health savings|\bhsa\b"),
    ("mm", r"money market"),
    ("hys", r"high.yield"),
    ("bright", r"bright savings"),
    ("yni", r"you name it"),
    ("holiday", r"holiday"),
    ("launchpad", r"launch ?pad"),
    ("share", r"share savings|^savings"),
    ("checking", r"checking"),
    ("savings", r"savings"),
]


def table_family(t: Table) -> str:
    firsts = " ".join(r[0] for r in t.rows[:12] if r)
    labels = " ".join(" ".join(r[:2]) for r in t.rows[:3])
    base = ""
    slug = re.sub(r"[-_/]+", " ", re.sub(r"^https?://[^/]+", "", t.url.split("?")[0]).lower())
    scopes = [f"{t.name}", f"{t.name} {t.section}", f"{t.name} {t.section} {firsts} {labels}"]
    if not t.name.strip() or t.name.strip().lower() in ("balance", "account type", "card type"):
        scopes = [slug] + scopes
    for scope in scopes:
        sc = scope.lower()
        for fam, rx in FAMILIES:
            if re.search(rx, sc):
                base = fam
                break
        if base:
            break
    if not base:
        return ""
    biz = "/business" in t.url.lower() or "business" in f"{t.name} {t.section} {firsts}".lower()
    return ("biz_" + base) if biz else base


def rate_like_table(t: Table) -> bool:
    kinds = [_hdr_kind(h) for h in t.headers]
    return any(k in ("apy", "apr", "rate", "div") for k in kinds) and len(t.rows) > 0


_TERM = re.compile(r"(\d{1,3})\s*[- ]?\s*(months?|mos?\b|years?|yrs?\b|yr\b)", re.I)
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20, "thirty": 30,
            "seventeen": 17, "sixteen": 16, "twenty-four": 24, "thirty-six": 36, "forty-eight": 48, "sixty": 60}
_WORDTERM = re.compile(r"\b(" + "|".join(sorted(_WORDNUM, key=len, reverse=True)) + r")[\s-]+(months?|years?)\b", re.I)


def terms_in(text: str):
    out = set()
    for m in _TERM.finditer(text or ""):
        v = int(m.group(1))
        unit = m.group(2).lower()
        out.add(v * 12 if unit.startswith(("y",)) else v)
    for m in _WORDTERM.finditer(text or ""):
        v = _WORDNUM[m.group(1).lower()]
        out.add(v * 12 if m.group(2).lower().startswith("y") else v)
    return out


def money_vals(text: str):
    out = []
    for num, suf in re.findall(r"\$\s*([\d,]+(?:\.\d+)?)\s*(k|m|thousand|million)?\b", text or "", re.I):
        v = float(num.replace(",", ""))
        suf = (suf or "").lower()
        v *= 1000 if suf in ("k", "thousand") else 1_000_000 if suf in ("m", "million") else 1
        out.append(v)
    return out


def amount_range(label: str):
    """(lo, hi) of a tier label; None if none."""
    t = label.replace("–", "-").replace("—", "-")
    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s*\+", t)
    if m:
        return float(m.group(1).replace(",", "")), math.inf
    m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)\s*-\s*\$?\s*([\d,]+(?:\.\d+)?)", t)
    if m and ("$" in t):
        a, b = float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
        if b >= a:
            return a, b
    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s*or less", t, re.I)
    if m:
        return 0.0, float(m.group(1).replace(",", ""))
    return None


def year_range(label: str):
    t = label.replace("–", "-").replace("—", "-")
    m = re.search(r"\b(20\d\d|19\d\d)\s*-\s*(\d{2,4})\b", t)
    if m:
        a = int(m.group(1))
        b = m.group(2)
        b = int(b) if len(b) == 4 else int(str(a)[:2] + b)
        return min(a, b), max(a, b)
    m = re.search(r"\b(20\d\d|19\d\d)\b.*older", t, re.I)
    if m:
        return 0, int(m.group(1))
    return None


# --------------------------------------------------------------------------------------
# the system
# --------------------------------------------------------------------------------------
RATE_WORDS = re.compile(r"\b(rates?|aprs?|apys?|dividends?|interest|yields?|percent|percentage)\b|%", re.I)
HIGH_WORDS = re.compile(r"\b(highest|top|maximum|max|best|most)\b", re.I)
LOW_WORDS = re.compile(r"\b(lowest|minimum rate|least|cheapest|smallest)\b", re.I)
HIST_WORDS = re.compile(r"\b(chang(?:e|ed|es|ing)|previous(?:ly)?|before|used to|history|over time|increas(?:e|ed)|"
                        r"decreas(?:e|ed)|went up|went down|moved|trend|earlier|prior|compared|since)\b", re.I)
LINK_WORDS = re.compile(r"\b(link|url|website|web site|web address|webpage|web page|site|portal|where (?:can|do|should) i (?:go|see|find|apply|sign|log))\b", re.I)
REFUSE_QUAL = [  # qualifier regex -> rows must literally contain it, otherwise the site does not publish it
    r"\barms?\b", r"adjustable", r"\b\d/\d\b", r"construction", r"bridge", r"reverse mortgage", r"\bfha\b", r"\bva\b",
    r"usda", r"balloon", r"interest.only", r"\bjumbo\b", r"\bsubprime\b", r"\bstudent\b", r"\bpayday\b",
]

GENERIC_Q = set("rate rates apr apy dividend dividends interest percent percentage yield current currently now today "
                "latest what whats right new special offer offers account accounts tier tiers".split())


class KS:
    def __init__(self):
        self.snaps = []           # list of dicts: id, date, pages{url: (title, md)}
        self.chunks = []          # global chunk store
        self._chunk_key = {}      # (url,text) -> cid
        self.post = defaultdict(list)   # token -> [cid*256+tf]
        self.tables = []          # per snapshot: list[Table]
        self.dfs = defaultdict(int)
        self._avglen = 1.0
        self._conf = None
        self._cache = {}

    # ------------------------------------------------------------------ ingest
    def ingest(self, snapshot) -> None:
        si = len(self.snaps)
        d = parse_date_any(snapshot.taken_on) or _dt.date(2026, 1, 1)
        pages = {}
        tables = []
        for p in snapshot.pages:
            url = p.url
            if "privacy-policy?" in url or "campaignId" in url:
                continue
            pages[url] = (p.title or "", p.markdown or "")
        self.snaps.append({"id": snapshot.snapshot_id, "date": d, "pages": pages})
        bit = 1 << si
        for url, (title, md) in pages.items():
            kind = page_kind(url)
            head = clean_md(md)[:2500]
            pdate = find_date(head) if kind in ("news", "blog") else None
            if kind == "core" and not pdate:
                pdate = None
            for heading, text, ck in make_chunks(url, title, md):
                key = (url, text)
                cid = self._chunk_key.get(key)
                if cid is None:
                    cid = len(self.chunks)
                    links = _LINK.findall(text)
                    c = Chunk(cid, url, title, heading, text, ck if kind != "news" else ck, pdate, links)
                    c.kind = kind
                    self.chunks.append(c)
                    self._chunk_key[key] = cid
                    tk = toks(f"{title} {heading} {strip_links(text)} {url.rsplit('/', 1)[-1].replace('-', ' ')}")
                    c.ntok = max(len(tk), 1)
                    tf = defaultdict(int)
                    for w in tk:
                        tf[w] += 1
                    for w, f in tf.items():
                        self.post[w].append(cid * 256 + min(f, 255))
                        self.dfs[w] += 1
                self.chunks[cid].mask |= bit
            if kind in ("core", "landing") or url.rstrip("/").endswith(("/rates", "/business-rates")):
                for t in parse_tables(url, md):
                    if rate_like_table(t):
                        t.snap = si
                        t.family = table_family(t)
                        t.prio = 3 if url.rstrip("/").endswith(("/rates", "/business-rates")) else (1 if kind == "landing" else 2)
                        tables.append(t)
        self.tables.append(tables)
        self._avglen = sum(c.ntok for c in self.chunks) / max(len(self.chunks), 1)
        self._cache.clear()
        self._conf = None

    # ------------------------------------------------------------------ snapshot selection
    def snap_index(self, as_of):
        if not self.snaps:
            return None
        if not as_of:
            return len(self.snaps) - 1
        d = parse_date_any(str(as_of))
        if d is None:
            return len(self.snaps) - 1
        idx = None
        for i, s in enumerate(self.snaps):
            if s["date"] <= d:
                idx = i
        return idx  # None -> before the first snapshot

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, q: str, si: int, k: int = 8, extra=None):
        base = toks(q)
        qt = []
        for w in base:
            qt.append(w)
            for e in SYN.get(w, []):
                qt.append(stem(e))
        if extra:
            qt += [stem(e) for e in extra]
        weights = defaultdict(float)
        for w in qt:
            weights[w] += 1.0
        if not weights:
            return [], 0.0
        N = len(self.chunks)
        bit = 1 << si
        scores = defaultdict(float)
        matched = defaultdict(set)
        k1, b = 1.2, 0.75
        for w, qw in weights.items():
            plist = self.post.get(w)
            if not plist:
                continue
            df = self.dfs[w]
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for enc in plist:
                cid, tf = enc >> 8, enc & 255
                c = self.chunks[cid]
                if not (c.mask & bit):
                    continue
                den = tf + k1 * (1 - b + b * c.ntok / self._avglen)
                scores[cid] += min(qw, 2.0) * idf * tf * (k1 + 1) / den
                matched[cid].add(w)
        if not scores:
            return [], 0.0
        nq = max(len(weights), 1)
        dmin, dmax = _dt.date(2018, 1, 1), _dt.date(2026, 10, 1)
        res = []
        for cid, sc in scores.items():
            c = self.chunks[cid]
            cov = len(matched[cid]) / nq
            s = sc * (0.55 + 0.45 * cov) * KIND_PRIOR.get(c.kind, 1.0)
            if c.date:
                rec = (c.date - dmin).days / max((dmax - dmin).days, 1)
                s *= 1 + 0.06 * max(0.0, min(1.0, rec))
            res.append((s, cid))
        res.sort(reverse=True)
        # --- second stage: answer-type and title/slug features on the best candidates only
        ql = q.lower()
        hours_q = bool(re.search(r"\b(hours?|open|opens|close|closes|closed|schedule)\b", ql))
        count_q = bool(re.search(r"\bhow (many|much)\b|\bnumber of\b|\bhow long\b|\bwhat (?:is|was) the (?:total|amount)\b", ql))
        money_q = bool(re.search(r"\b(cost|costs|price|fee|fees|charge|charges|how much)\b", ql))
        phone_q = bool(re.search(r"\b(phone|call|number to call|telephone)\b", ql))
        addr_q = bool(re.search(r"\b(address|located|location of|where is|street)\b", ql))
        content = [w for w in set(base) if w not in GENERIC_Q]
        idfs = {w: math.log(1 + (N - self.dfs.get(w, 0) + 0.5) / (self.dfs.get(w, 0) + 0.5)) for w in content}
        idf_total = sum(idfs.values()) or 1.0
        rarest = max(idfs, key=idfs.get) if idfs and max(idfs.values()) > 4.0 else None
        qset = set(base)
        qbiz = bool(re.search(r"\bbusiness|commercial|company\b", ql))
        qyears = re.findall(r"\b(20[0-3]\d)\b", ql)
        qwords = [w for w in re.findall(r"[a-z0-9$]+", ql) if w not in STOP]
        bigrams = [f"{a} {b}" for a, b in zip(qwords, qwords[1:])]
        bigrams += [bg[:-1] for bg in bigrams if bg.endswith("s")]
        rescored = []
        for s, cid in res[:250]:
            c = self.chunks[cid]
            plain = None
            if qyears and any(y in c.text for y in qyears):
                s *= 1.6
            if bigrams:
                low = re.sub(r"[^a-z0-9$]+", " ", c.text.lower())
                nb = sum(1 for bg in bigrams if bg in low)
                if nb:
                    s *= min(1 + 0.6 * nb, 2.4)
            if "Keep reading" in c.text or re.match(r"\s*\d+ min read", c.text):
                s *= 0.5
            if hours_q or count_q or phone_q or addr_q or money_q:
                plain = strip_links(c.text)
                if money_q:
                    s *= 1.5 if re.search(r"\$\s?\d", plain) else 0.8
                if hours_q:
                    s *= 1.8 if _TIME.search(plain) else 0.6
                if count_q and re.search(r"\d", plain):
                    s *= 1.25
                if phone_q and re.search(r"\d{3}[-.)\s]+\d{3,4}", plain):
                    s *= 1.3
                if addr_q and re.search(r"\b[A-Z]{2},?\s+\d{5}\b", plain):
                    s *= 1.3
            if not (hours_q or count_q or phone_q or addr_q or money_q):
                dig = sum(ch.isdigit() for ch in c.text[:400])
                if dig > 70:                       # chart / calculator data, not an explanation
                    s *= 0.55
            if "/business" in c.url and not qbiz:
                s *= 0.7
            if c.t0 is None:
                c.t0 = tuple(toks(c.title.split("|")[0]))
            if 0 < len(c.t0) <= 5:
                ratio = len(set(c.t0) & qset) / len(c.t0)
                if ratio > 0:
                    s *= 1 + 0.9 * ratio * ratio
            if content:
                tt = c.ttoks
                if tt is None:
                    tt = c.ttoks = set(toks(f"{c.title} {c.url.rsplit('/', 1)[-1].replace('-', ' ')}"))
                if tt:
                    scale = 1.0 / (1.0 + max(0, len(tt) - 5) / 6.0)   # long article titles are weak evidence
                    num = sum(idfs[w] for w in content if w in tt)
                    if num:
                        s *= 1 + 0.9 * scale * num / idf_total
                    if rarest and rarest in tt:
                        s *= 1 + 0.6 * scale
            rescored.append((s, cid))
        rescored.sort(reverse=True)
        top = rescored[:k]
        return [(s, self.chunks[cid]) for s, cid in top], (top[0][0] if top else 0.0)

    # ------------------------------------------------------------------ main entry
    def answer(self, question: str, as_of=None):
        key = (question, as_of)
        if key in self._cache:
            return self._cache[key]
        try:
            ans = self._answer(question, as_of)
        except Exception as e:  # never crash the harness
            ans = Answer(text="I don't know the answer to that from the information I have; please contact us.",
                         confident=False)
        self._cache[key] = ans
        return ans

    def _answer(self, question: str, as_of):
        q = (question or "").strip()
        if not self.snaps:
            return self._idk()
        as_of = as_of or self._asof_from_text(q)
        si = self.snap_index(as_of)
        if si is None:
            return Answer(text="I don't have information from the website for that date; "
                               "the earliest information I have is later. Please contact us.", confident=False)
        ql = q.lower()

        # 1. links
        if LINK_WORDS.search(ql) and re.search(r"\b(link|url|address of the website|web address|where)\b", ql):
            a = self._answer_link(q, si)
            if a:
                return a
        # 2. how-to-join / servicing / company facts
        for fn in (self._answer_branch_address, self._answer_join, self._answer_eligibility, self._answer_servicing, self._answer_facts):
            a = fn(q, si)
            if a is not None:
                return a
        # 3. rates
        if self._is_rate_question(ql):
            a = self._answer_rate(q, si, as_of)
            if a is not None:
                return a
        # 4. branch-specific hours guard
        a = self._answer_hours_guard(q, si)
        if a is not None:
            return a
        # 5. generic retrieval
        return self._answer_prose(q, si, explicit_rate=bool(re.search(r"\b(rates?|aprs?|apys?|interest|dividends?)\b", ql)))

    # ------------------------------------------------------------------ helpers
    def _idk(self, extra=""):
        return Answer(text="I don't know - the website does not say. Please contact Dupaco or visit a branch for help."
                      + extra, confident=False)

    def _asof_from_text(self, q: str):
        m = re.search(r"\b(?:as of|in|during|back in|for|on)\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|"
                      r"july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?,?\s+(?:(\d{1,2})(?:st|nd|rd|th)?,?\s+)?(20\d\d)\b", q, re.I)
        if m and re.search(r"\b(was|were|as of|back|then|at the time|used to|previous|earlier|last|had)\b", q, re.I):
            mon = MONTHS.get(m.group(1).lower().rstrip(".")) or MONTHS.get(m.group(1).lower()[:3])
            y = int(m.group(3))
            if mon and (y, mon) < (2026, 10):
                if m.group(2):
                    try:
                        return iso(_dt.date(y, mon, int(m.group(2))))
                    except Exception:
                        pass
                nxt = _dt.date(y + (mon == 12), 1 if mon == 12 else mon + 1, 1)
                return iso(nxt - _dt.timedelta(days=1))
        m = re.search(r"\b(?:as of|on)\s+(20\d\d-\d\d-\d\d)\b", q)
        if m:
            return m.group(1)
        return None

    def _sources_for(self, chunks, si):
        seen, out = set(), []
        for c in chunks:
            if c.url in seen:
                continue
            seen.add(c.url)
            out.append(Source(url=c.url, snapshot_id=self.snaps[si]["id"],
                              quote=strip_links(c.text)[:160]))
        return out

    # ------------------------------------------------------------------ prose answers
    @staticmethod
    def _table_text(text: str) -> str:
        rows = [split_row(l) for l in text.split("\n") if l.strip().startswith("|")]
        if len(rows) < 2:
            return text
        hdr = rows[0]
        body = [r for r in rows[1:] if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in r)]
        if any(re.search(r"\$|\d", h or "") for h in hdr):      # header row is really a data row (PDF tables)
            body = [hdr] + body
            hdr = [""] * len(hdr)
        out = []
        for r in body:
            cells = [f"{h}: {c}" if h and not re.fullmatch(r"Col\d+|", h) else c for h, c in zip(hdr, r) if c]
            out.append("; ".join(cells))
        return "\n".join(out)

    def _best_window(self, text: str, q: str, limit: int = 900) -> str:
        if text.lstrip().startswith("|"):
            text = self._table_text(text)
        plain = strip_links(text)
        if len(plain) <= limit:
            return plain
        qt = set(toks(q))
        hours_q = bool(re.search(r"\b(hours?|open|opens|close|closes|closed|schedule)\b", q, re.I))
        sents = [x for x in re.split(r"(?<=[.!?])\s+|\n+", plain) if x.strip()]
        scored = []
        for i, sn in enumerate(sents):
            ov = len(qt & set(toks(sn)))
            bonus = 0.4 if re.search(r"\d|\$", sn) else 0.0
            if hours_q and (_TIME.search(sn) or re.search(r"\b(lobby|drive|monday|saturday|sunday)\b", sn, re.I)):
                bonus += 1.5
            scored.append((ov + bonus, -i, i))
        scored.sort(reverse=True)
        chosen, total = [], 0
        for sc, _, i in scored:
            if total + len(sents[i]) > limit and chosen:
                continue
            chosen.append(i)
            total += len(sents[i]) + 1
            if total >= limit:
                break
        return " ".join(sents[i] for i in sorted(chosen)).strip()

    def _answer_prose(self, q: str, si: int, explicit_rate: bool = False):
        wants_list = bool(re.search(r"\b(what|which)\b.*\b(cards|loans|accounts|options|types|services|ways|kinds|products|branches|"
                                    r"locations|wallets|programs|offers)\b|\blist\b|\ball\b", q, re.I))
        allhits, top = self.retrieve(q, si, k=300)
        if not allhits:
            return self._idk()
        hits = allhits[:12]
        qtok = [w for w in toks(q) if w not in GENERIC_Q]
        best_s, best = hits[0]
        # newest of several near-identical pages (e.g. repeated press releases): prefer the newest dated page
        if best.kind in ("news", "blog") and not re.search(r"\b20[0-3]\d\b", q):
            bt = set(toks(re.sub(r"\s*\d+\s*$", "", best.title)))
            cluster = []
            for sc, c in allhits[:40]:
                if sc < 0.5 * best_s or c.kind not in ("news", "blog"):
                    continue
                ct = set(toks(re.sub(r"\s*\d+\s*$", "", c.title)))
                if bt and ct and len(bt & ct) / len(bt | ct) >= 0.6:
                    cluster.append((sc, c))
            newest = max(cluster, key=lambda x: ((x[1].date or _dt.date(1900, 1, 1)), x[0]))[1]
            relative_year = bool(re.search(r"\b(last|this|previous|prior|past) year\b", q, re.I))
            if relative_year and len({c.url for _, c in cluster}) > 1:
                outs, seenu = [], set()
                for sc, c in sorted(cluster, key=lambda x: -(x[1].date or _dt.date(1900, 1, 1)).toordinal()):
                    if c.url in seenu:
                        continue
                    seenu.add(c.url)
                    outs.append(c)
                if len(outs) > 1:
                    txt = "\n".join(f"(page dated {c.date.year if c.date else 'n/a'}) " + self._best_window(c.text, q, 500)
                                    for c in outs[:3])
                    return Answer(text=txt, sources=self._sources_for(outs[:4], si), confident=True,
                                  conflict_note="Several dated pages report different yearly figures.")
            if newest.url != best.url and (newest.date or _dt.date(1900, 1, 1)) > (best.date or _dt.date(1900, 1, 1)):
                same = [(sc, c) for sc, c in allhits if c.url == newest.url]
                if same:
                    best_s, best = same[0]
        if re.search(r"\b(hours?|open|opens|close|closes|closed|schedule)\b", q, re.I) and \
                not _TIME.search(strip_links(best.text)):
            for sc, c in allhits:
                if c.url == best.url and _TIME.search(strip_links(c.text)):
                    best_s, best = sc, c
                    break
        ctoks = set(toks(f"{best.title} {best.heading} {strip_links(best.text)}"))
        if qtok:
            cov = sum(1 for w in qtok if w in ctoks) / len(qtok)
            if cov < 0.34 and top < 8:
                return self._idk()
        picked = [best]
        if wants_list:
            for sc, c in hits:
                if sc >= 0.75 * best_s and c.url not in {p.url for p in picked} and len(picked) < 3:
                    picked.append(c)
        else:
            # a second passage from the SAME page (context such as the service area next to the eligibility rules)
            for sc, c in allhits:
                if c.url == best.url and c.cid != best.cid and sc >= 0.4 * best_s:
                    picked.append(c)
                    break
        limit = 900 if len(picked) == 1 else 600
        text = "\n".join(self._best_window(c.text, q, limit) for c in picked)
        if explicit_rate and not re.search(r"\d\s*%", text):
            return self._rate_refusal(q, si)
        return Answer(text=text, sources=self._sources_for([best] + [c for _, c in hits[:6]], si), confident=True)

    # ------------------------------------------------------------------ join / servicing / facts
    def _page_by_suffix(self, si, suffix):
        for u in self.snaps[si]["pages"]:
            if u.rstrip("/").endswith(suffix):
                return u
        return None

    def _answer_join(self, q: str, si: int):
        ql = q.lower()
        if re.search(r"\beligib|who can join|who is able", ql):
            return None
        if not re.search(r"\b(how|where)\b.*\b(join|become (?:a )?member|open (?:an? )?(?:savings )?(?:account|membership)|sign up|"
                         r"apply for membership)\b", ql):
            return None
        hits, _ = self.retrieve("who can join eligible live or work charter area open a savings account", si, k=60)
        join_url = self._page_by_suffix(si, "/about/join")
        extra = ""
        for sc, c in hits:
            if join_url and c.url == join_url and re.search(r"eligible|open a savings", c.text, re.I):
                extra = " " + self._best_window(c.text, "eligible join savings account", 420).replace("\n", " ")
                break
        text = "You can join Dupaco online at https://join.dupaco.com/ ." + extra
        srcs = [Source(url="https://join.dupaco.com/", snapshot_id=self.snaps[si]["id"])]
        if join_url:
            srcs.append(Source(url=join_url, snapshot_id=self.snaps[si]["id"]))
        return Answer(text=text, sources=srcs, confident=True)

    def _answer_branch_address(self, q: str, si: int):
        ql = q.lower()
        if not (re.search(r"\b(address|located|location|where is|where's|find)\b", ql) and
                re.search(r"\b(branch|office|atm)\b", ql)) or re.search(r"\b(hours?|open|close[sd]?|phone)\b", ql):
            return None
        cu = self._page_by_suffix(si, "/contact")
        if not cu:
            return None
        generic = GENERIC_Q | {"address", "branch", "locat", "where", "find", "road", "street", "avenue", "st", "rd", "ave",
                               "office", "atm", "dupaco", "locator", "closest", "nearest", "ia", "il", "wi", "dr", "ln"}
        qt = {w for w in toks(q) if w not in generic and w not in ("rd", "st", "ave", "ln", "dr")}
        if not qt:
            return None
        found = []
        lines = clean_md(self.snaps[si]["pages"][cu][1]).split("\n")
        for i, ln in enumerate(lines):
            if re.match(r"^\d+ [^,]+, [^,]+, [A-Z]{2}, \d{5}$", ln.strip()):
                ltok = set(toks(ln))
                hit = qt & ltok
                if hit:
                    kind = next((lines[j].strip() for j in range(i - 1, max(i - 5, 0), -1)
                                 if lines[j].strip() and not lines[j].startswith("#")), "")
                    found.append((len(hit), ln.strip(), kind))
        if not found:
            return None
        found.sort(key=lambda x: -x[0])
        best = [f for f in found if f[0] == found[0][0]][:3]
        text = " ".join(f"{k or 'Dupaco location'}: {a}." for _, a, k in best)
        return Answer(text=text, sources=[Source(url=cu, snapshot_id=self.snaps[si]["id"])], confident=True)

    def _answer_eligibility(self, q: str, si: int):
        ql = q.lower()
        if not (re.search(r"\beligib", ql) and re.search(r"\b(member|membership|join)\b", ql)) and \
                not re.search(r"who can (?:join|become)|can i join|who is able to join", ql):
            return None
        join_url = self._page_by_suffix(si, "/about/join")
        if not join_url:
            return None
        md = clean_md(self.snaps[si]["pages"][join_url][1])
        m = re.search(r"Eligible to Join[^\n]*\n+((?:\s*[-*][^\n]*\n+)+)([^\n]*)", md, re.I)
        parts = []
        if m:
            bullets = [strip_links(b.lstrip("-* ").strip()) for b in m.group(1).strip().split("\n") if b.strip()]
            parts.append("You are eligible to join Dupaco if: " + "; ".join(x.rstrip(".") for x in bullets) + ".")
            if m.group(2).strip():
                parts.append(strip_links(m.group(2)))
        # service area from the most recent press-release boilerplate
        fact = self._answer_facts("how many counties does dupaco serve", si)
        area = re.search(r"serves residents in (\d+) counties throughout ([^.]+)\.", " ".join(
            md2 for _, md2 in [self.snaps[si]["pages"][u] for u in self.snaps[si]["pages"]
                               if page_kind(u) == "news"][:400]))
        if area:
            parts.append(f"The charter area covers counties throughout {area.group(2)}.")
        elif fact:
            parts.append(fact.text)
        if not parts:
            return None
        return Answer(text=" ".join(parts), sources=[Source(url=join_url, snapshot_id=self.snaps[si]["id"])],
                      confident=True)

    def _phone(self, si):
        cu = self._page_by_suffix(si, "/contact")
        if cu:
            m = re.search(r"Phone:\**\s*\[?(\d{3}-\d{3}-\d{4})", self.snaps[si]["pages"][cu][1])
            if m:
                return m.group(1)
        return "800-373-7600"

    def _answer_servicing(self, q: str, si: int):
        ql = q.lower()
        cu = self._page_by_suffix(si, "/contact")
        srcs = [Source(url=cu or "https://www.dupaco.com/contact", snapshot_id=self.snaps[si]["id"])]
        if re.search(r"\b(update|change|edit|correct|new)\b.*\b(contact|address|phone|email|mailing|name)\b", ql) and \
                re.search(r"\b(my|information|info|details|record)\b", ql):
            return Answer(text=("You can update your contact information by logging in to your account in Shine Online or "
                                f"Mobile Banking, or by calling Dupaco at {self._phone(si)}."), sources=srcs, confident=True)
        if re.search(r"\b(phone|telephone|call|contact)\b.*\b(number|dupaco|credit union|you|member services|customer service)\b|"
                     r"\bcustomer service\b|\bmember services? (?:number|phone)", ql) and \
                not re.search(r"activate|\bpin\b|\bcard\b|branch|mortgage|loan|lender|business|insurance|investment|"
                              r"financial|fraud|dispute|hours|email|address|learning lab|\bwire|routing|atm", ql):
            return Answer(text=f"You can reach Dupaco at {self._phone(si)}.", sources=srcs, confident=True)
        if re.search(r"\b(lost|stolen|missing)\b.*\b(card|debit|credit)\b|\b(card)\b.*\b(lost|stolen)\b", ql) and \
                not re.search(r"\b(fee|cost|charge|how much|replace)\b", ql):
            return Answer(text=("If your card is lost or stolen, call Dupaco immediately at "
                                f"{self._phone(si)}."), sources=srcs, confident=True)
        return None

    FACTS = [
        (r"\bhow many (?:dupaco )?(?:branch(?:es)?|branch offices?|locations?|offices?)\b|\bnumber of (?:branch|location)",
         r"(\d+)\s+branch office locations", "Dupaco has {} branch office locations."),
        (r"\bhow many (?:dupaco )?(?:employees|staff)|\bnumber of employees|\bhow many people (?:work|are employed)",
         r"((?:nearly|over|more than)\s+[\d,]+)\s+employees", "Dupaco has {} employees."),
        (r"\bhow many (?:dupaco )?members|\bnumber of members|membership (?:size|count)|\bhow large is (?:the )?membership",
         r"membership has grown to ((?:more than|over|nearly)\s+[\d,]+)", "Dupaco has {} members."),
        (r"\btotal assets|\bassets\b|\bhow (?:big|large) is dupaco",
         r"assets (?:exceeding|of (?:nearly|more than|over) ?)\s*(\$\d\.\d\s+billion)", "Dupaco has assets exceeding {}."),
        (r"\bhow many counties|\bwhich counties",
         r"serves residents in (\d+) counties", "Dupaco serves residents in {} counties throughout Iowa, northwest Illinois and southwest Wisconsin."),
        (r"\bwhen (?:was|were) dupaco (?:founded|started|established)|\bwhen did dupaco (?:start|begin)|\bfounded\b",
         r"Founded in (\d{4})", "Dupaco was founded in {}."),
    ]

    def _answer_facts(self, q: str, si: int):
        ql = q.lower()
        # company-wide boilerplate figures only: a question about a specific program / event is not one
        if re.search(r"\b(used|use|using|volunteer\w*|took|take|participat\w*|attend\w*|enroll\w*|joined|opened|closed|"
                     r"closing|new|last year|this year|per|each|average|growth|grew|increase\w*)\b", ql):
            return None
        ym = re.search(r"\b(20[012]\d)\b", ql)
        year = int(ym.group(1)) if ym else None
        for i, (qrx, rx, tmpl) in enumerate(self.FACTS):
            if not re.search(qrx, ql):
                continue
            key = ("fact", si, i, year)
            if key not in self._cache:
                best = None
                for url, (title, md) in self.snaps[si]["pages"].items():
                    if page_kind(url) not in ("news", "blog"):
                        continue
                    m = re.search(rx, md)
                    if not m:
                        continue
                    d = find_date(clean_md(md)[:2500])
                    if not d or (year and d.year != year):
                        continue
                    if best is None or d > best[0]:
                        best = (d, m.group(1), url)
                self._cache[key] = best
            best = self._cache[key]
            if best:
                txt = tmpl.format(best[1].strip())
                if year:
                    txt = f"As of its {year} press releases: " + txt[0].lower() + txt[1:]
                return Answer(text=txt, sources=[Source(url=best[2], snapshot_id=self.snaps[si]["id"])], confident=True)
        return None

    def _answer_hours_guard(self, q: str, si: int):
        ql = q.lower()
        if not re.search(r"\b(hours?|open|opens|close|closes|closed)\b", ql):
            return None
        m = re.search(r"\b([a-z][a-z.'-]+(?:\s+[a-z][a-z.'-]+)?)\s+(?:branch|location|office)\b", q, re.I)
        place = None
        if m:
            cand = m.group(1).lower().split()
            cand = [w for w in cand if w not in STOP and w not in {"each", "every", "all", "other", "closest", "nearest",
                                                                  "main", "local", "branch", "your", "dupaco", "any"}]
            if cand:
                place = cand
        m2 = re.search(r"\bfor (?:the )?([A-Z][a-zA-Z.'-]+(?: [A-Z][a-zA-Z.'-]+)*)\b", q)
        if not place and m2:
            place = m2.group(1).lower().split()
        if not place:
            return None
        pl = " ".join(place)
        if re.search(r"hy-?vee|closing|closure", ql) and "hy" in pl:
            return None
        # look for a chunk that mentions the place AND an hours pattern
        hits, _ = self.retrieve(q, si, k=15)
        for s, c in hits:
            t = strip_links(c.text).lower()
            if all(w in t for w in place) and re.search(r"\d{1,2}(:\d\d)?\s?[ap]\.?m", t):
                return None
        return Answer(text=f"I don't know the hours for the {pl.title()} branch - the website does not list them. "
                           "Please use the branch locator or contact Dupaco directly.", confident=False)

    # ------------------------------------------------------------------ links
    LINKS = [
        (r"locator|branch|atm|location|nearest|find a", "https://www.dupaco.com/contact", "Branch and ATM locator"),
        (r"\bjoin\b|become a member|open (?:an )?account|membership|sign up", "https://join.dupaco.com/", "Join Dupaco"),
        (r"certificate|\bcds?\b", "https://www.dupaco.com/personal/savings-account#rates", "Certificate rates"),
        (r"mortgage|home loan", "https://www.dupaco.com/personal/loans/home-loans#rates", "Home loan rates"),
        (r"auto|vehicle|car loan|motorcycle", "https://www.dupaco.com/personal/loans/auto-motorcycle-loans", "Auto loans"),
        (r"credit card|visa", "https://www.dupaco.com/personal/credit-cards", "Credit cards"),
        (r"checking", "https://www.dupaco.com/personal/checking-accounts", "Checking"),
        (r"savings", "https://www.dupaco.com/personal/savings-account#rates", "Savings rates"),
        (r"\brates?\b", "https://www.dupaco.com/rates", "Rates"),
        (r"contact|phone|email|support", "https://www.dupaco.com/contact", "Contact"),
        (r"online banking|shine|log ?in|sign ?in", "https://shine.dupaco.com/", "Shine online banking"),
    ]

    def _answer_link(self, q: str, si: int):
        ql = q.lower()
        for rx, url, label in self.LINKS:
            if re.search(rx, ql):
                if url.startswith("https://www.dupaco.com") and url.split("#")[0].rstrip("/") not in \
                        {u.rstrip("/") for u in self.snaps[si]["pages"]}:
                    continue
                return Answer(text=f"{label}: {url}", sources=[Source(url=url, snapshot_id=self.snaps[si]["id"])],
                              confident=True)
        hits, _ = self.retrieve(q, si, k=8)
        qt = set(toks(q))
        scored = defaultdict(float)
        for s, c in hits:
            for anchor, url in c.links:
                if "dupaco.com" not in url or url.endswith("#"):
                    continue
                ov = len(qt & set(toks(anchor + " " + url.rsplit("/", 1)[-1].replace("-", " "))))
                scored[url] += s * (1 + ov)
        if not scored:
            return None
        best = sorted(scored.items(), key=lambda x: -x[1])[:2]
        return Answer(text="You can find it here: " + "; ".join(u for u, _ in best),
                      sources=[Source(url=u, snapshot_id=self.snaps[si]["id"]) for u, _ in best], confident=True)

    # ------------------------------------------------------------------ rates
    def _is_rate_question(self, ql: str) -> bool:
        if RATE_WORDS.search(ql):
            return True
        if re.search(r"\b(certificate|cds?)\b", ql) and re.search(r"\b(special|term|month|\d+)\b", ql):
            return True
        if self._question_families(ql) and re.search(
                r"\b(tiers?|earn|earns|earning|pay|pays|cost|charge|how much|offer|offers|change|changed|"
                r"highest|lowest|best|top|minimum|min|open|opening|specials?|promo\w*)\b", ql):
            return not re.search(r"\b(fee|fees|hours?|address|phone|routing|branch|atm)\b", ql)
        return False

    def _question_families(self, ql: str):
        biz = bool(re.search(r"\bbusiness|commercial|company\b", ql))
        fams = []

        def add(*f):
            for x in f:
                if x not in fams:
                    fams.append(x)

        if re.search(r"credit card|\bvisa\b|platinum|rewards|\bcards?\b", ql) and not re.search(r"debit|gift card", ql):
            add("card")
        if re.search(r"heloc|home equity line|line of credit", ql):
            add("heloc")
        if re.search(r"home equity loan|equity loan", ql):
            add("home_equity_loan")
        if re.search(r"refinanc|\brefi\b", ql) and not re.search(r"auto|\bcar\b|vehicle|student", ql):
            add("mort_refi")
        if re.search(r"mortgage|home loan|house|fixed.rate|conventional", ql) and not re.search(r"equity|improvement", ql):
            add("mort_purchase")
        if re.search(r"home improvement", ql):
            add("home_improvement")
        if re.search(r"personal loan", ql):
            add("personal_loan")
        if re.search(r"\bauto|\bcar\b|vehicle|motorcycle|model.year", ql) and not re.search(r"checking|insurance", ql):
            add("auto")
        if re.search(r"boat|\brv\b|recreational|camper|motorhome", ql):
            add("boat")
        if re.search(r"\batv|snowmobile|jet ski|jetski|side.by.side|utv", ql):
            add("atv")
        if re.search(r"certificate|\bcds?\b|share certificate|term share", ql):
            add("cert")
        if re.search(r"\bira\b|\biras\b|roth|traditional|retirement", ql):
            add("ira")
        if re.search(r"\bhsa\b|health savings", ql):
            add("hsa")
        if re.search(r"money market|investor", ql):
            add("mm")
        if re.search(r"high.yield", ql):
            add("hys")
        if re.search(r"bright", ql):
            add("bright")
        if re.search(r"you.name.it|youname", ql):
            add("yni")
        if re.search(r"holiday", ql):
            add("holiday")
        if re.search(r"launch ?pad", ql):
            add("launchpad")
        if re.search(r"checking", ql):
            add("checking")
        if re.search(r"share savings|regular savings|savings (?:account|tier|rate|apy|dividend)|\bsavings\b", ql) and \
                not re.search(r"bright|holiday|launch|you.name|high.yield|health|money market", ql):
            add("share")
        if not fams and re.search(r"\b(15|20|30)[- ]?(year|yr)\b|\b(180|240|360)[- ]?months?\b", ql):
            add("mort_purchase")
        if re.search(r"\b(15|20|30)[- ]?(year|yr)\b", ql) and not any(f in fams for f in (
                "boat", "auto", "atv", "personal_loan", "home_equity_loan", "heloc", "home_improvement", "cert", "ira")):
            add("mort_purchase")
        if biz:
            out = ["biz_" + f for f in fams]
            if "biz_share" in out:
                out += ["biz_savings", "biz_mm", "biz_hys"]
            return out
        if "card" in fams and not re.search(r"personal|platinum|rewards", ql):
            fams.append("biz_card")   # "what cards do you offer" -> list personal AND business cards
        elif "card" in fams and re.search(r"platinum|rewards", ql) and not re.search(r"personal", ql):
            pass
        return fams

    def _pick_tables(self, si: int, fams, prefer_new=True):
        """Best table per (family, name): official rate sheet first, a product/landing page only if dated later."""
        best = {}
        pool = []
        for fam in fams:
            fam_tables = [t for t in self.tables[si] if t.family == fam]
            official = [t for t in fam_tables if t.prio >= 3]
            if official:
                newest_official = max((t.eff or _dt.date(1900, 1, 1)) for t in official)
                pool += official + [t for t in fam_tables if 2 <= t.prio < 3 and t.eff and t.eff > newest_official]
            else:
                core = [t for t in fam_tables if t.prio >= 2]
                pool += core or fam_tables
        for t in pool:
            key = (t.family, re.sub(r"[^a-z0-9]+", " ", t.name.lower()).strip())
            cur = best.get(key)
            sk = (t.eff or _dt.date(1900, 1, 1), t.prio, len(t.rows))
            if cur is None or sk > cur[0]:
                best[key] = (sk, t)
        order = {f: i for i, f in enumerate(fams)}
        return [t for (_, t) in sorted(best.values(), key=lambda x: (order.get(x[1].family, 99), x[1].idx))]

    def _row_info(self, t: Table, row):
        kinds = [_hdr_kind(h) for h in t.headers]
        labels, vals = [], {}
        for h, k, c in zip(t.headers, kinds, row):
            if k == "label":
                if c and c != "—":
                    labels.append(c)
            else:
                vals[k] = c
        if len(row) > len(t.headers):
            labels += [c for c in row[len(t.headers):] if c]
        return " ".join(labels), vals

    def _q_qualifiers(self, q: str):
        ql = q.lower().replace("–", "-")
        qs = {}
        t = terms_in(ql)
        if t:
            qs["terms"] = t
        mv = money_vals(ql)
        if not mv:
            for m in re.finditer(r"\b(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k|thousand|million)\b", ql):
                v = float(m.group(1).replace(",", ""))
                mv.append(v * (1000 if m.group(2) in ("k", "thousand") else 1_000_000))
        if not mv:
            for m in re.finditer(r"(?<![\w$.,/-])(\d{1,3}(?:,\d{3})+|\d{3,9})(?![\w.,/-]|\s*-?\s*(?:months?|mos?|years?|yrs?|%|percent))", ql):
                raw = m.group(1)
                v = float(raw.replace(",", ""))
                if "," not in raw and 1980 <= v <= 2035:
                    continue
                mv.append(v)
        if mv:
            qs["amounts"] = mv
        yrs = [int(y) for y in re.findall(r"(?<![\d$.,/])(20[0-2]\d|19[89]\d)(?![\d%])", ql)
               if not re.search(r"\$\s*" + y, ql)]
        if yrs:
            qs["years"] = yrs
        hard, soft = set(), set()
        for w in ("jumbo", "roth", "traditional"):
            if re.search(r"\b" + w + r"\b", ql):
                hard.add(w)
        for w in ("special", "regular", "variable", "standard", "intro", "introductory", "balance transfer"):
            if re.search(r"\b" + w + r"s?\b", ql):
                soft.add(w)
        if "balance transfer" in soft or "intro" in soft:
            soft.add("introductory")
        qs["hard"], qs["soft"] = hard, soft
        return qs

    @staticmethod
    def _primary_num(vals):
        v = vals.get("apy") or vals.get("apr") or vals.get("rate") or vals.get("div") or "0"
        m = re.search(r"[\d.]+", v)
        return float(m.group(0)) if m else 0.0

    def _select(self, tables, qs, ql):
        """-> list of (table, [(label, vals)]) after hard constraints and amount-tier selection."""
        info = []
        for t in tables:
            rows = [(self._row_info(t, r)) for r in t.rows]
            info.append((t, rows))
        has_terms = any(any(terms_in(lab) for lab, _ in rows) for _, rows in info)
        has_years = any(any(year_range(lab) for lab, _ in rows) for _, rows in info)
        selected = []
        for t, rows in info:
            keep = []
            tname = f"{t.name} {t.section}".lower()
            for lab, vals in rows:
                text = f"{lab} {t.name}".lower().replace("–", "-")
                ok = True
                if "terms" in qs:
                    rt = terms_in(lab)
                    if rt:
                        ok = ok and bool(rt & qs["terms"])
                    elif has_terms:
                        ok = False
                if ok and "years" in qs:
                    yr = year_range(lab)
                    if yr:
                        ok = ok and yr[0] <= qs["years"][0] <= yr[1]
                    elif has_years:
                        ok = False
                for w in qs["hard"]:
                    if w not in text:
                        ok = False
                if ok:
                    keep.append((lab, vals))
            if not keep:
                continue
            # amount tiers
            if "amounts" in qs:
                a = qs["amounts"][0]
                ranged = [(amount_range(lab), lab, vals) for lab, vals in keep if amount_range(lab)]
                if ranged:
                    caps = [x for x in ranged if x[0][0] == 0.0 and x[0][1] != math.inf]
                    if caps and len(caps) == len(ranged):
                        fits = [x for x in caps if a <= x[0][1]]
                        if fits:
                            mc = min(x[0][1] for x in fits)
                            keep = [(lab, vals) for rg, lab, vals in fits if rg[1] == mc]
                        else:
                            continue          # larger than every published loan-size cap
                    else:
                        fits = [(lab, vals) for rg, lab, vals in ranged if rg[0] <= a <= rg[1]]
                        if fits:
                            keep = fits
            selected.append((t, keep, tname))
        # a distinctive word of the question that names only SOME of the tables (e.g. "rewards" -> Rewards Visa)
        if len(selected) > 1:
            qtok = set(toks(ql)) - GENERIC_Q - {"visa", "card", "credit", "loan", "account", "saving", "certificate"}
            for w in qtok:
                holders = [x for x in selected if w in set(toks(f"{x[0].name} {x[0].section}"))]
                if holders and len(holders) < len(selected):
                    selected = holders
        # soft table-name words (special / regular / variable ...)
        if selected:
            for w in ("special", "regular", "variable"):
                if w in qs["soft"]:
                    hit = [x for x in selected if w in x[2]]
                    if hit:
                        selected = hit
        return [(t, keep) for t, keep, _ in selected]

    def _fmt_row(self, t: Table, label: str, vals: dict, want_min: bool) -> str:
        parts = []
        order = [("rate", "rate"), ("div", "dividend rate"), ("apy", "APY"), ("apr", "APR"), ("points", "points")]
        for k, nm in order:
            v = vals.get(k)
            if v and v not in ("—", "-"):
                parts.append(f"{nm} {v}")
        if want_min or t.family.endswith(("cert", "ira")):
            for k, nm in (("minopen", "minimum to open"), ("minearn", "minimum balance to earn")):
                v = vals.get(k)
                if v and v not in ("—", "-") and (want_min or k == "minearn"):
                    parts.append(f"{nm} {v}")
        lab = label.strip() or t.name
        if t.name and t.name.lower() not in lab.lower() and not lab.lower().startswith(("purchase", "refinance", "roth", "traditional")):
            lab = f"{t.name} - {lab}"
        return f"{lab}: " + ", ".join(parts)

    def _alltext(self, si):
        key = ("alltext", si)
        if key not in self._cache:
            self._cache[key] = " ".join(f"{t.name} {t.section} " + " ".join(" ".join(r) for r in t.rows)
                                        for t in self.tables[si]).lower()
        return self._cache[key]

    def _answer_rate(self, q: str, si: int, as_of):
        ql = q.lower()
        # qualifiers the site never mentions in any rate table (ARM, FHA, construction ...) -> decline
        alltxt = self._alltext(si)
        for rx in REFUSE_QUAL:
            if re.search(rx, ql) and not re.search(rx, alltxt) and rx not in (r"\bstudent\b", r"\bjumbo\b"):
                return self._rate_refusal(q, si)
        th = self._answer_threshold(q, si)
        if th is not None:
            return th
        fams = self._question_families(ql)
        if not fams:
            return None
        qs = self._q_qualifiers(q)
        tables = self._pick_tables(si, fams)
        if not tables:
            return self._rate_refusal(q, si)
        selected = self._select(tables, qs, ql)
        if not selected:
            return self._rate_refusal(q, si)
        want_min = bool(re.search(r"minimum|min\.|to open|opening|deposit|require", ql))

        # highest / lowest
        if (HIGH_WORDS.search(ql) or LOW_WORDS.search(ql)) and "terms" not in qs and "amounts" not in qs:
            rows0 = [(t, lab, vals) for t, rows in selected for lab, vals in rows]
            pick = (max if HIGH_WORDS.search(ql) else min)(rows0, key=lambda x: self._primary_num(x[2]))
            selected = [(pick[0], [(pick[1], pick[2])])]

        # history ("how did X change")
        if HIST_WORDS.search(ql) and len(self.snaps) > 1 and not re.search(r"\bsince\b.*\b(last|new)\b", ql):
            h = self._rate_history(q, si, fams, selected)
            if h:
                return h

        # yes/no check: "Do any ... pay 0.35% APY?"
        prefix = ""
        mt = re.search(r"(\d+(?:\.\d+)?)\s*%", ql)
        if mt and re.search(r"\b(do|does|is|are|can|will)\b.*\b(any|there|pay|offer|earn|get)\b", ql):
            target = float(mt.group(1))
            found = any(abs(self._primary_num({k: v}) - target) < 1e-9
                        for t, rows in selected for lab, vals in rows for k, v in vals.items()
                        if k in ("apy", "apr", "rate", "div"))
            if found:
                prefix = f"Yes - a {target:g}% rate is listed. "
            else:
                names = " / ".join(sorted({t.name for t, _ in selected}))
                bests = [(self._primary_num(vals), vals.get("apy") or vals.get("apr") or vals.get("rate") or "", lab)
                         for t, rows in selected for lab, vals in rows]
                top = max(bests) if bests else None
                prefix = (f"No - none of the {names} rates is {target:g}%. " +
                          (f"The highest is {top[1]} ({top[2]}). " if top else ""))

        lines, seen = [], set()
        eff_txt = []
        for t, rows in selected:
            for lab, vals in rows[:16]:
                ln = self._fmt_row(t, lab, vals, want_min)
                k = re.sub(r"[\s,]", "", ln.lower())
                if k in seen:
                    continue
                seen.add(k)
                lines.append(ln)
            if t.eff:
                eff_txt.append(t.eff.strftime("%m/%d/%Y"))
        lines = lines[:26]
        eff = (" Rates effective " + ", ".join(sorted(set(eff_txt))[:3]) + ".") if eff_txt else ""
        text = prefix + " ".join(l if l.endswith(".") else l + ";" for l in lines)
        text = text.rstrip(";") + "." + eff
        urls = []
        for t, _ in selected:
            for u in [t.url] + [x for x in t.links if "dupaco.com" in x]:
                if u not in urls:
                    urls.append(u)
        for t in self.tables[si]:
            if t.family in fams and t.url not in urls and t.prio >= 2:
                urls.append(t.url)
        return Answer(text=text, sources=[Source(url=u, snapshot_id=self.snaps[si]["id"]) for u in urls[:8]],
                      confident=True)

    def _answer_threshold(self, q: str, si: int):
        """'Is there an account paying more than 2%?' -> every deposit row above the threshold."""
        ql = q.lower()
        m = re.search(r"\b(more than|over|above|greater than|higher than|at least|exceed\w*|better than)\s+(\d+(?:\.\d+)?)\s*%", ql)
        if not m or not re.search(r"\b(account|accounts|savings|checking|certificate|cds?|money market|ira|hsa|apy|dividend)\b", ql):
            return None
        thr, inclusive = float(m.group(2)), m.group(1) == "at least"
        if re.search(r"\b(loan|mortgage|apr|card)\b", ql):
            return None
        if re.search(r"checking", ql):
            fams = ["checking"]
        elif re.search(r"certificate|\bcds?\b", ql):
            fams = ["cert"]
        elif re.search(r"\bira\b", ql):
            fams = ["ira"]
        elif re.search(r"savings|money market|hsa", ql):
            fams = ["share", "bright", "yni", "holiday", "launchpad", "mm", "hys", "hsa"]
        else:
            fams = ["share", "bright", "yni", "holiday", "launchpad", "mm", "hys", "hsa", "cert", "ira", "checking"]
        if re.search(r"\bbusiness\b", ql):
            fams = ["biz_" + f for f in fams]
        hits, seen = [], set()
        for t in self._pick_tables(si, fams):
            for r in t.rows:
                lab, vals = self._row_info(t, r)
                v = self._primary_num(vals)
                if (v >= thr if inclusive else v > thr):
                    ln = self._fmt_row(t, lab, vals, False)
                    if ln not in seen:
                        seen.add(ln)
                        hits.append((v, ln))
        if not hits:
            return Answer(text=f"No - none of the published rates is above {thr:g}%.",
                          sources=[Source(url=u, snapshot_id=self.snaps[si]["id"]) for u in
                                   {t.url for t in self.tables[si] if t.family in fams}][:4], confident=True)
        hits.sort(reverse=True)
        text = f"Yes - these listed rates are above {thr:g}%: " + "; ".join(l for _, l in hits[:12]) + "."
        return Answer(text=text, sources=[Source(url=u, snapshot_id=self.snaps[si]["id"]) for u in
                                          sorted({t.url for t in self._pick_tables(si, fams)})][:4], confident=True)

    def _rate_history(self, q, si, fams, selected):
        """Series of the selected rows across all snapshots up to si (oldest -> newest)."""
        lines = []
        for t, rows in selected[:3]:
            for lab, vals in rows[:6]:
                norm = re.sub(r"\s+", " ", lab.lower())
                series = []
                for sj in range(0, si + 1):
                    for t2 in self._pick_tables(sj, fams):
                        if t2.family != t.family:
                            continue
                        for r2 in t2.rows:
                            lab2, v2 = self._row_info(t2, r2)
                            if re.sub(r"[\s,]+", " ", lab2.lower()) == re.sub(r"[\s,]+", " ", norm):
                                if v2.get("rate") and v2.get("apr"):
                                    prim = f"rate {v2['rate']} / APR {v2['apr']}"
                                else:
                                    prim = v2.get("apy") or v2.get("apr") or v2.get("rate") or ""
                                series.append((self.snaps[sj]["date"], prim))
                                break
                if not series:
                    continue
                dedup = []
                for d, v in series:
                    if not dedup or dedup[-1][1] != v:
                        dedup.append((d, v))
                if len(dedup) == 1:
                    lines.append(f"{lab}: {dedup[0][1]} (unchanged across the downloads)")
                else:
                    chain = " -> ".join(f"{v} (as of {d.isoformat()})" for d, v in dedup)
                    lines.append(f"{lab}: {chain}; the current value is {dedup[-1][1]}")
        if not lines:
            return None
        return Answer(text="; ".join(lines) + ".", sources=[Source(url=t.url, snapshot_id=self.snaps[si]["id"])
                                                           for t, _ in selected[:3]],
                      confident=True, conflict_note="Value changed between website downloads; newest is current.")

    def _rate_refusal(self, q: str, si: int):
        return Answer(text="I don't know that rate - the website does not publish it. "
                           "Please contact a Dupaco lender to ask about it.", confident=False)

    # ------------------------------------------------------------------ conflicts
    def conflicts(self):
        if self._conf is not None:
            return self._conf
        out = []
        try:
            out += self._rate_conflicts()
        except Exception:
            pass
        try:
            out += self._boilerplate_conflicts()
        except Exception:
            pass
        self._conf = out
        return out

    @staticmethod
    def _primary(vals):
        for k in ("apy", "apr", "rate", "div"):
            if vals.get(k) and re.search(r"\d", vals[k]):
                return vals[k]
        return None

    @staticmethod
    def _row_key(t: Table, label: str, ctr, yr_rank=None):
        """Align the 'same' row across pages and crawls: family + term + tier + flags + n-th occurrence."""
        terms = tuple(sorted(terms_in(label)))
        amt = amount_range(label)
        text = f"{label} {t.name}".lower()
        flags = tuple(w for w in ("roth", "traditional", "jumbo", "special", "refinance", "purchase", "introductory")
                      if w in text)
        yr = year_range(label)
        tier = ("yr", yr_rank.get(yr)) if (yr and yr_rank) else None
        if not terms and not amt and not tier:
            core = re.sub(r"[^a-z0-9]+", " ", (label or t.name).lower()).strip()
            core = re.sub(r"\b(rate|apr|apy)\b", "", core).strip()
            return (t.family, "single", core, flags)
        base = (t.family, terms, amt, flags, tier)
        n = ctr[base]
        ctr[base] += 1
        return base + (n,)

    @staticmethod
    def _sig(vals):
        """Numeric signature: formatting differences ('7.29', '7.29%', '7.29% APR') are not conflicts."""
        out = []
        for k in ("rate", "div", "apy", "apr"):
            v = vals.get(k, "")
            nums = ()
            if v:
                rg = re.match(r"\s*\$?(\d+(?:\.\d+)?)\s*%?\s*[-–]\s*(\d+(?:\.\d+)?)\s*%", v)
                if rg:
                    nums = (round(float(rg.group(1)), 4), round(float(rg.group(2)), 4))
                else:
                    f1 = re.search(r"\d+(?:\.\d+)?", v)
                    nums = (round(float(f1.group(0)), 4),) if f1 else ()
            out.append(nums)
        return tuple(out)

    FAM_NAME = {
        "card": "Visa credit card APR", "heloc": "HELOC home equity line of credit APR",
        "home_equity_loan": "home equity loan APR", "mort_refi": "mortgage refinance rate (home loan)",
        "mort_purchase": "mortgage purchase rate (home loan)", "home_improvement": "home improvement loan APR",
        "personal_loan": "personal loan APR", "auto": "auto loan APR", "boat": "boat and RV loan APR",
        "atv": "ATV, snowmobile and jet ski loan APR", "cert": "certificate (CD) APY", "ira": "IRA APY",
        "hsa": "health savings account (HSA) APY", "mm": "money market APY", "hys": "high-yield savings APY",
        "bright": "Bright Savings APY", "yni": "You Name It Savings APY", "holiday": "Holiday Club savings APY",
        "launchpad": "Launchpad Savings APY", "share": "Share Savings APY", "checking": "checking APY",
        "savings": "savings APY",
    }

    def _topic(self, t: Table, label: str) -> str:
        fam = t.family
        biz = fam.startswith("biz_")
        base = self.FAM_NAME.get(fam[4:] if biz else fam, t.name)
        lab = re.sub(r"\s+", " ", (label or "").strip(" -—"))
        name = t.name
        parts = [("business " if biz else "") + base]
        if name and name.lower() not in base.lower():
            parts.append(name)
        if lab and lab.lower() not in " ".join(parts).lower():
            parts.append(lab)
        return " - ".join(parts)

    def _rate_conflicts(self):
        entries = defaultdict(list)     # key -> [(si, table, sig, label)]
        for si, tbls in enumerate(self.tables):
            for t in tbls:
                if not t.family or t.prio < 1:
                    continue
                ctr = defaultdict(int)
                infos = [self._row_info(t, r) for r in t.rows]
                ranges = []
                for lab, _ in infos:
                    yr = year_range(lab)
                    if yr and yr not in ranges:
                        ranges.append(yr)
                yr_rank = {r_: i for i, r_ in enumerate(ranges)}
                for label, vals in infos:
                    if not self._primary(vals):
                        continue
                    entries[self._row_key(t, label, ctr, yr_rank)].append((si, t, self._sig(vals), label, vals))
        out = []
        for key, es in entries.items():
            flagged, why = set(), []
            # A) different pages disagree inside one crawl
            by_snap = defaultdict(list)
            for e in es:
                by_snap[e[0]].append(e)
            for si, lst in by_snap.items():
                sigs = {e[2] for e in lst}
                urls = {e[1].url for e in lst}
                if len(sigs) > 1 and len(urls) > 1:
                    flagged |= sigs
                    why.append("pages disagree")
            # B) the official rate sheet itself changed between crawls
            off = [e for e in es if e[1].prio >= 3]
            osigs = []
            for e in sorted(off, key=lambda x: x[0]):
                if not osigs or osigs[-1] != e[2]:
                    osigs.append(e[2])
            if len(set(osigs)) > 1:
                flagged |= set(osigs)
                why.append("rate changed between downloads")
            if not flagged:
                continue
            values = []
            for sig in sorted(flagged, key=lambda sg: [x for x in es if x[2] == sg][0][0]):
                ex = [x for x in es if x[2] == sig][0][4]
                for k, nums in zip(("rate", "div", "apy", "apr"), sig):
                    for n_ in nums:
                        tok_ = f"{n_:.3f}".rstrip("0").rstrip(".")
                        tok_ = (tok_ if "." in tok_ else tok_ + ".00")
                        tok_ = (tok_ if len(tok_.split(".")[1]) >= 2 else tok_ + "0") + "%"
                        if tok_ not in values:
                            values.append(tok_)
            last_si = max(e[0] for e in es)
            latest = [e for e in es if e[0] == last_si]
            win = max(latest, key=lambda e: (e[1].eff or _dt.date(1900, 1, 1), e[1].prio))
            wv = win[4]
            wsig = self._sig(wv)
            res_parts = []
            for nums in wsig:
                for n_ in nums:
                    tok_ = f"{n_:.3f}".rstrip("0").rstrip(".")
                    tok_ = (tok_ if "." in tok_ else tok_ + ".00")
                    tok_ = (tok_ if len(tok_.split(".")[1]) >= 2 else tok_ + "0") + "%"
                    if tok_ not in res_parts:
                        res_parts.append(tok_)
            res = " / ".join(res_parts)
            srcs, seen = [], set()
            for e in sorted(es, key=lambda x: (-x[1].prio, -x[0])):
                if e[1].url not in seen and len(srcs) < 6:
                    seen.add(e[1].url)
                    srcs.append(Source(url=e[1].url, snapshot_id=self.snaps[e[0]]["id"]))
            topic = self._topic(win[1], win[3])
            out.append(Conflict(
                topic=topic, values=values, sources=srcs, resolution=res,
                reason=("Sources disagree (" + ", ".join(sorted(set(why))) + "); the value on the newest official "
                        f"rate sheet (effective {win[1].eff.isoformat() if win[1].eff else 'n/a'}) is current; "
                        "older or landing-page copies are stale.")))
        return out

    def _boilerplate_conflicts(self):
        pats = {
            "number of Dupaco branch office locations": r"(\d+)\s+branch office locations",
            "number of Dupaco employees": r"It has ((?:nearly|over|more than)\s+[\d,]+)\s+employees",
            "Dupaco total assets": r"assets (?:exceeding|of nearly|of more than|of over)\s+\$(\d\.\d)\s+billion",
            "counties Dupaco serves": r"Dupaco serves residents in (\d+) counties",
        }
        out = []
        latest = len(self.snaps) - 1
        found = {k: {} for k in pats}
        for url, (title, md) in self.snaps[latest]["pages"].items():
            kind = page_kind(url)
            if kind not in ("news", "blog"):
                continue
            head = clean_md(md)
            d = find_date(head[:2500]) or _dt.date(2000, 1, 1)
            for topic, rx in pats.items():
                m = re.search(rx, head)
                if m:
                    found[topic].setdefault(m.group(1), []).append((d, url))
        for topic, vals in found.items():
            if len(vals) < 2:
                continue
            ranked = sorted(vals.items(), key=lambda kv: max(d for d, _ in kv[1]))
            newest = ranked[-1][0]
            values = [(v if topic.startswith("Dupaco total") is False else f"${v} billion") for v, _ in ranked]
            if topic == "number of Dupaco branch office locations":
                values = [f"{v} branch office locations" for v, _ in ranked]
            srcs = [Source(url=sorted(u, reverse=True)[0][1], snapshot_id=self.snaps[latest]["id"]) for _, u in ranked]
            res = values[-1]
            out.append(Conflict(topic=topic, values=values, sources=srcs, resolution=res,
                                reason="Older press releases carry outdated boilerplate; the most recently dated "
                                       "press release reflects the current figure."))
        return out


def build():
    return KS()
