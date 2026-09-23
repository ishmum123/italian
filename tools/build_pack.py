#!/usr/bin/env python3
"""
Build the Italian A1-B1 vocab pack (pack/pack.json, words.json, sentences.json,
attribution.json) and tools/REPORT.md from public frequency, dictionary and
sentence-corpus sources.

v2: part of speech, lemma and sense are chosen from CORPUS USAGE, not from
dictionary structure. The Tatoeba Italian corpus is POS-tagged once with spaCy
(it_core_news_sm); every downstream decision (lemma, POS, Wiktionary entry,
sense order, sentence word links) reads the tagged corpus.

Usage:
    python3 tools/build_pack.py                  # full pipeline, uses .cache/
    python3 tools/build_pack.py --stage corpus   # Tatoeba ita+eng+links+audio -> compact corpus
    python3 tools/build_pack.py --stage tag      # spaCy-tag the corpus (cached, version-stamped)
    python3 tools/build_pack.py --stage lex      # kaikki Wiktionary -> compact lexicon + form map
    python3 tools/build_pack.py --stage freq     # corpus-resolved (lemma, POS) frequency blend
    python3 tools/build_pack.py --stage words    # word selection -> pack/words.json
    python3 tools/build_pack.py --stage all      # everything incl. sentences + REPORT.md

Each stage caches its derived output under .cache/derived/ (file names carry
a version stamp), so re-runs are fast and byte-identical.
"""
import argparse
import bz2
import gzip
import hashlib
import io
import json
import math
import re
import sys
import tarfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
DERIVED = CACHE / "derived"
PACK = ROOT / "pack"
TOOLS = ROOT / "tools"

SOURCES = {
    "it_full.txt": "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/it/it_full.txt",
    "kaikki_it.jsonl.gz": "https://kaikki.org/dictionary/Italian/kaikki.org-dictionary-Italian.jsonl.gz",
    "ita_detailed.tsv.bz2": "https://downloads.tatoeba.org/exports/per_language/ita/ita_sentences_detailed.tsv.bz2",
    "ita_cc0.tsv.bz2": "https://downloads.tatoeba.org/exports/per_language/ita/ita_sentences_CC0.tsv.bz2",
    "eng_sentences.tsv.bz2": "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2",
    "links.tar.bz2": "https://downloads.tatoeba.org/exports/links.tar.bz2",
    "audio.tar.bz2": "https://downloads.tatoeba.org/exports/sentences_with_audio.tar.bz2",
    "kelly_it.json": "https://raw.githubusercontent.com/kotoshu/frequency-list-kelly/main/data/it.json",
}

SPACY_MODEL = "it_core_news_sm"
CORPUS_VERSION = "c2"
TAG_VERSION = "t3"
LEX_VERSION = "l6"

N_WORDS = 2000
BANDS = [("A1", 600), ("A2", 700), ("B1", 700)]

STATS = {}  # everything REPORT.md prints; filled by the stages


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def stat(key, value):
    STATS[key] = value


def dump_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def _remote_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "italian-pack-builder/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def ensure_downloaded(check_remote=False):
    """Download missing sources. With check_remote, re-download when the
    cached size differs from the server's (sources are updated weekly)."""
    CACHE.mkdir(exist_ok=True)
    for name, url in SOURCES.items():
        dest = CACHE / name
        if dest.exists() and dest.stat().st_size > 0:
            if not check_remote:
                continue
            expected = _remote_size(url)
            if expected is None or dest.stat().st_size == expected:
                continue
            log(f"  {name}: cached size {dest.stat().st_size} != remote {expected}, re-downloading")
        log(f"downloading {name} ...")
        t0 = time.time()
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "italian-pack-builder/2.0"})
                with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.replace(dest)
                break
            except Exception as e:
                log(f"  attempt {attempt+1} failed: {e}")
                if attempt == 4:
                    raise
        log(f"  {name}: {dest.stat().st_size} bytes in {time.time()-t0:.1f}s")

    gz = CACHE / "kaikki_it.jsonl.gz"
    plain = CACHE / "kaikki_it.jsonl"
    if gz.exists() and not plain.exists():
        log("decompressing kaikki_it.jsonl.gz ...")
        with gzip.open(gz, "rb") as fin, open(plain, "wb") as fout:
            while True:
                chunk = fin.read(1 << 20)
                if not chunk:
                    break
                fout.write(chunk)


def file_sig(path):
    st = path.stat()
    return f"{path.name}:{st.st_size}"


# ---------------------------------------------------------------------------
# Stage corpus: Italian sentences that have an English translation
# ---------------------------------------------------------------------------

PERMISSIVE_AUDIO = {"CC BY 2.0 FR", "CC BY-SA 3.0", "CC BY-SA 4.0", "CC BY 4.0", "CC0 1.0"}


def corpus_path():
    sig = hashlib.sha1("|".join([CORPUS_VERSION] + [file_sig(CACHE / n) for n in
                       ("ita_detailed.tsv.bz2", "eng_sentences.tsv.bz2", "links.tar.bz2", "audio.tar.bz2")]
                       ).encode()).hexdigest()[:10]
    return DERIVED / f"corpus_{CORPUS_VERSION}_{sig}.json.gz"


def stage_corpus():
    out = corpus_path()
    if out.exists():
        with gzip.open(out, "rt", encoding="utf-8") as f:
            return json.load(f)
    DERIVED.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ita = {}
    data = bz2.decompress((CACHE / "ita_detailed.tsv.bz2").read_bytes()).decode("utf-8")
    for line in data.split("\n"):
        p = line.split("\t")
        if len(p) >= 3 and p[1] == "ita":
            ita[int(p[0])] = (p[2], p[3] if len(p) > 3 else "")
    eng = {}
    data = bz2.decompress((CACHE / "eng_sentences.tsv.bz2").read_bytes()).decode("utf-8")
    for line in data.split("\n"):
        p = line.split("\t")
        if len(p) >= 3:
            eng[int(p[0])] = p[2]
    del data
    # links.csv: sentence_id <tab> translation_id (both directions listed)
    best_en = {}
    with tarfile.open(CACHE / "links.tar.bz2", "r:bz2") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("links.csv"))
        for raw in io.TextIOWrapper(tf.extractfile(member), encoding="utf-8"):
            p = raw.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            a, b = int(p[0]), int(p[1])
            if a in ita and b in eng:
                if a not in best_en or b < best_en[a]:
                    best_en[a] = b  # lowest English id: deterministic choice
    # sentences_with_audio.csv columns: sentence_id, audio_id, username,
    # license, attribution_url. (v1 read these two columns swapped, which
    # attached unrelated recordings to sentences; verified against the
    # Tatoeba API: sentence 4369 -> audio 1009973.)
    audio = {}
    with tarfile.open(CACHE / "audio.tar.bz2", "r:bz2") as tf:
        member = next(m for m in tf.getmembers() if "sentences_with_audio" in m.name)
        for raw in io.TextIOWrapper(tf.extractfile(member), encoding="utf-8"):
            p = raw.rstrip("\n").split("\t")
            if len(p) < 4 or not p[0].isdigit() or not p[1].isdigit():
                continue
            sid, aid, lic = int(p[0]), int(p[1]), p[3]
            if sid in ita and lic in PERMISSIVE_AUDIO:
                if sid not in audio or aid < audio[sid][0]:
                    audio[sid] = (aid, lic)
    rows = []
    for sid in sorted(best_en):
        text, user = ita[sid]
        aid, lic = audio.get(sid, (None, None))
        rows.append([sid, text, user, eng[best_en[sid]], aid, lic])
    result = {"rows": rows, "n_ita": len(ita), "n_with_en": len(rows),
              "n_audio": sum(1 for r in rows if r[4])}
    with gzip.GzipFile(out, "wb", mtime=0) as g:
        g.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    log(f"corpus: {len(ita)} ita sentences, {len(rows)} with English, "
        f"{result['n_audio']} with permissive audio ({time.time()-t0:.0f}s)")
    return result


# ---------------------------------------------------------------------------
# Stage tag: spaCy over the corpus, truecasing the sentence-initial token
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
MORPH_KEEP = ("Gender", "Number", "Tense", "Mood", "VerbForm", "Person", "Clitic")


def truecase_stats(rows):
    """For each lowercase word: (#mid-sentence lowercase, #mid-sentence
    capitalised) occurrences in the corpus."""
    low, cap = Counter(), Counter()
    for r in rows:
        text = r[1]
        for m in WORD_RE.finditer(text):
            before = text[:m.start()].rstrip(" \"'«»“”‘’-—()")
            if not before or before[-1] in ".!?:;…":
                continue            # sentence-initial (also after an internal full stop)
            t = m.group(0)
            if t[0].isupper():
                if t[1:].islower() or len(t) == 1:
                    cap[t.lower()] += 1
            else:
                low[t] += 1
    return low, cap


def truecase(text, low, cap):
    """Lowercase the sentence-initial letter unless the word is mostly
    capitalised mid-sentence (a proper noun); the sm lemmatiser and tagger
    are case-sensitive and mis-handle 'Chiudi', 'Odio', 'Portami' etc."""
    m = WORD_RE.search(text)
    if not m or not text[m.start()].isupper():
        return text
    w = m.group(0)
    lw = w.lower()
    if w != w[0] + w[1:].lower():  # all-caps / camel: leave
        return text
    if low[lw] >= cap[lw] and (low[lw] > 0 or cap[lw] == 0):
        return text[:m.start()] + lw[0] + text[m.start() + 1:]
    return text


def tagged_path(corpus_file):
    import spacy
    model = spacy.util.get_package_version(SPACY_MODEL)
    sig = hashlib.sha1(f"{TAG_VERSION}|{spacy.__version__}|{SPACY_MODEL}-{model}|{corpus_file.name}".encode()).hexdigest()[:10]
    return DERIVED / f"tagged_{TAG_VERSION}_{sig}.jsonl.gz", f"spaCy {spacy.__version__}, {SPACY_MODEL} {model}"


def stage_tag(corpus):
    out, desc = tagged_path(corpus_path())
    stat("tagger", desc)
    meta_path = out.with_suffix(".meta.json")
    if out.exists() and meta_path.exists():
        stat("tag_meta", json.loads(meta_path.read_text()))
        return out
    import spacy
    t0 = time.time()
    rows = corpus["rows"]
    low, cap = truecase_stats(rows)
    texts = [truecase(r[1], low, cap) for r in rows]
    n_lowered = sum(1 for r, t in zip(rows, texts) if r[1] != t)
    nlp = spacy.load(SPACY_MODEL, exclude=["parser", "ner"])
    tmp = out.with_suffix(".part")
    n_tok = 0
    with gzip.GzipFile(tmp, "wb", mtime=0) as g:
        for r, doc in zip(rows, nlp.pipe(texts, batch_size=2000, n_process=6)):
            toks = []
            for t in doc:
                if t.is_space:
                    continue
                md = t.morph.to_dict()
                ms = "|".join(f"{k}={md[k]}" for k in MORPH_KEEP if k in md)
                toks.append([t.text, t.lemma_, t.pos_, ms])
            n_tok += len(toks)
            g.write((json.dumps([r[0], toks], ensure_ascii=False) + "\n").encode("utf-8"))
    tmp.replace(out)
    meta = {"sentences": len(rows), "tokens": n_tok, "seconds": round(time.time() - t0, 1),
            "truecased_initials": n_lowered}
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    stat("tag_meta", meta)
    log(f"tag: {len(rows)} sentences, {n_tok} tokens in {meta['seconds']}s")
    return out


def iter_tagged(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)



# ---------------------------------------------------------------------------
# Stage lex: kaikki (Wiktionary) -> compact lexicon + inflection/form map
# ---------------------------------------------------------------------------

LEX_WORD_RE = re.compile(r"^[a-zàáèéìíòóùúç']+$")
FORM_TAGS = {"form-of"}
ALT_TAGS = {"alt-of"}
DEMOTE_TAGS = {"archaic", "obsolete", "rare", "dated", "historical", "regional",
               "dialectal", "Tuscany", "Southern-Italy", "Northern-Italy", "Rome", "Naples"}
# an entry whose every sense carries one of these is not a learner lemma
# (giuro "oath" [Tuscany, literary], bisogna "matter" [literary])
MARKED_TAGS = {"archaic", "obsolete", "dated", "historical", "regional", "dialectal", "literary",
               "poetic", "Tuscany", "Southern-Italy", "Northern-Italy", "Rome", "Naples"}


NONDEF_RE = re.compile(
    r"^(alternative |obsolete |archaic |dialectal |regional |past |present |future |imperfect |perfect )*"
    r"(form|forms|inflection|participle|gerund|singular|plural|masculine|feminine|imperative|"
    r"subjunctive|indicative|conditional|ellipsis|abbreviation|initialism|acronym|apocopic|elided|"
    r"superlative|diminutive|augmentative|first-person|second-person|third-person|compound of|"
    r"synonym|reflexive)\b.*\bof\b",
    re.IGNORECASE)
FORM_OF_ANY_RE = re.compile(r"^\S+(\s+\S+){0,6}\s+forms? of\b", re.IGNORECASE)


def _gender(d):
    g = None
    for ht in d.get("head_templates", []):
        a = ht.get("args", {})
        if ht.get("name") == "it-noun":
            g = a.get("1")
        elif ht.get("name") == "head":
            g = a.get("g")
        if g:
            break
    return g


def _borrow_en(d):
    for t in d.get("etymology_templates", []):
        a = t.get("args", {})
        n = t.get("name", "")
        if n in ("bor", "bor+", "lbor", "ubor", "der", "der+") and a.get("2") == "en":
            return True
        if n == "ety" and str(a.get("3", "")).startswith("en:"):
            return True
    return bool(re.search(r"\b(borrowing|borrowed) from English\b", d.get("etymology_text", "") or ""))


def lex_path():
    sig = hashlib.sha1(f"{LEX_VERSION}|{file_sig(CACHE / 'kaikki_it.jsonl')}".encode()).hexdigest()[:10]
    return DERIVED / f"lex_{LEX_VERSION}_{sig}.json.gz"


def stage_lex():
    out = lex_path()
    if out.exists():
        with gzip.open(out, "rt", encoding="utf-8") as f:
            return json.load(f)
    DERIVED.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    entries = defaultdict(list)
    formmap = defaultdict(set)
    names = set()   # lowercase forms of capitalised proper-name headwords
    n = 0
    with open(CACHE / "kaikki_it.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"lang_code": "it"' not in line:
                continue
            d = json.loads(line)
            if d.get("lang_code") != "it":
                continue
            word, pos = d.get("word", ""), d.get("pos", "")
            if pos == "name" and word[:1].isupper():
                names.add(word.lower())
            if not LEX_WORD_RE.match(word):
                continue
            n += 1
            senses = []
            for s in d.get("senses", []):
                gl = s.get("glosses") or []
                tags = sorted(set(s.get("tags", [])))
                tset = set(tags)
                kind = ""
                target = None
                if s.get("form_of") or tset & FORM_TAGS:
                    kind = "form"
                    target = (s.get("form_of") or [{}])[0].get("word")
                elif "misspelling" in tset:
                    kind = "miss"
                elif s.get("alt_of") or tset & ALT_TAGS:
                    kind = "alt"
                    target = (s.get("alt_of") or [{}])[0].get("word")
                elif "compound-of" in tset:
                    kind = "comp"
                elif gl and (NONDEF_RE.match(gl[0]) or NONDEF_RE.match(gl[-1]) or FORM_OF_ANY_RE.match(gl[-1])):
                    # form-of senses that Wiktionary left untagged
                    # ("first-person plural present indicative of potere")
                    kind = "form"
                    m = re.search(r"\bof ([a-zàèéìòù']+)", gl[0] + " " + gl[-1])
                    target = m.group(1) if m else None
                if kind in ("form", "alt") and target and LEX_WORD_RE.match(target):
                    formmap[word].add((target, pos, kind))
                if not gl:
                    continue
                senses.append([gl[-1].strip(), gl[0].strip() if len(gl) > 1 else "", tags, kind])
                if kind == "form" and "; " in gl[-1]:
                    # "comparative degree of molto; more": the part after ';' is a translation
                    senses.append([gl[-1].split("; ")[-1].strip(), "",
                                   sorted(set(t for t in tags if t != "form-of") | {"from-form-sense"}), ""])
            if any(s[3] == "comp" for s in senses):
                for t in d.get("etymology_templates", []):
                    if t.get("name") == "af":
                        base = str(t.get("args", {}).get("2", "")).split("<")[0]
                        if LEX_WORD_RE.match(base):
                            formmap[word].add((base, pos, "comp"))
                        break
            if not senses:
                continue
            ent = {"p": pos, "s": senses[:30]}
            g = _gender(d)
            if g:
                ent["g"] = g
            if _borrow_en(d):
                ent["b"] = 1
            entries[word].append(ent)
    result = {
        "entries": dict(entries),
        "formmap": {k: sorted(v) for k, v in formmap.items()},
        "names": sorted(names),
        "n_entries": n,
    }
    with gzip.GzipFile(out, "wb", mtime=0) as g:
        g.write(json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    log(f"lex: {n} single-word entries, {len(result['formmap'])} inflected/alt surfaces ({time.time()-t0:.0f}s)")
    return result



# ---------------------------------------------------------------------------
# Token resolution: (surface, spaCy lemma, UPOS) -> (lemma, POS group)
# ---------------------------------------------------------------------------

GROUP_OF = {"AUX": "VERB", "CCONJ": "CONJ", "SCONJ": "CONJ"}
SKIP_UPOS = {"PUNCT", "SYM", "X", "SPACE"}
# UD (corpus) POS -> Wiktionary POS headers, in preference order. The first
# is the direct match; the rest cover systematic convention differences
# between UD Italian and Wiktionary (UD ADJ altro/stesso/tale = Wiktionary
# det; UD ADV però/tuttavia = conj, fino = prep; UD ADP come = adv/conj;
# UD NOUN milione = num).
GROUP_KPOS = {
    "NOUN": ["noun", "num"], "PROPN": ["name", "noun"], "VERB": ["verb"], "ADJ": ["adj", "det", "num"],
    "ADV": ["adv", "conj", "prep"], "DET": ["article", "det", "adj", "pron", "num"],
    "ADP": ["prep", "adv", "conj"], "PRON": ["pron", "det"], "CONJ": ["conj", "adv"],
    "NUM": ["num", "adj", "noun"],
    "INTJ": ["intj", "particle"], "PART": ["particle", "adv"],
}
GROUP_LABEL = {"NOUN": "noun", "VERB": "verb", "ADJ": "adj", "ADV": "adv", "DET": "det",
               "ADP": "prep", "PRON": "pron", "CONJ": "conj", "NUM": "num", "INTJ": "intj",
               "PART": "part"}
CONTENT_GROUPS = {"NOUN", "VERB", "ADJ", "ADV"}
FUNCTION_UPOS = {"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART"}
INFORMAL_TAGS = {"informal", "slang", "vulgar", "derogatory", "offensive", "colloquial",
                 "humorous", "euphemistic"}
CLITIC_RE = re.compile(r"^(.+?)((?:glie|me|te|ce|ve|se)(?:lo|la|li|le|ne)|lo|la|li|le|mi|ti|ci|vi|si|ne|gli)$")


ART_PREP = {}
for _base, _stem in (("a", "a"), ("da", "da"), ("di", "de"), ("in", "ne"), ("su", "su")):
    for _suf in ("l", "llo", "lla", "ll'", "i", "gli", "lle"):
        ART_PREP[_stem + _suf] = _base
ART_PREP.update({"col": "con", "coi": "con", "pel": "per", "pei": "per"})
# nouns used (almost) only in the plural: kept as their own lemma
PLURALIA_TANTUM = {"soldi", "occhiali", "pantaloni", "forbici", "nozze", "ferie", "dintorni",
                   "stoviglie", "mutande", "calzoni", "spiccioli"}
COPULAS = {"essere", "diventare", "sembrare", "rimanere", "restare", "stare", "diventato"}
REFL_CLITICS = {"mi": ("1", "Sing"), "ti": ("2", "Sing"), "si": ("3", None), "ci": ("1", "Plur"), "vi": ("2", "Plur")}
VERB_ENDINGS = ("are", "ere", "ire", "rre", "arsi", "ersi", "irsi", "rsi")
# unstressed object pronouns whose Wiktionary entry is only "clitic form of X"
CLITIC_OF = {"mi": "io", "ti": "tu"}
MONO_IMPERATIVE = {"da": "dare", "di": "dire", "fa": "fare", "sta": "stare", "va": "andare"}
ARTICLE_FORMS = {"il": {"il", "lo", "la", "l'", "i", "gli", "le"}, "uno": {"un", "uno", "una", "un'"}}


KPOS_GROUP = {"noun": "NOUN", "verb": "VERB", "adj": "ADJ", "adv": "ADV", "pron": "PRON",
              "det": "DET", "article": "DET", "prep": "ADP", "conj": "CONJ", "num": "NUM",
              "intj": "INTJ"}
RARE_ZIPF = 2.5
RARE_MARGIN = 1.5
_ZIPF = {}


def zipf(w):
    if w not in _ZIPF:
        from wordfreq import zipf_frequency
        _ZIPF[w] = zipf_frequency(w, "it")
    return _ZIPF[w]


def best_by_freq(cands):
    """Deterministic tie-break between dictionary-valid lemmas: the more
    frequent word (wordfreq), then alphabetical."""
    from wordfreq import zipf_frequency
    return sorted(cands, key=lambda c: (-zipf_frequency(c, "it"), c))[0]


def group_of(upos):
    return GROUP_OF.get(upos, upos)


def header_tags(ent):
    """Tags kaikki copies onto every sense from headword-level qualifiers
    (stazione 'f,m<l:archaic>', gente [archaic, poetic] on all 6 senses).
    They qualify an alternative form or gender, not the senses."""
    ht = set(re.findall(r"[A-Za-z-]+", " ".join(re.findall(r"<[^>]*>", ent.get("g", "")))))
    if ht & {"outdated", "old-fashioned", "obsolescent"}:
        ht |= {"archaic", "dated", "obsolete"}       # kaikki's normalised names for these
    defs = [set(sn[2]) for sn in ent["s"] if sn[3] == ""]
    if len(defs) >= 3:
        ht |= set.intersection(*defs)
    return ht


def sense_tags(ent, sn):
    return set(sn[2]) - ent["ht"]


class Lexicon:
    def __init__(self, lex):
        self.E = lex["entries"]
        for ents in self.E.values():
            for e in ents:
                e["ht"] = header_tags(e)
        self.F = lex["formmap"]
        self.names = set(lex["names"])
        self._cache = {}
        self._hp = {}
        self.n_unresolved = Counter()
        self.n_clitic = 0
        self.n_num_as_verb = 0
        self.n_rare_override = 0
        self.n_after_article = 0
        self.n_copula_adj = 0
        self.n_pron_as_article = 0
        self.n_imperative = 0

    @staticmethod
    def entry_usable(ent):
        # a translation recovered from a form-of line ("...of molto; more")
        # glosses the word but does not make the entry a lemma of its own
        defs = [s for s in ent["s"] if s[3] == "" and "from-form-sense" not in s[2]]
        if not defs or all(sense_tags(ent, s) & MARKED_TAGS for s in defs):
            return False            # no senses, or archaic/literary/regional-only
        has_form = any(s[3] in ("form", "alt") for s in ent["s"])
        if not has_form:
            return True
        # "female equivalent of X" + only an informal/archaic extra sense
        # (bambina -> "babe") is a feminine-of, not a lemma in its own right
        return any(not (sense_tags(ent, s) & (DEMOTE_TAGS | INFORMAL_TAGS)) for s in defs)

    def usable_entries(self, word, kpos):
        return [e for e in self.E.get(word, []) if (kpos is None or e["p"] in kpos) and self.entry_usable(e)]

    def chase(self, word, kpos, depth=0):
        """Follow form/alt/compound pointers until a usable lemma entry."""
        if self.usable_entries(word, kpos):
            return word
        if depth >= 3:
            return None
        for tgt, pos, kind in self.F.get(word, []):
            if kpos is None or pos in kpos:
                r = self.chase(tgt, kpos, depth + 1)
                if r:
                    return r
        return None

    def candidates(self, s, kpos):
        c = set()
        if self.usable_entries(s, kpos):
            c.add(s)
        for tgt, pos, kind in self.F.get(s, []):
            if kpos is None or pos in kpos:
                r = self.chase(tgt, kpos)
                if r:
                    c.add(r)
        return c

    def readings(self, s):
        """All dictionary-valid (lemma, group) readings of a surface."""
        out = set()
        for pos, grp in KPOS_GROUP.items():
            for lem in self.candidates(s, [pos]):
                if grp != "VERB" or lem.endswith(VERB_ENDINGS):
                    out.add((lem, grp))
        return sorted(out)

    def historic_past(self, s):
        """Wiktionary lists the surface as a passato remoto form ("cadde",
        "vendette"): catches tokens the tagger gave no Tense=Past morph."""
        if s not in self._hp:
            self._hp[s] = any(sn[3] == "form" and {"historic", "past"} <= set(sn[2])
                              for e in self.E.get(s, []) if e["p"] == "verb" for sn in e["s"])
        return self._hp[s]

    def plural_pointer(self, s):
        return any(sn[3] == "form" and "plural" in sn[2]
                   for e in self.E.get(s, []) if e["p"] == "noun" for sn in e["s"])

    def clitic_verb(self, s):
        """farlo -> fare, dimmi -> dire, portami -> portare, dirglielo -> dire."""
        stems = []
        cur = s
        for _ in range(2):
            m = CLITIC_RE.match(cur)
            if not m:
                break
            cur, cl = m.group(1), m.group(2)
            stems.append((cur, cl))
        for stem, cl in stems:
            tries = [stem, stem + "e", stem + "'"]
            if len(stem) > 2 and stem[-1] == cl[0] and stem[:-1] in MONO_IMPERATIVE:
                # monosyllabic imperatives double the clitic consonant:
                # dimmi -> di', fallo -> fa', dacci -> da', stammi -> sta'
                self.n_clitic += 1
                return MONO_IMPERATIVE[stem[:-1]]
            if stem in MONO_IMPERATIVE and cl.startswith("gli"):
                self.n_clitic += 1
                return MONO_IMPERATIVE[stem]
            for t in tries:
                if len(t) < 2:
                    continue
                c = self.candidates(t, ["verb"])
                if c:
                    self.n_clitic += 1
                    return best_by_freq(c)
        return None

    def imperative_form(self, s):
        """Verb lemma if Wiktionary lists s as a second-person imperative."""
        for e in self.E.get(s, []):
            if e["p"] != "verb":
                continue
            for sn in e["s"]:
                if sn[3] == "form" and "imperative" in sn[2] and "second-person" in sn[2]:
                    c = self.candidates(s, ["verb"])
                    c = {x for x in c if x.endswith(VERB_ENDINGS)}
                    if c:
                        return best_by_freq(c)
        return None

    def imperative_clitic(self, s):
        """s splits into an imperative verb form + enclitic ("fallo" = fa' + lo,
        "dimmi" = di' + mi, "portalo" = porta + lo)."""
        cur = s
        for _ in range(2):
            m = CLITIC_RE.match(cur)
            if not m:
                return False
            cur, cl = m.group(1), m.group(2)
            stems = [cur, cur + "'"]
            if len(cur) > 2 and cur[-1] == cl[0]:
                stems += [cur[:-1], cur[:-1] + "'"]
            for st in stems:
                if st in MONO_IMPERATIVE or any(
                        sn[3] == "form" and "imperative" in sn[2]
                        for e in self.E.get(st, []) if e["p"] == "verb" for sn in e["s"]):
                    return True
        return False

    def resolve_sentence(self, toks, groups=None):
        """Resolve every token of a tagged sentence, with context rules on top
        of the tagger: a numeral that is also a verb form and is not followed
        by a noun is the verb ("Sei sicuro?", "Tu sei...": you are, not six)."""
        out = []
        for i, (text, sl, upos, ms) in enumerate(toks):
            if upos == "NUM" and text.isalpha():
                after = [t[2] for t in toks[i + 1:i + 4] if t[2] != "PUNCT"][:2]
                if after[:1] == ["ADJ"]:
                    after = after[1:]           # "sei belle ragazze"
                prev_tok = toks[i - 1][0] if i else "."
                prev = next((t[2] for t in reversed(toks[:i]) if t[2] != "PUNCT"), None)
                nominal = bool(after) and after[0] in ("NOUN", "NUM", "PROPN")
                clause_q = next((t[0] for t in toks[i + 1:] if t[0] in (".", "!", "?")), "") == "?"
                initial_question = prev_tok in (".", "!", "?", "…") and clause_q   # "Sei Elijah?"
                if prev in ("ADP", "DET"):
                    pass                        # "alle sei", "ogni sei mesi": the numeral
                elif prev == "PRON" or not nominal or initial_question:
                    c = self.candidates(text.lower(), ["verb"])
                    c = {x for x in c if x.endswith(VERB_ENDINGS)}
                    if c:
                        self.n_num_as_verb += 1
                        out.append((best_by_freq(c), "VERB"))
                        continue
            if upos == "NOUN" and i:
                # predicate after a copula with no determiner is an adjective
                # ("sono fiera", "diventare matto"), not the noun homograph
                j = i - 1
                while j >= 0 and toks[j][2] == "ADV":
                    j -= 1
                if j >= 0 and toks[j][2] in ("AUX", "VERB"):
                    vr = self.resolve(*toks[j])
                    adj = self.candidates(text.lower(), ["adj"])
                    if vr and vr[0] in COPULAS and adj:
                        a = best_by_freq(adj)
                        inflected = a != text.lower()
                        adj_major = groups is not None and groups.get(a) and \
                            groups[a]["ADJ"] > groups[a]["NOUN"]
                        if inflected or adj_major:
                            self.n_copula_adj += 1
                            out.append((a, "ADJ"))
                            continue
            low = text.lower()
            nxt = next((t for t in toks[i + 1:i + 3] if t[2] != "ADJ"), None)
            if upos == "PRON" and low in ARTICLE_FORMS["il"] and nxt is not None and \
                    nxt[2] != "PUNCT" and self.candidates(nxt[0].lower(), ["noun"]) and \
                    not self.candidates(nxt[0].lower(), ["verb"]):
                # the dictionary decides, not the tag of the next word: the small
                # tagger calls "schiavo" a verb after "Lo" and "vidi" a noun after
                # "lo"; only a word with a noun and no verb reading counts
                # "Lo schiavo scappò": an article form before a noun is the article
                self.n_pron_as_article += 1
                out.append(("il", "DET"))
                continue
            if upos == "NOUN" and (i == 0 or toks[i - 1][0] in (".", "!", "?", "…", ":", ";")) and \
                    i + 1 < len(toks) and \
                    toks[i + 1][2] == "DET" and self.imperative_form(low):
                # clause-initial noun homograph + determiner is an imperative
                # ("Traccia una linea", "Porta un ombrello")
                self.n_imperative += 1
                out.append((self.imperative_form(low), "VERB"))
                continue
            if upos in ("ADV", "VERB") and i and toks[i - 1][2] == "DET" and \
                    toks[i - 1][0].lower() in ARTICLE_FORMS["il"] | ARTICLE_FORMS["uno"]:
                # right after an article the word is a noun ("gettò l'ancora")
                nouns = self.candidates(text.lower(), ["noun"])
                if nouns:
                    self.n_after_article += 1
                    out.append((best_by_freq(nouns), "NOUN"))
                    continue
            out.append(self.resolve(text, sl, upos, ms))
        return out

    def resolve(self, text, slemma, upos, morph=""):
        """Return (lemma, group) or None for punctuation/symbols/digits."""
        plur = upos == "NOUN" and "Number=Plur" in morph
        key = (text, slemma, upos, plur)
        if key in self._cache:
            return self._cache[key]
        r = self._resolve(text.lower(), slemma.lower(), upos, plur)
        if r and r[1] not in ("PROPN",) and zipf(r[0]) < RARE_ZIPF:
            # implausibly rare reading picked by the tagger ("carino" as a
            # form of cariare): take a far more common reading of the surface
            best = max((x for x in self.readings(text.lower()) if x[1] != r[1]),
                       key=lambda x: (zipf(x[0]), x), default=None)
            if best and zipf(best[0]) >= zipf(r[0]) + RARE_MARGIN:
                self.n_rare_override += 1
                r = best
        if r and r[1] == "VERB" and r[0].endswith("rsi"):
            base = r[0][:-2] + "e"          # farsi -> fare, muoversi -> muovere
            if any(not (sense_tags(e, sn) & DEMOTE_TAGS) for e in self.usable_entries(base, ["verb"])
                   for sn in e["s"] if sn[3] == ""):
                r = (base, "VERB")
        self._cache[key] = r
        return r

    def _resolve(self, s, sl, upos, plur=False):
        if upos in SKIP_UPOS or not WORD_RE.search(s):
            return None
        if any(ch.isdigit() for ch in s):
            return None
        g = group_of(upos)
        if " " in sl:
            sl = sl.split()[0]          # 'su il' (sulla), 'fare lo' (farlo)
        if g == "PROPN":
            return (s, "PROPN")
        kp = GROUP_KPOS.get(g)
        if s in ART_PREP:
            return (ART_PREP[s], "ADP")  # sulla -> su, nell' -> in (also when tagged DET: "del pane")
        if g == "DET" and sl in ("il", "uno") and s in ARTICLE_FORMS[sl]:
            return (sl, g)              # la/le/gli/l' -> il, una/un' -> uno
        cands = self.candidates(s, kp)
        if g == "VERB":
            cands = {c for c in cands if c.endswith(VERB_ENDINGS)}
        if g == "NOUN" and s not in PLURALIA_TANTUM and sl != s and (plur or self.plural_pointer(s)):
            if self.usable_entries(sl, kp):
                return (sl, g)          # giorni -> giorno (plural never its own lemma)
            cands.discard(s)
        if g == "NOUN" and s in cands:
            return (s, g)               # signora, soldi: the surface is its own noun lemma
        if g == "VERB" and s in cands:
            return (s, g)
        if g == "PRON" and s in CLITIC_OF:
            return (CLITIC_OF[s], g)    # mi -> io, ti -> tu (spaCy gives 'si' for ti)
        if sl in cands:
            return (sl, g)
        if len(cands) == 1:
            return (next(iter(cands)), g)
        if cands:
            return (best_by_freq(cands), g)
        if CLITIC_RE.match(s):
            v = self.clitic_verb(s)
            if v:
                return (v, "VERB")
        # nothing with the tagged POS: trust spaCy's lemma when it is a real headword
        if sl and self.usable_entries(sl, kp):
            return (sl, g)
        # ...else the surface's reading under another POS (tagger error:
        # "Lo giuro" tagged NOUN -> giurare)
        alt = self.readings(s)
        if alt:
            return max(alt, key=lambda r: (zipf(r[0]), r))
        self.n_unresolved[g] += 1
        return (sl or s, g)


# ---------------------------------------------------------------------------
# Stage freq: corpus usage statistics + (lemma, POS) frequency blend
# ---------------------------------------------------------------------------

SUB_TOKEN_RE = re.compile(r"^[a-zàèéìíòóùú]+'?$")
MIN_SHARE = 0.025   # "stato" the noun is ~3% of "stato" tokens
N_SUB_SURFACES = 100000


def corpus_usage(tagged, lexicon, groups=None):
    """One pass over the tagged corpus."""
    t0 = time.time()
    surf = defaultdict(Counter)       # surface -> Counter((lemma, group))
    raw_upos = defaultdict(Counter)   # (lemma, group) -> Counter(UPOS)
    morph = defaultdict(Counter)      # (lemma, group) -> Counter("Gender=..", "Number=..") on surface==lemma
    refl = Counter()                  # (lemma, VERB) -> tokens carrying a reflexive clitic
    initial = Counter()               # (lemma, group) -> sentence/clause-initial tokens
    stative = Counter()               # (lemma, VERB) -> essere + participle without a clitic
    for sid, toks in iter_tagged(tagged):
        for i, ((text, sl, upos, ms), r) in enumerate(zip(toks, lexicon.resolve_sentence(toks, groups))):
            if r is None:
                continue
            if i == 0 or toks[i - 1][2] == "PUNCT":
                initial[r] += 1
            if r[1] == "VERB":
                if is_reflexive(toks, i):
                    refl[r] += 1
                elif "VerbForm=Part" in ms and essere_aux(toks, i):
                    stative[r] += 1       # "è innamorato", "sono impegnato": neither reading
            s = text.lower()
            surf[s][r] += 1
            raw_upos[r][upos if group_of(upos) == r[1] else "VERB"] += 1
            if s == r[0] and r[1] == "NOUN":
                for kv in ms.split("|"):
                    if kv.startswith(("Gender=", "Number=")):
                        morph[r][kv] += 1
    log(f"usage pass: {len(surf)} surfaces, {len(raw_upos)} (lemma,POS) keys ({time.time()-t0:.0f}s)")
    refl["_stative"] = stative
    return surf, raw_upos, morph, refl, initial


def carries_refl_clitic(toks, i):
    """Lenient check for the -rsi gate: a reflexive-form clitic attached to the
    verb surface ("fidarti", "muoviti": spaCy's lemma is often garbled there)
    or before it through auxiliaries/adverbs/non ("mi fidassi", "si fidi"),
    without the person agreement the tagger's morphology often gets wrong."""
    s = toks[i][0].lower()
    if len(s) > 4 and s[-2:] in REFL_CLITICS and s[-3] in "aeiouàèéìòùr":   # muoviti, fidarti
        return True
    j = i - 1
    while j >= 0 and toks[j][2] in ("AUX", "ADV"):
        j -= 1
    return j >= 0 and toks[j][2] == "PRON" and toks[j][0].lower() in REFL_CLITICS


def essere_aux(toks, i):
    j = i - 1
    while j >= 0 and toks[j][2] == "ADV":
        j -= 1
    return j >= 0 and toks[j][2] in ("AUX", "VERB") and toks[j][1].lower() == "essere"


def is_reflexive(toks, i):
    """The verb token carries a reflexive clitic: attached ("lamentarsi",
    spaCy lemma 'lamentare si') or right before it or its auxiliaries, in the
    same person ("mi sono lamentato", "si fida")."""
    text, sl, upos, ms = toks[i]
    parts = sl.lower().split()
    if len(parts) == 2 and parts[1] in REFL_CLITICS and parts[1] != "ci":
        return True
    j = i - 1
    finite = ms
    while j >= 0 and toks[j][2] in ("AUX", "ADV"):
        if toks[j][2] == "AUX":
            finite = toks[j][3] or finite
        j -= 1
    if j < 0 or toks[j][2] != "PRON":
        return False
    cl = toks[j][0].lower()
    if cl not in REFL_CLITICS:
        return False
    person, number = REFL_CLITICS[cl]
    m = dict(kv.split("=") for kv in finite.split("|") if "=" in kv)
    if m.get("Person") != person:
        return False
    return number is None or m.get("Number") in (None, number)


ACCENT_VARIANTS = {"a": "à", "e": "èé", "i": "ì", "o": "ò", "u": "ù"}


def is_profane(w):
    return w in PROFANITY or w.startswith(PROFANE_STEMS)


def distribute(surface, count, surf, fallback, stats):
    if is_profane(surface):
        stats["profane_skipped"] += 1
        return []
    dist = surf.get(surface) or surf.get(surface + "'")
    if not dist and surface[-1:] in ACCENT_VARIANTS:
        # subtitle text often drops the final accent (citta, perche, piu):
        # use the accented form when only that one occurs in the corpus
        for acc in ACCENT_VARIANTS[surface[-1]]:
            d2 = surf.get(surface[:-1] + acc)
            if d2:
                dist = d2
                stats["accent_restored"] += 1
                break
    if dist and sum(dist.values()) >= 2:
        tot = sum(dist.values())
        keep = {k: v for k, v in dist.items() if v / tot >= MIN_SHARE}
        kt = sum(keep.values())
        stats["corpus"] += 1
        return [(k, count * v / kt) for k, v in sorted(keep.items())]
    stats["fallback"] += 1
    return [((fallback(surface), "?"), float(count))]


def accent_split(counts, surf, stats, moved):
    """Unaccented spellings inflated in a frequency list (pero for però, da
    for dà): move the share above what the corpus predicts to the accented
    spelling. Expected share of s = corpus(s) / (corpus(s) + corpus(s'))."""
    out = dict(counts)
    for s_, n in counts.items():
        if s_[-1:] not in ACCENT_VARIANTS:
            continue
        cs = sum(surf.get(s_, {}).values())
        for acc in ACCENT_VARIANTS[s_[-1]]:
            s2 = s_[:-1] + acc
            ca = sum(surf.get(s2, {}).values())
            if not ca or s2 not in counts:
                continue
            expected = cs / (cs + ca)
            observed = n / (n + counts[s2])
            if observed > expected:
                excess = (observed - expected) * (n + counts[s2])
                out[s_] -= excess
                out[s2] += excess
                stats["accent_excess_moved"] += 1
                if excess > 1000 and len(moved) < 40:
                    moved.append(f"{s_}->{s2} {excess / n:.0%}")
    return out


def stage_freq(surf, raw_upos):
    import simplemma
    from wordfreq import zipf_frequency, top_n_list
    stats = Counter()
    sm = lambda w: simplemma.lemmatize(w, lang="it")
    moved = []
    raw = {}
    with open(CACHE / "it_full.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != 2 or not SUB_TOKEN_RE.match(parts[0]):
                continue
            raw[parts[0]] = int(parts[1])
            if len(raw) >= N_SUB_SURFACES:
                break
    sub = Counter()
    for w, c in accent_split(raw, surf, stats, moved).items():
        for k, cc in distribute(w, c, surf, sm, stats):
            sub[k] += cc
    sub_stats = dict(stats)
    stats = Counter()
    raw = {}
    for w in top_n_list("it", 30000):
        if SUB_TOKEN_RE.match(w):
            z = zipf_frequency(w, "it")
            if z > 0:
                raw[w] = 10 ** z
    wf = Counter()
    for w, c in accent_split(raw, surf, stats, moved).items():
        for k, cc in distribute(w, c, surf, sm, stats):
            wf[k] += cc
    wf_stats = dict(stats)
    stat("accent_moves", moved)

    # "?"-POS keys (surface unseen in the corpus): attach to the lemma's
    # corpus-majority POS when the lemma is known from other surfaces.
    lemma_best = {}
    for (lem, g), c in sorted(raw_upos.items(), key=lambda kv: (kv[0][0], -sum(kv[1].values()), kv[0][1])):
        if g != "PROPN" and lem not in lemma_best:
            lemma_best[lem] = g
    n_q = 0
    for counter in (sub, wf):
        for (lem, g) in sorted(k for k in counter if k[1] == "?"):
            c = counter.pop((lem, g))
            if lem in lemma_best:
                counter[(lem, lemma_best[lem])] += c
            else:
                counter[(lem, "?")] += c
                n_q += 1

    def ranks(counter):
        return {k: i + 1 for i, (k, _) in enumerate(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))}
    sr, wr = ranks(sub), ranks(wf)
    common = sorted(set(sr) & set(wr))
    blended = sorted(((math.log(sr[k]) + math.log(wr[k])) / 2.0, k) for k in common)
    stat("freq", {"sub_surfaces": sub_stats, "wf_surfaces": wf_stats, "unknown_pos_keys": n_q,
                  "sub_keys": len(sr), "wf_keys": len(wr), "common_keys": len(common)})
    return [(k, sc, sr[k], wr[k], sub[k] + 0.0) for sc, k in blended]



# ---------------------------------------------------------------------------
# English side: crude stemming for gloss <-> translation overlap
# ---------------------------------------------------------------------------

EN_WORD_RE = re.compile(r"[a-z]+")
EN_STOP = set("""a an the of to in on at for with by from as and or but not no is are was were be
been being am do does did have has had it its this that these those there here i you he she we they
me him her us them my your his our their what which who whom whose when where why how all any some
one ones something someone somebody thing things very so too just than then also only into out up
down about over after before again if will would shall should can could may might must let s t
don isn didn doesn won aren wasn weren haven hasn ll re ve d m used denote denotes indicate
indicates indicating expressing express expresses especially particularly etc sense senses kind
type form""".split())
EN_IRREG = {
    "is": "be", "are": "be", "was": "be", "were": "be", "am": "be", "been": "be", "being": "be",
    "has": "have", "had": "have", "having": "have", "did": "do", "does": "do", "done": "do",
    "went": "go", "gone": "go", "goes": "go", "said": "say", "says": "say", "made": "make",
    "saw": "see", "seen": "see", "took": "take", "taken": "take", "came": "come", "knew": "know",
    "known": "know", "got": "get", "gotten": "get", "gave": "give", "given": "give",
    "thought": "think", "told": "tell", "found": "find", "left": "leave", "felt": "feel",
    "brought": "bring", "bought": "buy", "ate": "eat", "eaten": "eat", "wrote": "write",
    "written": "write", "spoke": "speak", "spoken": "speak", "ran": "run", "sat": "sit",
    "stood": "stand", "understood": "understand", "met": "meet", "paid": "pay", "sold": "sell",
    "sent": "send", "spent": "spend", "built": "build", "kept": "keep", "slept": "sleep",
    "drank": "drink", "drunk": "drink", "began": "begin", "begun": "begin", "chose": "choose",
    "chosen": "choose", "fell": "fall", "fallen": "fall", "held": "hold", "lost": "lose",
    "meant": "mean", "put": "put", "read": "read", "heard": "hear", "taught": "teach",
    "caught": "catch", "fought": "fight", "won": "win", "wore": "wear", "worn": "wear",
    "broke": "break", "broken": "break", "forgot": "forget", "forgotten": "forget",
    "children": "child", "men": "man", "women": "woman", "people": "person", "feet": "foot",
    "teeth": "tooth", "mice": "mouse", "better": "good", "best": "good", "worse": "bad",
    "worst": "bad", "lives": "life", "wives": "wife", "knives": "knife", "leaves": "leaf",
    "died": "die", "dying": "die", "lying": "lie", "lied": "lie",
}


EN_VOCAB = set()   # raw English word forms seen in the corpus translations
_STEM = {}


def en_stem(w):
    """Map an inflected English word to its base form, choosing only bases
    attested in the corpus vocabulary (making -> make, stopped -> stop,
    boxes -> box), so unrelated words never collide (plane != plan)."""
    if w in _STEM:
        return _STEM[w]
    r = EN_IRREG.get(w)
    if r is None:
        r = w
        tries = []
        if len(w) > 4 and w.endswith("ies"):
            tries = [w[:-3] + "y"]
        elif len(w) > 3 and w.endswith("es"):
            tries = [w[:-2], w[:-1]]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            tries = [w[:-1]]
        elif len(w) > 5 and w.endswith("ing"):
            b = w[:-3]
            tries = [b + "e", b, b[:-1] if len(b) > 2 and b[-1] == b[-2] else None]
        elif len(w) > 4 and w.endswith("ed"):
            b = w[:-2]
            tries = [w[:-1], b, b[:-1] if len(b) > 2 and b[-1] == b[-2] else None,
                     b[:-1] + "y" if b.endswith("i") else None]
        for t in tries:
            if t and t in EN_VOCAB:
                r = t
                break
    _STEM[w] = r
    return r


EN_CONTRACTION_RE = re.compile(r"(n't|'s|'re|'ll|'ve|'m|'d)\b")


def en_stems(text, keep_stop=False):
    out = []
    text = text.lower().replace("’", "'").replace("can't", "can not").replace("won't", "will not")
    text = EN_CONTRACTION_RE.sub(lambda m: " not" if m.group(1) == "n't" else " ", text)
    for w in EN_WORD_RE.findall(text):
        if not keep_stop and w in EN_STOP:
            continue
        out.append(en_stem(w))
    return out


# ---------------------------------------------------------------------------
# Glosses
# ---------------------------------------------------------------------------

MAX_GLOSS = 45
GLOSS_LABEL_RE = re.compile(
    r"^(transitive|intransitive|ambitransitive|reflexive|pronominal|figuratively|figurative|broadly|"
    r"colloquial|informal|formal|euphemistic|rare|dated|archaic|literary|by extension|chiefly|"
    r"especially|usually|often|sometimes|in the plural|plural|singular|uncountable|countable),?:?\s+",
    re.IGNORECASE)
DEFINITIONAL_RE = re.compile(
    r"^(used|denotes|denoting|indicates|indicating|expresses|expressing|introduces|forms|"
    r"a |an |the |one who|someone who|something that|any |of or |relating|pertaining|"
    r"in the sense|with the meaning|translated|equivalent|see )", re.IGNORECASE)


def cap_parts(parts):
    """At most 3 comma alternatives and MAX_GLOSS chars."""
    parts = parts[:3]
    g = ", ".join(parts)
    if len(g) > MAX_GLOSS:
        out = []
        for p in parts:
            if len(", ".join(out + [p])) > MAX_GLOSS:
                break
            out.append(p)
        g = ", ".join(out) if out else g[:MAX_GLOSS].rsplit(" ", 1)[0]
    return g


EN_PROFANE_RE = re.compile(r"\b(fuck\w*|motherfuck\w*|shit\w*|cunt\w*)\b", re.I)


def clean_gloss(g, cap=True, all_groups=False):
    g = g.strip()
    for _ in range(4):
        g2 = GLOSS_LABEL_RE.sub("", g.strip())
        g2 = re.sub(r"^\([^)]*\)\s*", "", g2)
        if g2 == g:
            break
        g = g2
    g = re.sub(r"\[[^\[\]]*\]", "", g)
    stripped = g
    for _ in range(4):
        stripped = re.sub(r"\s*\([^()]*\)", "", stripped)
    if "(" in stripped:
        stripped = stripped.split("(")[0]
    stripped = stripped.replace(")", "")
    if len(stripped.strip()) >= 1:
        g = stripped
    if ". " in g:
        segs = [x.strip() for x in g.split(". ") if x.strip()]
        g = segs[-1] if len(segs[-1].split()) <= 4 else segs[0]
    if ": " in g:
        head, tail = g.rsplit(": ", 1)
        if tail and DEFINITIONAL_RE.match(head + " "):
            g = tail
    g = re.sub(r"^(with the same meaning|in the same sense|same as)\s*:?\s*", "", g, flags=re.I)
    g = re.sub(r"^the (need|desire)(?: or (?:need|desire))? (for|to)\s+", "", g, flags=re.I)
    g = re.sub(r"([!?]) (?=\S)", r"\1, ", g)       # "here it is! there you have it!"
    if len(g.split()) <= 4:
        g = re.sub(r"^(a|an|the) (?!(lot|little|bit|few|while|long)\b)", "", g)   # "a full moon" -> "full moon"
    g = re.sub(r",?\s*\betc\.?$", "", g.strip())
    g = re.sub(r"\s{2,}", " ", g)
    g = re.sub(r"\s+([,.;:])", r"\1", g).strip().rstrip(".").strip()
    g = re.sub(r"[,;:]\s*$", "", g).strip()
    if ";" in g:
        # relational adjectives ("home; national, domestic", "sex; sexual")
        # keep every group so the adjectival one can lead
        g = g.replace(";", ",") if all_groups else g.split(";")[0].strip()
    parts = [p.strip() for p in g.split(",") if re.search(r"[A-Za-z]", p) and p.strip() and not re.match(r"^(pl\.?|see|cf\.?) ", p.strip())]
    # strong English profanity never leads a learner gloss (fregare "to fuck, to screw")
    parts = [p for p in parts if not EN_PROFANE_RE.search(p)] or parts
    # "shop, a store" -> "shop, store": a bare article on a short alternative
    parts = [re.sub(r"^(a|an|the) (?!(lot|little|bit|few|while|long)\b)", "", p) if len(p.split()) <= 3 else p
             for p in parts]
    return cap_parts(parts) if cap else parts


def is_definitional(g, group):
    if DEFINITIONAL_RE.match(g):
        return True
    if group == "VERB" and not g.startswith("to "):
        return True
    return len(g.split(",")[0].split()) > 5


GLOSS_IGNORE = {"to", "a", "an", "the", "of", "or", "and", "in", "on", "at", "for", "with", "by",
                "from", "as", "one", "someon", "someth", "somebody", "oneself", "etc", "thing",
                "used", "us", "denot", "indicat", "express", "especially", "particularly"}


ADJISH_RE = re.compile(r"(al|ic|ous|ive|ary|ful|less|able|ible|ed|ing|ish|an|ese|ent|ant|ile|ar|ory|y|ern|en|ior|ite|ate|ual|ular|ior)$")


def sense_candidates(entry, group, df, nsent, closed, bg, bgn):
    """Scored, cleaned senses of an entry, best first. A sense scores by its
    best English word's association with the translations of the corpus
    sentences that use this (lemma, POS): p * log2(p / q), where p is the
    share of those translations containing the word and q its share over all
    translations (so generic words like 'be', 'thing' do not win)."""
    rows = []
    for idx, (gl, hdr, tags, kind) in enumerate(entry["s"]):
        if kind:
            continue
        parts = clean_gloss(gl, cap=False, all_groups=group == "ADJ" and "relational" in tags)
        if not parts:
            continue

        def tok_scores(text):
            st = set(en_stems(text, keep_stop=True))
            if not closed:
                st -= GLOSS_IGNORE
            st.discard("to")
            out = {}
            for t in st:
                pp = df.get(t, 0) / nsent if nsent else 0.0
                q = (bg.get(t, 0) + 1) / bgn
                out[t] = pp * math.log2(pp / q) if pp > q else 0.0
            return out
        # score each comma alternative; the best-supported one leads the
        # displayed gloss ("to marry, to cause to get married")
        if group == "ADJ" and "relational" in tags and any(ADJISH_RE.search(p) for p in parts):
            # relational senses list the noun first ("law; legal"): an
            # adjective gloss must be adjectival
            parts = [p for p in parts if ADJISH_RE.search(p)]
        if group == "VERB" and any(p.startswith("to ") for p in parts):
            parts = [p if p.startswith("to ") or p.split()[0].lower() in EN_STOP else "to " + p
                     for p in parts]
        sc = [tok_scores(p) for p in parts]
        best = [max(d.values(), default=0.0) for d in sc]
        order = sorted(range(len(parts)), key=lambda i: (-round(best[i], 3),
                                                         sum(1 for v in sc[i].values() if v < 0.01), i))
        parts = [parts[i] for i in order]
        lead = sc[order[0]]
        g = cap_parts(parts)
        stems = set().union(*[set(sc[i]) for i in order[:3]])
        score = best[order[0]]
        unmatched = sum(1 for v in lead.values() if v < 0.01)
        demote = bool((set(tags) - entry.get("ht", set())) & DEMOTE_TAGS)
        if group == "ADJ" and "relational" in tags and not ADJISH_RE.search(parts[0]):
            demote = True
        rows.append({"idx": idx, "g": g, "score": round(score, 4), "demote": demote,
                     "defn": is_definitional(g, group), "stems": stems, "unmatched": unmatched,
                     "tags": set(tags), "pscore": {parts[j]: best[i] for j, i in enumerate(order)}})
    # translation-like senses with corpus support first, then definitional
    # ones ("Used as ...") only when nothing translation-like matched; ties go
    # to the sense whose leading alternative has fewer unsupported words,
    # then Wiktionary order.
    rows.sort(key=lambda r: (r["demote"], 0 if r["score"] > 0 and not r["defn"] else 1 if r["score"] > 0 else 2,
                             r["defn"], -round(r["score"], 3), r["unmatched"] if r["score"] > 0 else 0, r["idx"]))
    return rows


def compose_gloss(rows):
    if not rows:
        return None, None
    top = rows[0]
    gloss = top["g"]
    second = None
    for r in rows[1:]:
        if r["demote"] or r["defn"] or top["score"] <= 0 or r["score"] < 0.5 * top["score"]:
            continue
        if r["stems"] & top["stems"] or r["g"].lower() == gloss.lower():
            continue
        if len(gloss) + 2 + len(r["g"]) <= MAX_GLOSS:
            second = r
        break
    if second:
        gloss = f"{gloss}; {second['g']}"
    return gloss, top


# ---------------------------------------------------------------------------
# Forced inclusion (closed sets + A1 core) and small explicit override tables
# ---------------------------------------------------------------------------

DAYS = "lunedì martedì mercoledì giovedì venerdì sabato domenica".split()
MONTHS = ("gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre "
          "novembre dicembre").split()
NUMBERS = ("zero uno due tre quattro cinque sei sette otto nove dieci undici dodici tredici "
           "quattordici quindici sedici diciassette diciotto diciannove venti trenta quaranta "
           "cinquanta sessanta settanta ottanta novanta cento mille").split()
COLOURS = "rosso blu verde giallo nero bianco grigio marrone rosa azzurro arancione viola".split()
NATIONALITIES = "italiano tedesco francese spagnolo inglese americano".split()
SEASONS = "primavera estate autunno inverno".split()
# A1 core: nouns/verbs/adjectives/adverbs any A1 syllabus covers
A1_CORE = {
    "NOUN": """casa porta finestra pane latte uovo acqua vino caffè tè birra formaggio mela carne
        pesce frutta verdura pasta pizza riso zucchero sale naso occhio bocca mano testa piede
        braccio gamba orecchio capello faccia dente sedia tavolo letto cucina bagno camera stanza
        libro penna scuola lavoro ufficio negozio città strada treno macchina autobus aereo
        bicicletta stazione albergo ristorante mercato giorno notte mattina sera settimana mese
        anno ora minuto famiglia madre padre fratello sorella figlio figlia marito moglie amico
        bambino ragazzo ragazza uomo donna nome telefono cane gatto tempo sole pioggia mare
        montagna chiave biglietto colazione pranzo cena medico lezione vestito scarpa mattino
        televisione bicchiere doccia cugino zia zio orologio pollo cappello neve torta bottiglia
        tasca forchetta cucchiaio coltello tazza camicia giacca gonna ombrello matita arancia""".split(),
    "VERB": """essere avere fare andare venire stare dire vedere sapere potere volere dovere
        mangiare bere dormire parlare capire leggere scrivere lavorare abitare vivere comprare
        aprire chiudere prendere dare aspettare chiamare cercare trovare piacere amare conoscere
        guardare ascoltare arrivare partire tornare uscire entrare pagare studiare""".split(),
    "ADJ": """buono bello grande piccolo nuovo vecchio caldo freddo alto basso lungo corto felice
        stanco contento giovane facile difficile""".split(),
    "ADV": "oggi domani ieri qui là sempre mai molto poco bene male anche ancora già adesso".split(),
}
FORCED = ([(w, "NOUN") for w in DAYS + MONTHS + SEASONS] + [(w, "NUM") for w in NUMBERS] +
          [(w, "ADJ") for w in COLOURS + NATIONALITIES] + [("buonanotte", "INTJ")] +
          [("sì", "INTJ"), ("no", "INTJ"), ("ciao", "INTJ"), ("grazie", "INTJ"), ("prego", "INTJ"),
           ("scusa", "INTJ"), ("buongiorno", "INTJ"), ("buonasera", "INTJ"), ("arrivederci", "INTJ"), ("per favore", "PHRASE"), ("a", "ADP"), ("e", "CONJ"), ("o", "CONJ"),
           ("è", "FORM")] +
          [(w, g) for g, ws in A1_CORE.items() for w in ws])
NO_ARTICLE = set(DAYS + MONTHS)
ALLOWED_NUM = set(NUMBERS)

# Explicit small override tables (closed-class items whose Wiktionary gloss is
# grammatical description rather than a translation).
FIXED_GLOSS = {
    ("il", "DET"): "the (il, lo, l', la, i, gli, le)", ("uno", "DET"): "a, an (un, uno, una, un')", ("è", "FORM"): "is (from essere)",
    ("per favore", "PHRASE"): "please",
}
FIXED_WORD = {("il", "DET"): ("il", ["lo", "la", "l'", "i", "gli", "le"]),
              ("uno", "DET"): ("un", ["uno", "una", "un'"])}
MULTIWORD = {"per favore": ("per", "favore")}
# apocopated forms link to their lemma (link forms, not typing alts)
APOCOPE = {"nessun": "nessuno", "quel": "quello", "bel": "bello", "buon": "buono", "gran": "grande",
           "san": "santo", "ciascun": "ciascuno", "alcun": "alcuno"}
# final-QA hand drops: entries whose sentences are all another word's use
# (fine adj: every sentence is "fine settimana") or a duplicate of a forced
# entry (no adv beside the interjection); their tokens link as mapped
DROP_KEYS = {("fine", "ADJ"): None, ("no", "ADV"): ("no", "INTJ")}
PRONOMINAL_LINKED_SHARE = 0.6   # -rsi label needs this share of its linked sentences reflexive
GLOSS_OVERRIDES = {k: v for k, v in json.loads((TOOLS / "gloss_overrides.json").read_text()).items()
                   if not k.startswith("_")}
SECOND_ENTRY_SHARE = 0.20     # a second (lemma, POS) entry needs this share of the lemma's tokens
IMPERATIVE_INITIAL_SHARE = 0.5
FOLD_IGNORE = {"a", "an", "the", "of", "to", "one", "thing", "person", "female", "male", "woman", "man",
               "f", "m", "or", "and"}
COLLISION_SUPPORT = 0.75      # a replacement lead sense/alternative needs this share of the top score
SECOND_SENSE_OVERLAP = 0.25   # max share of a second POS's translations using the first entry's words
PRONOMINAL_SHARE = 0.5        # verbs used mostly with a reflexive clitic are shown as -rsi

PROFANE_STEMS = ("cazz", "merd", "stronz", "puttan", "coglion", "fott", "incazz", "vaffancul")
PROFANITY = {"cazzo", "cazzata", "merda", "stronzo", "stronza", "puttana", "vaffanculo",
             "coglione", "cagare", "scopare", "fica", "figa", "culo", "troia", "bastardo",
             "bastarda", "porco", "porca", "fottere", "fottuto", "puttanata", "incazzare"}


# ---------------------------------------------------------------------------
# Articles / gender
# ---------------------------------------------------------------------------

VOWELS = "aeiouàèéìíòóùú"


def parse_gender(g):
    """Wiktionary gender spec -> ('m'|'f'|'mf'|None, plural_only). Qualified
    alternatives such as 'f,m<l:archaic>' keep only the unqualified gender."""
    if not g:
        return None, False
    specs = [x for x in str(g).split(",") if x and "<" not in x] or [str(g).split("<")[0]]
    g = ",".join(specs).replace("bysense", "")
    plural = "-p" in g or (g.endswith("p") and g not in ("m", "f"))
    has_m, has_f = "m" in g, "f" in g
    if has_m and has_f:
        return "mf", plural
    return ("m" if has_m else "f" if has_f else None), plural


def article_for(gender, lemma, plural=False):
    first = lemma[0]
    vowel = first in VOWELS or first == "h"
    lo_type = (lemma[:2] in ("gn", "ps", "pn") or first in "zxy" or
               (first == "s" and len(lemma) > 1 and lemma[1] not in VOWELS) or
               (first == "i" and len(lemma) > 1 and lemma[1] in VOWELS))
    if gender == "f":
        if plural:
            return "le"
        return "l'" if vowel and not (first == "i" and lemma[1:2] in tuple(VOWELS)) else "la"
    if plural:
        return "gli" if (vowel or lo_type) else "i"
    if lo_type:
        return "lo"
    return "l'" if vowel else "il"


def with_article(art, lemma):
    return f"l'{lemma}" if art == "l'" else f"{art} {lemma}"



# ---------------------------------------------------------------------------
# Stage words
# ---------------------------------------------------------------------------

EN_BAG_CAP = 400
POOL_KEYS = 6500   # distinct lemmas considered (all their POS keys)
ID_MAP = TOOLS / "id_map_v1.json"


def english_bags(tagged, lexicon, keys, en_by_sid, groups=None):
    """(lemma, group) -> (#sentences, Counter(English stem -> #sentences))."""
    keys = set(keys)
    bags = {k: [0, Counter()] for k in keys}
    for sid, toks in iter_tagged(tagged):
        seen = set()
        for r in lexicon.resolve_sentence(toks, groups):
            if r in keys and r not in seen:
                seen.add(r)
        if not seen:
            continue
        stems = set(en_stems(en_by_sid[sid], keep_stop=True))
        for r in seen:
            b = bags[r]
            if b[0] < EN_BAG_CAP:
                b[0] += 1
                b[1].update(stems)
    return bags


def build_words(ctx):
    lexicon, raw_upos, morph, blended = ctx["lexicon"], ctx["raw_upos"], ctx["morph"], ctx["blended"]
    low, cap = ctx["truecase"]
    from wordfreq import top_n_list
    en_top = set(top_n_list("en", 2000))

    lemma_groups = ctx["lemma_groups"]

    def share(lem, g):
        lg = lemma_groups.get(lem, Counter())
        tot = sum(c for gg, c in lg.items() if gg != "PROPN")
        return lg[g] / tot if tot else 0.0

    def gloss_words(gl):
        # exact English words: amica "friend" folds into amico "friend";
        # volta "time" stays apart from volto "face"
        return {x for x in re.findall(r"[a-z]+", re.sub(r"\(.*?\)", " ", gl.lower()))
                if x not in FOLD_IGNORE}

    def same_words(a, b):
        # "this, these" (PRON) vs "this, these" (DET): stopword-only glosses
        f = lambda x: set(en_stems(re.sub(r"\(.*?\)", " ", x), keep_stop=True)) - {"to", "a", "an", "the", "of"}
        return bool(f(a) & f(b))

    def gloss_stems(gl):
        return set(en_stems(re.sub(r"\(.*?\)", " ", gl))) - GLOSS_IGNORE

    def is_proper(lem):
        lg = lemma_groups.get(lem)
        tot = sum(lg.values()) if lg else 0
        if tot and lg["PROPN"] / tot > 0.5:
            return True
        return cap[lem] >= 3 and cap[lem] > low[lem]

    excluded = Counter()
    ex_examples = defaultdict(list)

    def exclude(reason, lem):
        excluded[reason] += 1
        if len(ex_examples[reason]) < 25:
            ex_examples[reason].append(lem)

    # --- candidate pool: one key per lemma (its best-ranked POS), plus articles
    forced_keys = []
    for w, g in FORCED:
        if g is None:
            lg = lemma_groups.get(w, Counter())
            opts = [(c, gg) for gg, c in lg.items() if gg != "PROPN"]
            g = max(opts)[1] if opts else "INTJ"
        forced_keys.append((w, g))
    forced_set = set(forced_keys)
    forced_lemmas = {k[0] for k in forced_keys}
    order = {}
    pool = []
    lemmas_in_pool = set()
    for i, (k, sc, sr, wr, subc) in enumerate(blended):
        order[k] = i
        lem, g = k
        if g == "PROPN" or k in DROP_KEYS:
            continue
        is_article = g == "DET" and lem in ("il", "uno")
        if lem in forced_lemmas and k not in forced_set and not is_article and \
                share(lem, g) < SECOND_ENTRY_SHARE:
            continue       # the forced POS wins for forced lemmas (e.g. rosa ADJ, sei NUM)
        if len(lemmas_in_pool) < POOL_KEYS or lem in lemmas_in_pool:
            pool.append(k)
            lemmas_in_pool.add(lem)
    for k in forced_keys:
        if k not in pool:
            pool.append(k)
    stat("pool_keys", len(pool))

    t0 = time.time()
    bags = english_bags(ctx["tagged"], lexicon, pool, ctx["en_by_sid"], lemma_groups)
    log(f"english bags for {len(pool)} keys ({time.time()-t0:.0f}s)")

    multi_pos = 0
    fem_glossed = []
    records = {}
    done_lemma = {}
    second_entries, pronominal, overridden, second_diag = [], [], [], []
    for k in pool:
        lem, g = k
        forced = k in forced_set
        is_article = g == "DET" and lem in ("il", "uno")
        second = lem in done_lemma and not is_article
        displaced = None
        if second and not forced:
            if share(lem, g) < SECOND_ENTRY_SHARE:
                continue
            if len(done_lemma[lem]) >= 2:
                # the second slot goes to the POS with the larger corpus share
                # (come: ADV "how" 27% over CONJ 25%)
                prev = done_lemma[lem][1]
                if records[prev]["forced"] or share(lem, g) <= share(lem, prev[1]):
                    continue
                displaced = prev       # one entry per lemma (its best-ranked POS with a usable entry), plus
            #                a second POS holding >=20% of the lemma's tokens with a different sense
        if g == "PHRASE" or g == "FORM":
            records[k] = {"lemma": lem, "group": g, "pos": "phrase" if g == "PHRASE" else "verb",
                          "en": FIXED_GLOSS[k], "w": lem, "alt": None, "forced": True,
                          "entry_pos": None, "sense_idx": None, "gender": None}
            continue
        if not forced:
            if is_proper(lem):
                exclude("proper noun (corpus PROPN/capitalised majority)", lem); continue
            if is_profane(lem):
                exclude("profanity (hand list)", lem); continue
            if g == "INTJ":
                exclude("interjection (not in forced greetings)", lem); continue
            if g == "NUM" and lem not in ALLOWED_NUM:
                exclude("numeral outside 0-20/tens/100/1000", lem); continue
        if len([gg for gg in lemma_groups.get(lem, {}) if gg != "PROPN" and lemma_groups[lem][gg] >= 5]) > 1:
            multi_pos += 1
        kpos = GROUP_KPOS.get(g) if g != "?" else None
        ents = lexicon.usable_entries(lem, kpos)
        if not ents and forced:
            ents = lexicon.usable_entries(lem, None)
        if not ents:
            exclude("no usable Wiktionary entry for corpus POS", lem); continue
        nsent, df = bags.get(k, [0, Counter()])
        closed = g not in CONTENT_GROUPS
        best = None
        for ei, e in enumerate(ents):
            rows = sense_candidates(e, g, df, nsent, closed, ctx["en_bg"], ctx["en_bgn"])
            if not rows:
                continue
            top = rows[0]
            q = (top["demote"], 0 if top["score"] > 0 and not top["defn"] else 1 if top["score"] > 0 else 2,
                 top["defn"], -round(top["score"], 3), ei)
            if best is None or q < best[0]:
                best = (q, e, rows)
        if best is not None and best[0][1] > 0 and g != "ADJ":
            # only definitional senses ("used to call someone's attention"):
            # a translation-like sense under another POS header wins (ecco)
            for ei, e in enumerate(lexicon.usable_entries(lem, None)):
                if e in ents:
                    continue
                rows = sense_candidates(e, g, df, nsent, closed, ctx["en_bg"], ctx["en_bgn"])
                if rows and not rows[0]["defn"]:
                    top = rows[0]
                    q = (top["demote"], 0 if top["score"] > 0 else 2, False, -round(top["score"], 3), 100 + ei)
                    if q < best[0]:
                        best = (q, e, rows)
        if best is None:
            exclude("no usable sense after cleaning", lem); continue
        _, ent, rows = best
        if (not forced and ent.get("b") and lem in en_top and
                sum(raw_upos.get(k, Counter()).values()) < 10):
            exclude("English loanword unattested in Italian corpus", lem); continue
        gloss, top = compose_gloss(rows)
        tokens = sum(raw_upos.get(k, Counter()).values())
        if g == "NOUN" and not forced and tokens:
            # an imperative + clitic homograph ("Fallo subito." = do it, not the
            # noun "foul/phallus"): the tokens open the sentence with no
            # determiner (measured: fallo 94%, real nouns <=5%), or the noun's
            # only senses are vulgar
            if lexicon.imperative_clitic(lem):
                noun_tags = [set(sn[2]) for e in lexicon.E.get(lem, []) if e["p"] == "noun"
                             for sn in e["s"] if not sn[3]]
                vulgar = bool(noun_tags) and all(t & {"vulgar", "offensive"} for t in noun_tags)
                if ctx["initial"][k] / tokens >= IMPERATIVE_INITIAL_SHARE or vulgar:
                    exclude("noun homograph of an imperative+clitic (sentence-initial or vulgar)", lem)
                    continue
        disp = None
        base_gloss = None
        active = tokens - ctx["refl"]["_stative"][k]
        if g == "VERB" and active >= 10 and ctx["refl"][k] / active >= PRONOMINAL_SHARE:
            # used mostly pronominally ("mi lamento", "si fida"): show the -rsi
            # verb with its own sense
            rsi = (lem[:-2] if lem.endswith("rre") else lem[:-1]) + "si"
            prow = None
            for e in lexicon.usable_entries(rsi, ["verb"]):
                r2 = [r for r in sense_candidates(e, g, df, nsent, closed, ctx["en_bg"], ctx["en_bgn"])
                      if not r["defn"]]
                if r2 and (prow is None or r2[0]["score"] > prow[0]["score"]):
                    prow = r2
            if prow is None:
                prow = [r for r in rows if r["tags"] & {"pronominal", "reflexive"}] or None
            if prow:
                base_gloss = gloss
                gloss, top = compose_gloss(prow)
                disp = rsi
                pronominal.append(rsi)
        fem_of = []
        if g == "NOUN":
            # (target, via_noun): "female equivalent of amico" in the noun
            # entry, or only an adjective inflection line ("feminine singular
            # of santo"), which folds only into a same-meaning masculine noun
            fem_of = sorted({(m.group(1), e["p"] == "noun") for e in lexicon.E.get(lem, [])
                             if e["p"] in ("noun", "adj")
                             for sn in e["s"] if sn[3] == "form"
                             for m in [re.match(r"(?:female equivalent|(?:singular )?feminine(?: singular)?) of ([a-zàèéìòù]+)",
                                                sn[0].lower())] if m})
        if g == "NOUN" and not forced:
            # Feminine forms with an extra sense of their own (la piccola "small
            # beer", l'amica "girlfriend", la fisica "physics"): compare the
            # corpus support for the masculine lemma's reading with the
            # entry's own senses.
            fem = sorted({m.group(1) for e in lexicon.E.get(lem, []) if e["p"] in ("noun", "adj")
                          for sn in e["s"] if sn[3] == "form"
                          for m in [re.match(r"(?:female equivalent|(?:singular )?feminine(?: singular)?) of ([a-zàèéìòù]+)",
                                             sn[0].lower())] if m})
            drop = False
            for t in fem:
                t_adj = lemma_groups.get(t) and lemma_groups[t].most_common(1)[0][0] == "ADJ"
                tents = lexicon.usable_entries(t, ["adj"] if t_adj else ["noun"])
                tbest = None
                for te in tents:
                    trows = sense_candidates(te, g, df, nsent, closed, ctx["en_bg"], ctx["en_bgn"])
                    if trows and (tbest is None or trows[0]["score"] > tbest[0]["score"]):
                        tbest = trows
                if not tbest or tbest[0]["demote"]:
                    continue
                if t_adj and top["defn"] and tbest[0]["score"] >= top["score"]:
                    drop = True          # nominalised adjective: la piccola = the little one
                elif not t_adj and tbest[0]["score"] >= 1.0 and tbest[0]["score"] > top["score"] * 1.5:
                    gloss, top = compose_gloss(tbest)
                    fem_glossed.append(lem)
            if drop:
                exclude("feminine of an adjective used as a noun", lem); continue
        if k in FIXED_GLOSS:
            gloss = FIXED_GLOSS[k]
        okey = f"{lem}|{GROUP_LABEL.get(g, g.lower())}"
        if okey in GLOSS_OVERRIDES:
            gloss = GLOSS_OVERRIDES[okey]
            overridden.append(okey)
        if second:
            fk = done_lemma[lem][0]
            first = records[fk]
            # the second POS is a distinct sense only if its sentences' English
            # does not keep using the first entry's words (il malato = "the sick")
            fw = gloss_stems(first["en"].split(";")[0].split(",")[0])
            overlap = max((df.get(t, 0) / nsent for t in fw), default=0.0) if nsent else 1.0
            second_diag.append(f"{lem} {g} '{gloss}' vs {first['group']} '{first['en']}' overlap={overlap:.2f}")
            nominalised = {g, first["group"]} == {"NOUN", "ADJ"} and \
                re.search(r"\b(person|people|man|woman|one|soul)\b", gloss)
            if gloss_stems(gloss) & gloss_stems(first["en"]) or first["group"] == g or \
                    same_words(gloss, first["en"]) or nominalised or \
                    top["score"] <= 0 or top["defn"] or top["demote"] or \
                    overlap > SECOND_SENSE_OVERLAP:
                if not forced or first["forced"]:
                    exclude("second POS entry without a distinct sense", lem); continue
                del records[fk]              # the forced POS replaces a same-sense entry
                done_lemma[lem].remove(fk)
            else:
                second_entries.append(f"{lem} {g}")
                if displaced:
                    del records[displaced]
                    done_lemma[lem].remove(displaced)
                    second_entries.append(f"(replaces {lem} {displaced[1]})")
        if g == "?":
            inv = {"noun": "NOUN", "verb": "VERB", "adj": "ADJ", "adv": "ADV", "pron": "PRON",
                   "prep": "ADP", "conj": "CONJ", "num": "NUM", "intj": "INTJ", "det": "DET",
                   "article": "DET"}
            g = inv.get(ent["p"], "NOUN")
        if not is_article:
            done_lemma.setdefault(lem, []).append(k)
        records[k] = {"lemma": lem, "group": g, "en": gloss, "forced": forced, "pos_from_dict": k[1] == "?",
                      "entry_pos": ent["p"], "sense_idx": top["idx"], "sense_score": top["score"],
                      "gender": ent.get("g"), "nsent": nsent, "rows": rows, "display": disp, "base_en": base_gloss,
                      "fem_of": fem_of, "fixed": okey in GLOSS_OVERRIDES or k in FIXED_GLOSS}
    stat("excluded", dict(excluded))
    stat("excluded_examples", {k: v for k, v in ex_examples.items()})
    stat("multi_pos_lemmas_in_pool", multi_pos)
    stat("feminine_glossed_from_masculine", sorted(set(fem_glossed)))
    stat("second_pos_entries_in_pool", second_entries)
    stat("second_pos_diagnostics", second_diag)
    stat("pronominal_verbs_shown_as_rsi", sorted(pronominal))
    stat("gloss_overrides", {"applied": sorted(set(overridden)),
                             "unused": sorted(set(GLOSS_OVERRIDES) - set(overridden))})

    # --- selection + levels
    ranked = [k for k in pool if k in records and not records[k]["forced"]]
    ranked.sort(key=lambda k: order.get(k, 10**9))
    forced_ok = [k for k in forced_keys if k in records]
    stat("forced", {"requested": len(forced_keys), "included": len(forced_ok),
                    "missing": [k[0] for k in forced_keys if k not in records]})
    need = N_WORDS - len(forced_ok)
    # a feminine noun whose masculine lemma is in the pack (l'amica, la santa,
    # l'unica) is folded into the masculine entry as an alt; refill to 2000
    fem_folded = {}
    while True:
        chosen = [k for k in ranked if k not in fem_folded][:need]
        in_pack = defaultdict(list)
        for kk in forced_ok + chosen:
            in_pack[records[kk]["lemma"]].append(kk)
        new = {}
        for k in chosen:
            for m, via_noun in records[k]["fem_of"]:
                if m == records[k]["lemma"] or k in new:
                    continue
                for mk in in_pack.get(m, []):
                    if not via_noun and mk[1] != "NOUN":
                        continue
                    if gloss_words(records[k]["en"]) & gloss_words(records[mk]["en"]):
                        new[k] = mk
                        break
        if not new:
            break
        fem_folded.update(new)
    fem_alt = defaultdict(list)
    for k, mk in sorted(fem_folded.items()):
        fem_alt[mk].append(records[k]["lemma"])
    stat("feminine_folded_into_masculine", sorted(f"{records[k]['lemma']}->{records[m]['lemma']}"
                                                  for k, m in fem_folded.items()))
    a1_rest = BANDS[0][1] - len(forced_ok)
    level_of = {}
    for k in forced_ok:
        level_of[k] = "A1"
    for i, k in enumerate(chosen):
        level_of[k] = "A1" if i < a1_rest else ("A2" if i < a1_rest + BANDS[1][1] else "B1")
    final = forced_ok + chosen
    stat("words_pos_from_dictionary", sorted(records[k]["lemma"] for k in final if records[k].get("pos_from_dict")))
    rank_order = sorted(final, key=lambda k: (order.get(k, 10**9), k))
    rank_of = {k: i + 1 for i, k in enumerate(rank_order)}

    # a content word whose main English word is already the main gloss of a
    # higher-ranked word of the same POS leads with its next-best supported
    # sense instead (stare "to stay" after essere "to be")
    def main_word(gl):
        head = re.split(r"[;,]", re.sub(r"\(.*?\)", "", gl))[0].strip().lower()
        return re.sub(r"^(to|a|an|the) ", "", head)
    taken = defaultdict(dict)
    collisions = []
    for k in rank_order:
        rec = records[k]
        g = rec["group"]
        if g not in CONTENT_GROUPS:
            continue
        mw = main_word(rec["en"])
        if mw in taken[g] and not rec["fixed"]:
            segs = [x.strip() for x in rec["en"].split(";")]
            parts = [x.strip() for x in segs[0].split(",")]
            top = rec["rows"][0] if rec["rows"] else None
            ps = top["pscore"] if top else {}
            lead = ps.get(parts[0], 0.0)
            # the next-best alternative must have its own corpus support
            alt_part = next((i for i, x in enumerate(parts) if i and main_word(x) not in taken[g]
                             and lead > 0 and ps.get(x, 0.0) >= COLLISION_SUPPORT * lead), None)
            new = None
            if alt_part:
                parts.insert(0, parts.pop(alt_part))
                new = "; ".join([", ".join(parts)] + segs[1:])
            elif top:
                for r in rec["rows"][1:]:
                    if r["demote"] or r["defn"] or r["score"] < COLLISION_SUPPORT * top["score"] or \
                            top["score"] <= 0:
                        continue
                    if main_word(r["g"]) in taken[g] or mw in r["g"].lower():
                        continue          # vetro "object made of glass" still scores on "glass"
                    new = r["g"] if len(r["g"]) + 2 + len(segs[0]) > MAX_GLOSS else f"{r['g']}; {segs[0]}"
                    break
            if new:
                collisions.append(f"{rec['lemma']}: {rec['en']} -> {new} (after {taken[g][mw]})")
                rec["en"] = new
                mw = main_word(new)
        taken[g].setdefault(mw, rec["lemma"])
    stat("gloss_collisions_resolved", collisions)

    words = []
    for k in final:
        rec = records[k]
        lem, g = rec["lemma"], rec["group"]
        w, alt, en = lem, None, rec["en"]
        gender = None
        pos = GROUP_LABEL.get(g, rec.get("pos", "other"))
        if g == "PHRASE":
            pos = "phrase"
        elif g == "FORM":
            pos = "verb"
        if k in FIXED_WORD:
            w, alt = FIXED_WORD[k][0], list(FIXED_WORD[k][1])
            pos = "art"
        elif g == "NOUN" and lem not in NO_ARTICLE:
            gender, plural = parse_gender(rec["gender"])
            mc = morph.get(k, Counter())
            if gender is None:
                gm, gf = mc["Gender=Masc"], mc["Gender=Fem"]
                gender = "m" if gm > gf else "f" if gf > gm else ("f" if lem.endswith("a") else "m")
            # plural articles only for dictionary plural-only heads and the
            # pluralia tantum list; corpus plural majorities are not evidence
            plural = plural or lem in PLURALIA_TANTUM
            if gender == "mf":
                am, af = article_for("m", lem, plural), article_for("f", lem, plural)
                if am == af == "l'":
                    w = with_article("l'", lem)
                    en = f"{en} (m/f)"
                else:
                    w = f"{am}/{af} {lem}"
            else:
                art = article_for(gender, lem, plural)
                w = with_article(art, lem)
                if art == "l'":
                    en = f"{en} ({gender})"
            alt = [lem]
        if rec.get("display"):
            lem, w, alt = rec["display"], rec["display"], [rec["lemma"]]
        if fem_alt.get(k):
            alt = (alt or []) + [x for x in fem_alt[k] if x not in (alt or [])]
        word = {"lemma": lem, "w": w, "pos": pos, "en": en, "lv": level_of[k], "rank": rank_of[k]}
        if alt:
            word["alt"] = alt
        word["_key"] = k
        if rec.get("display"):
            word["_base"] = (rec["lemma"], rec["base_en"])
        word["_gender"] = gender if g == "NOUN" else None
        word["_epos"] = rec.get("entry_pos")
        word["_sidx"] = rec.get("sense_idx") if rec.get("sense_idx") is not None else 99
        words.append(word)
    lvl = {"A1": 0, "A2": 1, "B1": 2}
    words.sort(key=lambda x: (lvl[x["lv"]], x["rank"]))

    idmap = json.loads(ID_MAP.read_text()) if ID_MAP.exists() else {}
    used = set()
    reused = 0
    for wd in words:
        old = idmap.get(f"{wd['lemma']}|{wd['pos']}")
        if old and old not in used:
            wd["id"] = old
            used.add(old)
            reused += 1
    nxt = max([int(v[1:]) for v in idmap.values()] + [0]) + 1
    for wd in words:
        if "id" not in wd:
            wd["id"] = f"w{nxt:04d}"
            nxt += 1
    stat("ids_reused_from_v1", reused)

    top3000 = set()
    for k, *_ in blended:
        if k[1] != "PROPN":
            top3000.add(k[0])
        if len(top3000) >= 3000:
            break
    return words, records, top3000



# ---------------------------------------------------------------------------
# Stage sentences: in-context word links
# ---------------------------------------------------------------------------

SENT_END_RE = re.compile(r"[.!?…][\"'»”)]*$")
TARGET_LEN = {"A1": 5, "A2": 7, "B1": 8}
MIN_LEN = {"A1": 4, "A2": 4, "B1": 5}
MAX_LEN = 14
LV_ORD = {"A1": 0, "A2": 1, "B1": 2}


def is_remoto(ms):
    return "Tense=Past" in ms and "Mood=Ind" in ms and "VerbForm=Fin" in ms


def det_gender(toks, i):
    """Gender of the article/articulated preposition governing noun token i
    (skipping adjectives), or None."""
    j = i - 1
    while j >= 0 and toks[j][2] == "ADJ":
        j -= 1
    if j < 0 or toks[j][2] not in ("DET", "ADP"):
        return None
    ms = toks[j][3]
    return "m" if "Gender=Masc" in ms else "f" if "Gender=Fem" in ms else None


def sentence_links(toks, lexicon, key_to_id, allowed, text, groups=None, gender_of=None, epos_to_id=None,
                   lemma_ids=None):
    """Word ids linked by (lemma, POS) in context, or None if the sentence
    has a content lemma outside the pack/top-3000."""
    links = []
    initial = True
    resolved = lexicon.resolve_sentence(toks, groups)
    for i, (text_t, sl, upos, ms) in enumerate(toks):
        if upos == "PUNCT":
            if text_t in (".", "!", "?", "…"):
                initial = True
            continue
        was_initial, initial = initial, False
        r = resolved[i]
        low = text_t.lower()
        if lemma_ids:
            later = toks[i + 1][0] if i + 1 < len(toks) else ""
            direct = None
            if low in APOCOPE and (was_initial or not text_t[:1].isupper()) and not later[:1].isupper():
                direct = lemma_ids.get(APOCOPE[low])     # nessun -> nessuno, quel -> quello
            elif (r is None or key_to_id.get(r) is None) and (
                    (low, "INTJ") in key_to_id or (was_initial and low in lemma_ids)):
                # "Grazie per..." read as the noun grazie; a sentence-initial
                # capitalised pack word the tagger took for a name
                direct = key_to_id.get((low, "INTJ")) or lemma_ids[low]
            if direct:
                if direct not in links:
                    links.append(direct)
                continue
        if r is None:
            continue
        lem, g = r
        if g == "PROPN" or (not was_initial and text_t[:1].isupper()):
            continue
        if g in CONTENT_GROUPS and lem not in allowed:
            return None
        nxt = toks[i + 1][2] if i + 1 < len(toks) else "PUNCT"
        if text_t.lower() == "è" and ("è", "FORM") in key_to_id:
            wid = key_to_id[("è", "FORM")]
        elif (text_t.lower(), "INTJ") in key_to_id and nxt == "PUNCT" and g != "INTJ":
            # "Prego." / "Scusa, ..." / "Grazie!": standalone greeting use
            wid = key_to_id[(text_t.lower(), "INTJ")]
        else:
            wid = key_to_id.get((lem, g))
            if wid is None and DROP_KEYS.get((lem, g)):
                wid = key_to_id.get(DROP_KEYS[(lem, g)])   # "dire di no", "o no" -> no (intj)
            if wid is None and epos_to_id and g in GROUP_KPOS:
                # no entry for this POS: the lemma's entry built from the same
                # Wiktionary headword (come ADV "Come stai?" -> come "how",
                # whose gloss comes from the adverb headword)
                wid = epos_to_id.get((lem, GROUP_KPOS[g][0]))
            if wid and g == "NOUN" and gender_of and gender_of.get(wid) in ("m", "f"):
                dg = det_gender(toks, i)
                if dg and dg != gender_of[wid]:
                    wid = None       # la moto is not il moto, il fine is not la fine
        if wid and wid not in links:
            links.append(wid)
    low = text.lower()
    for phrase in MULTIWORD:
        if re.search(r"\b" + re.escape(phrase) + r"\b", low) and (phrase, "PHRASE") in key_to_id:
            wid = key_to_id[(phrase, "PHRASE")]
            if wid not in links:
                links.append(wid)
    return links


# "bel" before a vowel is an error for "bell'" ("Non ho un bel aspetto")
BAD_ITALIAN_RE = re.compile(r"\bbel [aeiouàèéìòù]", re.I)
EN_REGISTER_RE = re.compile(r"\b(ain't|gonna|wanna|gotta|y'all|dunno|lemme|gimme|innit|ya|yer)\b"
                            r"|buzzed the control tower", re.I)


def build_sentences(ctx, words, top3000):
    lexicon = ctx["lexicon"]
    groups = ctx["lemma_groups"]
    gender_of = {w["id"]: w.get("_gender") for w in words}
    ep = defaultdict(list)
    for w in words:
        if w.get("_epos"):
            ep[(w["_key"][0], w["_epos"])].append((w["_sidx"], w["id"]))
    # several entries from one headword: the one carrying its primary sense
    epos_to_id = {k: min(v)[1] for k, v in ep.items()}
    lemma_ids = {}
    for w in sorted(words, key=lambda x: x["rank"]):
        if w["pos"] != "art":
            lemma_ids.setdefault(w["_key"][0], w["id"])
    rsi_ids = {w["id"] for w in words if w.get("_base")}
    refl_by_sid = {}
    rows = ctx["rows_by_sid"]
    key_to_id = {w["_key"]: w["id"] for w in words}
    lv_of = {w["id"]: w["lv"] for w in words}
    allowed = {w["lemma"] for w in words} | top3000
    st = Counter()
    info = {}
    cands = defaultdict(list)
    for sid, toks in iter_tagged(ctx["tagged"]):
        text = rows[sid][1]
        if not SENT_END_RE.search(text.strip()):
            st["no_terminal_punct"] += 1
            continue
        if BAD_ITALIAN_RE.search(text):
            st["ungrammatical_italian"] += 1
            continue
        if EN_REGISTER_RE.search(rows[sid][3]):
            st["non_standard_english_register"] += 1
            continue
        n = sum(1 for t in toks if t[2] not in SKIP_UPOS)
        if n < 3 or n > MAX_LEN:
            st["length_out_of_range"] += 1
            continue
        links = sentence_links(toks, lexicon, key_to_id, allowed, text, groups, gender_of, epos_to_id,
                               lemma_ids)
        if links and rsi_ids & set(links):
            refl_by_sid[sid] = {key_to_id.get(r) for j, r in enumerate(lexicon.resolve_sentence(toks, groups))
                                if r and r[1] == "VERB" and carries_refl_clitic(toks, j)}
        if links is None:
            st["content_lemma_outside_pack_top3000"] += 1
            continue
        if not links:
            continue
        remoto = any(is_remoto(t[3]) or (r is not None and r[1] == "VERB" and lexicon.historic_past(t[0].lower()))
                     for t, r in zip(toks, lexicon.resolve_sentence(toks, groups)))
        maxlv = max((lv_of[w] for w in links), key=lambda l: LV_ORD[l])
        if remoto:
            st["candidates_with_passato_remoto"] += 1
        info[sid] = (n, remoto, rows[sid][4] is not None, maxlv, links)
        for w in links:
            cands[w].append(sid)
    st["candidates"] = len(info)

    use = Counter()
    chosen = defaultdict(list)
    primary = {}
    remoto_blocked = Counter()
    order = sorted(cands, key=lambda w: (len(cands[w]), w))
    for wid in order:
        lv = lv_of[wid]
        ok = []
        for sid in cands[wid]:
            n, remoto, aud, maxlv, links = info[sid]
            if remoto and lv != "B1":
                remoto_blocked[sid] += 1
                continue
            ok.append(sid)
        good = [s for s in ok if MIN_LEN[lv] <= info[s][0]]
        if lv == "A1" and len(good) < 2:
            good += [s for s in ok if info[s][0] == 3]
        good.sort(key=lambda s: (LV_ORD[info[s][3]] > LV_ORD[lv], not info[s][2],
                                 abs(info[s][0] - TARGET_LEN[lv]), use[s] == 0, s))
        for s in good[:2]:
            chosen[wid].append(s)
            use[s] += 1
            primary.setdefault(s, wid)

    sel = sorted(primary)
    sentences = []
    users = set()
    for i, sid in enumerate(sel):
        n, remoto, aud, maxlv, links = info[sid]
        lv = "B1" if remoto else maxlv
        row = rows[sid]
        rec = {"id": f"s{i+1:04d}", "t": re.sub(r"\s+([?!])", r"\1", row[1]), "en": row[3], "lv": lv,
               "words": links}
        if row[4] is not None:
            rec["audio"] = f"https://tatoeba.org/audio/download/{row[4]}"
        sentences.append(rec)
        if row[2]:
            users.add(row[2])
    # -rsi label only when the pack's own sentences show it: >=60% of the
    # sentences linking the word carry a reflexive clitic on it; otherwise the
    # base infinitive with a combined gloss
    reverted = []
    for w in words:
        if not w.get("_base"):
            continue
        linked = [sid for sid in sel if w["id"] in info[sid][4]]
        if not linked:
            continue
        rs = sum(1 for sid in linked if w["id"] in refl_by_sid.get(sid, ()))
        if rs / len(linked) < PRONOMINAL_LINKED_SHARE:
            base, base_en = w["_base"]
            rsi, rsi_en = w["lemma"], w["en"]
            w["lemma"], w["w"] = base, base
            w.pop("alt", None)
            head = lambda x: re.split(r"[,;]", x)[0].strip()
            w["en"] = f"{head(base_en)}; {rsi}: {head(rsi_en)}" if base_en else w["en"]
            reverted.append(f"{rsi} ({rs}/{len(linked)})")
    st_rev = sorted(reverted)
    cov = Counter(min(len(chosen.get(w["id"], [])), 2) for w in words)
    lens = Counter(info[s][0] for s in sel)
    st = dict(st)
    st.update({
        "final_sentences": len(sentences),
        "with_audio": sum(1 for s in sentences if "audio" in s),
        "coverage_0": cov[0], "coverage_1": cov[1], "coverage_2": cov[2],
        "remoto_candidates_blocked_for_A1_A2": len(remoto_blocked),
        "remoto_in_final": sum(1 for s in sel if info[s][1]),
        "length_hist": {str(k): lens[k] for k in sorted(lens)},
        "primary_by_level": dict(Counter(lv_of[primary[s]] for s in sel)),
        "zero_sentence_words": [w["w"] for w in words if not chosen.get(w["id"])],
        "rsi_reverted_to_base": st_rev,
    })
    stat("sentences", dict(st))
    return sentences, sorted(users), primary


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def build_pack_json(words, raw_upos):
    fids = []
    for w in words:
        k = w["_key"]
        if k[1] == "FORM":
            fids.append(w["id"])
            continue
        c = raw_upos.get(k)
        if k[1] == "VERB" and k[0] not in ("essere", "avere"):
            continue       # modal/aux verbs (potere, stare...) are drilled as words
        if k in FIXED_WORD or (c and c.most_common(1)[0][0] in FUNCTION_UPOS):
            fids.append(w["id"])
    return {
        "key": "it",
        "name": "Italian (A1–B1)",
        "tts": "it-IT",
        "stt": "it-IT",
        "ttsRate": 0.9,
        "levels": [{"id": "A1", "label": "A1"}, {"id": "A2", "label": "A2"}, {"id": "B1", "label": "B1"}],
        "setSize": 10,
        "placement": [["A1", 4], ["A2", 4], ["B1", 4]],
        "functionWords": sorted(fids),
        "typing": {"caseSensitive": False, "accents": "lenient", "strictFromLevel": "B1"},
        "showPron": False,
        "hasLessons": False,
    }


def audio_recorders(ctx, sentences):
    """{licence: [{"recorder": username, "clips": n}]} for the clips the pack links."""
    used = {int(s["audio"].rsplit("/", 1)[1]) for s in sentences if "audio" in s}
    meta = {}
    with tarfile.open(CACHE / "audio.tar.bz2", "r:bz2") as tf:
        member = next(m for m in tf.getmembers() if "sentences_with_audio" in m.name)
        for raw in io.TextIOWrapper(tf.extractfile(member), encoding="utf-8"):
            p = raw.rstrip("\n").split("\t")
            if len(p) >= 4 and p[1].isdigit() and int(p[1]) in used:
                meta[int(p[1])] = (p[2], p[3])
    per = defaultdict(Counter)
    for aid in used:
        user, lic = meta.get(aid, ("unknown", "unknown"))
        per[lic or "unknown"][user or "unknown"] += 1
    return {lic: [{"recorder": u, "clips": n} for u, n in sorted(c.items(), key=lambda x: (-x[1], x[0]))]
            for lic, c in sorted(per.items())}


def write_json(path, obj, compact=True):
    if compact:
        txt = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    else:
        txt = json.dumps(obj, ensure_ascii=False, indent=2)
    path.write_text(txt + "\n")


def prepare(ctx):
    corpus = stage_corpus()
    ctx["rows_by_sid"] = {r[0]: r for r in corpus["rows"]}
    ctx["en_by_sid"] = {r[0]: r[3] for r in corpus["rows"]}
    ctx["truecase"] = truecase_stats(corpus["rows"])
    for r in corpus["rows"]:
        EN_VOCAB.update(EN_WORD_RE.findall(r[3].lower()))
    bg = Counter()
    for r in corpus["rows"]:
        bg.update(set(en_stems(r[3], keep_stop=True)))
    ctx["en_bg"], ctx["en_bgn"] = bg, len(corpus["rows"])
    stat("corpus", {k: v for k, v in corpus.items() if k != "rows"})
    ctx["tagged"] = stage_tag(corpus)
    ctx["lexicon"] = Lexicon(stage_lex())
    # two passes: the first gives each lemma's POS mix, which the copula rule
    # of the second uses ("è ridicolo" = the adjective)
    lx = ctx["lexicon"]
    for _ in range(2):
        surf, raw_upos, morph, refl, initial = corpus_usage(ctx["tagged"], lx, ctx.get("lemma_groups"))
        lg = defaultdict(Counter)
        for (lem, g), c in raw_upos.items():
            lg[lem][g] += sum(c.values())
        ctx["lemma_groups"] = lg
    ctx.update(surf=surf, raw_upos=raw_upos, morph=morph, refl=refl, initial=initial)
    ctx["blended"] = stage_freq(surf, raw_upos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "corpus", "tag", "lex", "freq", "words", "sentences", "final"])
    parser.add_argument("--check-remote", action="store_true",
                        help="re-download sources whose remote size changed")
    args = parser.parse_args()
    t0 = time.time()
    ensure_downloaded(args.check_remote)
    if args.stage == "corpus":
        stage_corpus(); return
    if args.stage == "tag":
        stage_tag(stage_corpus()); log(STATS.get("tag_meta")); return
    if args.stage == "lex":
        stage_lex(); return
    ctx = {}
    prepare(ctx)
    lx = ctx["lexicon"]
    stat("resolution", {"unresolved_tokens_by_pos": dict(lx.n_unresolved),
                        "clitic_compounds_resolved": lx.n_clitic,
                        "numeral_tokens_read_as_verb": lx.n_num_as_verb,
                        "rare_reading_overrides": lx.n_rare_override,
                        "adv_verb_after_article_read_as_noun": lx.n_after_article,
                        "noun_after_copula_read_as_adjective": lx.n_copula_adj,
                        "pronoun_before_noun_read_as_article": lx.n_pron_as_article,
                        "initial_noun_before_determiner_read_as_imperative": lx.n_imperative})
    if args.stage == "freq":
        log(json.dumps(STATS.get("freq"))); return
    words, records, top3000 = build_words(ctx)
    PACK.mkdir(exist_ok=True)
    out_words = [{k: w[k] for k in ("id", "w", "lemma", "pos", "en", "lv", "rank", "alt") if k in w}
                 for w in words]
    write_json(PACK / "words.json", out_words)
    if args.stage == "words":
        return
    sentences, users, primary = build_sentences(ctx, words, top3000)
    out_words = [{k: w[k] for k in ("id", "w", "lemma", "pos", "en", "lv", "rank", "alt") if k in w}
                 for w in words]
    write_json(PACK / "words.json", out_words)     # -rsi gate may revert entries
    write_json(PACK / "sentences.json", sentences)
    write_json(PACK / "pack.json", build_pack_json(words, ctx["raw_upos"]), compact=False)
    attribution = {
        "spoken_freq": {"source": "hermitdave/FrequencyWords", "licence": "CC-BY-SA-4.0",
                        "url": SOURCES["it_full.txt"]},
        "written_freq": {"source": "wordfreq (Python package)", "licence": "CC-BY-SA-4.0"},
        "dictionary": {"source": "kaikki.org Italian Wiktionary extract", "licence": "CC-BY-SA-3.0/GFDL",
                       "url": SOURCES["kaikki_it.jsonl.gz"]},
        "tagger": {"source": f"spaCy (MIT) + {SPACY_MODEL} model (CC BY-NC-SA 3.0, trained on UD Italian ISDT)",
                   "licence": "CC BY-NC-SA 3.0 (model)",
                   "note": "Used at build time only; the pack ships no model files. This project is non-commercial."},
        "sentences": {"source": "Tatoeba ita_sentences_detailed.tsv", "licence": "CC-BY 2.0 FR",
                      "url": SOURCES["ita_detailed.tsv.bz2"], "contributor_usernames": users},
        "audio": {"source": "Tatoeba sentences_with_audio.tsv",
                  "licences": "per clip; recorders listed per licence",
                  "recorders": audio_recorders(ctx, sentences)},
    }
    write_json(PACK / "attribution.json", attribution, compact=False)
    ctx.update(words=words, records=records, sentences=sentences, primary=primary)
    stat("wall_seconds_this_run", round(time.time() - t0, 1))
    dump_json(DERIVED / "build_stats.json", STATS)
    write_report(ctx)
    log(f"done in {time.time()-t0:.1f}s")


MANUAL_BEGIN = "<!-- manual:begin -->"
MANUAL_END = "<!-- manual:end -->"


def kelly_crosscheck(words):
    path = CACHE / "kelly_it.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    tag = {}

    def walk(o):
        if isinstance(o, dict):
            w, lv = o.get("word") or o.get("lemma"), o.get("cefr") or o.get("level")
            if isinstance(w, str) and isinstance(lv, str) and lv in ("A1", "A2", "B1", "B2", "C1", "C2"):
                for part in re.split(r"[,/]\s*", w):
                    tag.setdefault(part.strip().lower(), lv)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(data)
    if not tag:
        return None
    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    conf = Counter()
    for w in words:
        k = tag.get(w["lemma"])
        if k:
            conf[(w["lv"], k)] += 1
    n = sum(conf.values())
    exact = sum(conf[(a, a)] for a in ("A1", "A2", "B1"))
    near = sum(v for (a, b), v in conf.items() if abs(levels.index(a) - levels.index(b)) <= 1)
    return {"matched": n, "tagged": len(tag), "exact": exact, "within1": near,
            "table": {a: [conf[(a, b)] for b in levels] for a in ("A1", "A2", "B1")}}


def write_report(ctx):
    words = ctx["words"]
    S = STATS
    path = TOOLS / "REPORT.md"
    manual = ""
    if path.exists():
        old = path.read_text()
        if MANUAL_BEGIN in old and MANUAL_END in old:
            manual = old[old.index(MANUAL_BEGIN): old.index(MANUAL_END) + len(MANUAL_END)]
    if not manual:
        manual = f"{MANUAL_BEGIN}\n{MANUAL_END}"
    L = []
    a = L.append
    a("# Build report - Italian A1-B1 pack (v2, corpus-tagged)\n")
    a("Generated by `tools/build_pack.py` on every full run. Everything outside the "
      "`manual` markers is regenerated; the manual section (QA verdicts, check output) is preserved.\n")
    tm = S.get("tag_meta", {})
    a("## Tagging\n")
    a(f"- Tagger: {S.get('tagger')} (parser and NER disabled).")
    a(f"- Corpus: {S['corpus']['n_ita']:,} Tatoeba Italian sentences; {S['corpus']['n_with_en']:,} have an "
      f"English translation and were tagged ({tm.get('tokens', 0):,} tokens).")
    a(f"- Tagging wall time: **{tm.get('seconds')}s** (6 processes; cached in `.cache/derived/tagged_*.jsonl.gz`, "
      f"keyed by tag version + spaCy/model version + corpus signature).")
    a(f"- Sentence-initial capitals lowercased before tagging (truecasing, proper nouns kept): "
      f"{tm.get('truecased_initials', 0):,} sentences.")
    a(f"- Sentences with a permissively licensed recording: {S['corpus']['n_audio']:,}.\n")
    fr = S["freq"]
    a("## Lemma + POS resolution\n")
    a("Each subtitle / wordfreq surface form is split over the (lemma, POS) pairs it takes in the tagged "
      f"corpus (pairs under {int(MIN_SHARE*100)}% share dropped). Surfaces unseen in the corpus fall back "
      "to simplemma.\n")
    a("| list | surfaces resolved from corpus | simplemma fallback |")
    a("|---|---|---|")
    a(f"| subtitles (top {N_SUB_SURFACES:,} surfaces) | {fr['sub_surfaces'].get('corpus', 0):,} | "
      f"{fr['sub_surfaces'].get('fallback', 0):,} |")
    a(f"| wordfreq top 30,000 | {fr['wf_surfaces'].get('corpus', 0):,} | {fr['wf_surfaces'].get('fallback', 0):,} |")
    rs = S.get("resolution", {})
    a(f"\nFallback lemmas whose POS could not be attached from another surface: {fr['unknown_pos_keys']:,}. "
      f"Clitic compounds resolved to the verb by suffix stripping: {rs.get('clitic_compounds_resolved', 0):,} "
      f"distinct forms. Tokens with no dictionary-validated lemma (kept, never selected as words): "
      f"{sum(rs.get('unresolved_tokens_by_pos', {}).values()):,}.\n")
    a("## Word-selection funnel\n")
    a(f"Candidate pool: {S['pool_keys']:,} (lemma, POS) keys in blended-rank order, one POS per lemma "
      "(its best-ranked corpus POS; articles il/uno kept alongside), plus forced items.\n")
    a("| Exclusion | Count | Examples |")
    a("|---|---|---|")
    for r, c in sorted(S["excluded"].items(), key=lambda kv: -kv[1]):
        a(f"| {r} | {c:,} | {', '.join(S['excluded_examples'].get(r, [])[:12])} |")
    fo = S["forced"]
    a(f"\nForced A1 items (days, months, numbers 0-20 + tens + cento/mille, colours, greetings, a/e/o/è, "
      f"A1 core list): {fo['included']}/{fo['requested']} included"
      + (f"; missing: {', '.join(fo['missing'])}" if fo["missing"] else "") + ".")
    a(f"Word ids reused from v1 for unchanged (lemma, pos): {S['ids_reused_from_v1']:,}; the rest are new "
      "ids above w2000 (v1 ids of words whose POS/lemma was wrong are retired, not reassigned).\n")
    lv = Counter(w["lv"] for w in words)
    pos = Counter(w["pos"] for w in words)
    a(f"Levels: {dict(sorted(lv.items()))}. POS: {dict(pos.most_common())}.\n")
    st = S["sentences"]
    a("## Sentences\n")
    a(f"- Final sentences: **{st['final_sentences']:,}**, {st['with_audio']:,} with audio "
      "(`https://tatoeba.org/audio/download/<audio_id>`).")
    a(f"- Word coverage: 0 = {st['coverage_0']}, 1 = {st['coverage_1']}, 2 = {st['coverage_2']}.")
    if st["zero_sentence_words"]:
        a(f"- Words with no sentence: {', '.join(st['zero_sentence_words'])}.")
    a(f"- Candidate sentences (terminal punctuation, 3-{MAX_LEN} tokens, content lemmas in pack/top-3000, "
      f">=1 link): {st['candidates']:,}. Rejected for a content lemma outside pack/top-3000: "
      f"{st.get('content_lemma_outside_pack_top3000', 0):,}.")
    a(f"- Passato remoto: {st.get('candidates_with_passato_remoto', 0):,} candidates contain one; "
      f"{st['remoto_candidates_blocked_for_A1_A2']:,} were blocked for A1/A2 words; "
      f"{st['remoto_in_final']} in the final set (all lv B1).")
    a(f"- Primary word level of each sentence: {st['primary_by_level']}.")
    a("- Token-length distribution of the final set:\n")
    a("| tokens | " + " | ".join(st["length_hist"]) + " |")
    a("|---|" + "---|" * len(st["length_hist"]))
    a("| sentences | " + " | ".join(str(v) for v in st["length_hist"].values()) + " |\n")
    kc = kelly_crosscheck(words)
    if kc:
        a("## Kelly CEFR cross-check (sanity only, not shipped)\n")
        a(f"{kc['matched']:,} of 2,000 lemmas matched Kelly. Exact level agreement {kc['exact']}/{kc['matched']} "
          f"= {100*kc['exact']/max(kc['matched'],1):.1f}%; within one level {kc['within1']}/{kc['matched']} "
          f"= {100*kc['within1']/max(kc['matched'],1):.1f}%.\n")
        a("| pack \\ kelly | A1 | A2 | B1 | B2 | C1 | C2 |")
        a("|---|---|---|---|---|---|---|")
        for k, row in kc["table"].items():
            a(f"| **{k}** | " + " | ".join(str(x) for x in row) + " |")
        a("")
    a("## Top 100 by rank (lemma [pos] gloss)\n")
    a("```")
    for w in sorted(words, key=lambda x: x["rank"])[:100]:
        a(f"{w['rank']:>4} {w['w']} [{w['pos']}] {w['en']}")
    a("```\n")
    a(manual)
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
