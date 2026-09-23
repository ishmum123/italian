#!/usr/bin/env python3
"""Schema + referential integrity checks for the built pack.

Run after tools/build_pack.py. Exits non-zero on any failure.
"""
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "pack"
sys.path.insert(0, str(ROOT / "tools"))
from build_pack import PLURALIA_TANTUM, article_for  # noqa: E402  (stdlib-only module top)

SINGULAR_ARTICLES = {"il", "lo", "la", "l'", "il/la"}
PLURAL_ARTICLES = {"i", "gli", "le"}


def check_noun_article(w):
    """A noun's displayed article must be singular (il, lo, la, l', il/la) and
    the right form for the lemma's first letters; plural articles only for
    pluralia tantum."""
    shown, lemma = w["w"], (w.get("alt") or [w["lemma"]])[0]
    if shown == lemma:
        return None                      # no article (days, months)
    if shown == f"l'{lemma}":
        art = "l'"
    elif shown.endswith(" " + lemma):
        art = shown[: -len(lemma) - 1]
    else:
        return f"noun {w['id']} {shown!r}: display does not end in its lemma {lemma!r}"
    if art in PLURAL_ARTICLES:
        if lemma not in PLURALIA_TANTUM:
            return f"noun {w['id']} {shown!r}: plural article on a lemma outside PLURALIA_TANTUM"
        return None
    if art not in SINGULAR_ARTICLES:
        return f"noun {w['id']} {shown!r}: article {art!r} not in {sorted(SINGULAR_ARTICLES)}"
    if art == "il/la":
        ok = article_for("m", lemma) == "il" and article_for("f", lemma) == "la"
    elif art == "l'":
        ok = "l'" in (article_for("m", lemma), article_for("f", lemma))
    else:
        ok = art in (article_for("m", lemma), article_for("f", lemma))
    return None if ok else f"noun {w['id']} {shown!r}: wrong article form for {lemma!r}"

VALID_LEVELS = {"A1", "A2", "B1"}
FAILS = []
WARNS = []


def fail(msg):
    FAILS.append(msg)


def warn(msg):
    WARNS.append(msg)


def main():
    pack = json.loads((PACK / "pack.json").read_text())
    words = json.loads((PACK / "words.json").read_text())
    sentences = json.loads((PACK / "sentences.json").read_text())

    # -- pack.json basics --
    for key in ("key", "name", "tts", "ttsRate", "levels", "setSize", "placement",
                "functionWords", "typing", "showPron", "hasLessons"):
        if key not in pack:
            fail(f"pack.json missing key: {key}")

    # -- words.json --
    if len(words) != 2000:
        fail(f"words.json has {len(words)} entries, expected 2000")

    ids_seen = set()
    lv_counts = Counter()
    for w in words:
        for key in ("id", "w", "lemma", "pos", "en", "lv", "rank"):
            if key not in w:
                fail(f"word {w.get('id','?')} missing key {key}")
        if w["id"] in ids_seen:
            fail(f"duplicate word id: {w['id']}")
        ids_seen.add(w["id"])
        if w.get("pos") == "noun":
            err = check_noun_article(w)
            if err:
                fail(err)
        if w.get("lv") not in VALID_LEVELS:
            fail(f"word {w['id']} has invalid lv: {w.get('lv')}")
        else:
            lv_counts[w["lv"]] += 1

    function_word_ids = set(pack.get("functionWords", []))
    unknown_fw = function_word_ids - ids_seen
    if unknown_fw:
        fail(f"functionWords references unknown word ids: {sorted(unknown_fw)[:10]}")

    # -- sentences.json --
    sent_ids_seen = set()
    for s in sentences:
        for key in ("id", "t", "en", "lv", "words"):
            if key not in s:
                fail(f"sentence {s.get('id','?')} missing key {key}")
        if s["id"] in sent_ids_seen:
            fail(f"duplicate sentence id: {s['id']}")
        sent_ids_seen.add(s["id"])
        if s.get("lv") not in VALID_LEVELS:
            fail(f"sentence {s['id']} has invalid lv: {s.get('lv')}")
        for wid in s.get("words", []):
            if wid not in ids_seen:
                fail(f"sentence {s['id']} references unknown word id {wid}")

    # -- coverage: every word has >=1 sentence, ideally >=2 --
    word_sentence_count = Counter()
    for s in sentences:
        for wid in s.get("words", []):
            word_sentence_count[wid] += 1

    zero = [w["id"] for w in words if word_sentence_count[w["id"]] == 0]
    one = [w["id"] for w in words if word_sentence_count[w["id"]] == 1]
    two_plus = [w["id"] for w in words if word_sentence_count[w["id"]] >= 2]

    pct_ge1 = 100.0 * (len(words) - len(zero)) / len(words)
    pct_ge2 = 100.0 * len(two_plus) / len(words)

    if pct_ge1 < 90:
        fail(f"only {pct_ge1:.1f}% of words have >=1 sentence (need >=90%)")
    if pct_ge2 < 70:
        warn(f"only {pct_ge2:.1f}% of words have >=2 sentences (target >=70%)")

    print("=== check_pack summary ===")
    print(f"words: {len(words)} total; levels: {dict(lv_counts)}")
    print(f"sentences: {len(sentences)} total")
    print(f"word sentence coverage: 0={len(zero)} 1={len(one)} 2+={len(two_plus)} "
          f"(>=1: {pct_ge1:.1f}%, >=2: {pct_ge2:.1f}%)")
    print(f"function words: {len(function_word_ids)}")

    if WARNS:
        print("\nWARNINGS:")
        for w in WARNS:
            print(f"  - {w}")

    if FAILS:
        print("\nFAILURES:")
        for f in FAILS:
            print(f"  - {f}")
        print(f"\ncheck_pack: FAILED ({len(FAILS)} failures)")
        sys.exit(1)
    else:
        print("\ncheck_pack: PASSED")


if __name__ == "__main__":
    main()
