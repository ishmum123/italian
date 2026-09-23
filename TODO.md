# TODO (v4 candidates)

Residuals from the v3 QA rounds. See `tools/REPORT.md` for the rules
already in place and their counts.

## Sentence links
- "Dai," at the start of a sentence (come on) links to the preposition da.
- The conjunction che is sometimes tagged PRON and links to che (pron).
- Compounds are read word by word. "fine settimana" (weekend) links nothing
  useful, and "come se" (as if) links come "how". A compound table or
  multi-word links would fix both.
- Second-entry link swaps. Some sentences go to the other entry of the same
  lemma: via noun vs adv, ufficiale adj vs noun, and cosa noun getting
  pronoun sentences. The tagger's POS decides, so its errors show here.

## Words and glosses
- The piano adverb ("slowly, quietly") is absent. It is about 2% of piano
  tokens, below the 20% second-entry share.
- rapporto leads with "report". "relationship" is as common.
- The -rsi gate uses at most 2-4 linked sentences per verb, so it is noisy.
  Reverted verbs show a combined gloss ("affidare: to entrust; affidarsi:
  to rely on").

## Engine (vocab-engine repo)
- Verb examples show conjugated forms: 462 verbs have at least one example
  without the infinitive. The engine could highlight the linked token.
- The "5. Sentences" label wraps at 390px width.

## Licence
- The build-time tagger model (spaCy it_core_news_sm) is CC BY-NC-SA 3.0.
  The pack ships no model files, and the project must stay non-commercial
  while this model is used. A commercial use would need a differently
  licensed tagger.
