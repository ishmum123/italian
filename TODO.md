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

## Builder flags added for Spanish (off for Italian; each a v4 follow-up)
- `prefer_headword_sentence`: pick example sentences that show the headword
  (or an alt) first, then a 3sg present verb form. In Spanish it cut words
  whose examples never show the headword from 656 to 17 of 1998. Turning it
  on here changes sentence choice, so it needs a QA round.
- `derived_form_tags`: Wiktionary diminutive/augmentative form-of senses stop
  being inflections (es: señorita is not a form of señora). Italian has the
  same pattern (casetta, ragazzino); check links before enabling.
- `phrase_token_spans`, `homograph_by_translation`, `initial_noun_verb_homograph`,
  `sensitive_re`, `translation_mismatch`: see engine/tools/packbuilder/README.md.
