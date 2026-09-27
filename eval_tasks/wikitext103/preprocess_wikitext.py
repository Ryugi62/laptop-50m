# Re-export lm-evaluation-harness' own WikiText detokenizer / metrics so scoring is identical to the built-in task.
from lm_eval.tasks.wikitext.preprocess_wikitext import process_results, wikitext_detokenizer  # noqa: F401
