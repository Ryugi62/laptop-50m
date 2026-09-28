"""Word-13-gram decontamination against reference texts (pure python)."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str):
    return _WORD.findall(text.lower())


class NgramIndex:
    def __init__(self, refs, n: int = 13):
        self.n = n
        self.grams = set()
        for r in refs:
            w = words(r)
            for i in range(len(w) - n + 1):
                self.grams.add(hash(tuple(w[i:i + n])))

    def is_contaminated(self, doc: str, stride: int = 1) -> bool:
        w = words(doc)
        n = self.n
        g = self.grams
        for i in range(0, len(w) - n + 1, stride):
            if hash(tuple(w[i:i + n])) in g:
                return True
        return False
