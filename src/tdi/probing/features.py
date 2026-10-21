"""Linguistic features to probe, defined by (example, token position, label) samples.

Each feature also says how samples are grouped for the train/test split, so a probe has to
generalize to unseen nouns (or unseen sentences) rather than memorize token identities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..encoding import EncodedExample
from ..synthetic import POS_OF_ROLE

POS_CLASSES = ["DET", "NOUN", "ADV", "PREP", "REL", "VERB"]


@dataclass
class Samples:
    ex: np.ndarray  # example index
    pos: np.ndarray  # token position within the encoded prefix
    y: np.ndarray  # class label (int)
    tok: np.ndarray  # token id at pos (used for the control task)
    group: np.ndarray  # split group; train and test never share a group
    n_classes: int

    def __len__(self) -> int:
        return len(self.y)


def _build(rows: list[tuple[int, int, int, int, int]], n_classes: int) -> Samples:
    a = np.array(rows, dtype=np.int64).reshape(-1, 5)
    return Samples(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], n_classes)


def _number(n: str) -> int:
    return 0 if n == "sg" else 1


def subj_number_at_subject(items: list[EncodedExample]) -> Samples:
    """Grammatical number of the subject, read at the subject noun itself."""
    rows = []
    for i, it in enumerate(items):
        p = it.word_end[1]
        rows.append((i, p, _number(it.example.subject_number), it.prefix_ids[p], it.example.subject_lexeme))
    return _build(rows, 2)


def subj_number_at_last(items: list[EncodedExample]) -> Samples:
    """Subject number read at the last prefix token, where the verb is predicted. In attractor
    conditions this token is not the subject, so the number must have been carried forward."""
    rows = []
    for i, it in enumerate(items):
        p = it.word_end[-1]
        rows.append((i, p, _number(it.example.subject_number), it.prefix_ids[p], it.example.subject_lexeme))
    return _build(rows, 2)


def attractor_number_at_attractor(items: list[EncodedExample]) -> Samples:
    """Number of the (last) attractor noun at that noun. It is irrelevant to the correct verb
    form, so it is our candidate for a feature that is decodable but not causally used."""
    rows = []
    for i, it in enumerate(items):
        att = it.example.attractor_positions
        if not att:
            continue
        w = att[-1]
        p = it.word_end[w]
        rows.append((i, p, _number(it.example.attractor_number), it.prefix_ids[p], it.example.lexemes[w]))
    return _build(rows, 2)


def pos_at_word(items: list[EncodedExample]) -> Samples:
    """Coarse part of speech of every prefix word, read at the word's last token."""
    rows = []
    for i, it in enumerate(items):
        for w, role in enumerate(it.example.roles):
            p = it.word_end[w]
            rows.append((i, p, POS_CLASSES.index(POS_OF_ROLE[role]), it.prefix_ids[p], i))
    return _build(rows, len(POS_CLASSES))


FEATURES = {
    "subj_number@subj": subj_number_at_subject,
    "subj_number@last": subj_number_at_last,
    "attr_number@attr": attractor_number_at_attractor,
    "pos@word": pos_at_word,
}
