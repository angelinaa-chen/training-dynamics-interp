"""Templated subject-verb agreement grammar.

Serves three purposes:
  1. a small corpus where the correct mechanism is known (pipeline validation),
  2. the source of minimal-pair evaluation sets for any model (including WikiText-trained ones),
  3. controlled inputs for probing and activation patching.

Every sentence has the form `<prefix> <verb> .` where the verb's form is determined by the
number of the subject. Attractor nouns between subject and verb can match or mismatch it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

# (singular, plural)
NOUNS: list[tuple[str, str]] = [
    ("key", "keys"), ("dog", "dogs"), ("teacher", "teachers"), ("doctor", "doctors"),
    ("student", "students"), ("author", "authors"), ("farmer", "farmers"), ("driver", "drivers"),
    ("singer", "singers"), ("player", "players"), ("artist", "artists"), ("worker", "workers"),
    ("manager", "managers"), ("captain", "captains"), ("pilot", "pilots"), ("nurse", "nurses"),
    ("writer", "writers"), ("painter", "painters"), ("lawyer", "lawyers"), ("judge", "judges"),
    ("book", "books"), ("car", "cars"), ("house", "houses"), ("tree", "trees"),
    ("door", "doors"), ("window", "windows"), ("table", "tables"), ("chair", "chairs"),
    ("cabinet", "cabinets"), ("bridge", "bridges"), ("river", "rivers"), ("mountain", "mountains"),
    ("road", "roads"), ("letter", "letters"), ("picture", "pictures"), ("garden", "gardens"),
    ("boat", "boats"), ("bottle", "bottles"), ("lamp", "lamps"), ("clock", "clocks"),
    ("child", "children"), ("man", "men"), ("woman", "women"), ("mouse", "mice"),
    ("bird", "birds"), ("horse", "horses"), ("farm", "farms"), ("village", "villages"),
    ("tower", "towers"), ("shop", "shops"), ("train", "trains"), ("bus", "buses"),
    ("cat", "cats"), ("king", "kings"), ("soldier", "soldiers"), ("guard", "guards"),
    ("visitor", "visitors"), ("neighbor", "neighbors"), ("friend", "friends"), ("leader", "leaders"),
]

# (singular form, plural form)
VERBS: list[tuple[str, str]] = [
    ("runs", "run"), ("sleeps", "sleep"), ("walks", "walk"), ("smiles", "smile"),
    ("waits", "wait"), ("works", "work"), ("laughs", "laugh"), ("sings", "sing"),
    ("falls", "fall"), ("stands", "stand"), ("is", "are"), ("was", "were"),
    ("has", "have"), ("seems", "seem"), ("looks", "look"), ("stays", "stay"),
    ("grows", "grow"), ("speaks", "speak"), ("lives", "live"), ("arrives", "arrive"),
]

PREPOSITIONS = ["near", "behind", "beside", "with", "by", "under", "above", "outside"]
RELATIVE_VERBS = ["saw", "liked", "met", "helped", "heard", "found"]  # past tense: no agreement
ADVERBS = ["often", "never", "always", "still", "also"]

CONDITIONS = ["simple", "adverb", "pp_match", "pp_mismatch", "pp2_mismatch", "rel_mismatch"]


NOUN_INDEX = {n: i for i, n in enumerate(NOUNS)}

POS_OF_ROLE = {"DET": "DET", "SUBJ": "NOUN", "ATTR": "NOUN", "ADV": "ADV", "PREP": "PREP",
               "REL": "REL", "RELV": "VERB"}


@dataclass
class Lexicon:
    """Nouns split into train and held-out sets (held-out nouns are never in the synthetic corpus)."""

    train: list[tuple[str, str]]
    test: list[tuple[str, str]]

    def nouns(self, split: str) -> list[tuple[str, str]]:
        return {"train": self.train, "test": self.test, "all": self.train + self.test}[split]


def make_lexicon(held_out_frac: float = 0.2, seed: int = 0) -> Lexicon:
    rng = random.Random(seed)
    nouns = NOUNS[:]
    rng.shuffle(nouns)
    k = int(len(nouns) * held_out_frac)
    return Lexicon(train=nouns[k:], test=nouns[:k])


@dataclass
class Example:
    words: list[str]  # prefix words, e.g. ["The", "key", "near", "the", "cabinets"]
    roles: list[str]  # per-word role: DET, SUBJ, ADV, PREP, ATTR, REL, RELV
    lexemes: list[int | None]  # per-word index into NOUNS for SUBJ/ATTR words, else None
    verb_idx: int  # index into VERBS
    correct: str  # correct verb form, e.g. "is"
    wrong: str  # incorrect verb form, e.g. "are"
    subject_number: str  # "sg" | "pl"
    attractor_number: str | None  # number of the nearest attractor, if any
    condition: str
    subject: str  # the subject noun in the form used
    distance: int  # number of tokens (words) between subject and verb

    @property
    def prefix(self) -> str:
        return " ".join(self.words)

    @property
    def text(self) -> str:
        return f"{self.prefix} {self.correct} ."

    @property
    def subject_lexeme(self) -> int:
        return self.lexemes[1]  # type: ignore[return-value]

    @property
    def attractor_positions(self) -> list[int]:
        return [i for i, r in enumerate(self.roles) if r == "ATTR"]

    def flipped(self) -> "Example":
        """Counterfactual with the subject's number flipped (everything else identical)."""
        num = "pl" if self.subject_number == "sg" else "sg"
        words = self.words[:]
        words[1] = _form(NOUNS[self.subject_lexeme], num)
        verb = VERBS[self.verb_idx]
        correct, wrong = (verb[0], verb[1]) if num == "sg" else (verb[1], verb[0])
        return replace(
            self, words=words, correct=correct, wrong=wrong, subject_number=num,
            subject=words[1],
        )


def _form(noun: tuple[str, str], number: str) -> str:
    return noun[0] if number == "sg" else noun[1]


def sample(
    rng: random.Random, lex: Lexicon, split: str = "train", condition: str | None = None
) -> Example:
    pool = lex.nouns(split)
    condition = condition or rng.choice(CONDITIONS)
    subj = rng.choice(pool)
    snum = rng.choice(["sg", "pl"])
    other = "pl" if snum == "sg" else "sg"
    verb_idx = rng.randrange(len(VERBS))
    verb = VERBS[verb_idx]

    words: list[str] = ["The", _form(subj, snum)]
    roles: list[str] = ["DET", "SUBJ"]
    lexemes: list[int | None] = [None, NOUN_INDEX[subj]]

    def add(word: str, role: str, lexeme: int | None = None) -> None:
        words.append(word)
        roles.append(role)
        lexemes.append(lexeme)

    def add_attractor(num: str) -> None:
        n = rng.choice([x for x in pool if x is not subj])
        add("the", "DET")
        add(_form(n, num), "ATTR", NOUN_INDEX[n])

    anum: str | None = None
    if condition == "simple":
        pass
    elif condition == "adverb":
        add(rng.choice(ADVERBS), "ADV")
    elif condition in ("pp_match", "pp_mismatch"):
        anum = snum if condition == "pp_match" else other
        add(rng.choice(PREPOSITIONS), "PREP")
        add_attractor(anum)
    elif condition == "pp2_mismatch":
        anum = other
        for _ in range(2):
            add(rng.choice(PREPOSITIONS), "PREP")
            add_attractor(anum)
    elif condition == "rel_mismatch":
        anum = other
        add("that", "REL")
        add_attractor(anum)
        add(rng.choice(RELATIVE_VERBS), "RELV")
    else:
        raise ValueError(f"unknown condition {condition!r}")

    correct, wrong = (verb[0], verb[1]) if snum == "sg" else (verb[1], verb[0])
    return Example(
        words=words, roles=roles, lexemes=lexemes, verb_idx=verb_idx,
        correct=correct, wrong=wrong, subject_number=snum, attractor_number=anum,
        condition=condition, subject=words[1], distance=len(words) - 2,
    )


def generate_corpus(n: int, seed: int, lex: Lexicon, split: str = "train") -> list[str]:
    rng = random.Random(seed)
    return [sample(rng, lex, split).text for _ in range(n)]


def build_pairs(
    n_per_condition: int,
    seed: int,
    lex: Lexicon,
    split: str = "test",
    conditions: list[str] | None = None,
) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []
    for c in conditions or CONDITIONS:
        out += [sample(rng, lex, split, condition=c) for _ in range(n_per_condition)]
    return out
