"""Fictional facts: entities, the people tied to them, and near-miss distractors.

Every name a question can be answered with is invented and checked against the whole WikiText corpus
(no surname, entity stem or town appears there as a word), so the answer can only come from the
document, and the closed-book control checks that no model knows it.

An item is one fictional entity of a type (an observatory, a bridge, an expedition, an overture), the
person tied to it by the type's relation (founded, designed, led, composed), and:

- three near-miss distractors, each an entity with the same relation and a different person:
  `same_stem` (same name, another type of the same relation: the Vellmoor Library beside the Vellmoor
  Observatory), `same_prefix` (same type, the name's ending changed: the Vellmark Observatory) and
  `same_suffix` (same type, the name's beginning changed: the Ostmoor Observatory);
- a second-hop fact (the person's birthplace) and two decoy birthplaces of other invented people, for
  the two-hop question ("In which town was the founder of the Vellmoor Observatory born?").
"""

from __future__ import annotations

import numpy as np

# relation -> question templates and the entity types that take it. Each type has three intro
# sentences (the key and its two same-type distractors each use a different one) and a fact sentence.
RELATIONS = {
    "founded": {
        "question": "Who founded the {name}?",
        "twohop": "In which town was the founder of the {name} born?",
        "types": {
            "Observatory": {
                "intro": ["The {name} stands on a low ridge above the river and opens to visitors on clear evenings.",
                          "The {name} keeps a long record of weather readings alongside its star charts.",
                          "The {name} houses a brass refracting telescope in a domed tower of grey stone."],
                "fact": "It was founded in {year} by the astronomer {person}."},
            "Conservatory": {
                "intro": ["The {name} trains about two hundred students a year in performance and composition.",
                          "The {name} gives a free concert in its courtyard on the first evening of every month.",
                          "The {name} occupies a terrace of four townhouses joined by a covered walkway."],
                "fact": "It was founded in {year} by the pianist {person}."},
            "Library": {
                "intro": ["The {name} holds a large collection of maps, ledgers and early printed almanacs.",
                          "The {name} lends books by post to readers in the outlying farms.",
                          "The {name} has a long reading room lit by a row of tall arched windows."],
                "fact": "It was founded in {year} by the bookseller {person}."},
            "Museum": {
                "intro": ["The {name} occupies a former grain warehouse near the harbour.",
                          "The {name} shows clocks, compasses and other instruments made in the region.",
                          "The {name} is known for its rooms of carved ship figureheads."],
                "fact": "It was founded in {year} by the collector {person}."},
        },
    },
    "designed": {
        "question": "Who designed the {name}?",
        "twohop": "In which town was the designer of the {name} born?",
        "types": {
            "Lighthouse": {
                "intro": ["The {name} marks a shoal at the mouth of the bay and is still in use.",
                          "The {name} is a white tower of dressed granite on a rocky headland.",
                          "The {name} can be reached on foot at low tide along a shingle spit."],
                "fact": "It was designed by the engineer {person} and first lit in {year}."},
            "Bridge": {
                "intro": ["The {name} carries a single road across the gorge on three stone arches.",
                          "The {name} replaced a ferry crossing that had operated for two centuries.",
                          "The {name} is a narrow iron footbridge over the canal basin."],
                "fact": "It was designed by the engineer {person} and opened in {year}."},
            "Pavilion": {
                "intro": ["The {name} is a small glass building at the edge of the public gardens.",
                          "The {name} hosts a flower show every spring under its timber roof.",
                          "The {name} faces the bandstand across a lawn lined with lime trees."],
                "fact": "It was designed by the architect {person} and completed in {year}."},
            "Viaduct": {
                "intro": ["The {name} carries the railway across a wide valley on twenty-two brick piers.",
                          "The {name} curves gently over the river meadows at the edge of the town.",
                          "The {name} was strengthened with steel bands after a severe winter flood."],
                "fact": "It was designed by the engineer {person} and opened in {year}."},
        },
    },
    "led": {
        "question": "Who led the {name}?",
        "twohop": "In which town was the leader of the {name} born?",
        "types": {
            "Expedition": {
                "intro": ["The {name} spent two winters mapping the northern coast and its islands.",
                          "The {name} crossed the salt flats with a train of forty pack animals.",
                          "The {name} brought back the first measured drawings of the glacier."],
                "fact": "It was led by the surveyor {person}, and it returned in {year}."},
            "Survey": {
                "intro": ["The {name} produced the first reliable charts of the inland lakes.",
                          "The {name} recorded the depth of every well in the district.",
                          "The {name} set out boundary stones along the length of the old forest."],
                "fact": "It was led by the geologist {person} and completed in {year}."},
            "Mission": {
                "intro": ["The {name} brought supplies and medical staff to the mountain villages after the floods.",
                          "The {name} set up a field hospital in a disused school.",
                          "The {name} vaccinated several thousand people along the river valley."],
                "fact": "It was led by the physician {person} in {year}."},
        },
    },
    "composed": {
        "question": "Who composed the {name}?",
        "twohop": "In which town was the composer of the {name} born?",
        "types": {
            "Overture": {
                "intro": ["The {name} is a concert piece in a single movement, often played to open summer festivals.",
                          "The {name} quotes a fishermen's song in its slow middle section.",
                          "The {name} calls for an unusually large brass section."],
                "fact": "It was composed in {year} by {person}."},
            "Suite": {
                "intro": ["The {name} is a set of five short dances for string orchestra.",
                          "The {name} was written for a touring puppet theatre.",
                          "The {name} ends with a fast reel for solo fiddle and strings."],
                "fact": "It was composed in {year} by {person}."},
            "Sonata": {
                "intro": ["The {name} is a three-movement work for violin and piano.",
                          "The {name} is often paired in recitals with the composer's songs.",
                          "The {name} opens with a long unaccompanied melody for the cello."],
                "fact": "It was composed in {year} by {person}."},
        },
    },
}

HOP2 = "{person} was born in the town of {town}{tail}."
HOP2_Q = "In which town was {person} born?"
TAILS = [", and moved to the capital as a young adult", ", where the family kept a clock shop",
         ", on the edge of the marshes", ", a river port known for its boatyards", ", and was taught at home",
         ", a market town on the old coach road", ", and later returned there to live",
         ", in a house beside the parish church"]

FIRST = ["Odile", "Hallam", "Ines", "Tobiah", "Maren", "Corin", "Elspeth", "Anselm", "Liesel", "Rafe", "Thea",
         "Bram", "Ysolde", "Casimir", "Wren", "Lucan", "Ottilie", "Fenna", "Idris", "Margery", "Silas", "Agathe",
         "Emrys", "Livia", "Ansel", "Briony", "Cosmo", "Dagny", "Evander", "Hester", "Ivo", "Jessamy", "Leopold",
         "Mireille", "Nils", "Orla", "Piers", "Rosalind", "Soren", "Tamsin", "Ulric", "Verity", "Willa", "Yara",
         "Aurelie", "Benedikt", "Clemence", "Dorian", "Eleni", "Florian", "Greta", "Henrike", "Isidore",
         "Jolyon", "Katrin", "Lorcan", "Magda", "Niamh", "Osric", "Perpetua", "Quill", "Rosamund", "Sabine",
         "Tancred", "Ulla", "Viggo", "Winifred", "Zelie", "Alaric", "Bettina", "Cyprian", "Delphine"]
SUR_PRE = ["Ash", "Brin", "Cal", "Dun", "Fen", "Gar", "Hal", "Kes", "Lor", "Mar", "Nor", "Orm", "Pel", "Quil",
           "Ros", "Sel", "Tam", "Ven", "Wex", "Yel", "Zor", "Brack", "Cran", "Drey", "Gild", "Harl", "Merr",
           "Thorn", "Wick", "Elm", "Ferr", "Gast", "Hob", "Lark", "Mott", "Pask", "Rook", "Sall", "Tull", "Varn"]
SUR_SUF = ["wick", "combe", "thorpe", "ley", "more", "den", "holt", "worth", "field", "ridge", "ford", "stead",
           "well", "hurst", "gate", "cott", "shaw", "brook", "wood", "low", "wright", "aker", "ett", "ingham"]
STEM_PRE = ["Vell", "Ost", "Brae", "Carn", "Dray", "Esk", "Fal", "Grim", "Hask", "Isk", "Jor", "Kell", "Lund",
            "Mor", "Nesk", "Orl", "Pry", "Rath", "Skel", "Tarn", "Ull", "Vask", "Wend", "Yarr", "Quen", "Sorr",
            "Thal", "Brisk", "Corv", "Delv"]
STEM_SUF = ["moor", "mark", "dell", "wyn", "haven", "brook", "fall", "crest", "vale", "march", "fold", "reach",
            "stone", "water", "wold", "mere", "garth", "hollow"]
TOWN_PRE = ["Ambel", "Bresk", "Cawl", "Dorn", "Eller", "Frisk", "Glim", "Hollin", "Ivel", "Kern", "Lisk",
            "Marrow", "Nethel", "Ottar", "Pember", "Quarr", "Rill", "Stave", "Tresk", "Umber", "Wyst", "Yelver"]
TOWN_SUF = ["bury", "wich", "minster", "mouth", "port", "stow", "cester", "ham", "thwaite", "by", "ton", "leigh"]


class NamePool:
    """Invented names drawn without replacement from a seeded shuffle, skipping corpus words."""

    def __init__(self, vocab: frozenset, seed: int):
        self.rng = np.random.default_rng(seed)

        def combos(pre, suf):
            return [p + s for p in pre for s in suf if (p + s).lower() not in vocab]

        self.surnames = self._shuffled(combos(SUR_PRE, SUR_SUF))
        self.towns = self._shuffled(combos(TOWN_PRE, TOWN_SUF))
        self.first = self._shuffled(FIRST)        # first names may be real words; surnames decide the check
        self.valid_stems = frozenset(combos(STEM_PRE, STEM_SUF))
        self.stems = self._shuffled([(p, s) for p in STEM_PRE for s in STEM_SUF if p + s in self.valid_stems])
        self.used: set[str] = set()

    def _shuffled(self, xs: list) -> list:
        return [xs[i] for i in self.rng.permutation(len(xs))]

    def _take(self, pool: list[str]) -> str:
        while pool:
            x = pool.pop()
            if x.lower() not in self.used:
                self.used.add(x.lower())
                return x
        raise RuntimeError("name pool exhausted")

    def surname(self) -> str:
        return self._take(self.surnames)

    def town(self) -> str:
        return self._take(self.towns)

    def people(self, n: int) -> list[dict]:
        """n people with distinct first names (within the call) and surnames never used before."""
        firsts = [self.first[i] for i in self.rng.choice(len(self.first), size=n, replace=False)]
        out = []
        for f in firsts:
            last = self.surname()
            out.append({"first": f, "last": last, "name": f"{f} {last}"})
        return out

    def stem_family(self) -> tuple[str, str, str]:
        """A stem and two near misses: the same beginning with another ending, and the same ending with
        another beginning. All three are invented, unused and distinct."""
        def free(x):
            return x in self.valid_stems and x.lower() not in self.used

        while self.stems:
            p, s = self.stems.pop()
            key = p + s
            alt_a = [p + x for x in STEM_SUF if x != s and free(p + x)]
            alt_b = [x + s for x in STEM_PRE if x != p and free(x + s)]
            if not free(key) or not alt_a or not alt_b:
                continue
            a = alt_a[int(self.rng.integers(len(alt_a)))]
            b = alt_b[int(self.rng.integers(len(alt_b)))]
            self.used.update({key.lower(), a.lower(), b.lower()})
            return key, a, b
        raise RuntimeError("stem pool exhausted")
