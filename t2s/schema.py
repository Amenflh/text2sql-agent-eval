"""Schema introspection and schema linking.

Schema linking narrows the database down to the tables, columns and cell values a
question is about. The result is used twice: the offline generator builds its
query from it, and LLM backends receive only the pruned DDL in their prompt.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- text


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def tokenize(text: str) -> tuple[list[str], list[str]]:
    """Return (original tokens, normalised tokens) aligned index by index."""
    orig = re.findall(r"\d+(?:\.\d+)?|\w+", text.replace("'", " ").replace("’", " "))
    return orig, [strip_accents(t.lower()) for t in orig]


def norm(text: str) -> str:
    return " ".join(tokenize(text)[1])


# ------------------------------------------------------------------------ lexicon
# Normalised (lower-case, accent-free) words, English and French.

TABLE_WORDS = {
    "customers": ["customer", "customers", "client", "clients", "clientele", "who"],
    "accounts": ["account", "accounts", "compte", "comptes"],
    "transactions": ["transaction", "transactions", "operation", "operations", "payment", "payments",
                     "paiement", "paiements"],
    "loans": ["loan", "loans", "credit", "credits", "pret", "prets", "borrower", "borrowers",
              "emprunteur", "emprunteurs"],
    "branches": ["branch", "branches", "agence", "agences"],
}

# measure word -> (column name the generator uses, table it belongs to)
MEASURE_WORDS = {
    "balance": ("balance", "accounts"), "balances": ("balance", "accounts"),
    "solde": ("balance", "accounts"), "soldes": ("balance", "accounts"),
    "deposits held": ("balance", "accounts"),
    "amount": ("amount", None), "amounts": ("amount", None), "montant": ("amount", None),
    "montants": ("amount", None), "volume": ("amount", None), "value": ("amount", None),
    "principal": ("principal", "loans"),
    "interest rate": ("interest_rate", "loans"), "rate": ("interest_rate", "loans"),
    "taux": ("interest_rate", "loans"), "taux d interet": ("interest_rate", "loans"),
    "term": ("term_months", "loans"), "duree": ("term_months", "loans"),
}

# dimension word -> logical dimension
DIMENSION_WORDS = {
    "city": "city", "cities": "city", "ville": "city", "villes": "city",
    "region": "region", "regions": "region",
    "branch": "branch", "branches": "branch", "agence": "branch", "agences": "branch",
    "segment": "segment", "segments": "segment",
    "gender": "gender", "sexe": "gender", "genre": "gender",
    "type": "type", "types": "type",
    "channel": "channel", "channels": "channel", "canal": "channel", "canaux": "channel",
    "currency": "currency", "currencies": "currency", "devise": "currency", "devises": "currency",
    "status": "status", "statut": "status", "statuts": "status",
    "category": "category", "categories": "category", "categorie": "category",
    "month": "month", "months": "month", "mois": "month",
    "year": "year", "years": "year", "annee": "year", "annees": "year", "an": "year",
    "customer": "customer", "client": "customer",
}

# French / English synonyms that translate to a stored categorical value
VALUE_SYNONYMS = {
    ("accounts", "account_type"): {"epargne": "savings", "courant": "checking", "courants": "checking",
                                   "cheque": "checking", "professionnel": "business",
                                   "business account": "business", "business accounts": "business",
                                   "professionnels": "business"},
    ("accounts", "status"): {"actif": "active", "actifs": "active", "ferme": "closed", "fermes": "closed",
                             "clos": "closed", "cloture": "closed", "clotures": "closed", "gele": "frozen",
                             "geles": "frozen", "bloque": "frozen", "bloques": "frozen"},
    ("accounts", "currency"): {"dinar": "TND", "dinars": "TND", "euro": "EUR", "euros": "EUR",
                               "dollar": "USD", "dollars": "USD"},
    ("loans", "loan_type"): {"immobilier": "mortgage", "immobiliers": "mortgage", "home": "mortgage",
                             "voiture": "auto", "voitures": "auto",
                             "automobile": "auto", "automobiles": "auto", "car": "auto", "personnel": "personal",
                             "personnels": "personal", "consommation": "personal",
                             "business loan": "business", "business loans": "business",
                             "credit professionnel": "business", "credits professionnels": "business"},
    ("loans", "status"): {"en retard": "late", "overdue": "late", "defaut": "default",
                          "rembourse": "repaid", "rembourses": "repaid", "en cours": "current",
                          "current loans": "current"},
    ("transactions", "txn_type"): {"depot": "deposit", "depots": "deposit", "deposits": "deposit",
                                   "retrait": "withdrawal", "retraits": "withdrawal",
                                   "withdrawals": "withdrawal", "virement": "transfer",
                                   "virements": "transfer", "transfers": "transfer",
                                   "card payment": "card_payment", "card payments": "card_payment",
                                   "paiement par carte": "card_payment", "paiements par carte": "card_payment",
                                   "frais": "fee", "fees": "fee"},
    ("transactions", "channel"): {"en ligne": "online", "internet": "online", "dab": "atm", "gab": "atm",
                                  "guichet": "branch", "at the branch": "branch", "in branch": "branch",
                                  "en agence": "branch", "mobile app": "mobile", "application mobile": "mobile"},
    ("transactions", "merchant_category"): {"alimentation": "grocery", "epicerie": "grocery",
                                            "carburant": "fuel", "voyage": "travel", "voyages": "travel",
                                            "sante": "health", "electronique": "electronics",
                                            "factures": "utilities", "restauration": "restaurants"},
    ("customers", "segment"): {"particuliers": "retail", "particulier": "retail",
                               "entreprises": "business", "professionnels": "business",
                               "business customers": "business", "business clients": "business",
                               "business segment": "business"},
    ("customers", "gender"): {"women": "F", "woman": "F", "female": "F", "femmes": "F", "femme": "F",
                              "men": "M", "man": "M", "male": "M", "hommes": "M", "homme": "M"},
    ("branches", "region"): {"nord": "North", "sud": "South", "grand tunis": "Greater Tunis"},
}

# columns whose distinct values are indexed for value linking
VALUE_COLUMNS = [("customers", "city"), ("customers", "segment"), ("branches", "region"),
                 ("accounts", "account_type"), ("accounts", "currency"), ("accounts", "status"),
                 ("loans", "loan_type"), ("loans", "status"), ("transactions", "txn_type"),
                 ("transactions", "channel"), ("transactions", "merchant_category")]
# raw values too ambiguous to link without a synonym ("branch" is also a table word)
UNLINKED_VALUES = {"branch", "current", "business"}


# ------------------------------------------------------------------------- schema


@dataclass
class Schema:
    tables: dict[str, list[str]]                      # table -> columns
    fks: list[tuple[str, str, str, str]]              # (child, child_col, parent, parent_col)
    ddl: dict[str, str]

    @classmethod
    def from_db(cls, con: sqlite3.Connection) -> Schema:
        tables, fks, ddl = {}, [], {}
        for name, sql in con.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"):
            tables[name] = [r[1] for r in con.execute(f"PRAGMA table_info({name})")]
            ddl[name] = sql
            for r in con.execute(f"PRAGMA foreign_key_list({name})"):
                fks.append((name, r[3], r[2], r[4]))
        return cls(tables, fks, ddl)

    def pk(self, table: str) -> str:
        return self.tables[table][0]

    def tables_with(self, column: str) -> list[str]:
        return [t for t, cols in self.tables.items() if column in cols]

    def parent_of(self, table: str) -> tuple[str, str, str] | None:
        """(fk column, parent table, parent column) — the schema is a tree."""
        for child, col, parent, pcol in self.fks:
            if child == table:
                return col, parent, pcol
        return None

    def path(self, src: str, dst: str) -> list[tuple[str, str, str]]:
        """Steps from ``src`` to ``dst`` along FKs: (direction 'up'|'down', from, to)."""
        def ancestors(t):
            chain = [t]
            while (p := self.parent_of(chain[-1])):
                chain.append(p[1])
            return chain
        a, b = ancestors(src), ancestors(dst)
        common = next(t for t in a if t in b)
        up = a[: a.index(common) + 1]
        down = list(reversed(b[: b.index(common)]))
        steps = [("up", up[i], up[i + 1]) for i in range(len(up) - 1)]
        prev = common
        for t in down:
            steps.append(("down", prev, t))
            prev = t
        return steps

    def join_condition(self, a: str, b: str) -> str:
        for child, col, parent, pcol in self.fks:
            if {child, parent} == {a, b}:
                return f"{child}.{col} = {parent}.{pcol}"
        raise ValueError(f"no foreign key between {a} and {b}")

    def prompt_ddl(self, tables: list[str] | None = None) -> str:
        return "\n".join(self.ddl[t] + ";" for t in (tables or list(self.tables)))


@dataclass
class ValueMatch:
    table: str
    column: str
    value: str        # canonical value stored in the database
    surface: str      # text as typed by the user
    start: int        # token span in the question
    end: int
    via_synonym: bool


@dataclass
class Link:
    """Output of schema linking for one question."""
    orig: list[str]
    toks: list[str]
    table_hits: list[tuple[int, str]] = field(default_factory=list)       # (position, table)
    measure_hits: list[tuple[int, str, str | None]] = field(default_factory=list)
    dimension_hits: list[tuple[int, str]] = field(default_factory=list)
    values: list[ValueMatch] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.toks)

    @property
    def tables(self) -> list[str]:
        seen = []
        for t in [t for _, t in self.table_hits] + [v.table for v in self.values] + \
                 [m[2] for m in self.measure_hits if m[2]]:
            if t not in seen:
                seen.append(t)
        return seen


def _phrase_hits(toks: list[str], phrases, used: set[int]):
    """Longest-first, non-overlapping phrase matching on normalised tokens."""
    hits = []
    for phrase in sorted(phrases, key=lambda p: -len(p.split())):
        words = phrase.split()
        n = len(words)
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == words and not used & set(range(i, i + n)):
                hits.append((i, i + n, phrase))
                used.update(range(i, i + n))
    return hits


class Linker:
    def __init__(self, con: sqlite3.Connection, schema: Schema):
        self.schema = schema
        self.value_index: dict[str, list[tuple[str, str, str, bool]]] = {}
        for table, col in VALUE_COLUMNS:
            for (val,) in con.execute(f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL"):
                key = norm(str(val)).replace("_", " ")
                if key not in UNLINKED_VALUES:
                    self.value_index.setdefault(key, []).append((table, col, str(val), False))
        for (table, col), mapping in VALUE_SYNONYMS.items():
            for word, val in mapping.items():
                self.value_index.setdefault(word, []).append((table, col, val, True))

    def link(self, question: str) -> Link:
        orig, toks = tokenize(question)
        lk = Link(orig, toks)
        used: set[int] = set()
        # values first (multi-word synonyms such as "paiement par carte" win over table words)
        value_hits = _phrase_hits(toks, self.value_index, used)
        for start, end, phrase in value_hits:
            cands = self.value_index[phrase]
            lk.values.append(self._pick(cands, toks, start, end, orig))
        for start, _, phrase in _phrase_hits(toks, MEASURE_WORDS, used):
            col, table = MEASURE_WORDS[phrase]
            lk.measure_hits.append((start, col, table))
        free = set()
        for t, words in TABLE_WORDS.items():
            for start, _, _ in _phrase_hits(toks, words, set()):
                lk.table_hits.append((start, t))
                free.add(start)
        for start, _, phrase in _phrase_hits(toks, DIMENSION_WORDS, set(used)):
            lk.dimension_hits.append((start, DIMENSION_WORDS[phrase]))
        lk.table_hits.sort()
        lk.values.sort(key=lambda v: v.start)
        return lk

    def _pick(self, cands, toks, start, end, orig) -> ValueMatch:
        """Disambiguate a value present in several columns using nearby table words."""
        best = cands[0]
        if len(cands) > 1:
            window = toks[max(0, start - 3): end + 3]
            for c in cands:
                if any(w in window for w in TABLE_WORDS[c[0]]):
                    best = c
                    break
        table, col, val, syn = best
        surface = val if syn else " ".join(orig[start:end])
        return ValueMatch(table, col, val, surface, start, end, syn)
