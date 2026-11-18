"""Deterministic, offline SQL generator (no LLM, no network).

A small semantic parser over the schema-linking output: it detects the
aggregation, the measure, group-by dimensions, filters, thresholds, dates,
top-N and existence constraints, then renders SQL along the foreign-key tree.

It writes a *first draft* the way a hurried human or an LLM often does: cell
values are copied as the user typed them, columns are left unqualified and the
generic word "amount" is mapped to a column called ``amount``. Those drafts
fail in realistic ways (empty results, ambiguous or unknown columns) and are
fixed by the agent's correction loop, not by this module. ``repair`` returns
None: the offline backend relies only on the deterministic repair tools.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from ..schema import TABLE_WORDS, Link, Schema, tokenize

COUNT_W = ["how many", "number of", "nombre de", "nombre d", "combien", "count"]
AVG_W = ["average", "mean", "avg", "moyenne", "moyen", "moyens"]
SUM_W = ["total", "sum", "somme", "cumul", "overall"]
MAX_W = ["highest", "largest", "biggest", "maximum", "max", "most", "plus eleve", "plus elevee",
         "plus eleves", "plus elevees", "plus grand", "plus grande", "plus grands", "plus gros",
         "plus importants", "plus important", "le plus de", "la plus de"]
MIN_W = ["lowest", "smallest", "minimum", "min", "least", "fewest", "plus faible", "plus faibles",
         "plus petit", "plus petite", "plus bas", "le moins de"]
WRITE_W = {"delete": "delete", "remove": "delete", "wipe": "delete", "supprime": "delete",
           "supprimer": "delete", "supprimez": "delete", "efface": "delete", "effacer": "delete",
           "drop": "drop", "update": "update", "modifie": "update", "modifier": "update",
           "modifiez": "update", "set": "update", "mets": "update", "insert": "insert",
           "ajoute": "insert", "truncate": "delete", "vide": "delete", "vider": "delete"}
GROUP_MARKERS = {"by", "per", "par", "each", "chaque", "selon", "across"}
SUBJECT_MARKERS = {"which", "what", "quel", "quelle", "quels", "quelles"}
LIST_MARKERS = {"list", "show", "display", "liste", "listez", "affiche", "afficher", "donne", "which",
                "quels", "quelles", "who", "qui"}
WITHOUT_W = {"without", "sans", "no", "aucun", "aucune", "never", "jamais", "pas"}
WITH_W = {"with", "have", "has", "having", "avec", "ayant", "ont", "possedant", "holding", "hold"}
COMPARATORS = [("greater than", ">"), ("more than", ">"), ("at least", ">="), ("at most", "<="),
               ("exceeding", ">"), ("above", ">"), ("over", ">"), ("less than", "<"), ("under", "<"),
               ("below", "<"), ("superieur a", ">"), ("superieure a", ">"), ("superieurs a", ">"),
               ("superieures a", ">"), ("au dessus de", ">"), ("depassant", ">"),
               ("depassent", ">"), ("depasse", ">"), ("exceeds", ">"), ("exceed", ">"), ("plus de", ">"),
               ("au moins", ">="), ("moins de", "<"), ("inferieur a", "<"), ("inferieure a", "<"),
               ("inferieurs a", "<"), ("en dessous de", "<")]
MONTHS = {m: i for i, names in enumerate([
    ("january", "janvier"), ("february", "fevrier"), ("march", "mars"), ("april", "avril"),
    ("may", "mai"), ("june", "juin"), ("july", "juillet"), ("august", "aout"),
    ("september", "septembre"), ("october", "octobre"), ("november", "novembre"),
    ("december", "decembre")], start=1) for m in names}
DATE_COL = {"transactions": "txn_date", "loans": "start_date", "accounts": "opened_date",
            "customers": "joined_date", "branches": "opened_date"}
DEFAULT_MEASURE = {"accounts": "balance", "transactions": "amount", "loans": "principal"}
DISPLAY = {"customers": ["customer_id", "first_name", "last_name", "city"],
           "accounts": ["account_id", "customer_id", "account_type", "balance"],
           "transactions": ["transaction_id", "account_id", "txn_date", "amount"],
           "loans": ["loan_id", "customer_id", "loan_type", "principal"],
           "branches": ["branch_id", "name", "city"]}
TYPE_COL = {"accounts": "account_type", "loans": "loan_type", "transactions": "txn_type"}
BRANCH_CITY = ["branch city", "city of the branch", "ville de l agence", "ville d agence",
               "ville des agences", "branch cities"]


@dataclass
class Group:
    select: str
    group: str
    table: str
    alias: str


@dataclass
class Spec:
    main: str
    agg: str = "list"                     # list | count | sum | avg | max | min
    measure: str | None = None
    groups: list[Group] = field(default_factory=list)
    filters: list[tuple[str, str]] = field(default_factory=list)    # (table, condition)
    exists: list[tuple[str, bool]] = field(default_factory=list)    # (child table, negated)
    having: tuple[str, str, int] | None = None                      # (child, op, n)
    order: str | None = None              # 'DESC' | 'ASC'
    limit: int | None = None
    subject: str | None = None            # table whose rows are listed, if not ``main``


def _has(text: str, words) -> str | None:
    for w in words:
        if f" {w} " in text:
            return w
    return None


def _numbers(toks: list[str]) -> list[tuple[int, float]]:
    """Numbers with their token position; joins '10 000' into 10000."""
    out, i = [], 0
    while i < len(toks):
        if re.fullmatch(r"\d+(\.\d+)?", toks[i]):
            start, val = i, toks[i]
            while i + 1 < len(toks) and re.fullmatch(r"\d{3}", toks[i + 1]) and "." not in val:
                val += toks[i + 1]
                i += 1
            out.append((start, float(val)))
        i += 1
    return out


class OfflineBackend:
    name = "offline"

    def __init__(self, schema: Schema):
        self.schema = schema

    # ------------------------------------------------------------------ public
    def generate(self, question: str, link: Link, lang: str) -> str:
        injected = ""
        if ";" in question:
            head, injected = (p.strip() for p in question.split(";", 1))
            link = self._truncate(link, len(tokenize(head)[0]))
        write = self._write_intent(link)
        if write:
            return write
        sql = self._render(self._parse(link))
        # A naive generator copies SQL embedded in the question; the guard must catch it.
        return f"{sql}; {injected}" if injected else sql

    def repair(self, question: str, sql: str, feedback: str, link: Link, lang: str) -> str | None:
        return None

    @staticmethod
    def _truncate(lk: Link, n: int) -> Link:
        """Keep only what was linked in the first ``n`` tokens."""
        return replace(lk, orig=lk.orig[:n], toks=lk.toks[:n],
                       table_hits=[h for h in lk.table_hits if h[0] < n],
                       measure_hits=[h for h in lk.measure_hits if h[0] < n],
                       dimension_hits=[h for h in lk.dimension_hits if h[0] < n],
                       values=[v for v in lk.values if v.end <= n])

    # ----------------------------------------------------------------- parsing
    def _write_intent(self, lk: Link) -> str | None:
        for i, tok in enumerate(lk.toks):
            kind = WRITE_W.get(tok)
            if kind and (kind != "update" or tok != "set" or i == 0):
                table = lk.table_hits[0][1] if lk.table_hits else "customers"
                if kind == "drop":
                    return f"DROP TABLE {table}"
                if kind == "update":
                    col = lk.measure_hits[0][1] if lk.measure_hits else self.schema.tables[table][1]
                    return f"UPDATE {table} SET {col} = 0"
                if kind == "insert":
                    return f"INSERT INTO {table} DEFAULT VALUES"
                return f"DELETE FROM {table}"
        return None

    def _parse(self, lk: Link) -> Spec:
        text = f" {lk.text} "
        toks = lk.toks
        nums = _numbers(toks)

        # ---- aggregation
        if _has(text, AVG_W):
            agg = "avg"
        elif _has(text, COUNT_W):
            agg = "count"
        elif _has(text, SUM_W):
            agg = "sum"
        else:
            agg = "list"
        superlative = "DESC" if _has(text, MAX_W) else "ASC" if _has(text, MIN_W) else None

        # ---- top-N
        limit = None
        m = re.search(r" (?:top|les|the|first|premiers|premieres) (\d+) ", text)
        if m:
            limit = int(m.group(1))
        elif superlative:
            limit = 1

        # ---- group-by dimensions
        branch_city = bool(_has(text, BRANCH_CITY))
        group_dims, subject_dim = [], None
        for pos, dim in lk.dimension_hits:
            before = toks[max(0, pos - 3):pos]
            if branch_city and dim == "branch" and pos + 1 < len(toks) and toks[pos + 1] in ("city", "cities"):
                continue
            if branch_city and dim == "branch" and pos > 0 and toks[pos - 1] in ("l", "d", "des"):
                continue
            if any(w in GROUP_MARKERS for w in before):
                group_dims.append((pos, dim))
            elif toks[pos - 1:pos] and toks[pos - 1] in SUBJECT_MARKERS | {"top"} or \
                    (pos >= 2 and re.fullmatch(r"\d+", toks[pos - 1]) and toks[pos - 2] in ("top", "les", "the")):
                subject_dim = (pos, dim)
        dim_positions = {p for p, _ in group_dims} | ({subject_dim[0]} if subject_dim else set())

        # ---- entity tables
        entity_hits = [(p, t) for p, t in lk.table_hits if p not in dim_positions]
        measure = None
        if lk.measure_hits:
            _, col, table = lk.measure_hits[0]
            if table is None:   # "amount" / "montant": the generator guesses the column name
                mentioned = {t for _, t in entity_hits}
                table = "loans" if "loans" in mentioned and "transactions" not in mentioned else "transactions"
            measure = (col, table)

        subject = entity_hits[0][1] if entity_hits else None
        if measure and (agg in ("sum", "avg") or superlative or limit):
            main = measure[1]
        elif subject:
            main = subject
        elif lk.values:
            main = lk.values[0].table
        else:
            main = "customers"
        if agg == "list" and superlative and not measure and not subject_dim and not group_dims:
            agg = "list"
        spec = Spec(main=main, agg=agg, limit=limit)
        if measure:
            spec.measure = measure[0]

        # ---- group-by: explicit, subject dimension ("which city has the most..."),
        #      or entity ("top 5 customers by total balance")
        for _, dim in group_dims:
            spec.groups.append(self._dimension(dim, main, lk, branch_city))
        if subject_dim:
            spec.groups.append(self._dimension(subject_dim[1], main, lk, branch_city))
        if subject and measure and subject != main and agg in ("sum", "avg", "count") \
                and not spec.groups and subject == "customers" and (limit or superlative):
            spec.groups.append(Group("first_name, last_name", "customer_id", "customers", ""))
        elif subject and measure and subject != main and agg == "list":
            spec.subject = subject

        # ---- aggregation refinements
        if spec.groups and agg == "list":
            spec.agg = "sum" if measure else "count"
        if not spec.groups and agg == "list" and measure and superlative and not _has(text, list(LIST_MARKERS)):
            spec.agg = "max" if superlative == "DESC" else "min"
            spec.limit = None
        if spec.groups and (superlative or limit):
            spec.order = superlative or "DESC"
        elif spec.agg == "list" and (superlative or limit):
            spec.order = superlative or "DESC"
            if not spec.measure:
                spec.measure = DEFAULT_MEASURE.get(main)
        if spec.agg in ("count", "sum", "avg", "max", "min") and not spec.groups:
            spec.limit = None

        # ---- value filters (a currency word right after a number is a unit: "1000 dinars")
        num_pos = {p for p, _ in nums}
        for v in lk.values:
            if v.column == "currency" and v.start - 1 in num_pos:
                continue
            table = v.table
            if v.column == "city" and (branch_city or main == "branches"):
                table = "branches"
            spec.filters.append((table, f"{v.column} = '{v.surface}'"))

        # ---- thresholds and HAVING ("more than 2 accounts")
        for phrase, op in COMPARATORS:
            for mm in re.finditer(f" {phrase} ", text):
                pos = len(text[:mm.start() + 1].split())
                after = pos + len(phrase.split())
                num = next((n for p, n in nums if p == after), None)
                if num is None:
                    continue
                nxt = toks[after + 1] if after + 1 < len(toks) else ""
                child = next((t for t, words in TABLE_WORDS.items() if nxt in words), None)
                if child and child != main:
                    spec.having = (child, op, int(num))
                    continue
                col = self._nearest_measure(lk, pos) or DEFAULT_MEASURE.get(main, "balance")
                tbl = measure[1] if measure and measure[0] == col else \
                    next(iter(self.schema.tables_with(col)), main)
                spec.filters.append((tbl, f"{col} {op} {num:g}"))

        # ---- dates
        for pos, n in nums:
            if not (2000 <= n <= 2100) or any(f" {c} {int(n)} " in text for c, _ in COMPARATORS):
                continue
            year = int(n)
            col = DATE_COL[main]
            prev = toks[pos - 1] if pos else ""
            if prev in MONTHS:
                spec.filters.append((main, f"strftime('%Y-%m', {col}) = '{year}-{MONTHS[prev]:02d}'"))
            elif prev in ("since", "depuis", "after", "apres"):
                op = ">=" if prev in ("since", "depuis") else ">"
                bound = f"{year}-01-01" if op == ">=" else f"{year}-12-31"
                spec.filters.append((main, f"{col} {op} '{bound}'"))
            elif prev in ("before", "avant"):
                spec.filters.append((main, f"{col} < '{year}-01-01'"))
            else:
                spec.filters.append((main, f"strftime('%Y', {col}) = '{year}'"))

        # ---- existence: "customers with a loan" / "clients sans crédit"
        filtered = {t for t, _ in spec.filters}
        for pos, t in entity_hits:
            if t == main or t in filtered or (spec.having and spec.having[0] == t):
                continue
            if any(s[0] != "down" for s in self.schema.path(main, t)):
                continue
            before = set(toks[max(0, pos - 4):pos])
            if before & WITHOUT_W:
                spec.exists.append((t, True))
            elif before & WITH_W:
                spec.exists.append((t, False))
        return spec

    def _nearest_measure(self, lk: Link, pos: int) -> str | None:
        if not lk.measure_hits:
            return None
        return min(lk.measure_hits, key=lambda m: abs(m[0] - pos))[1]

    def _dimension(self, dim: str, main: str, lk: Link, branch_city: bool) -> Group:
        if dim == "city":
            table = "branches" if branch_city or main == "branches" else "customers"
            return Group("city", "city", table, "city")
        if dim in ("region",):
            return Group("region", "region", "branches", "region")
        if dim == "branch":
            return Group("name", "name", "branches", "branch")
        if dim in ("segment", "gender"):
            return Group(dim, dim, "customers", dim)
        if dim == "channel":
            return Group("channel", "channel", "transactions", "channel")
        if dim == "category":
            return Group("merchant_category", "merchant_category", "transactions", "merchant_category")
        if dim == "currency":
            return Group("currency", "currency", "accounts", "currency")
        if dim in ("type", "status"):
            table = main
            for p, t in lk.table_hits:   # "account type", "type de crédit"
                if any(abs(p - q) <= 3 for q, d in lk.dimension_hits if d == dim) and t != main:
                    table = t
            if dim == "type" and table not in TYPE_COL:
                table = next((t for _, t in lk.table_hits if t in TYPE_COL), "accounts")
            if dim == "status" and table not in ("accounts", "loans"):
                table = "loans" if any(t == "loans" for _, t in lk.table_hits) else "accounts"
            col = TYPE_COL[table] if dim == "type" else "status"
            return Group(col, col, table, col)
        if dim in ("month", "year"):
            fmt = "%Y-%m" if dim == "month" else "%Y"
            expr = f"strftime('{fmt}', {DATE_COL[main]})"
            return Group(expr, expr, main, dim)
        return Group("first_name, last_name", "customer_id", "customers", "")

    # --------------------------------------------------------------- rendering
    def _render(self, s: Spec) -> str:
        sc = self.schema
        joins: list[str] = []
        joined = [s.main]
        where: list[str] = []

        def join_to(target: str):
            prev = s.main
            for _, a, b in sc.path(s.main, target):
                if b not in joined:
                    joins.append(f"JOIN {b} ON {sc.join_condition(a, b)}")
                    joined.append(b)
                prev = b
            return prev

        for g in s.groups:
            join_to(g.table)
        if s.subject:
            join_to(s.subject)
        if s.having:
            join_to(s.having[0])

        by_table: dict[str, list[str]] = {}
        for t, cond in s.filters:
            by_table.setdefault(t, []).append(cond)
        for t, conds in by_table.items():
            steps = sc.path(s.main, t)
            if all(d == "up" for d, _, _ in steps) or t in joined:
                join_to(t)
                where.extend(conds)
            else:
                for d, a, b in steps:
                    if d != "up":
                        break
                    if b not in joined:
                        joins.append(f"JOIN {b} ON {sc.join_condition(a, b)}")
                        joined.append(b)
                where.append(self._subquery(s.main, t, conds))
        for t, negated in s.exists:
            where.append(self._subquery(s.main, t, [], negated))

        pk = sc.pk(s.main)
        count = f"COUNT(DISTINCT {pk})" if len(joined) > 1 else "COUNT(*)"
        measure = s.measure or DEFAULT_MEASURE.get(s.main, pk)
        agg_expr = {"count": count, "sum": f"ROUND(SUM({measure}), 2)", "avg": f"ROUND(AVG({measure}), 2)",
                    "max": f"MAX({measure})", "min": f"MIN({measure})"}.get(s.agg)
        alias = {"count": "n", "sum": f"total_{measure}", "avg": f"avg_{measure}",
                 "max": f"max_{measure}", "min": f"min_{measure}"}.get(s.agg, "")

        if s.having:
            child, op, n = s.having
            inner = (f"SELECT {', '.join(DISPLAY[s.main][:3])} FROM {s.main} {' '.join(joins)}"
                     + (f" WHERE {' AND '.join(where)}" if where else "")
                     + f" GROUP BY {pk} HAVING COUNT(*) {op} {n}")
            return f"SELECT COUNT(*) AS n FROM ({inner})" if s.agg == "count" else inner

        if s.groups:
            sel = ", ".join(g.select + (f" AS {g.alias}" if g.alias and g.alias != g.select else "")
                            for g in s.groups)
            sql = f"SELECT {sel}, {agg_expr} AS {alias} FROM {s.main}"
        elif agg_expr:
            sql = f"SELECT {agg_expr} AS {alias} FROM {s.main}"
        else:
            cols = list(DISPLAY[s.subject] if s.subject else DISPLAY[s.main])
            if s.subject and measure not in cols:
                cols.append(measure)
            sql = f"SELECT {', '.join(cols)} FROM {s.main}"
        if joins:
            sql += " " + " ".join(joins)
        if where:
            sql += " WHERE " + " AND ".join(where)
        if s.groups:
            sql += " GROUP BY " + ", ".join(g.group for g in s.groups)
        if s.order:
            sql += f" ORDER BY {alias if s.groups else measure} {s.order}"
        if s.limit and (s.groups or s.agg == "list"):
            sql += f" LIMIT {s.limit}"
        return sql

    def _subquery(self, main: str, target: str, conds: list[str], negated: bool = False) -> str:
        """``main.pk IN (SELECT ...)`` for constraints on a child table."""
        sc = self.schema
        steps = sc.path(main, target)
        ups = [st for st in steps if st[0] == "up"]
        downs = [st for st in steps if st[0] == "down"]
        pivot = ups[-1][2] if ups else main
        first_child = downs[0][2]
        fk_col, _, parent_col = sc.parent_of(first_child)
        inner_joins = " ".join(f"JOIN {b} ON {sc.join_condition(a, b)}" for _, a, b in downs[1:])
        inner = f"SELECT {first_child}.{fk_col} FROM {first_child} {inner_joins}".strip()
        if conds:
            inner += " WHERE " + " AND ".join(conds)
        return f"{pivot}.{parent_col} {'NOT IN' if negated else 'IN'} ({inner})"
