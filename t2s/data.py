"""Generate the SYNTHETIC banking database used by the agent and the benchmark.

Every row is produced by a seeded pseudo-random generator: no real customer,
bank or employer data is involved. Running the script twice with the same seed
gives a byte-identical database, which keeps the benchmark reproducible.

    python -m t2s.data --db data/bank.db --seed 42
"""
from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "bank.db"

SCHEMA = """
CREATE TABLE branches (
    branch_id    INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    city         TEXT NOT NULL,
    region       TEXT NOT NULL,
    opened_date  TEXT NOT NULL            -- ISO date
);
CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    first_name   TEXT NOT NULL,
    last_name    TEXT NOT NULL,
    gender       TEXT NOT NULL,           -- 'F' or 'M'
    birth_date   TEXT NOT NULL,
    city         TEXT NOT NULL,
    segment      TEXT NOT NULL,           -- retail | premium | business
    branch_id    INTEGER NOT NULL REFERENCES branches(branch_id),
    joined_date  TEXT NOT NULL
);
CREATE TABLE accounts (
    account_id   INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    account_type TEXT NOT NULL,           -- checking | savings | business
    currency     TEXT NOT NULL,           -- TND | EUR | USD
    balance      REAL NOT NULL,
    opened_date  TEXT NOT NULL,
    status       TEXT NOT NULL            -- active | closed | frozen
);
CREATE TABLE transactions (
    transaction_id    INTEGER PRIMARY KEY,
    account_id        INTEGER NOT NULL REFERENCES accounts(account_id),
    txn_date          TEXT NOT NULL,
    amount            REAL NOT NULL,      -- always positive
    txn_type          TEXT NOT NULL,      -- deposit | withdrawal | transfer | card_payment | fee
    channel           TEXT NOT NULL,      -- branch | atm | online | mobile
    merchant_category TEXT                -- only for card payments
);
CREATE TABLE loans (
    loan_id       INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    loan_type     TEXT NOT NULL,          -- mortgage | auto | personal | business
    principal     REAL NOT NULL,
    interest_rate REAL NOT NULL,          -- percent per year
    start_date    TEXT NOT NULL,
    term_months   INTEGER NOT NULL,
    status        TEXT NOT NULL           -- current | late | default | repaid
);
CREATE INDEX idx_accounts_customer ON accounts(customer_id);
CREATE INDEX idx_txn_account ON transactions(account_id);
CREATE INDEX idx_loans_customer ON loans(customer_id);
"""

BRANCHES = [
    ("Tunis Lac", "Tunis", "Greater Tunis"), ("Tunis Centre", "Tunis", "Greater Tunis"),
    ("Ariana Ennasr", "Ariana", "Greater Tunis"), ("Ben Arous Medina", "Ben Arous", "Greater Tunis"),
    ("Bizerte Port", "Bizerte", "North"), ("Nabeul Centre", "Nabeul", "North"),
    ("Sousse Corniche", "Sousse", "Sahel"), ("Monastir Marina", "Monastir", "Sahel"),
    ("Sfax Ville", "Sfax", "Centre"), ("Kairouan Medina", "Kairouan", "Centre"),
    ("Gabès Sud", "Gabès", "South"), ("Sfax Thyna", "Sfax", "Centre"),
]
FIRST_F = ["Amira", "Sarra", "Ines", "Mariem", "Yasmine", "Nour", "Rania", "Salma", "Emna", "Hiba", "Lina", "Asma"]
FIRST_M = ["Mohamed", "Ahmed", "Youssef", "Ali", "Omar", "Karim", "Sami", "Walid", "Hamza", "Bilel", "Anis", "Mehdi"]
LAST = ["Ben Salah", "Trabelsi", "Gharbi", "Jaziri", "Mansouri", "Bouaziz", "Hammami", "Chaabane",
        "Khelifi", "Ayari", "Dridi", "Sassi", "Riahi", "Mejri", "Baccouche", "Zouari"]
MERCHANTS = ["grocery", "restaurants", "fuel", "travel", "electronics", "health", "utilities"]

N_CUSTOMERS, N_ACCOUNTS, N_TRANSACTIONS, N_LOANS = 600, 900, 3200, 350
START, END = date(2023, 1, 1), date(2025, 12, 31)


def _rand_date(rng: random.Random, start: date, end: date) -> str:
    return (start + timedelta(days=rng.randint(0, (end - start).days))).isoformat()


def build(db_path: Path = DEFAULT_DB, seed: int = 42) -> Path:
    """(Re)create the database at ``db_path`` and return the path."""
    rng = random.Random(seed)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    branches = []
    for i, (name, city, region) in enumerate(BRANCHES, start=1):
        branches.append((i, name, city, region, _rand_date(rng, date(2005, 1, 1), date(2020, 12, 31))))
    con.executemany("INSERT INTO branches VALUES (?,?,?,?,?)", branches)

    customers = []
    for cid in range(1, N_CUSTOMERS + 1):
        branch = rng.choice(branches)
        gender = rng.choice("FM")
        first = rng.choice(FIRST_F if gender == "F" else FIRST_M)
        # most customers live in the city of their branch
        city = branch[2] if rng.random() < 0.8 else rng.choice(BRANCHES)[1]
        segment = rng.choices(["retail", "premium", "business"], weights=[70, 18, 12])[0]
        customers.append((cid, first, rng.choice(LAST), gender,
                          _rand_date(rng, date(1955, 1, 1), date(2004, 12, 31)), city, segment,
                          branch[0], _rand_date(rng, date(2015, 1, 1), END)))
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?)", customers)

    accounts = []
    for aid in range(1, N_ACCOUNTS + 1):
        cust = customers[aid - 1] if aid <= N_CUSTOMERS else rng.choice(customers)  # everyone has >= 1
        a_type = "business" if cust[6] == "business" and rng.random() < 0.6 else \
            rng.choices(["checking", "savings"], weights=[60, 40])[0]
        currency = rng.choices(["TND", "EUR", "USD"], weights=[85, 10, 5])[0]
        scale = {"retail": 6_000, "premium": 60_000, "business": 120_000}[cust[6]]
        balance = round(rng.lognormvariate(0, 0.9) * scale, 2)
        status = rng.choices(["active", "closed", "frozen"], weights=[88, 9, 3])[0]
        accounts.append((aid, cust[0], a_type, currency, balance, _rand_date(rng, START, END), status))
    con.executemany("INSERT INTO accounts VALUES (?,?,?,?,?,?,?)", accounts)

    txns = []
    for tid in range(1, N_TRANSACTIONS + 1):
        acc = rng.choice(accounts)
        t_type = rng.choices(["deposit", "withdrawal", "transfer", "card_payment", "fee"],
                             weights=[22, 20, 18, 35, 5])[0]
        channel = {"card_payment": rng.choice(["online", "mobile", "atm"]),
                   "fee": "online"}.get(t_type) or rng.choice(["branch", "atm", "online", "mobile"])
        amount = round(rng.uniform(2, 25) if t_type == "fee" else rng.lognormvariate(5, 1.1), 2)
        merchant = rng.choice(MERCHANTS) if t_type == "card_payment" else None
        txns.append((tid, acc[0], _rand_date(rng, START, END), amount, t_type, channel, merchant))
    con.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?)", txns)

    loans = []
    for lid in range(1, N_LOANS + 1):
        cust = rng.choice(customers)
        l_type = rng.choices(["mortgage", "auto", "personal", "business"], weights=[30, 25, 35, 10])[0]
        principal = round({"mortgage": rng.uniform(80_000, 450_000), "auto": rng.uniform(15_000, 90_000),
                           "personal": rng.uniform(2_000, 40_000), "business": rng.uniform(50_000, 600_000)}[l_type], 2)
        rate = round(rng.uniform(6.5, 13.5), 2)
        term = rng.choice({"mortgage": [180, 240, 300], "auto": [48, 60, 84],
                           "personal": [12, 24, 36, 60], "business": [36, 60, 84]}[l_type])
        status = rng.choices(["current", "late", "default", "repaid"], weights=[70, 12, 5, 13])[0]
        loans.append((lid, cust[0], l_type, principal, rate, _rand_date(rng, START, END), term, status))
    con.executemany("INSERT INTO loans VALUES (?,?,?,?,?,?,?,?)", loans)

    con.commit()
    con.close()
    return db_path


def ensure_db(db_path: Path = DEFAULT_DB) -> Path:
    """Build the database on first use."""
    db_path = Path(db_path)
    if not db_path.exists():
        build(db_path)
    return db_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    path = build(args.db, args.seed)
    con = sqlite3.connect(path)
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("branches", "customers", "accounts", "transactions", "loans")}
    con.close()
    print(f"synthetic database written to {path}: {counts} (total {sum(counts.values())} rows)")


if __name__ == "__main__":
    main()
