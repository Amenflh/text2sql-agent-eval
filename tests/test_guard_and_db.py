import sqlite3

import pytest

from t2s.db import connect_ro, execute, lock
from t2s.guard import validate


@pytest.mark.parametrize("sql", [
    "DELETE FROM loans",
    "DROP TABLE customers",
    "UPDATE accounts SET balance = 0",
    "INSERT INTO branches DEFAULT VALUES",
    "SELECT * FROM customers; DROP TABLE accounts",
    "PRAGMA table_info(customers)",
    "ATTACH DATABASE 'x.db' AS x",
    "WITH x AS (SELECT 1) DELETE FROM loans",
    "",
])
def test_guard_blocks(sql):
    assert not validate(sql).ok


def test_guard_allows_select_and_adds_limit():
    res = validate("SELECT name FROM branches")
    assert res.ok and res.sql.endswith("LIMIT 200")
    assert validate("SELECT name FROM branches LIMIT 5").sql == "SELECT name FROM branches LIMIT 5"


def test_guard_ignores_keywords_inside_strings_and_comments():
    assert validate("SELECT * FROM customers WHERE last_name = 'Drop; Delete'").ok
    assert validate("SELECT 1 -- update later").ok


def test_authorizer_blocks_writes_even_without_guard(db_path):
    con = lock(connect_ro(db_path))
    for sql in ("DELETE FROM loans", "PRAGMA table_info(loans)", "CREATE TABLE t(x)"):
        assert execute(con, sql).error
    assert execute(con, "SELECT COUNT(*) FROM loans").rows == [(350,)]


def test_connection_is_read_only_at_file_level(db_path):
    con = connect_ro(db_path)          # no authorizer: the file mode alone must refuse writes
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM loans")


def test_time_budget(db_path):
    con = lock(connect_ro(db_path))
    slow = ("WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r) SELECT COUNT(*) FROM r")
    assert "time budget" in execute(con, slow, timeout_s=0.2).error
