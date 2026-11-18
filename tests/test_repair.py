import sqlite3

from t2s.db import connect_ro
from t2s.repair import fix_ambiguous_column, fix_empty_result, fix_unknown_column, fix_unknown_table
from t2s.schema import Schema


def _schema(db_path):
    return Schema.from_db(sqlite3.connect(db_path))


def test_unknown_column_uses_synonyms(db_path):
    sql, how = fix_unknown_column("SELECT SUM(amount) FROM loans", "no such column: amount", _schema(db_path), "")
    assert sql == "SELECT SUM(principal) FROM loans" and "principal" in how


def test_unknown_column_fuzzy(db_path):
    sql, _ = fix_unknown_column("SELECT AVG(intrest_rate) FROM loans", "no such column: intrest_rate",
                                _schema(db_path), "")
    assert "interest_rate" in sql


def test_ambiguous_column_prefers_table_named_in_question(db_path):
    sql = ("SELECT city, COUNT(*) FROM customers JOIN branches ON customers.branch_id = branches.branch_id "
           "GROUP BY city")
    fixed, _ = fix_ambiguous_column(sql, "ambiguous column name: city", _schema(db_path), "customers by branch city")
    assert fixed.startswith("SELECT branches.city") and fixed.endswith("GROUP BY branches.city")
    fixed, _ = fix_ambiguous_column(sql, "ambiguous column name: city", _schema(db_path), "clients par ville")
    assert fixed.startswith("SELECT customers.city")


def test_unknown_table(db_path):
    fixed, _ = fix_unknown_table("SELECT COUNT(*) FROM clients", "no such table: clients", _schema(db_path), "")
    assert fixed == "SELECT COUNT(*) FROM customers"


def test_value_linking_case_and_accents(db_path):
    con = connect_ro(db_path)
    fixed, how = fix_empty_result("SELECT COUNT(*) FROM customers WHERE city = 'gabes'", con, _schema(db_path))
    assert "'Gabès'" in fixed and "value linking" in how
    assert fix_empty_result("SELECT COUNT(*) FROM customers WHERE city = 'Paris'", con, _schema(db_path)) is None


def test_year_compared_as_integer(db_path):
    fixed, _ = fix_empty_result("SELECT COUNT(*) FROM loans WHERE strftime('%Y', start_date) = 2024",
                                connect_ro(db_path), _schema(db_path))
    assert fixed.endswith("= '2024'")
