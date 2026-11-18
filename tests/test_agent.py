from t2s.agent import Agent, detect_lang
from t2s.backends.llm import OpenAICompatBackend, extract_sql
from t2s.schema import tokenize


def test_simple_question(agent):
    s = agent.run("How many customers live in Sfax?")
    assert s.status == "answered" and s.attempts == 1
    assert s.result.rows[0][0] > 0 and s.answer.startswith("Answer:")


def test_value_linking_repair_loop(agent):
    s = agent.run("How many customers live in sousse?")
    assert s.attempts == 2 and "'Sousse'" in s.sql
    assert s.first_result.rows == [(0,)] and s.result.rows[0][0] > 0
    assert any(t["node"] == "repair" and t["applied"] for t in s.trace)


def test_unknown_column_repair_loop_in_french(agent):
    s = agent.run("Montant total des crédits immobiliers")
    assert s.lang == "fr" and "principal" in s.sql and s.first_result.error
    assert s.answer.startswith("Réponse")


def test_no_repair_when_max_attempts_is_one(db_path):
    s = Agent(db_path, backend="offline", max_attempts=1).run("How many customers live in sousse?")
    assert s.attempts == 1 and s.result.rows == [(0,)]


def test_write_requests_are_blocked(agent):
    for q in ("Delete all transactions", "Show customers; DROP TABLE accounts", "Supprime tous les clients"):
        s = agent.run(q)
        assert s.status == "blocked", q
        assert s.result is None


def test_language_detection():
    assert detect_lang(tokenize("Combien de clients habitent à Sfax ?")[1]) == "fr"
    assert detect_lang(tokenize("How many customers live in Sfax?")[1]) == "en"


def test_extract_sql():
    assert extract_sql("Sure:\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


class FakeLLM(OpenAICompatBackend):
    """LLM backend with the network call replaced: first a bad draft, then a fix."""
    def __init__(self, schema):
        super().__init__(schema)
        self.calls = []

    def _chat(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return "```sql\nSELECT COUNT(*) FROM customer\n```"   # wrong table name
        return "```sql\nSELECT COUNT(*) FROM customers WHERE segment = 'premium'\n```"


def test_llm_backend_prompt_and_reprompt(db_path):
    agent = Agent(db_path, backend="offline")
    agent.backend = FakeLLM(agent.schema)
    s = agent.run("How many premium customers?")
    prompt = agent.backend.calls[0][1]["content"]
    assert "CREATE TABLE customers" in prompt and "segment = 'premium'" in prompt   # pruned DDL + value hint
    assert s.status == "answered" and s.attempts == 2   # 'customer' -> 'customers' by the deterministic tool
    assert len(agent.backend.calls) == 1


def test_llm_reprompt_when_no_rule_applies(db_path):
    class Bad(FakeLLM):
        def _chat(self, messages):
            self.calls.append(messages)
            return "SELECT COUNT(*) FROM customers WHERE" if len(self.calls) == 1 else \
                "SELECT COUNT(*) FROM customers"
    agent = Agent(db_path, backend="offline")
    agent.backend = Bad(agent.schema)
    s = agent.run("How many customers?")
    assert s.status == "answered" and len(agent.backend.calls) == 2
    assert "syntax error" in agent.backend.calls[1][-1]["content"]
