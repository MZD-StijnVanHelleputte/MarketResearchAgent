import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def auto_approve_gates():
    """Prevent LangGraph interrupt() from raising outside the graph context.

    Integration tests that call graph nodes directly (not through the compiled
    graph) hit the same issue as unit tests: interrupt() requires a LangGraph
    runnable context. Mock it to return "approve" for all such tests.
    """
    with patch("core.graph.interrupt", return_value="approve"):
        yield
