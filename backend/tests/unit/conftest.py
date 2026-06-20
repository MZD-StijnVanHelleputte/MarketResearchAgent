import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def auto_approve_gates():
    """Prevent LangGraph interrupt() from raising outside the graph context.

    Nodes pause at gates by calling interrupt(). Unit tests call nodes directly
    (no LangGraph runnable context), so interrupt() raises RuntimeError.
    Mocking it to return "approve" simulates a human approving at every gate
    and lets the node complete normally.
    """
    with patch("core.graph.interrupt", return_value="approve"):
        yield
