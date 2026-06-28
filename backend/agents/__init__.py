"""Domain collection agents.

There is a single generic ``DomainAgent`` whose persona (role/goal/backstory) and
domain key come from the domain registry (core/domains.py). ``DOMAIN_AGENTS`` maps
each domain key to a zero-argument factory so the dispatch controller in
core/graph.py can keep calling ``DOMAIN_AGENTS[domain]()``. Adding or renaming a
domain is therefore a single edit to core/domains.py — no new agent class.
"""
from functools import partial

from agents.base_domain_agent import BaseDomainAgent
from agents.grounding_agent import GroundingAgent
from core.domains import DOMAINS, DomainSpec


class DomainAgent(BaseDomainAgent):
    """Config-driven domain collection agent built from a DomainSpec."""

    def __init__(self, spec: DomainSpec) -> None:
        # Instance attributes shadow BaseDomainAgent's empty class-level defaults.
        self.DOMAIN = spec.key
        self.ROLE = spec.role
        self.GOAL = spec.goal
        self.BACKSTORY = spec.backstory
        super().__init__()


def make_domain_agent(domain: str) -> DomainAgent:
    """Instantiate the collection agent for a domain key."""
    spec = DOMAINS.get(domain)
    if spec is None:
        raise KeyError(f"Unknown domain '{domain}'. Known: {', '.join(DOMAINS)}")
    return DomainAgent(spec)


# domain key → zero-arg factory, consumed by core/graph.py's parallel dispatch.
DOMAIN_AGENTS: dict[str, "partial[DomainAgent]"] = {
    key: partial(DomainAgent, spec) for key, spec in DOMAINS.items()
}

__all__ = ["BaseDomainAgent", "DomainAgent", "GroundingAgent", "DOMAIN_AGENTS", "make_domain_agent"]
