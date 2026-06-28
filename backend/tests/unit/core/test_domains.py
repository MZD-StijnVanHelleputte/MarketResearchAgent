"""Integrity tests for the domain + leaf-type registry (core/domains.py)."""
from core import domains


def test_nine_domains_registered():
    assert set(domains.domain_keys()) == {
        "commodities", "competition", "mining_operators", "construction_companies",
        "specialized_customers", "distributors", "mining_projects", "macroeconomics",
        "general_search",
    }


def test_every_domain_has_valid_leaf_types_and_tools():
    for key, spec in domains.DOMAINS.items():
        assert spec.leaf_types, f"{key} has no leaf types"
        for lt in spec.leaf_types:
            assert lt in domains.LEAF_TOOLSETS, f"{key} references unknown leaf type {lt}"
        assert domains.domain_tools(key), f"{key} resolves to no tools"


def test_ownership_order_is_total_and_unique():
    order = domains.ownership_order()
    assert len(order) == len(set(order)) == len(domains.DOMAINS)
    priorities = [s.ownership_priority for s in domains.DOMAINS.values()]
    assert len(priorities) == len(set(priorities)), "ownership priorities must be unique"


def test_masterdata_sources_are_known_getters():
    valid = {"competitors", "operators", "construction", "others", "distributors", "commodities"}
    for spec in domains.DOMAINS.values():
        if spec.masterdata_source is not None:
            assert spec.masterdata_source in valid


def test_domain_tools_is_union_of_leaf_toolsets_order_preserving():
    # general_search holds both topic and company leaves.
    tools = domains.domain_tools("general_search")
    assert tools[: len(domains.LEAF_TOOLSETS["topic"])] == domains.LEAF_TOOLSETS["topic"]
    assert "get_company_financials" in tools  # contributed by the company leaf type
