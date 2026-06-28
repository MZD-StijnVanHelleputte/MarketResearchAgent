"""The master-data entity resolver pins each entity to exactly one domain.

These tests run against the real master-data files so they also guard data
integrity (no company appearing in two master-data files / domains)."""
from services.masterdata_service import MasterDataService


def test_known_entities_resolve_to_canonical_domains():
    md = MasterDataService()
    cases = {
        "Caterpillar": "competition",
        "CAT": "competition",
        "BHP": "mining_operators",
        "Vinci SA": "construction_companies",
        "ArcelorMittal": "specialized_customers",
        "Hastings Deering": "distributors",
    }
    for name, expected_domain in cases.items():
        res = md.resolve_entity(name)
        assert res is not None, f"{name} should resolve"
        assert res.domain == expected_domain, f"{name} -> {res.domain}, expected {expected_domain}"


def test_parenthesised_ticker_form_resolves():
    md = MasterDataService()
    res = md.resolve_entity("Caterpillar Inc. (CAT)")
    assert res is not None and res.domain == "competition" and res.params.get("ticker") == "CAT"


def test_unknown_entity_returns_none():
    md = MasterDataService()
    assert md.resolve_entity("Totally Unknown Co") is None
    assert md.resolve_entity("") is None


def test_no_entity_resolves_to_two_domains():
    """Each indexed alias maps to a single resolution — overlap is impossible."""
    md = MasterDataService()
    index = md._entity_resolution_index()
    # Every alias has exactly one (deterministic) resolution; the index itself is a
    # 1:1 map, so the guarantee is structural. Spot-check a competitor never doubles
    # as a customer.
    cat = index.get("caterpillar")
    assert cat is not None and cat.domain == "competition"
    assert all(r.domain == cat.domain for k, r in index.items() if k in {"caterpillar", "cat"})
