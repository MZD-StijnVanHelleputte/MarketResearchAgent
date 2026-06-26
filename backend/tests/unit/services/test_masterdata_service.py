"""Tests for MasterDataService — no mocking; loads actual JSON files from data/."""
import pytest
from services.masterdata_service import MasterDataService


@pytest.fixture(scope="module")
def svc():
    return MasterDataService()


def test_get_equipment_nonempty(svc):
    equipment = svc.get_equipment()
    assert isinstance(equipment, list)
    assert len(equipment) > 0


def test_get_operators_nonempty(svc):
    operators = svc.get_operators()
    assert isinstance(operators, list)
    assert len(operators) > 0


def test_get_competitors_nonempty(svc):
    competitors = svc.get_competitors()
    assert isinstance(competitors, list)
    assert len(competitors) > 0


def test_get_distributors_nonempty(svc):
    distributors = svc.get_distributors()
    assert isinstance(distributors, list)
    assert len(distributors) > 0


def test_all_equipment_have_website(svc):
    for item in svc.get_equipment():
        assert "website" in item and item["website"], f"Missing website for equipment: {item.get('model')}"


def test_all_equipment_have_required_fields(svc):
    required = {"model", "category", "application", "website"}
    for item in svc.get_equipment():
        assert required.issubset(item.keys()), f"Missing fields in: {item}"


def test_all_competitors_have_ticker(svc):
    # At least half of competitors should have a ticker (some are private)
    competitors = svc.get_competitors()
    with_ticker = [c for c in competitors if c.get("ticker")]
    assert len(with_ticker) >= len(competitors) // 2


def test_all_competitors_have_website(svc):
    for item in svc.get_competitors():
        assert "website" in item and item["website"], f"Missing website for: {item.get('name')}"


def test_competitors_have_is_private_and_no_stale_revenue(svc):
    for item in svc.get_competitors():
        assert "is_private" in item, f"Missing is_private for: {item.get('name')}"
        assert "revenue_usd_bn_fy2023" not in item, f"Stale revenue field on: {item.get('name')}"


def test_all_operators_have_website(svc):
    for item in svc.get_operators():
        assert "website" in item and item["website"], f"Missing website for operator: {item.get('name')}"


def test_operators_have_required_fields_and_private_flag(svc):
    for item in svc.get_operators():
        assert "is_private" in item, f"Missing is_private for: {item.get('name')}"
        assert "primary_commodities" in item and item["primary_commodities"], \
            f"Missing primary_commodities for: {item.get('name')}"
        # Listed operators must carry a ticker; private ones must not.
        if item["is_private"]:
            assert item.get("ticker") is None, f"Private operator should have null ticker: {item.get('name')}"
        else:
            assert item.get("ticker"), f"Listed operator missing ticker: {item.get('name')}"


def test_all_distributors_have_website(svc):
    for item in svc.get_distributors():
        assert "website" in item and item["website"], f"Missing website for distributor: {item.get('name')}"


# --- lookup() tests ---

def test_lookup_distributors_all(svc):
    results = svc.lookup("distributors")
    assert isinstance(results, list)
    assert len(results) > 0


def test_lookup_competitors_all(svc):
    results = svc.lookup("competitors")
    assert isinstance(results, list)
    assert len(results) > 0


def test_lookup_operators_all(svc):
    results = svc.lookup("operators")
    assert isinstance(results, list)
    assert len(results) > 0


def test_lookup_equipment_all(svc):
    results = svc.lookup("equipment")
    assert isinstance(results, list)
    assert len(results) > 0


def test_lookup_distributors_region_filter(svc):
    results = svc.lookup("distributors", region="Asia-Pacific")
    assert len(results) > 0
    for item in results:
        assert "Asia-Pacific" in item.get("region", "") or any(
            "Asia-Pacific" in str(v) for v in item.values()
        )


def test_lookup_distributors_region_filter_returns_subset(svc):
    all_results = svc.lookup("distributors")
    filtered = svc.lookup("distributors", region="Asia-Pacific")
    assert len(filtered) < len(all_results)


def test_lookup_competitors_keyword_filter(svc):
    results = svc.lookup("competitors", keyword="CAT")
    assert len(results) > 0
    assert any("CAT" in item.get("ticker", "") or "Caterpillar" in item.get("name", "") for item in results)


def test_lookup_keyword_case_insensitive(svc):
    upper = svc.lookup("competitors", keyword="CAT")
    lower = svc.lookup("competitors", keyword="cat")
    assert len(upper) == len(lower)


def test_lookup_combined_region_and_keyword(svc):
    results = svc.lookup("distributors", region="Asia-Pacific", keyword="Australia")
    assert all("Australia" in str(item) for item in results)


def test_lookup_unknown_entity_type_raises(svc):
    with pytest.raises(ValueError, match="Unknown entity_type"):
        svc.lookup("invalid")


# --- commodity_tickers tests ---

def test_get_commodities_nonempty(svc):
    commodities = svc.get_commodities()
    assert isinstance(commodities, list)
    assert len(commodities) > 0


def test_get_commodities_have_required_fields(svc):
    required = {"Ticker", "Name", "Category", "Type"}
    for item in svc.get_commodities():
        assert required.issubset(item.keys()), f"Missing fields in: {item}"


def test_get_commodities_tickers_nonempty(svc):
    for item in svc.get_commodities():
        assert item.get("Ticker"), f"Empty ticker in row: {item}"


def test_lookup_commodities_all(svc):
    results = svc.lookup("commodities")
    assert isinstance(results, list)
    assert len(results) > 0


def test_lookup_commodities_keyword_gold(svc):
    results = svc.lookup("commodities", keyword="Gold")
    assert len(results) > 0
    assert all("gold" in str(item).lower() for item in results)


def test_lookup_commodities_keyword_copper(svc):
    results = svc.lookup("commodities", keyword="Copper")
    tickers = [item["Ticker"] for item in results]
    assert "HG" in tickers or "COPX" in tickers


def test_lookup_commodities_by_category(svc):
    results = svc.lookup("commodities", keyword="Precious Metals")
    assert len(results) > 0
    assert all("Precious Metals" in item.get("Category", "") for item in results)


def test_lookup_commodities_by_type_etf(svc):
    results = svc.lookup("commodities", keyword="ETF")
    assert len(results) > 0
    assert all(item.get("Type") == "ETF" for item in results)


def test_lookup_commodities_returns_subset(svc):
    all_results = svc.lookup("commodities")
    gold_results = svc.lookup("commodities", keyword="Gold")
    assert len(gold_results) < len(all_results)
