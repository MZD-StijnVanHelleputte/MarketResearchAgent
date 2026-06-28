from core.result_quality import has_usable_data


def test_empty_collection_shapes_are_not_usable_data():
    assert has_usable_data({"articles": []}) is False
    assert has_usable_data({"results": []}) is False
    assert has_usable_data({"rows": []}) is False
    assert has_usable_data({"pages": []}) is False
    assert has_usable_data({"filings": []}) is False
    assert has_usable_data({"ticker": "CAT", "source": "provider"}) is False


def test_non_empty_collection_shapes_are_usable_data():
    assert has_usable_data({"articles": [{"title": "CAT earnings"}]}) is True
    assert has_usable_data({"results": [{"url": "https://example.com"}]}) is True
    assert has_usable_data({"rows": [{"date": "2026-01-01", "value": 10}]}) is True
    assert has_usable_data({"pages": [{"url": "https://example.com", "content": "text"}]}) is True
    assert has_usable_data({"filings": [{"form_type": "10-K"}]}) is True


def test_latest_technical_reports_and_citations_are_usable_data():
    assert has_usable_data({"latest": {"date": "2026-01-01", "value": 9500}}) is True
    assert has_usable_data({"latest": {"date": "2026-01-01", "value": None}}) is False
    assert has_usable_data({"technical_report": {"exhibit_url": "https://example.com/report"}}) is True
    assert has_usable_data({"report": "summary", "citations": []}) is True
    assert has_usable_data({"citations": ["https://example.com/source"]}) is True
