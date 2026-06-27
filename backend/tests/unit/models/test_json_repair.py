from models.json_repair import extract_json_field


def test_valid_json():
    assert extract_json_field('{"domain": "customers", "text": "hello"}', "text") == "hello"


def test_json_with_wrapper_prose():
    raw = 'Sure, here is the result:\n{"domain": "customers", "text": "hello"}\nLet me know!'
    assert extract_json_field(raw, "text") == "hello"


def test_json_with_markdown_fences():
    raw = '```json\n{"domain": "customers", "text": "hello"}\n```'
    assert extract_json_field(raw, "text") == "hello"


def test_json_with_escaped_quote_in_text():
    raw = '{"domain": "customers", "text": "Komatsu said \\"strong demand\\" persists."}'
    assert extract_json_field(raw, "text") == 'Komatsu said "strong demand" persists.'


def test_regex_fallback_when_sibling_key_is_malformed():
    # contradiction_flags is broken (unterminated string) but text is intact.
    raw = '{"domain": "customers", "text": "Recovered prose.", "contradiction_flags": ["unterminated}'
    assert extract_json_field(raw, "text") == "Recovered prose."


def test_totally_unparseable_returns_none():
    assert extract_json_field("not json at all, just prose.", "text") is None


def test_missing_field_returns_none():
    assert extract_json_field('{"domain": "customers"}', "text") is None
