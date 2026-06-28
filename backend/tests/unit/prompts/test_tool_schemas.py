from prompts.tool_schemas import WEB_EXTRACT_SCHEMA


def test_web_extract_urls_schema_is_array():
    urls = WEB_EXTRACT_SCHEMA["function"]["parameters"]["properties"]["urls"]

    assert urls["type"] == "array"
    assert urls["items"]["type"] == "string"
    assert urls["minItems"] == 1
    assert urls["maxItems"] == 20

