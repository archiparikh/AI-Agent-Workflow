"""
tests/test_tools.py – Unit tests for the built-in agent tools.
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.tools import calculator, text_search, get_weather, summarize_text, ToolRegistry, ToolSpec


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

class TestCalculator:
    def test_simple_addition(self):
        assert calculator("2 + 3") == "5"

    def test_multiplication(self):
        assert calculator("6 * 7") == "42"

    def test_order_of_operations(self):
        assert calculator("2 + 3 * 4") == "14"

    def test_float_result(self):
        result = calculator("10 / 3")
        assert result.startswith("3.333")

    def test_power(self):
        assert calculator("2^10") == "1024"

    def test_unsafe_expression(self):
        result = calculator("__import__('os').system('ls')")
        assert "unsafe" in result.lower() or "error" in result.lower()

    def test_invalid_expression(self):
        result = calculator("2 +")
        assert "error" in result.lower()


# ---------------------------------------------------------------------------
# text_search
# ---------------------------------------------------------------------------

class TestTextSearch:
    _kb = json.dumps([
        "Python is a popular programming language.",
        "JavaScript runs in the browser.",
        "Machine learning uses statistical models.",
    ])

    def test_match_found(self):
        result = text_search("python", self._kb)
        assert "Python" in result

    def test_case_insensitive(self):
        result = text_search("PYTHON", self._kb)
        assert "Python" in result

    def test_no_match(self):
        result = text_search("Rust", self._kb)
        assert "No results" in result

    def test_invalid_kb(self):
        result = text_search("python", "not-json")
        assert "Error" in result


# ---------------------------------------------------------------------------
# get_weather
# ---------------------------------------------------------------------------

class TestGetWeather:
    def test_known_city(self):
        result = get_weather("London")
        assert "\u00b0C" in result

    def test_case_insensitive(self):
        result_lower = get_weather("tokyo")
        result_mixed = get_weather("Tokyo")
        assert result_lower == result_mixed

    def test_unknown_city(self):
        result = get_weather("Atlantis")
        assert "not available" in result


# ---------------------------------------------------------------------------
# summarize_text
# ---------------------------------------------------------------------------

class TestSummarizeText:
    _text = "First sentence. Second sentence. Third sentence. Fourth sentence."

    def test_default_two_sentences(self):
        result = summarize_text(self._text)
        assert result == "First sentence. Second sentence."

    def test_custom_count(self):
        result = summarize_text(self._text, num_sentences=3)
        assert result == "First sentence. Second sentence. Third sentence."

    def test_single_sentence(self):
        result = summarize_text("Only one sentence here.", num_sentences=2)
        assert result == "Only one sentence here."


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_call(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(name="double", description="double a number",
                     func=lambda x: x * 2, parameters={"x": "int"})
        )
        assert registry.call("double", x=5) == 10

    def test_unknown_tool_raises(self):
        registry = ToolRegistry()
        try:
            registry.call("nonexistent")
            assert False, "Expected KeyError"
        except KeyError:
            pass

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(name="noop", description="does nothing", func=lambda: None)
        )
        assert any(s.name == "noop" for s in registry.list_tools())

    def test_describe(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(name="greet", description="greet user",
                     func=lambda name: f"Hello, {name}!", parameters={"name": "str"})
        )
        desc = registry.describe()
        assert "greet" in desc
        assert "greet user" in desc
