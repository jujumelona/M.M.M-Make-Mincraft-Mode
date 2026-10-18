"""Protect the native Java grammar against unbounded AliasHolder identifier loops."""
from minecraft_mod_ai.java_source_grammar import _resource, native_java_source_grammar


def test_native_java_identifier_suffix_is_finite_before_llama_compilation():
    grammar = native_java_source_grammar({
        "kind": "method", "prefix": "public void run() {", "suffix": "}",
    })
    source = _resource()["grammar"]
    # The generated Java grammar contains many keyword-avoidance trie rules.
    assert source.count("[A-Za-z0-9_$]*") > 200
    assert "[A-Za-z0-9_$]*" not in grammar
    assert grammar.count("[A-Za-z0-9_$]{0,127}") == source.count("[A-Za-z0-9_$]*")
    assert grammar.startswith('root ::= "public void run() {"')


def test_native_java_restriction_keeps_other_grammar_surfaces_unchanged():
    source = _resource()["grammar"]
    frame = {"kind": "field", "prefix": "private int count = ", "suffix": ";"}
    compiled = native_java_source_grammar(frame)
    restricted = source.replace("[A-Za-z0-9_$]*", "[A-Za-z0-9_$]{0,127}")
    assert compiled == 'root ::= "private int count = " g139 ";"\n' + restricted
    assert "java-string ::=" in compiled
