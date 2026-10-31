"""graphflow.text.chunk_text: structure-preserving semantic chunks."""

from graphflow.text import chunk_text


def test_headings_sentences_and_list_items_are_separate_chunks():
    text = "# Title\n\nFirst sentence here. Second one follows.\n\n- a list item\n"
    assert chunk_text(text) == ["# Title\n", "First sentence here. Second one follows.\n\n", "- a list item\n"]


def test_a_fenced_code_block_is_one_chunk():
    text = "Before the code.\n```py\nx = 1\ny = 2\n```\nAfter the code."
    assert chunk_text(text) == ["Before the code.\n", "```py\nx = 1\ny = 2\n```\n", "After the code."]
