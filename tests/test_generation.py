import pytest

from app.generation import has_content


@pytest.mark.parametrize(
    "answer", ["24 days. [1]", "No. [1]", "Yes, business class is allowed. [1]", "5 days [2][3]"]
)
def test_short_real_answers_count_as_content(answer):
    assert has_content(answer)


@pytest.mark.parametrize("answer", ["[1]", " [1][2] ", "", "[1, 2]"])
def test_bare_citations_are_not_content(answer):
    assert not has_content(answer)