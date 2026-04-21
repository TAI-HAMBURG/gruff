"""Helpers for aligning noun/article variants with pronoun variants."""

from typing import Iterator, Sequence, Tuple, TypeVar

T = TypeVar("T")

# Article index -> pronoun index.
# This swaps only the nonbinary pairings:
# - De + e-ending noun uses xier-paradigm (index 3)
# - Dier + *-noun uses en-paradigm (index 2)
ARTICLE_TO_PRONOUN_INDEX = {
    0: 0,
    1: 1,
    2: 3,
    3: 2,
}


def iter_paired_variants(
    pronouns: Sequence[str],
    referents: Sequence[T],
    articles: Sequence[str],
) -> Iterator[Tuple[int, str, str, T]]:
    """Yield aligned variants as: (article_idx, pronoun, article, referent)."""
    if len(pronouns) != len(articles) or len(referents) != len(articles):
        raise ValueError(
            "Expected pronouns, referents, and articles to have identical lengths."
        )

    for article_idx, (article, referent) in enumerate(zip(articles, referents)):
        pronoun_idx = ARTICLE_TO_PRONOUN_INDEX.get(article_idx, article_idx)
        if pronoun_idx >= len(pronouns):
            raise ValueError(
                f"Pronoun index {pronoun_idx} out of range for {len(pronouns)} pronouns."
            )
        yield article_idx, pronouns[pronoun_idx], article, referent
