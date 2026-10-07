"""Text normalisation for Kiswahili tweets.

The AfriSenti release has already removed emojis, punctuation, @mentions and
hashtags, so what remains to clean is:
  * HTML-entity residue left after punctuation stripping ("&gt;&gt;" -> "gtgt", "&amp;" -> "amp")
  * URL fragments ("http", "www...")
  * casing ("CCM" vs "ccm")
  * letter elongation used for emphasis ("sanaaaa" -> "sanaa")
  * numbers (dates, counts, prices), which carry little sentiment on their own

Each step can be switched off so that its effect can be measured as an ablation.
"""
import re

_URL = re.compile(r"\b(?:https?\w*|www\w*|\w+\.(?:com|co|org|tz|ke)\w*)\b")
_ENTITY = re.compile(r"\b(?:(?:gt|lt)+|amp|quot|nbsp)\b")
_ELONGATION = re.compile(r"(\w)\1{2,}")
_NUMBER = re.compile(r"\b\d+(?:\w*)\b")
_SPACES = re.compile(r"\s+")


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_entities: bool = True,
    squeeze_elongation: bool = True,
    replace_numbers: bool = True,
) -> str:
    """Normalise a single tweet. With every flag off it only collapses whitespace."""
    if remove_urls:
        text = _URL.sub(" ", text)
    if lowercase:
        text = text.lower()
    if remove_entities:
        text = _ENTITY.sub(" ", text)
    if squeeze_elongation:
        # Keep two letters: Swahili has genuine double vowels (e.g. "kuu", "saa").
        text = _ELONGATION.sub(r"\1\1", text)
    if replace_numbers:
        text = _NUMBER.sub(" <num> ", text)
    return _SPACES.sub(" ", text).strip()


def clean_series(texts, **kwargs):
    """Apply clean_text to an iterable / pandas Series and return a list."""
    return [clean_text(t, **kwargs) for t in texts]
