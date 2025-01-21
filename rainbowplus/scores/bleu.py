import nltk

from rainbowplus.scores.base import BaseScore
from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class BleuScoreNLTK(BaseScore):
    """
    BLEU score implementation using NLTK package.
    Suitable for sentence-level BLEU score calculation.
    """

    def __init__(self, weights: tuple = (0.25, 0.25, 0.25, 0.25)):
        """
        Initialize BLEU score calculator with custom weights.

        Args:
            weights: Tuple of weights for 1-gram through 4-gram scores.
                    Must sum to 1.0. Default is (0.25, 0.25, 0.25, 0.25).
        """
        self.weights = weights
        # Initialize NLTK's smoothing function
        self.smoother = SmoothingFunction()

        # Download required NLTK data if not already present
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
            nltk.download("punkt_tab")

    def get_name(self) -> str:
        """Return the name of the scoring method."""
        return "BLEU-NLTK"

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text using NLTK's word tokenizer.

        Args:
            text: Input text string

        Returns:
            List of tokens
        """
        return nltk.word_tokenize(text.lower())

    def score(self, response: str, response_: str) -> float:
        """
        Calculate BLEU score between candidate and reference texts using NLTK.

        Args:
            response: Candidate text
            response_: Reference text

        Returns:
            BLEU score between 0.0 and 1.0
        """
        candidate_tokens = self._tokenize(response)
        reference_tokens = [self._tokenize(response_)]

        try:
            return sentence_bleu(
                reference_tokens,
                candidate_tokens,
                weights=self.weights,
                smoothing_function=self.smoother.method4,
            )
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return 0.0
