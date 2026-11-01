from __future__ import annotations

from typing import List, Optional
import torch
from transformers import pipeline


class TransformerSentimentScorer:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: Optional[int] = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        self._pipeline = pipeline(
            task="sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            truncation=True,
        )

    @staticmethod
    def _label_to_score(label: str, score: float) -> float:
        label = label.upper()
        if label == "POSITIVE":
            return float(score)
        if label == "NEGATIVE":
            return -float(score)
        return 0.0

    def score_texts(self, texts: List[str]) -> List[float]:
        if not texts:
            return []

        outputs = self._pipeline(texts, batch_size=self.batch_size)
        scores = []
        for out in outputs:
            label = out["label"]
            confidence = float(out["score"])
            scores.append(self._label_to_score(label, confidence))
        return scores


def get_transformer_scores(
    texts: List[str],
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    batch_size: int = 16,
    device: Optional[int] = None,
) -> List[float]:
    scorer = TransformerSentimentScorer(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    return scorer.score_texts(texts)