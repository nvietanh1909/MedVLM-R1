import os
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.rewards.utils import extract_answer


class SemanticReward:
    SCALE = 1.0

    def __init__(self, model_name="models/S-PubMedBert-MS-MARCO"):
        os.environ.pop("http_proxy", None)
        self.model = SentenceTransformer(model_name)

    def __call__(self, completions, answer, **kwargs):
        preds = [extract_answer(c) for c in completions]
        gts = [a.strip() for a in answer]

        results = []
        for pred, gt in zip(preds, gts):
            if len(pred.strip()) < 5:
                results.append(0.0)
                continue

            emb_pred = self.model.encode([pred], convert_to_tensor=True)
            emb_gt = self.model.encode([gt], convert_to_tensor=True)
            sim = F.cosine_similarity(emb_pred, emb_gt, dim=1)
            normalized = max(0.0, (sim.item() - 0.3) / 0.7)
            results.append((normalized ** 3) * self.SCALE)

        return results
