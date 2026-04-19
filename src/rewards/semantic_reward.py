import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.rewards.utils import extract_answer


class SemanticReward:
    SCALE = 1.5

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, completions, answer, **kwargs):
        preds = [extract_answer(c) for c in completions]
        gts = [a.strip() for a in answer]

        emb_preds = self.model.encode(preds, convert_to_tensor=True)
        emb_gts = self.model.encode(gts, convert_to_tensor=True)

        similarities = F.cosine_similarity(emb_preds, emb_gts, dim=1)
        return [max(0.0, s.item()) * self.SCALE for s in similarities]
