from rouge_score import rouge_scorer

from src.rewards.utils import extract_answer


class CorrectnessReward:
    SCALE = 2.0

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def __call__(self, completions, answer=None, ground_truth=None, **kwargs):
        gt_list = answer if answer is not None else ground_truth
        rewards = []
        for completion, gt in zip(completions, gt_list):
            pred = extract_answer(completion)
            score = self.scorer.score(gt.strip(), pred)["rougeL"].fmeasure
            rewards.append(score * self.SCALE)
        return rewards
