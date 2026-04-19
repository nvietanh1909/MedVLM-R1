from src.rewards.utils import extract_content


class QualityReward:
    MIN_LENGTH = 20
    MAX_LENGTH = 2000
    MIN_UNIQUE_RATIO = 0.3

    def __call__(self, completions, **kwargs):
        rewards = []
        for completion in completions:
            clean = extract_content(completion).strip()

            if len(clean) < self.MIN_LENGTH:
                score = -1.0
            elif len(clean) > self.MAX_LENGTH:
                score = -0.5
            else:
                score = 0.5

            words = clean.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < self.MIN_UNIQUE_RATIO:
                    score = -1.0

            rewards.append(score)
        return rewards
