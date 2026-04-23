from src.rewards.utils import extract_content


class QualityReward:
    MIN_LENGTH = 20
    MAX_LENGTH = 2000
    MIN_UNIQUE_RATIO = 0.3

    def __call__(self, completions, **kwargs):
        rewards = []
        for completion in completions:
            clean = extract_content(completion).strip()

            if len(clean) < 100:
                score = -0.1
            elif len(clean) > 1000:
                score = -0.1
            else:
                score = 0.2

            words = clean.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < self.MIN_UNIQUE_RATIO:
                    score = -0.5

            rewards.append(score)
        return rewards
