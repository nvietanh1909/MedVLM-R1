import re

from src.rewards.utils import extract_content


class FormatReward:
    PATTERN = r"<think>[\s\S]+?</think>\s*<answer>[\s\S]+?</answer>"
    THINK_PATTERN = r"<think>([\s\S]+?)</think>"
    MIN_THINKING_LENGTH = 50

    def __call__(self, completions, **kwargs):
        rewards = []
        for completion in completions:
            text = extract_content(completion)
            if not re.search(self.PATTERN, text):
                rewards.append(0.0)
                continue

            think_match = re.search(self.THINK_PATTERN, text)
            think_len = len(think_match.group(1).strip()) if think_match else 0
            rewards.append(1.0 if think_len >= self.MIN_THINKING_LENGTH else 0.5)

        return rewards
