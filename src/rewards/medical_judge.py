from src.rewards.format_reward import FormatReward
from src.rewards.semantic_reward import SemanticReward
from src.rewards.medical_entity_reward import MedicalEntityReward

class MedicalJudgeReward:
    def __init__(self):
        self.format_judge = FormatReward()
        self.semantic_judge = SemanticReward()
        self.entity_judge = MedicalEntityReward()

    def __call__(self, completions, answer, modality=None, body_part=None, **kwargs):
        format_scores = self.format_judge(completions)
        semantic_scores = self.semantic_judge(completions, answer)
        entity_scores = self.entity_judge(completions, modality=modality, body_part=body_part)

        final_rewards = []
        for f_s, s_s, e_s in zip(format_scores, semantic_scores, entity_scores):
            format_part = f_s * 0.4
            semantic_part = max(0.0, s_s) * 0.4
            entity_part = e_s * 0.2
            total = format_part + semantic_part + entity_part
            final_rewards.append(total)

        return final_rewards
