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
            if f_s <= 0.1:
                final_rewards.append(0.0)
                continue
            
            score_modifier = 1.0
            if e_s < 0:
                score_modifier = 0.5
            elif e_s == 0:
                score_modifier = 0.8 
            
            total = (s_s * score_modifier) + f_s + (max(0.0, e_s) * 0.1)
            final_rewards.append(total)
            
        return final_rewards
