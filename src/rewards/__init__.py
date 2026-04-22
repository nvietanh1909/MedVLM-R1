REWARD_REGISTRY = {
    "format": "src.rewards.format_reward.FormatReward",
    "correctness": "src.rewards.correctness_reward.CorrectnessReward",
    "semantic": "src.rewards.semantic_reward.SemanticReward",
    "medical_entity": "src.rewards.medical_entity_reward.MedicalEntityReward",
    "quality": "src.rewards.quality_reward.QualityReward",
    "medical_judge": "src.rewards.medical_judge.MedicalJudgeReward",
}


def _import_class(dotted_path):
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_reward_funcs(cfg):
    reward_cfg = cfg.get("rewards", {})
    enabled = reward_cfg.get("enabled", list(REWARD_REGISTRY.keys()))

    funcs = []
    for name in enabled:
        if name not in REWARD_REGISTRY:
            raise ValueError(
                f"Unknown reward: '{name}'. Available: {list(REWARD_REGISTRY.keys())}"
            )
        cls = _import_class(REWARD_REGISTRY[name])
        if name == "semantic":
            model_name = reward_cfg.get("semantic_model", "all-MiniLM-L6-v2")
            funcs.append(cls(model_name=model_name))
        else:
            funcs.append(cls())

    return funcs
