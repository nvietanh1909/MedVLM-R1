from src.data.pubmedvision import PubMedVisionDataset
from src.data.pubmedvision_grpo import PubMedVisionGRPO

DATASET_REGISTRY = {
    "pubmedvision": PubMedVisionDataset,
    "pubmedvision_grpo": PubMedVisionGRPO,
}


def get_dataset(cfg: dict):
    dataset_cfg = cfg["dataset"]
    dataset_type = dataset_cfg["type"]

    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'. Available: {list(DATASET_REGISTRY.keys())}")

    dataset_cls = DATASET_REGISTRY[dataset_type]
    return dataset_cls(
        data_path=dataset_cfg["data_path"],
        json_file=dataset_cfg["json_file"],
        max_samples=dataset_cfg.get("max_samples"),
    )
