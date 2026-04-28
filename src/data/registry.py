from src.data.pubmedvision import PubMedVisionDataset
from src.data.ds_pubmed import PubMedVisionGRPO
from src.data.ds_pmcvqa import PMCVQAGRPODataset

DATASET_REGISTRY = {
    "pubmedvision": PubMedVisionDataset,
    "ds_pubmed": PubMedVisionGRPO,
    "ds_pmcvqa": PMCVQAGRPODataset,
}


def get_dataset(cfg: dict, tokenizer=None):
    dataset_cfg = cfg["dataset"]
    dataset_type = dataset_cfg["type"]

    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'. Available: {list(DATASET_REGISTRY.keys())}")

    dataset_cls = DATASET_REGISTRY[dataset_type]
    
    if dataset_type == "ds_pmcvqa":
        return dataset_cls(
            data_path=dataset_cfg["path"],
            tokenizer=tokenizer,
            max_samples=dataset_cfg.get("max_samples"),
            offset=dataset_cfg.get("offset", 0),
        )
    else:
        return dataset_cls(
            data_path=dataset_cfg.get("data_path", ""),
            json_file=dataset_cfg.get("json_file", ""),
            max_samples=dataset_cfg.get("max_samples"),
            offset=dataset_cfg.get("offset", 0),
        )
