import json
import os

from src.prompts import MEDICAL_THINKING_PROMPT

try:
    from datasets import Dataset as HFDataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False


def _build_records(data_path, json_file, max_samples=None):
    json_path = os.path.join(data_path, json_file)
    with open(json_path) as f:
        raw_data = json.load(f)

    if max_samples is not None:
        raw_data = raw_data[:max_samples]

    records = []
    for sample in raw_data:
        sample_images = sample.get("image", [])
        conversations = sample.get("conversations", [])

        if not sample_images or len(conversations) < 2:
            continue

        full_paths = [os.path.join(data_path, p) for p in sample_images]

        image_content = [{"type": "image", "image": p} for p in full_paths]
        image_content.append({"type": "text", "text": conversations[0]["value"]})

        prompt = [
            {"role": "system", "content": MEDICAL_THINKING_PROMPT},
            {"role": "user", "content": image_content},
        ]

        records.append({
            "prompt": prompt,
            "answer": conversations[1]["value"],
            "modality": sample.get("modality", ""),
            "body_part": sample.get("body_part", ""),
        })

    return records


class PubMedVisionGRPO:
    def __init__(self, data_path, json_file, max_samples=None):
        self.data = _build_records(data_path, json_file, max_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.data)))
            return [self.data[i] for i in indices]
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    @property
    def column_names(self):
        return list(self.data[0].keys()) if self.data else []

    def select(self, indices):
        subset = PubMedVisionGRPO.__new__(PubMedVisionGRPO)
        subset.data = [self.data[i] for i in indices]
        return subset

    def shuffle(self, seed=None):
        import random
        rng = random.Random(seed)
        shuffled = PubMedVisionGRPO.__new__(PubMedVisionGRPO)
        shuffled.data = list(self.data)
        rng.shuffle(shuffled.data)
        return shuffled

    def map(self, func, **kwargs):
        return self

    def with_format(self, *args, **kwargs):
        return self
