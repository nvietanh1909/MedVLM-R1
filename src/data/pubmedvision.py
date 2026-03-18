import json
import os

from PIL import Image

from src.data.base_dataset import BaseVisionDataset


class PubMedVisionDataset(BaseVisionDataset):

    def __init__(self, data_path: str, json_file: str, max_samples: int = None):
        json_path = os.path.join(data_path, json_file)
        with open(json_path, "r") as f:
            self.data = json.load(f)
        if max_samples is not None:
            self.data = self.data[:max_samples]
        self.data_path = data_path

    def __len__(self):
        return len(self.data)

    def _load_images(self, image_paths: list) -> list:
        images = []
        for path in image_paths:
            full_path = os.path.join(self.data_path, path)
            images.append(Image.open(full_path).convert("RGB"))
        return images

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        images = self._load_images(sample.get("image", []))
        conversations = sample["conversations"]

        messages = []
        for i in range(0, len(conversations), 2):
            human = conversations[i]
            gpt = conversations[i + 1]

            user_content = [{"type": "image", "image": img} for img in images]
            user_content.append({"type": "text", "text": human["value"]})

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": gpt["value"]}]})

        return {"messages": messages}
