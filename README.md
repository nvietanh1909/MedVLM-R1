# finetune

Fine-tune vision-language models with Unsloth on local datasets.

## Cấu trúc

```
finetune/
├── configs/                        # YAML config cho từng experiment
│   └── qwen3.5-2b_pubmedvision.yaml
├── src/
│   ├── data/
│   │   ├── base_dataset.py         # Abstract base class
│   │   ├── pubmedvision.py         # PubMedVision dataset loader
│   │   └── registry.py             # Dataset registry
│   ├── models/
│   │   └── loader.py               # Model + LoRA loader
│   ├── training/
│   │   └── trainer.py              # SFTTrainer builder
│   └── utils/
│       └── logger.py
├── scripts/
│   ├── train.py                    # Entry point training
│   ├── evaluate.py                 # Evaluation
│   └── infer.py                    # Interactive inference
├── dataset/                        # Dữ liệu
├── models/                         # Model weights
└── outputs/                        # Kết quả training (auto-generated)
```

## Cài đặt

```bash
pip install unsloth
pip install transformers==5.3.0
pip install --no-deps trl==0.22.2
pip install -r requirements.txt
```

## Training

```bash
python scripts/train.py --config configs/qwen3.5-2b_pubmedvision.yaml
```

Test nhanh với 1000 samples, 50 steps:

```bash
python scripts/train.py \
    --config configs/qwen3.5-2b_pubmedvision.yaml \
    --max_steps 50 \
    --max_samples 1000
```

## Inference

```bash
python scripts/infer.py \
    --config configs/qwen3.5-2b_pubmedvision.yaml \
    --adapter outputs/qwen3.5-2b_pubmedvision \
    --image /path/to/image.jpg \
    --question "What does this CT scan show?"
```

Với thinking mode:

```bash
python scripts/infer.py \
    --config configs/qwen3.5-2b_pubmedvision.yaml \
    --adapter outputs/qwen3.5-2b_pubmedvision \
    --image /path/to/image.jpg \
    --question "Analyze this medical image." \
    --thinking
```

## Evaluation

```bash
python scripts/evaluate.py \
    --config configs/qwen3.5-2b_pubmedvision.yaml \
    --adapter outputs/qwen3.5-2b_pubmedvision \
    --num_samples 200 \
    --output_file eval_results.json
```

## Thêm model mới

1. Tạo file `configs/new-model_new-dataset.yaml`
2. Đổi `model.model_path` và `dataset.json_file`
3. Nếu dataset có format mới → thêm class vào `src/data/`, đăng ký trong `src/data/registry.py`

## GPU Requirements

| Config | VRAM cần |
|--------|----------|
| 4bit LoRA (r=16) | ~8 GB |
| 16bit LoRA (r=16) | ~12 GB |
| 16bit LoRA (r=64) | ~14 GB |

Tesla T4 15GB → dùng `load_in_4bit: false` với gradient checkpointing là đủ.
