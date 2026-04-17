# EgoForesight — Usage Guide

## Prerequisites

```bash
cd ~/Desktop/sdp && source .venv/bin/activate   # Mac
cd /root/sdp                                     # GPU server
```

## V1/V2/V3 Inference

```bash
# With GLM API (default, requires ZHIPUAI_API_KEY)
python -m v1.run --video data/videos/<video>.mp4 --output data/results/
python -m v2.run --video data/videos/<video>.mp4 --output data/results/
python -m v3.run --video data/videos/<video>.mp4 --output data/results/

# With local Qwen (GPU server, base model, no adapter)
python -m v1.run --video data/videos/<video>.mp4 --backend qwen
python -m v2.run --video data/videos/<video>.mp4 --backend qwen
python -m v3.run --video data/videos/<video>.mp4 --backend qwen
```

## V4/V5 Training (GPU Server)

```bash
# Run 1: lr=2e-4, 3 epochs, full training data
python -m v4.train
python -m v5.train

# Run 2: lr=5e-5, 5 epochs, filtered training data
python -m v4.train_v2
python -m v5.train_v2
```

## V4/V5 Inference (GPU Server)

```bash
# Quick test on 5 samples
python -m v4.infer
python -m v5.infer

# Full inference pipeline
python -m v4.run --video data/videos/<video>.mp4 --output data/results/
python -m v5.run --video data/videos/<video>.mp4 --output data/results/

# Custom adapter path
python -m v4.run --video data/videos/<video>.mp4 --adapter /root/sdp/outputs/v4_run2/lora_adapter
python -m v5.run --video data/videos/<video>.mp4 --adapter /root/sdp/outputs/v5_run2/lora_adapter
```

## Annotation

```bash
# Annotate a single video (GLM API only)
python -m annotation.run_annotate \
    --video data/videos/<video_uid>.mp4 \
    --narration data/narrations/selected_narrations.csv \
    --output data/annotations/

# Batch annotate all videos
for video in data/videos/*.mp4; do
    python -m annotation.run_annotate \
        --video "$video" \
        --narration data/narrations/selected_narrations.csv \
        --output data/annotations/ --delay 0.5
done
```

## Build Training Data

```bash
# V4 training data (no history)
python -m annotation.build_training \
    --annotations data/annotations/ \
    --output data/training_v4/ \
    --n_frames 4 --pred_horizon 1

# V5 training data (with history)
python -m annotation.build_training \
    --annotations data/annotations/ \
    --output data/training_v5/ \
    --n_frames 4 --pred_horizon 1 \
    --with_history --history_steps 3
```

## Filter Training Data

```bash
# Filter both V4 and V5 (Jaccard > 0.5 removed)
python -m tools.filter_training_data

# Custom threshold
python -m tools.filter_training_data --threshold 0.4

# Filter specific version
python -m tools.filter_training_data --versions v4
```

## Evaluation

```bash
# BERTScore only (default)
python -m tools.evaluate --versions v1 v2 v3

# Semantic similarity only
python -m tools.evaluate --versions v1 v2 --metrics semantic

# Both metrics
python -m tools.evaluate --metrics bertscore semantic

# Multi-directory evaluation (10 versions across different result dirs)
python -m tools.evaluate \
  --version_dirs \
    glm_v1:report_data/glm_v1:v1 \
    glm_v2:report_data/glm_v2:v2 \
    glm_v3:report_data/glm_v3:v3 \
    qwen_v1:report_data/results_qwen:v1 \
    qwen_v2:report_data/results_qwen:v2 \
    qwen_v3:report_data/results_qwen:v3 \
    v4_run1:report_data/results_run1:v4 \
    v5_run1:report_data/results_run1:v5 \
    v4_run2:report_data/results_run2:v4 \
    v5_run2:report_data/results_run2:v5 \
  --metrics bertscore semantic \
  --model_type roberta-large \
  --output_dir report_data/evaluation

# Custom semantic model
python -m tools.evaluate --metrics semantic --semantic_model all-mpnet-base-v2

# Evaluate specific test videos
python -m tools.evaluate --video_list <uid1> <uid2> <uid3>
```

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--versions` | v1-v5 | Simple mode: versions in `--results_dir` |
| `--version_dirs` | — | Multi-dir mode: `name:dir:suffix` entries |
| `--metrics` | bertscore | `bertscore`, `semantic`, or both |
| `--model_type` | microsoft/deberta-xlarge-mnli | BERTScore model |
| `--semantic_model` | all-MiniLM-L6-v2 | Sentence-transformers model |
| `--offset` | 1.0 | GT target = t_end + offset (seconds) |
| `--tolerance` | 2.0 | Max distance to GT narration (seconds) |
| `--output_dir` | data/evaluation | Where to save results |

### Dependencies

```bash
pip install bert-score              # for BERTScore
pip install sentence-transformers   # for Semantic Similarity
```
