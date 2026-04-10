# BLIP-2 Explorer

**Course:** CMPT 420 — Simon Fraser University  
**Model:** [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)  
**Dataset:** [UICD — Underwater Image Captioning Dataset](https://www.kaggle.com/datasets/kevintang2048/underwater-image-captioning-dataset-uicd)  
**Accelerator:** Kaggle Notebooks — T4 x2

---

## Abstract

The application of large language models to image-to-text tasks in offline settings is often constrained by limited hardware resources, with deployment bottlenecks arising in embedded environments where high-accuracy models are too large for local operation. This limitation is particularly evident in marine imaging systems for ecological research, where real-time image captioning can enhance the utility of video footage by maintaining a searchable text-based record, yet has to operate entirely on-device in underwater environments with limited or unreliable internet connectivity.

To address this challenge, we introduce **BLIP-2 Explorer**, a lightweight image captioning model based on BLIP-2 that is specialised for marine image captioning through domain-specific fine-tuning. We combine knowledge distillation with low-rank adaptation (LoRA)-based parameter-efficient fine-tuning (PEFT) to substantially reduce the size of the original BLIP-2 architecture while preserving comparable image captioning performance in marine-specific contexts.

---

## Repository Structure

```
blip2-explorer/
├── blip2-explorer-lora.ipynb          # Teacher: BLIP-2 LoRA fine-tuning
├── blip2-explorer-lora-results.ipynb  # Teacher: results analysis
├── blip2_explorer_experiments.ipynb   # Experiments and ablations
├── blip1-explorer-distill.ipynb       # Student: BLIP-1 knowledge distillation
├── blip1-explorer-pseudo.ipynb        # Pseudo-label generation via teacher
├── experiments.ipynb                  # Additional experiment runs
├── environment.yml                    # Conda environment specification
├── graphs/                            # Training curves and metric plots
│   ├── loss_curves.png
│   ├── metric_curves.png
│   ├── composite_score_curve.png
│   ├── efficiency_curve.png
│   └── *_distill.png                  # Distillation equivalents
└── report/                            # Course report and presentation
    ├── report.pdf
    ├── report.tex
    ├── references.bib
    └── CMPT-420 Presentation Final Ver.pdf
```

---

## Approach

### Teacher Model — BLIP-2 + LoRA
- Base: `Salesforce/blip2-opt-2.7b` (~3.7 B parameters) frozen; LoRA adapters injected into `q_proj`, `k_proj`, `v_proj` of Q-Former cross-attention and OPT decoder layers.
- Training objective: cross-entropy loss + warm-started CLIP auxiliary alignment loss.
- Best checkpoint selected via weighted composite score (CLIP-prioritised).

### Student Model — BLIP-1 + LoRA + Knowledge Distillation
- Student: `Salesforce/blip-image-captioning-large` (~400 M parameters, 9× smaller than teacher).
- Trained on teacher pseudo-labels (`pseudo_labels.json`) with LoRA (r=32) and a CLIP auxiliary loss adapted for BLIP-1's decoder hidden states.
- Expected composite score: **80–88 % of teacher baseline** with the shared ViT-L/16 encoder protecting CLIP alignment.

---

## Setup

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate blip2-sentry
```

### 2. Run on Kaggle (recommended)

The notebooks are designed for **Kaggle Notebooks with T4 x2 GPUs**. Upload the notebooks and attach the UICD dataset from Kaggle.

For the distillation notebook, also attach the `kevintang2048/blip2-explorer-pseudo-labels` dataset as a Kaggle input.

---

## Results

Training curves and metric plots are available in the `graphs/` directory.

| Metric | Teacher (BLIP-2 + LoRA) | Student (BLIP-1 + Distill) |
|--------|------------------------|---------------------------|
| CLIP Score | — | — |
| CIDEr | — | — |
| METEOR | — | — |
| BLEU-4 | — | — |
| ROUGE-L | — | — |

> See `report/report.pdf` for full quantitative results and analysis.

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
