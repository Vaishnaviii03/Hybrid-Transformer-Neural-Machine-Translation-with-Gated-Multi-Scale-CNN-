# Hybrid Transformer NMT with Gated Multi-Scale CNN (GMSCNN)

**Hybrid-Transformer-Neural-Machine-Translation-with-Gated-Multi-Scale-CNN**

This repository contains code, notebooks, and pretrained artifacts for a Hybrid Transformer Neural Machine Translation model augmented with a Gated Multi-Scale CNN (GMSCNN). The hybrid architecture combines the global context modeling strength of Transformers with local multi-scale pattern extraction from gated CNN modules to improve translation quality, especially for morphologically rich or low-resource language pairs.

---

## Table of Contents

* [Highlights](#highlights)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Training](#training)

  * [Baseline models](#baseline-models)
  * [GMSCNN hybrid model](#gmscnn-hybrid-model)
  * [Checkpointing and resuming](#checkpointing-and-resuming)
* [Evaluation](#evaluation)

  * [Automatic metrics (BLEU, TER, METEOR)](#automatic-metrics-bleu-ter-meteor)
  * [Notebooks included](#notebooks-included)
* [Inference / Decoding](#inference--decoding)
* [Reproducing results](#reproducing-results)
* [Files Provided](#files-provided)
* [Contributing](#contributing)
* [Citations](#citations)
* [License](#license)

---

## Highlights

* Hybrid architecture: Transformer encoder-decoder with integrated Gated Multi-Scale CNN (GMSCNN) modules to capture local, multi-scale features.
* Notebooks for training baselines and evaluating results.
* Pretrained SentencePiece tokenizers (`spm_en.model`, `spm_hi.model`) and some metric checkpoint files included for quick experimentation.
* Example scripts and suggested commands to run experiments on a single GPU or multi-GPU environment.

---

## Repository Structure

```
README.md
Baseline Models Training.ipynb
Evaluation.ipynb
baseline_metrics.pt
cnn_metrics.pt
metrics_gmsc_greedy.pt
multiscale_metrics.pt
spm_en.model
spm_en.vocab
spm_hi.model
spm_hi.vocab
src/                    # (suggested) model, training, utils code
scripts/                # (suggested) training/eval/infer scripts
data/                   # expected data layout (not included)
checkpoints/            # where checkpoints will be saved
requirements.txt
```

> Note: The repository currently contains Jupyter notebooks and artifacts. If you add `src/` and `scripts/` please keep the structure consistent with the commands below.

---

## Requirements

Recommended Python environment (example):

* Python 3.8+
* PyTorch 1.12+ (or latest stable compatible with your CUDA)
* sentencepiece
* sacrebleu
* nltk (for METEOR; download required corpora)
* tqdm
* numpy, pandas

Example `requirements.txt` snippet (create if missing):

```
torch>=1.12.0
sentencepiece
sacrebleu
nltk
tqdm
numpy
pandas
```

Install with:

```bash
python -m pip install -r requirements.txt
```

If you plan to use GPU, install the PyTorch build that matches your CUDA version from [https://pytorch.org](https://pytorch.org).

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Vaishnaviii03/Hybrid-Transformer-Neural-Machine-Translation-with-Gated-Multi-Scale-CNN-.git
cd Hybrid-Transformer-Neural-Machine-Translation-with-Gated-Multi-Scale-CNN-
```

2. Create virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate    # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

3. (Optional) Install this repo in editable mode (if you create a `setup.py` or `pyproject.toml`):

```bash
pip install -e .
```

---

## Data Preparation

This repo expects parallel corpora in `data/` with separate training, validation, and test files (plain text, one sentence per line). Example structure:

```
data/
  train.en
  train.hi
  valid.en
  valid.hi
  test.en
  test.hi
```

Tokenization: sentencepiece models for English and Hindi are provided (`spm_en.model`, `spm_hi.model`). Use them to encode data or train your own.

Example tokenization command (Python pseudo-code):

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='spm_en.model')
encoded = sp.encode('<your sentence>', out_type=str)
```

Or use the included preprocessing scripts (if added) to produce integer token files, TFRecords, or Torch `.pt` dataset objects.

---

## Training

Two high-level training options are provided:

1. Baseline Transformer models (see `Baseline Models Training.ipynb`)
2. Hybrid Transformer + GMSCNN (the Gated Multi-Scale CNN) model

### Example training CLI (pseudo)

```bash
python train.py \
  --model hybrid_gmsc \
  --data_dir data/ \
  --src_lang en --tgt_lang hi \
  --batch_size 4096 \
  --max_tokens 8192 \
  --lr 2e-4 \
  --epochs 30 \
  --save_dir checkpoints/hybrid_gmsc
```

Adjust hyperparameters (learning rate, warmup steps, dropout, label smoothing) according to your compute.

### Checkpointing and resuming

Save checkpoints frequently. Example options:

* Save every N steps or every epoch.
* Keep a `latest.pt` symlink (or JSON) for easy resuming.

Resume training:

```bash
python train.py --resume checkpoints/hybrid_gmsc/latest.pt --other-flags ...
```

The notebooks included show how to save intermediate metrics (`baseline_metrics.pt`, `cnn_metrics.pt`, etc.) and resume experiments.

---

## Evaluation

We evaluate translations using standard metrics:

* BLEU (with `sacrebleu`) — automatic, widely used.
* TER — Translation Error Rate.
* METEOR — requires `nltk` and Java for full functionality in some implementations.

Example evaluation script (pseudo):

```bash
python evaluate.py --pred translations.txt --ref data/test.hi --metrics bleu,ter,meteor
```

### Automatic metrics (BLEU, TER, METEOR)

* BLEU: use `sacrebleu.corpus_bleu()` for reproducible scoring.
* TER: `sacrebleu.corpus_ter()` or implement using `tercom` tool.
* METEOR: `nltk.translate.meteor_score` (note: nltk's METEOR implementation is available but behaves differently from the Java METEOR jar). See the `Evaluation.ipynb` for working examples.

The repository includes `Evaluation.ipynb` which demonstrates computing metrics, saving metric files (`*.pt`), and comparing multiple model outputs.

---

## Inference / Decoding

Use beam search or sampling for decoding. Example:

```bash
python decode.py --model checkpoints/hybrid_gmsc/best.pt \
  --input data/test.en --output outputs/test.pred --beam 5 --max_len 200
```

When using the SentencePiece models for postprocessing, detokenize with:

```python
sp = sentencepiece.SentencePieceProcessor(model_file='spm_hi.model')
sp.decode(token_ids)
```

---

## Reproducing Results

To reproduce the results reported in accompanying notes/notebook:

1. Prepare data exactly as in the `data/` layout.
2. Use the provided SentencePiece models or train new ones with the same vocab size.
3. Run training with the same hyperparameters and random seed.
4. Evaluate with `sacrebleu` using `--tokenize none` if you already use SentencePiece detokenized references.

The notebooks contain concrete hyperparameter examples and plots used to compare baseline vs CNN vs GMSCNN models.

---

## Files Provided

* `Baseline Models Training.ipynb` — notebook to train and compare baseline models.
* `Evaluation.ipynb` — notebook showing how to compute BLEU, TER, METEOR and load saved metrics.
* `spm_en.model`, `spm_hi.model` — SentencePiece tokenizer models.
* `*.pt` metric files — example metric checkpoints/dumps.

---

## Contributing

Contributions welcome! Suggested ways to contribute:

* Add training and inference scripts under `scripts/`.
* Provide a `requirements.txt` or `environment.yml` for reproducible environments.
* Add unit tests and CI (GitHub Actions) to validate notebooks and scripts.

Please open issues or PRs — I'll review and merge improvements.

---

## Citations

If you use this code in published work, please cite the original Transformer and any GMSCNN-related papers or sources you used. Example citation placeholders:

* Vaswani, A. et al. (2017). *Attention is All You Need*.
* (GMSCNN paper / implementation reference if applicable)

---

## License

Add a license file (`LICENSE`). A permissive option is the MIT License. Example short statement:

```
This repository is released under the MIT License. See LICENSE for details.
```

---

## Contact

Repository owner: **Vaishnaviii03**

If you need help running experiments, or want me to expand this README with precise CLI commands, example `train.py`/`decode.py` implementations, or a `requirements.txt`, tell me which pieces you'd like and I will add them.
