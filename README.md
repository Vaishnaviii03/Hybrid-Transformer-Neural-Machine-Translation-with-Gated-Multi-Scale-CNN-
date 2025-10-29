# Hybrid Transformer Neural Machine Translation with Gated Multi-Scale CNN (GMSCNN)

**Repository:** [Vaishnaviii03/Hybrid-Transformer-Neural-Machine-Translation-with-Gated-Multi-Scale-CNN-](https://github.com/Vaishnaviii03/Hybrid-Transformer-Neural-Machine-Translation-with-Gated-Multi-Scale-CNN-)

This repository contains the implementation, training notebooks, and evaluation workflows for a **Hybrid Transformer-based Neural Machine Translation (NMT)** system enhanced with a **Gated Multi-Scale Convolutional Neural Network (GMSCNN)**. The hybrid model aims to combine the long-range dependency modeling power of Transformers with the local feature extraction capability of CNNs — improving translation accuracy, fluency, and robustness, especially for **morphologically rich and low-resource languages** like Hindi.

---

## 🧠 Overview

The proposed model integrates a **Transformer encoder-decoder** with **Gated Multi-Scale Convolutional (GMSC)** blocks that operate at multiple receptive fields. These GMSC modules are inserted within encoder and decoder layers to enhance the contextual representation, capturing both global dependencies and local linguistic patterns.

The model is trained and evaluated on the **AI4Bharat SAMANANTAR dataset** — a high-quality parallel corpus for English–Indic language translation, specifically **English ↔ Hindi** in this work.

📘 **Research Context:**
This implementation is part of a research project focusing on improving neural translation systems for Indian languages. You can cite this repository in your paper as the official codebase for experimental replication.

---

## 🧩 Dataset — SAMANANTAR

The **SAMANANTAR Dataset** is a large-scale parallel corpus for English ↔ Indic languages, released by **AI4Bharat**.
It provides millions of high-quality translation pairs across 11 Indian languages.

For this research, the **English–Hindi subset** was used.

**Dataset Link:** 🔗 [https://ai4bharat.iitm.ac.in/samanantar/](https://ai4bharat.iitm.ac.in/samanantar/)

### Dataset Statistics (EN–HI)

| Split | Sentence Pairs |
| ----- | -------------- |
| Train | ~1.4M          |
| Valid | ~10K           |
| Test  | ~10K           |

**License:** The dataset is open for research use under the terms provided by AI4Bharat.

---

## 📁 Repository Structure

```
README.md
Baseline Models Training.ipynb     # Transformer and CNN baseline training
Final_model.ipynb                  # Hybrid Transformer + GMSCNN final model
Evaluation.ipynb                   # BLEU, TER, METEOR evaluation
baseline_metrics.pt
cnn_metrics.pt
multiscale_metrics.pt
metrics_gmsc_greedy.pt
spm_en.model / spm_hi.model        # SentencePiece tokenizers
spm_en.vocab / spm_hi.vocab
```

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/Vaishnaviii03/Hybrid-Transformer-Neural-Machine-Translation-with-Gated-Multi-Scale-CNN-.git
cd Hybrid-Transformer-Neural-Machine-Translation-with-Gated-Multi-Scale-CNN-

python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

pip install -r requirements.txt
```

**Requirements (summary):**

```
torch>=1.12
sentencepiece
sacrebleu
nltk
tqdm
numpy
pandas
```

---

## 🏋️‍♀️ Training

Three main model configurations are provided via Jupyter notebooks:

### 1️⃣ Baseline Transformer

Implemented using standard Transformer encoder-decoder with positional encoding and attention.

### 2️⃣ CNN-Enhanced Transformer

Adds a single-scale CNN before attention layers to capture local dependencies.

### 3️⃣ Hybrid Transformer + Gated Multi-Scale CNN (GMSCNN)

Introduces **multi-scale convolutional filters** with **learnable gating** to dynamically weigh representations from different kernel sizes. This leads to better handling of compositional and morphological complexity.

#### Training Example

```python
!python train.py \
  --model hybrid_gmsc \
  --data_dir data/ \
  --src_lang en --tgt_lang hi \
  --epochs 30 --batch_size 4096 \
  --lr 2e-4 --save_dir checkpoints/hybrid_gmsc
```

#### Checkpointing & Resuming

The notebooks implement automatic checkpointing every **10k sentences** with the option to resume training seamlessly:

```python
checkpoint = torch.load('checkpoints/hybrid_gmsc/step_10000.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 📊 Evaluation

Evaluation metrics are computed using **BLEU**, **TER**, and **METEOR** scores on the test set.

### Example Code

```python
from nltk.translate.meteor_score import meteor_score
from sacrebleu import corpus_bleu, corpus_ter

refs = [open('data/test.hi').read().splitlines()]
preds = open('outputs/test_pred.txt').read().splitlines()

print('BLEU:', corpus_bleu(preds, refs).score)
print('TER:', corpus_ter(preds, refs).score)
print('METEOR:', sum(meteor_score([r], p) for r, p in zip(refs[0], preds)) / len(preds))
```

**Evaluation Notebook:** `Evaluation.ipynb` includes ready-to-run code for:

* Computing BLEU, TER, METEOR
* Comparing baseline and hybrid performance
* Visualizing improvement trends

---

## 🔍 Results Summary

| Model                    | BLEU ↑   | TER ↓    | METEOR ↑ |
| ------------------------ | -------- | -------- | -------- |
| Transformer Baseline     | 24.6     | 61.2     | 0.48     |
| CNN-Enhanced Transformer | 26.1     | 58.9     | 0.51     |
| Hybrid GMSCNN (Proposed) | **28.7** | **55.3** | **0.55** |

*↑ Higher is better, ↓ Lower is better.*

The GMSCNN-based hybrid model achieved consistent improvement across all metrics, demonstrating its ability to capture multi-scale linguistic nuances.

---

## 📜 Citation

If you use this work in your research or publication, please cite the dataset and this repository:

```bibtex
@inproceedings{vaishnavi2025gmscnn,
  title={Hybrid Transformer Neural Machine Translation with Gated Multi-Scale CNN},
  author={Vaishnavi Pandey},
  year={2025},
  howpublished={GitHub repository},
  url={https://github.com/Vaishnaviii03/Hybrid-Transformer-Neural-Machine-Translation-with-Gated-Multi-Scale-CNN-}
}

@article{ramesh2021samanantar,
  title={Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages},
  author={Ramesh, G and others},
  journal={arXiv preprint arXiv:2104.05596},
  year={2021}
}
```

---

## 🧾 License

This repository is released under the **MIT License**.
See [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

**Vaishnavi Pandey**
Hybrid Transformer NMT Research
📧 Contact: [via GitHub Issues](https://github.com/Vaishnaviii03)

---

## 🌐 References

* Vaswani et al. (2017), *Attention is All You Need.*
* AI4Bharat (2021), *Samanantar Dataset: English–Indic Parallel Corpora.*
* CNN/Transformer hybridization techniques in neural machine translation literature.

---

### 🧩 Future Work

* Extend to multilingual NMT using IndicBERT-style shared embeddings.
* Add contrastive loss fine-tuning for robustness.
* Deploy model for real-time inference with TorchScript or ONNX export.

---

