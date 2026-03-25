# NLU_Assignment2

**CSL 7640: Natural Language Understanding — Programming Assignment 2**
Indian Institute of Technology Jodhpur

> **Student:** Arjun Baidya (M25CSA006)
> **Course Instructor:** Dr. Anand Mishra

---

## Repository Structure

```
NLU_Assignment2/
├── PA2_Q1_M25CSA006.ipynb   # Problem 1: Word Embeddings
├── PA2_Q2_M25CSA006.ipynb   # Problem 2: Character-level Name Generation
├── corpus.txt               # IIT Jodhpur corpus (Problem 1)
├── TrainingNames.txt        # Indian names dataset (Problem 2)
└── README.md
```

---

## Problem 1: Learning Word Embeddings from IIT Jodhpur Data

**Notebook:** `PA2_Q1_M25CSA006.ipynb`
**Data file:** `corpus.txt` (place in the same directory as the notebook)

### Overview

This notebook builds Word2Vec embeddings from scratch using text scraped from IIT Jodhpur's academic portals, departmental pages, and faculty profiles. Both CBOW and Skip-gram with Negative Sampling (SG-NS) are implemented in pure Python/NumPy and benchmarked against Gensim.

### Tasks

| Task | Description |
|------|-------------|
| Task 1 | Corpus preparation, preprocessing pipeline, and frequency analysis |
| Task 2 | Hyperparameter grid search (54 runs: 3 dims × 3 windows × 3 neg-samples × 2 models) |
| Task 3 | Semantic analysis — nearest neighbours, analogies, Spearman ρ comparison |
| Task 4 | PCA visualisation of word embeddings with cluster labelling |

### How to Run

1. Make sure `corpus.txt` is in the **same directory** as `PA2_Q1_M25CSA006.ipynb`.
2. Install dependencies:
   ```bash
   pip install numpy gensim matplotlib scikit-learn wordcloud
   ```
3. Open and run the notebook cell by cell:
   ```bash
   jupyter notebook PA2_Q1_M25CSA006.ipynb
   ```

### Key Results

- Best CBOW config: `dim=300, win=2, neg_k=5` → final loss **2.57**
- Best SG-NS config: `dim=300, win=6, neg_k=5` → final loss **1.24**
- Gensim is ~40–160× faster than the scratch implementation
- Scratch SG-NS achieves Spearman ρ = **+0.84** vs. Gensim SG-NS (strong geometry agreement)

---

## Problem 2: Character-Level Name Generation using RNN Variants

**Notebook:** `PA2_Q2_M25CSA006.ipynb`
**Data file:** `TrainingNames.txt` (place in the same directory as the notebook)

### Overview

This notebook trains three sequence models from scratch in Python/NumPy for character-level Indian name generation. All models use the Adam optimiser and are evaluated on novelty, diversity, and qualitative realism.

### Models

| Model | Parameters | Final Loss |
|-------|-----------|-----------|
| Vanilla RNN | 94,007 | 9.68 |
| Bidirectional LSTM (BLSTM) | 667,191 | 5.77 |
| RNN with Bahdanau Attention | 370,743 | 3.43 |

### How to Run

1. Make sure `TrainingNames.txt` is in the **same directory** as `PA2_Q2_M25CSA006.ipynb`.
2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook PA2_Q2_M25CSA006.ipynb
   ```

### Key Results

| Model | Novelty | Diversity | Realism |
|-------|---------|-----------|---------|
| Vanilla RNN | 49.5% | 84.5% | High |
| BLSTM | 43.0% | 80.5% | High |
| RNN + Attention | 100.0% | 63.0% | Low |

The Vanilla RNN and BLSTM produce culturally plausible Indian names despite their simplicity. The Attention model achieves the best training loss but suffers from mode collapse on this small dataset (1,000 names), repeatedly generating nonsensical names starting with *Vikra-*. This highlights that training loss alone is a poor proxy for generation quality.

---

## Dependencies

```
numpy
matplotlib
scikit-learn
gensim
wordcloud
jupyter
```

Install all at once:

```bash
pip install numpy matplotlib scikit-learn gensim wordcloud jupyter
```

---

## Notes

- All models are implemented **from scratch** in Python/NumPy — no PyTorch or TensorFlow required.
- The corpus for Problem 1 was scraped from IIT Jodhpur's academic regulation documents, departmental pages, and faculty profile pages.
- The training names for Problem 2 were generated using a large language model and consist of 1,000 Indian names.
