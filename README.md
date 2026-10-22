# Lexical Shortcuts in Machine-Learning Depression Detection from Tweets: A Residual-Vocabulary Audit

Code for the paper by Harsh Gautam, Aditya Kumar, Raoshan Kumar Verma, Aashwit
Gupta and Deepinder Kaur (Department of Computer Science and Engineering
(AI&ML), KIET Deemed to be University, Ghaziabad, India).

`notebooks/shortcut_audit.ipynb` runs every experiment in the paper. It masks
the vocabulary used to retrieve the tweets in four tiers (M0 to M3), evaluates
seven classical classifiers on binary screening and six-class subtype
classification with stratified 5-fold cross-validation, and writes every table
and every result figure. `diagrams/make_diagrams.py` draws the two diagrams.
No deep learning is used.

## Data

The Multi-Class Depression Detection Dataset (Nusrat, Shahzad and Jamal, 2024)
is not redistributed here. Download it from Zenodo,
<https://doi.org/10.5281/zenodo.14233292> (CC BY 4.0), and save the CSV as
`train.csv`.

## Running

**Google Colab.** Put `train.csv` in `MyDrive/Major_Project/data/`, open the
notebook in Colab and choose Runtime > Run all. Results are written to
`MyDrive/Major_Project/results/`.

**Locally.** Put `train.csv` in `data/` at the repository root, install the
requirements, and run the notebook from the `notebooks/` folder:

```
pip install -r requirements.txt
jupyter notebook notebooks/shortcut_audit.ipynb
```

Each model configuration is fitted once and its out-of-fold predictions are
stored in `results/checkpoints/`, so an interrupted run resumes where it
stopped. Setting `SMOKE_N = 2000` in the configuration cell rehearses the
whole notebook on a stratified subsample in minutes.

## Figures

The notebook saves each result figure as PDF (vector) and PNG in
`results/figures/`:

| File | Content |
|---|---|
| `V1_corpus_and_leakage` | class sizes, retrieval-vocabulary leakage, share masked by each tier |
| `V2_confusion_M0_M2_M3` | subtype confusion matrices, unmasked, M2 and M3 |
| `V3_leakage_vs_fragility` | per-subtype leakage against F1 lost at M2 and M3 |
| `V4_headline_summary` | best configuration per tier and the v2 recovery |
| `V5_top_features_*` | largest Logistic Regression coefficients for one class |
| `V6_all_models` | macro-F1 of every classifier at every tier |
| `V7_roc_pr_binary`, `V8_roc_pr_subtype` | ROC and precision-recall curves |
| `V9_confusion_*` | confusion matrices at every tier |
| `V10_metric_heatmap_*` | per-class precision, recall and F1 by tier |
| `V11_protocol_comparison` | P-sym against P-dep at M2 and M3 |
| `V12_gain_decomposition` | gain from class weighting and from the v2 features |
| `V13_robustness` | masking cost under default C, tuned C and repeated folds |
| `V14_length_control` | binary macro-F1 on all tweets and on a length-matched subsample |

`diagrams/make_diagrams.py` needs only matplotlib and no data. It draws the
study overview (`D1_overview`) and the system architecture (`D2_architecture`)
into `diagrams/figures/`:

```
python diagrams/make_diagrams.py
```

## Licence

The code is released under the MIT licence (see `LICENSE`). The dataset
belongs to its creators and is licensed CC BY 4.0.
