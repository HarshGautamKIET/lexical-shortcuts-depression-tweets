# Lexical Shortcuts in Machine-Learning Depression Detection from Tweets: A Residual-Vocabulary Audit

Code for the paper by Harsh Gautam, Aditya Kumar, Raoshan Kumar Verma, Aashwit
Gupta and Deepinder Kaur (Department of Computer Science and Engineering
(AI&ML), KIET Group of Institutions, Ghaziabad, India).

`notebooks/shortcut_audit.ipynb` runs every experiment in the paper. It masks
the vocabulary used to retrieve the tweets in four tiers (M0 to M3), evaluates
seven classical classifiers on binary screening and six-class subtype
classification with stratified 5-fold cross-validation, and writes every table
and figure. No deep learning is used.

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

## Licence

The code is released under the MIT licence (see `LICENSE`). The dataset
belongs to its creators and is licensed CC BY 4.0.
