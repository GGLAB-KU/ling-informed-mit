# Linguistically-Informed Multilingual Instruction Tuning: Is There an Optimal Set of Languages to Tune?

This repository contains the official implementation for **Linguistically-Informed Multilingual Instruction Tuning: Is
There an Optimal Set of Languages to Tune?**

## Table of Contents

1. [Abstract](#abstract)
2. [Multilingual Instruction Tuning Dataset](#multilingual-instruction-tuning-dataset)
3. [Language Selection Algorithm](#language-selection-algorithm)
4. [Running the Experiments](#running-the-experiments)
    - Main Job Script
    - Experiment Sets
    - Running Individual Experiments
5. [Notebooks and Formatted Results](#notebooks-and-formatted-results)
6. [Raw Experiment Logs](#raw-experiment-logs)
7. [Setup Instructions - Creating the Conda Environment and Installing Dependencies](#setup-instructions---creating-the-conda-environment-and-installing-dependencies)
8. [.env File](#env-file)
9. [Acknowledgements](#acknowledgements)

### Abstract

Multilingual language models often show uneven performance across different languages due to limited generalization
capabilities for certain languages. This poses a challenge, especially given the growing interest in creating universal
language models that perform well across all languages. Instruction tuning with multilingual instruction-response pairs
has been shown to enhance model performance across various languages. However, this approach is constrained by high
computational costs, a lack of high-quality tuning data for all languages, and the "curse of multilinguality" — a
performance drop per language as the number of languages increases.

Recent studies suggest that working with datasets containing fewer languages and fewer instances can be beneficial.
However, no systematic investigation exists on how language choice affects multilingual instruction tuning. This study
proposes a method to select languages for instruction tuning based on linguistic features, aiming to improve performance
across languages and tasks. We employ a simple algorithm to select diverse languages and evaluate their effectiveness on
multiple benchmarks. Our results demonstrate that this strategic selection often yields better outcomes than random
selection, offering a straightforward way to improve multilingual models by choosing languages based on linguistic
features, thus guiding better dataset creation and model development.

---

### Multilingual Instruction Tuning Dataset

We use [Bactrian-X](https://github.com/mbzuai-nlp/bactrian-x), which is one of the most comprehensive multilingual
instruction datasets to date. This dataset is further enriched by translated English instructions and responses
generated by ChatGPT.

### Language Selection Algorithm

Our language selection algorithm is outlined in "Algorithm 1" in our paper. It is based on k-means clustering of
linguistic features, some of which are obtained from [Lang2Vec](https://github.com/antonisa/lang2vec). More details can
be found in "Section 3.1: Language Selection" of our paper.

Key files for this algorithm are:

- `src/notebooks_results_lang_selection/lang_to_tune.py`: contains two main functions
    - `lang_selection_for_main_experiments()`: creates language subsets of 14 based on various linguistic features.
    - `varying_number_langs_for_geo()`: uses geographical features to create varying numbers of language subsets.

### Running the Experiments

# TODO: @gsoykan - delete this disclaimer...
Note that some folder names, such as `rebuttal` and `journal_submission`, may be misleading due to the dependency of our
scripts on these names. If you wish to rename them, ensure that the related run scripts are also updated.

For running the experiments, you can find Python scripts and `.sh` files under `src/instruction_tuning`. We rely on a
SLURM cluster, and job scripts are provided to help you get started.

- **Main job script**: `src/instruction_tuning/scripts/sbatch_finetune_template.sh`. This script is currently specific
  to Koç University’s cluster and our directory structure, so make sure to update it according to your cluster setup.

Three sets of experiments are available under `src/instruction_tuning/scripts`:

- **analysis**: Contains scripts for "Section 7: Analysis and Discussion" (excluding Section 7.3).
- **geo_varying_langs**: Contains scripts for "Section 7.3: Effect of Varying Number of Languages."
- **journal_submission**: Contains scripts for "Section 6: Results," covering our main results.

Each of these directories includes a `run_replications.py` script that calls the relevant experiment script (e.g.,
`run_experiment_with_template_{exp_type}.py`).

If you want to run specific experiments individually, you can find them under each model directory (e.g.,
`src/instruction_tuning/scripts/journal_submission/bloom3b/finetune_lang_selection_typo_1.sh` runs the Bloom 3B model
with a subset based on the "Typological Feature Vector (TYPO)").

### Notebooks and Formatted Results

- `src/notebooks_results_lang_selection/language_subsets.ipynb`: Code for generating the Language x (Task + Model +
  Subset) Matrix and preliminary feature extraction examples.
- `src/notebooks_results_lang_selection/result_displayer.ipynb`: Code for generating formatted results and plots used in
  the paper.
- `src/notebooks_results_lang_selection/result_displayer_multiple_seeds_geo_varying_langs.ipynb`: Generates results and
  plots for "Section 7.3: Effect of Varying Number of Languages."

- Results from the above notebooks are saved as `.csv` files.

### Raw Experiment Logs

Raw experiment logs can be found under `./data/exp_logs`. This directory contains:

- `./data/exp_logs/additional_exp_logs`: Logs for "Section 7: Analysis and Discussion" (excluding Section 7.3).
- `./data/exp_logs/geo_varying_logs`: Logs for "Section 7.3: Effect of Varying Number of Languages."
- `./data/exp_logs/journal_exp_logs`: Logs for "Section 6: Results" (main results).

The logs include `.log` files for our jobs on the SLURM cluster (Koç University’s HPC cluster). `lm-evaluation-harness`
results are appended at the bottom of these logs in the following format:

```
|      Tasks      |Version|Filter|n-shot|Metric|Value |   |Stderr|
|-----------------|-------|------|-----:|------|-----:|---|-----:|
|pawsx            |N/A    |none  |     0|acc   |0.5106|±  |0.0265|
| - paws_de       |      0|none  |     0|acc   |0.4830|±  |0.0112|
| - paws_en       |      0|none  |     0|acc   |0.4750|±  |0.0112|
| - paws_es       |      0|none  |     0|acc   |0.4720|±  |0.0112|
| - paws_fr       |      0|none  |     0|acc   |0.5420|±  |0.0111|
| - paws_ja       |      0|none  |     0|acc   |0.5415|±  |0.0111|
| - paws_ko       |      0|none  |     0|acc   |0.5450|±  |0.0111|
| - paws_zh       |      0|none  |     0|acc   |0.5155|±  |0.0112|
|xcopa            |N/A    |none  |     0|acc   |0.5609|±  |0.0567|
| - xcopa_et      |      1|none  |     0|acc   |0.4820|±  |0.0224|
| - xcopa_ht      |      1|none  |     0|acc   |0.5040|±  |0.0224|
| - xcopa_id      |      1|none  |     0|acc   |0.6740|±  |0.0210|
| - xcopa_it      |      1|none  |     0|acc   |0.5160|±  |0.0224|
| - xcopa_qu      |      1|none  |     0|acc   |0.5060|±  |0.0224|
| - xcopa_sw      |      1|none  |     0|acc   |0.5340|±  |0.0223|
| - xcopa_ta      |      1|none  |     0|acc   |0.5440|±  |0.0223|
| - xcopa_th      |      1|none  |     0|acc   |0.5420|±  |0.0223|
| - xcopa_tr      |      1|none  |     0|acc   |0.5560|±  |0.0222|
| - xcopa_vi      |      1|none  |     0|acc   |0.6720|±  |0.0210|
| - xcopa_zh      |      1|none  |     0|acc   |0.6400|±  |0.0215|
|xnli             |N/A    |none  |     0|acc   |0.4005|±  |0.0491|
| - xnli_ar       |      1|none  |     0|acc   |0.3345|±  |0.0095|
| - xnli_bg       |      1|none  |     0|acc   |0.3663|±  |0.0097|
| - xnli_de       |      1|none  |     0|acc   |0.3863|±  |0.0098|
| - xnli_el       |      1|none  |     0|acc   |0.3466|±  |0.0095|
| - xnli_en       |      1|none  |     0|acc   |0.5309|±  |0.0100|
| - xnli_es       |      1|none  |     0|acc   |0.4751|±  |0.0100|
| - xnli_fr       |      1|none  |     0|acc   |0.4667|±  |0.0100|
| - xnli_hi       |      1|none  |     0|acc   |0.4382|±  |0.0099|
| - xnli_ru       |      1|none  |     0|acc   |0.3936|±  |0.0098|
| - xnli_sw       |      1|none  |     0|acc   |0.3442|±  |0.0095|
| - xnli_th       |      1|none  |     0|acc   |0.3353|±  |0.0095|
| - xnli_tr       |      1|none  |     0|acc   |0.3361|±  |0.0095|
| - xnli_ur       |      1|none  |     0|acc   |0.3932|±  |0.0098|
| - xnli_vi       |      1|none  |     0|acc   |0.4446|±  |0.0100|
| - xnli_zh       |      1|none  |     0|acc   |0.4161|±  |0.0099|
|xstorycloze      |N/A    |none  |     0|acc   |0.5803|±  |0.0535|
| - xstorycloze_ar|      1|none  |     0|acc   |0.5725|±  |0.0127|
| - xstorycloze_en|      1|none  |     0|acc   |0.6876|±  |0.0119|
| - xstorycloze_es|      1|none  |     0|acc   |0.6479|±  |0.0123|
| - xstorycloze_eu|      1|none  |     0|acc   |0.5606|±  |0.0128|
| - xstorycloze_hi|      1|none  |     0|acc   |0.5817|±  |0.0127|
| - xstorycloze_id|      1|none  |     0|acc   |0.6406|±  |0.0123|
| - xstorycloze_my|      1|none  |     0|acc   |0.4752|±  |0.0129|
| - xstorycloze_ru|      1|none  |     0|acc   |0.5050|±  |0.0129|
| - xstorycloze_sw|      1|none  |     0|acc   |0.5122|±  |0.0129|
| - xstorycloze_te|      1|none  |     0|acc   |0.5685|±  |0.0127|
| - xstorycloze_zh|      1|none  |     0|acc   |0.6314|±  |0.0124|
|xwinograd        |N/A    |none  |     0|acc   |0.6786|±  |0.0698|
| - xwinograd_en  |      1|none  |     0|acc   |0.7497|±  |0.0090|
| - xwinograd_fr  |      1|none  |     0|acc   |0.6386|±  |0.0531|
| - xwinograd_jp  |      1|none  |     0|acc   |0.5610|±  |0.0160|
| - xwinograd_pt  |      1|none  |     0|acc   |0.6654|±  |0.0292|
| - xwinograd_ru  |      1|none  |     0|acc   |0.5238|±  |0.0282|
| - xwinograd_zh  |      1|none  |     0|acc   |0.6845|±  |0.0207|

|  Groups   |Version|Filter|n-shot|Metric|Value |   |Stderr|
|-----------|-------|------|-----:|------|-----:|---|-----:|
|pawsx      |N/A    |none  |     0|acc   |0.5106|±  |0.0265|
|xcopa      |N/A    |none  |     0|acc   |0.5609|±  |0.0567|
|xnli       |N/A    |none  |     0|acc   |0.4005|±  |0.0491|
|xstorycloze|N/A    |none  |     0|acc   |0.5803|±  |0.0535|
|xwinograd  |N/A    |none  |     0|acc   |0.6786|±  |0.0698|
```

### Setup Instructions - Creating the Conda Environment and Installing Dependencies

Note: Please check the `requirements.txt` for any additional notes regarding package installation. You may need to
install `peft` and `lm-evaluation-harness` from PyPI via pip if required.
In that case you need to update the `requirements.txt` file.

1. **Create a Conda environment**:
   ```bash
   conda create --name <env_name> python=3.10
   ```
   Replace `<env_name>` with your desired environment name.

2. **Activate the Conda environment**:
   ```bash
   conda activate <env_name>
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### .env File

You are expected to create a `.env` file. An example is provided in `.env.example`, which includes environment variables
for WANDB project names and more.

### Acknowledgements

- Supported by the [Scientific and Technological Research Council of Türkiye (TÜBİTAK)](https://www.tubitak.gov.tr/) as
  part of the project *Automatic Learning of Procedural Language from Natural Language Instructions for Intelligent
  Assistance* (Project No: 121C132).
- Special thanks to [KUIS AI Lab](https://ai.ku.edu.tr/) for providing computational support.
- We are grateful to our anonymous reviewers and the members of GGLab for their valuable feedback on this paper.

