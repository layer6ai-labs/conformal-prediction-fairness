<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# Fairness of Conformal Prediction
This is the codebase accompanying the paper ["Conformal Prediction Sets Can Cause Disparate Impact"](https://arxiv.org/abs/2410.01888). 

## Environement Setup

The main prerequisite is to set up the python environment.

    conda env create -f environment.yaml
    conda activate conformal

## Dataset
We have implemented `fashion-mnist`, `bios`, `ravdess`, and `facet`. To prepare the datasets:

    mkdir data
    cd data
    mkdir BiosBias
    mkdir RAVDESS
    mkdir facet

For `bios`, we preprocessed the BiosBias data from [this repo](https://github.com/shauli-ravfogel/nullspace_projection) by encoding it with [bert-uncased model](https://huggingface.co/google-bert/bert-base-uncased). The data pkl files can be downloaded from [Google drive](https://drive.google.com/drive/folders/1TW6lFZCxuUPzy3A42_MSfHEWSwRP9zYP?usp=drive_link). After downloading, put it inside [data/BiosBias](data/BiosBias).

For `ravdess`, please download the dataset [here](https://zenodo.org/records/1188976) and put it inside [data/RAVDESS](data/RAVDESS).

For `facet`, please download the dataset [here](https://facet.metademolab.com/), unzip, and put it inside [data/facet](data/facet) with a folder structure:

    --images/
        ----imgs_1/
        ----imgs_2/
        ----imgs_3/

    --annotaions/
## Usage - `main.py`

The main script for creating datasets is unsurprisingly `main.py`.
This script loads raw datasets, splits them, loads or trains a model, performs conformal calibration, and generates conformal prediction sets for the test set data points.

The basic usage is as follows:

    python main.py --dataset <dataset>

where `<dataset>` is the desired dataset. We have implemented `fashion-mnist`, `bios`, `facet`, and `ravdess`.

### Dynamic Updating of Config Values

Dataset and calibration hyperparameters are loaded from the `config.py` file at runtime.
However, it is also possible to update the hyperparameters on the command line using the flag `--config`.
For each hyperparameter `<key>` that one wants to set to a new value `<value>`, add the following to the command line:

    --config <key>=<value>

This can be done multiple times for multiple keys. A full list of config values is visible in the `config.py` file.

### Run Directories

By default, the `main` command above will create a directory of the form `logs/<date>_<hh>-<mm>-<ss>`, e.g. `Apr26_09-38-37`, to store information about the run, including:

- Config files as `json`
- Experiment metrics / results as `json`
- `stderr` / `stdout` logs
- Output csvs containing conformal prediction sets for the test data

# Citing

    @article{cresswell2024conformal,
        title={Conformal Prediction Sets Can Cause Disparate Impact}, 
        author={Jesse C. Cresswell and Bhargava Kumar and Yi Sui and Mouloud Belbahri},
        year={2024},
        eprint={2410.01888},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2410.01888}, 
    }

# License
This data and code is licensed under the MIT License, copyright by Layer 6 AI.
