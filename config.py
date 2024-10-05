import ast

_VALID_DATASETS = {"fashion-mnist", "bios", "ravdess", "facet"}
_LOADER_KW = {
    "train_batch_size",
    "valid_batch_size",
    "test_batch_size",
    "calib_batch_size",
    "calib_val_batch_size",
    "n_calib",
    "n_test",
    "n_calib_val",  # HPO for conformal
    "n_train",
    "n_val",
    "m",
    "model_checkpoint",
    "save_model_ckpt",
}
_MODEL_KW = {
    "optimizer",
    "lr",
    "epochs",
    "model_size",
    "model_checkpoint",
    "save_model_ckpt",
}


def get_config(dataset):
    dataset = dataset.lower()
    assert dataset in _VALID_DATASETS, f"Unknown dataset {dataset}"

    base_config = CFG_MAP["base"]()

    dataset_config = CFG_MAP[dataset]()

    return {
        "dataset": dataset,
        **base_config,
        **dataset_config,  # dataset unpacked last, so overwrites base if there are duplicates
    }


def get_base_config():
    # Shared config applicable to all datasets
    cfg = {
        "seed": 0,  # random seed for reproducibility
        "alpha": 0.1,  # conformal error tolerance rate
        "test_batch_size": 256,
        "calib_batch_size": 256,
        "m": 10,
        "data_root": "data/",
        "logdir_root": "logs/",
        "k": 3,
        "score_fn": "raps",  # hinge, raps, saps
        "hpo_iterations": 50,
        "verbose": False,
        "model_checkpoint": None,
        "save_model_ckpt": False,
    }

    return cfg


def get_fmnist_config():
    # Add dataset-specific config parameters as required
    cfg = {
        "train_batch_size": 256,
        "valid_batch_size": 256,
        "n_calib": 2000,
        "n_test": 8000,
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 2,
        "h_params_raps": {
            "T": 1.0,
            "kreg": 3,
            "lamda": 0.5,
        },
        "h_params_saps": {
            "T": 1.0,
            "lamda": 0.5,
        },
        "h_params_hinge": {"T": 1.0},
    }

    return cfg

def get_ravdess_config():
    # Add dataset-specific config parameters as required
    cfg = {
        "alpha": 0.1,
        "n_calib": 0.5833333,
        "n_calib_val": 0.16666666,
        "n_test": 0.25,
        "test_batch_size": 64,
        "calib_val_batch_size": 64,
        "calib_batch_size": 64,
        "m": 8,
        "data_root": "data/RAVDESS/",
        "h_params_raps": {
            "T": 0.5,
            "kreg": 2,
            "lamda": 2.0,
        },
        "h_params_saps": {
            "T": 1.0,
            "lamda": 0.5,
        },
        "h_params_hinge": {"T": 1.0},
    }
    return cfg


def get_biosbias_config():
    # Add dataset-specific config parameters as required
    cfg = {
        "train_batch_size": 512,
        "test_batch_size": 512,
        "calib_batch_size": 512,
        "n_calib": 10000,
        "n_test": 2000,
        "n_calib_val": 5000,
        "n_train": 50000,
        "n_val": 5000,
        "data_root": "data/BiosBias/",
        "optimizer": "adam",
        "score_fn": "saps",
        "lr": 0.005,
        "epochs": 15,
        "h_params_raps": {
            "T": 0.2,
            "kreg": 3,
            "lamda": 0.5,
        },
        "h_params_saps": {
            "T": 1.0,
            "lamda": 0.5,
        },
        "h_params_hinge": {"T": 1.0},
    }

    return cfg

def get_facet_config():
    # Add dataset-specific config parameters as required
    cfg = {
        "n_calib": 4000,
        "n_calib_val": -1,  # use half of remaining examples for HPO
        "n_test": -1,  # use other half of remaining examples for test
        "m": 20,
        "data_root": "data/facet/",
        "model_size": "ViT-L/14",
        "h_params_raps": {
            "T": 1.0,
            "kreg": 3,
            "lamda": 0.5,
        },
        "h_params_saps": {
            "T": 1.0,
            "lamda": 0.5,
        },
        "h_params_hinge": {
            "T": 1.0
        }
    }
    return cfg


CFG_MAP = {
    "base": get_base_config,
    "fashion-mnist": get_fmnist_config,
    "ravdess": get_ravdess_config,
    "bios": get_biosbias_config,
    "facet": get_facet_config,
}


def parse_config_arg(key_value):
    assert "=" in key_value, "Must specify config items with format `key=value`"

    k, v = key_value.split("=", maxsplit=1)

    assert k, "Config item can't have empty key"
    assert v, "Config item can't have empty value"

    try:
        v = ast.literal_eval(v)
    except ValueError:
        v = str(v)

    return k, v
