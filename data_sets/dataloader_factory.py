from config import _VALID_DATASETS, _LOADER_KW
from data_sets.data_sets import *


DATASET_CLASS_MAP = {
    "fashion-mnist": Fashion_MNIST,
    "ravdess": RAVDESS,
    "bios": BiosBias,
    "facet": FACET,
}


def get_loaders(cfg):
    dataset = cfg["dataset"]
    data_root = cfg["data_root"]
    assert dataset in _VALID_DATASETS, f"Unknown dataset {dataset}"

    loader_kwargs = get_loader_kwargs(cfg)

    dataset_class = DATASET_CLASS_MAP[dataset]()

    output_dict = dataset_class.get_data(data_root, **loader_kwargs)
    
    if not hasattr(dataset_class, 'group_conformal_category'):
        raise ValueError(f"group_conformal_category must be specified in {dataset} dataset class")

    return dataset_class, output_dict


def get_loader_kwargs(cfg):
    # Create subdict of cfg for the keys relevant to dataloading
    loader_kw = _LOADER_KW
    loader_kwargs = {k: cfg[k] for k in cfg.keys() & loader_kw}

    return loader_kwargs
