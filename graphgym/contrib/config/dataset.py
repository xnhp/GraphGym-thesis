from yacs.config import CfgNode as CN

from graphgym.register import register_config


def set_cfg_dataset(cfg):

    # If True, ignore all complex species during reading of the dataset
    # This means these species will not appear in the graph.
    cfg.dataset.exclude_complex_species = False


register_config('dataset_config', set_cfg_dataset)
