from yacs.config import CfgNode as CN

from graphgym.register import register_config


def set_cfg_dataset(cfg):

    # If True, ignore all complex species during reading of the dataset
    # This means these species will not appear in the graph.
    cfg.dataset.exclude_complex_species = False

    # also see SBMLModel.min_node_degree

    # keys of features to normalise. These can also be features determined by GG`s feature augments.
    # note these usually begin with `node_`
    cfg.dataset.normalize_feats = []


    # Whether to save/load feature augment data to/from disk or recompute.
    # expected values:
    # - 'use_and_update': use data from cache if available; update cache if recomputed
    # - 'update_always' : always recompute and put into cache
    # - 'disabled'      : always recompute, do not touch cache at all
    cfg.dataset.feat_cache = 'use_and_update'

    # Possible species classes. This needs to be fixed beforehand because we include
    #   (one-hot) encodings of these classes as features. We cannot infer these possible values
    #   from the input data since we may supply multiple graphs and we cannot depend on the range
    #   or order or consistency of values they provide.
    cfg.dataset.possible_classes = ['PROTEIN', 'reaction', 'RNA', 'DEGRADED', 'UNKNOWN', 'SIMPLE_MOLECULE', 'ION',
                                    'GENE', 'PHENOTYPE', 'DRUG']


register_config('dataset_config', set_cfg_dataset)
