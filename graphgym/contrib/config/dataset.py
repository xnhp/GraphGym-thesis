from yacs.config import CfgNode as CN

from graphgym.register import register_config

from data.util import SpeciesClass


def set_cfg_dataset(cfg):
    # If true, complex species will be excluded from classification (but still appear
    #   in the input graph relevant for feature computation, message-passing etc).
    cfg.dataset.exclude_complex_species = True

    # If true, low-degree nodes will be excluded from classification (but still appear
    #   in the input graph relevant for feature computation, message-passing etc).
    # ↝ SBMLModel.min_node_degree for the value used
    cfg.dataset.exclude_low_degree = True

    # keys of features to normalise. These can also be features determined by GG`s feature augments.
    # note these usually begin with `node_`.
    # If such a feature is not present (in all graphs) in the dataset, nothing happens
    # ↝ graphgym.contrib.transform.normalize.normalize_scale
    # ↝ graphgym.contrib.transform.normalize.normalize_fit
    cfg.dataset.normalize_feats = [
        "node_degree",
        "node_degree_projection",
        "node_clustering_coefficient",
        "node_betweenness_centrality",
        "node_betweenness_centrality_projection",
        "node_ego_centralities",
        "node_ego_centralities_projection",
        "node_closeness_centrality",
        "node_closeness_centrality_projection",
        "node_eigenvector_centrality",
        "node_eigenvector_centrality_projection",
        "node_neighbour_centrality_statistics",
        "node_neighbour_centrality_statistics_projection",
        "node_distance_set_size",
        "node_distance_set_size_projection"
    ]

    # Names (in the sense of `util.get_dataset`) of datasets to use for training and testing resp.
    # A train/test split *within* the separate graphs is still possible
    cfg.dataset.train_names = []
    cfg.dataset.test_names = []

    # Whether to save/load feature augment data to/from disk or recompute.
    # expected values: ↝ graphgym.contrib.feature_augment.util.cache_wrap
    cfg.dataset.feat_cache = 'use_and_update'

    # Possible species classes. This needs to be fixed beforehand because we include
    #   (one-hot) encodings of these classes as features. We cannot infer these possible values
    #   from the input data since we may supply multiple graphs and we cannot depend on the range
    #   or order or consistency of values they provide.
    # Could technically also sweep over all given graphs first to determine the overall range but this
    #   is much simpler.
    cfg.dataset.possible_classes = [sc.value for sc in SpeciesClass]

    # whether to interpret the graph as simple graph, heterogeneous graph, bipartite, ...
    # will affect how some attributes (e.g. node_type) are set and what concrete subclass
    # of deepsnap.graph.Graph is created
    cfg.dataset.graph_interpretation = "simple"

    # If not None, undersampling on the negative class will be performed. The number of samples
    # to be taken for the negative class will be len(pos_class) * ratio
    # e.g. if we give 2 here, we will have twice as many negative as positive examples
    # if the given number results in more required samples as available, we will fall back to 1
    cfg.dataset.undersample_negatives_ratio = None

    # some loss functions (e.g. BCE) allow to specify weights for classes, classifications of a class with higher
    # weight will have more impact on loss value.
    cfg.dataset.minority_loss_weight = 1

register_config('dataset_config', set_cfg_dataset)
