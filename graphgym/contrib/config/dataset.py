from yacs.config import CfgNode as CN

from graphgym.register import register_config


def set_cfg_dataset(cfg):
    # If True, ignore all complex species during reading of the dataset
    # This means these species will not appear in the graph.
    cfg.dataset.exclude_complex_species = False

    # also see SBMLModel.min_node_degree

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
    cfg.dataset.possible_classes = ['PROTEIN', 'reaction', 'RNA', 'DEGRADED', 'UNKNOWN', 'SIMPLE_MOLECULE', 'ION',
                                    'GENE', 'PHENOTYPE', 'DRUG']

    # whether to interpret the graph as simple graph, heterogeneous graph, bipartite, ...
    # will affect how some attributes (e.g. node_type) are set and what concrete subclass
    # of deepsnap.graph.Graph is created
    cfg.dataset.graph_interpretation = "simple"

register_config('dataset_config', set_cfg_dataset)
