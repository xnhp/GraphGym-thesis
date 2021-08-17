# scaler supplied via kwargs

import torch
from graphgym.config import cfg
from graphgym.contrib.feature_augment.util import ds_get_interpretations
from sklearn.preprocessing import MinMaxScaler

import deepsnap


def normalize_scale(graph: deepsnap.graph.Graph, scalers=None):
    """
    Transform function to be used with GraphDataset.apply_transform. Serves to normalise node features.
    Current implementation is min-max scaling to [0,1].
    ↝ [[^#dcf41f]].
    :param scalers: dictionary of key -> scaler where key is node attribute key
    :param graph:
    :return:
    """
    label_index = graph.node_label_index
    assert scalers is not None
    assert label_index is not None

    print(f"normalizing features {list(scalers.keys())}")
    not_normalized_feats = [key for key, _ in graph if
                            key.startswith("node_") and key not in cfg.dataset.normalize_feats]
    if not_normalized_feats:
        print(f"some features are NOT normalized: {not_normalized_feats}")

    if cfg.dataset.normalize_feats is None:  # i.e. empty
        return
    for feat_key, scaler in scalers.items():
        # apply previously initialised scaler
        # only consider features in current split, i.e. those indicated by label_index
        # note that technically the entire tensor of size n is present but we only
        # touch those corresponding to the current split.

        # features computed on bipartite projection are of different size
        if feat_key.endswith('_projection'):
            # label_index corresponds to the entire graph (not bipartite projection)
            #   this is a problem if feature (in graph_key) are of smaller shape (if computed on bipartite projection)
            # TODO: this puts the computed bipartite projection into an attribute of the `Graph` object.
            #   When handling multiple graphs (`GraphDataSet` and its splits), the individual graphs are `copy`ed during
            #   split, i.e. we compute this multiple times but wouldn't need to
            #   (but then watch out for setting of attribute of node_label_index on bip proj)
            dsG_simple, dsG_bip = ds_get_interpretations(graph)
            graph_to_use = dsG_bip
            label_index = graph_to_use['node_label_index']
        else:
            label_index = graph.node_label_index
        # apply scalers to all graphs
        original = graph[feat_key][label_index]
        transformed = scaler.transform(original)
        target_dtype = graph[feat_key][label_index].dtype
        graph[feat_key][label_index] = torch.from_numpy(transformed).to(target_dtype)


def normalize_fit(dataset) -> dict:
    """
    Create a normalizer/scaler for each feature that is configured to be normalized and present
    in the input data
    :param dataset: Usually the training dataset on which to fit the scalers
    :return: A dict of feature key -> fitted scaler
    """
    if cfg.dataset.normalize_feats is None:
        return {}  # happens if yaml list in config file is empty

    # return a fitted scaler for feature each key
    # when given multiple graphs, we want to fit the scaler on the union of features from all graphs
    # we can do this by fitting the scaler on the concatenation of the individual graph feats

    def gather_feats(key) -> torch.Tensor:
        return torch.cat([
            graph[key] for graph in dataset
        ], dim=0)

    return {
        feat_key: MinMaxScaler().fit(gather_feats(feat_key))
        for feat_key in cfg.dataset.normalize_feats if
        all([feat_key in graph for graph in dataset])
    }
