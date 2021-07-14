# scaler supplied via kwargs
import torch
from graphgym.config import cfg
from sklearn.preprocessing import MinMaxScaler


def normalize_stateful(graph, scalers=None):
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

    # assume scalers are properly established
    assert len(cfg.dataset.normalize_feats) == len(scalers.keys())
    for key, scaler in scalers.items():
        # apply previously initialised scaler
        # only consider features in current split, i.e. those indicated by label_index
        # note that technically the entire tensor of size n is present but we only
        # touch those corresponding to the current split
        original = graph[key][label_index]
        transformed = scaler.transform(original)
        graph[key][label_index] = torch.from_numpy(transformed).to(torch.float)


def fit_normalizers(dataset) -> dict:
    # return a fitted scaler for each key
    graph = dataset.graphs[0]
    return {
        feat_key: create_fit_scaler(graph[feat_key])
        for feat_key in cfg.dataset.normalize_feats
    }


def create_fit_scaler(feat):
    scaler = MinMaxScaler()
    scaler.fit(feat)
    return scaler
