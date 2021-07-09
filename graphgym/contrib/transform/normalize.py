# scaler supplied via kwargs
import torch


def normalize_stateful(graph, scaler=None, label_index=None):
    """
    Transform function to be used with GraphDataset.apply_transform. Serves to normalise node features.
    Current implementation is min-max scaling to [0,1].
    ↝ [[^#dcf41f]].
    :param graph:
    :return:
    """
    assert scaler is not None
    assert label_index is not None
    # do this only on subset determined by g.node_label_index
    transformed = scaler.transform(graph.node_feature[label_index])
    transformed_tens = torch.from_numpy(transformed).to(torch.float)
    graph.node_feature[label_index] = transformed_tens
    # this is pretty freestyle, could well be wrong
