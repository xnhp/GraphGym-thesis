# directory containing dirs for each run (commonly called "config-{hyperparms}"
import csv
import os

import numpy as np
import pandas as pd
from graphgym.utils.io import json_to_dict_list
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics

from data.summary import print_model_summary, print_graph_summary

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment",
                    type=str,
                    help="name of experiment directory")
parser.add_argument("--config",
                    default="config-gnn",
                    type=str,
                    help="name of config file to consider (no suffix)")
parser.add_argument("--from_config",
                    help="if given, consider results created form invoking config file. Otherwise "
                         "read from output directory of running a grid file.",
                    action="store_true")
parser.add_argument("--grid_name",
                    help="name of grid file",
                    type=str,
                    default="grid")
args = parser.parse_args()

os.chdir("C:\\Users\\Ben\\Uni\\BA-Thesis")

# experiment_name = "train-on-many-gat"
# config_name = "config"

# experiment_name = "gcn-simple"
# config_name = "config-gnn"

# experiment_name = "small-playground"
# config_name = "config-gnn"

# experiment_name = "undersampling-lossweight"
# config_name = "config-gnn"

# experiment_name = "train-on-many-gcn"
# config_name = "config"

# experiment_name = "gcn-projection-reconmapolder"
# config_name = "config-gnn"

# experiment_name = "gcn-projection"
# config_name = "config-gnn"

# experiment_name = "svm-repro-reconmapolder"
# config_name = "config"

# experiment_name = "train-on-many"
# config_name = "config"

# experiment_name = "AD-reorg-playground"
# config_name = "config-gnn"

# experiment_name = "small-optimisations"
# config_name = "config-gnn"


# experiment_name = "feature-importance"
# config_name = "config-gnn-basic-both"


experiment_dir = os.path.join("GraphGym/run/", args.experiment)
grid_out_dir = os.path.join(experiment_dir, "generated-configs/", args.config + "_grid_grid")
plot_out_dir = os.path.join(experiment_dir, "plots")


def get_model_details(model_dir):
    """
    :param model_dir: directory containing results for repeats (dirs with numbers as names) and "agg" that contains
        results aggregated across repeats
    :return: dict
    """

    def find_stats(key):
        # rely on GG aggregation across repeats and only consider "agg" directory
        # (triggered at end of main.py)
        agg_path = os.path.join(model_dir, 'agg', key)
        best_path = os.path.join(agg_path, "best.json")
        stats_path = os.path.join(agg_path, "stats.json")

        if os.path.exists(best_path):
            # in case of GNN, stats.json contains a line for each eval epoch,
            # and additionally, best.json is created, selecting the best line according to
            # a configured metric
            return json_to_dict_list(best_path)[0]
        else:
            # in case of SVM we only print stats.json, which contains a single line (only one "epoch")
            # -- nothing to aggregate
            stats = json_to_dict_list(stats_path)
            assert len(stats) == 1
            return stats

    return {
               'name': os.path.basename(model_dir),
               'path': model_dir
           } | {
               key: find_stats(key)
               for key in ['train', 'val-graph']
           }


def read_model_results(out_dir):
    model_dirs = [dir for dir in os.scandir(out_dir) if dir.is_dir()]
    # dict from model name (arbitrary, lets make this the name of the directory)
    # to additional information such as performance metrics or predictions
    return [
        get_model_details(model_dir) for model_dir in model_dirs
    ]


def sort_models(split, key, model_details):
    key_func = lambda mdl: mdl[split][key]
    s = sorted(model_details,
               key=key_func,
               reverse=True)
    # return extracted values for convenience
    v = [key_func(mdl) for mdl in s]
    return list(zip(s, v))


def top_k_on_split(models, split, metric):
    k = 5
    s1 = sort_models(split, metric, models)[:k]
    s = ""
    split_desc = {
        'train': "internal train split",
        'val': "internal test/validate split",
        'val-graph': "external test/validate split"
    }
    s += ("top " + str(k) + " models on " + split_desc[split] + " by " + metric + "\n")
    for model, value in s1:
        s += ("\t" + model['name'] + ":\t " + str(value) + "\n")
    return s


def top_k_all_splits(models, metric):
    s = ""
    for split in ['train', 'val-graph']:
        s += top_k_on_split(models, split, metric)
    s += ("\n")
    return s


def split_info(mdls, split):
    # TODO: switch to name the csv files the same regardless of the split they are in
    #       so we dont have to switch cases here
    # dummy_mdl = mdls[0]
    # y_test = read_pd_csv(os.path.join(dummy_mdl['path'], "1", "val-graph", "Y_val.csv"))
    # print("foo")
    pass


def read_pd_csv(path):
    df = pd.read_csv(path)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df


def roc_thresh(model, split):
    y_true, y_proba = get_prediction_and_truth(model, split)
    return _roc_thresh(y_true, y_proba)


def _roc_thresh(y_true, y_proba):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return fpr, tpr, optimal_threshold, optimal_idx


def save_loss_plot(model, split, target_dir=plot_out_dir):
    try:
        targetpath = os.path.join(model['path'], "1", split, "stats.json")
    except FileNotFoundError:
        return
    dictlist = json_to_dict_list(targetpath)
    # plt.figure(figsize=(6, 3), facecolor='lightgray')
    plt.title(f"Loss ({split})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot([epoch['loss'] for epoch in dictlist])
    plt.savefig(os.path.join(target_dir, "loss_" + split))
    plt.close()

def save_conf_mat_aliases(model, target_dir=plot_out_dir):

    def write_list(filename, lines):
        with open(os.path.join(target_dir, filename), 'w') as f:
            lines = map(lambda x: x + "\n", lines)
            f.writelines(lines)

    conf_aliases = conf_mat_aliases(model)
    write_list('true-negatives', conf_aliases[0][0])
    write_list('false-negatives', conf_aliases[0][1])
    write_list('false-positives', conf_aliases[1][0])
    write_list('true-positives', conf_aliases[1][1])

def save_roc_plot(model, split="train", plot_random=True, target_dir=plot_out_dir):
    fpr, tpr, _, _ = roc_thresh(model, split)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title(f'Receiver Operating Characteristic ({split})')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)

    if plot_random:
        y_true, _ = get_prediction_and_truth(model, split)
        y_score = np.random.random(len(y_true))
        fpr, tpr, _, _ = _roc_thresh(y_true, y_score)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, 'r', label='AUC (random cl.) = %0.2f' % roc_auc)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ticks = np.append(np.arange(0, 1, step=0.25), 1)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(b=True, linestyle="dotted")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(target_dir, "roc_" + split))
    plt.close()


def get_filename(split):
    # ouch.
    if split == "train":
        return "train"
    elif split == "val":
        return "test"
    elif split == "val-graph":
        return "val"


def get_prediction_and_truth(model, split):
    svm_pred_path = os.path.join(model['path'], "1", split)
    # gnn_pred_base_path = os.path.join(model['path'], "1", split, "preds")
    if os.path.exists(os.path.join(svm_pred_path, "Y_" + get_filename(split) + ".csv")):
        y_test = read_pd_csv(os.path.join(svm_pred_path, "Y_" + get_filename(split) + ".csv"))
        # only probs of positive class
        probas = read_pd_csv(os.path.join(svm_pred_path, "pred_" + get_filename(split) + ".csv"))
        if probas.shape[1] == 2:  # in case we forgot to limit to column of positive class when writing
            probas = pd.DataFrame(probas['1'])
        return y_test, probas
    else:  # in case of gnn we write preds for each eval epoch ↝ train.py
        # eval_epoch_dirs = [dir for dir in os.scandir(gnn_pred_base_path)]
        # get preds of best epoch
        # find best performance on external validation split
        best = json_to_dict_list(os.path.join(model['path'], 'agg', 'val-graph', 'best.json'))
        best_epoch_ix = best[0]['epoch']
        best_pred_dir = os.path.join(model['path'], '1', split, 'preds', str(best_epoch_ix))
        y_test = read_pd_csv(os.path.join(best_pred_dir, "Y_" + get_filename(split) + ".csv"))
        probas = read_pd_csv(os.path.join(best_pred_dir, "pred_" + get_filename(split) + ".csv"))
        return y_test, probas


def read_identifier_mapping(model):
    # cleanup
    dir = os.path.join(model['path'], '1', 'val-graph')
    id_mapping_path = os.path.join(dir, 'mapping_int_to_alias.csv')
    identifier_mapping = {}
    with open(id_mapping_path) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        for row in reader:
            if len(row) == 2:
                identifier_mapping.update({int(row[0]): row[1]})
        # data_read = [row for row in reader]

    node_label_index_path = os.path.join(dir, 'node_label_index.csv')
    node_label_index = read_pd_csv(node_label_index_path)

    return identifier_mapping, node_label_index


def zip_safe(a, b, c):
    assert len(a) == len(b) == len(c)
    return zip(a, b, c)


def conf_mat_aliases(model):
    """
    Returns alias ids of true/false positives/negatives in shape of confusion matrix
    :param model:
    :return:
    """
    class_preds, y_proba, y_true, _ = get_class_preds(model, 'val-graph')
    identifier_mapping, node_label_index = read_identifier_mapping(model)
    y_proba = list(y_proba.get('0'))
    y_true = list(y_true.get('0'))
    node_label_index = list(node_label_index.get('0'))
    ids_for_nodes = [identifier_mapping[ix] for ix in node_label_index]
    # these should correspond
    # ok nice so far, but also need node_label_index
    preds_true_alias = list(zip_safe(class_preds, y_true, ids_for_nodes))

    # assemble sth like a confusion matrix
    tns = [alias for pred, true, alias in preds_true_alias if pred == 0 and true == 0]
    fns = [alias for pred, true, alias in preds_true_alias if pred == 0 and true == 1]
    fps = [alias for pred, true, alias in preds_true_alias if pred == 1 and true == 0]
    tps = [alias for pred, true, alias in preds_true_alias if pred == 1 and true == 1]
    r = [[tns, fns], [fps, tps]]
    return r


def save_conf_mat_plot(model, split="train", target_dir=plot_out_dir):
    class_preds, y_proba, y_true, tresh = get_class_preds(model, split)
    conf_mat = metrics.confusion_matrix(y_true, class_preds)
    fig, ax = plot_confusion_matrix(conf_mat=conf_mat)
    plt.title(f"Confusion Matrix (t={tresh} on {split})")
    plt.savefig(os.path.join(target_dir, "confusion_" + split))
    plt.close()


def get_class_preds(model, split, thresh=None):
    """
    Turn soft classifier confidence values into crisp class predictions based on threshold. If no threshold given,
    determine it automatically based on maximum margin between TPR and FPR
    :return:
    """
    y_true, y_proba = get_prediction_and_truth(model, split)
    if thresh is None or thresh == "auto":
        _, _, tresh, _ = roc_thresh(model, split)

    def decision_function(prob):
        return 0 if float(prob) < tresh else 1

    class_preds = [decision_function(prob) for prob in y_proba.values]
    return class_preds, y_proba, y_true, tresh


def read_yaml(path):
    import yaml
    with open(path) as f:
        dataMap = yaml.safe_load(f)
    return dataMap


def map_summary(identifier):
    model, s1 = print_model_summary(identifier)
    graph, s2 = print_graph_summary(model)
    return s1 + s2


def get_used_datasets(mdls):
    # obtain identifiers of used maps
    dummy_mdl = mdls[0]
    dummy_config = read_yaml(os.path.join(dummy_mdl['path'], "1", "config.yaml"))
    train_identifiers = dummy_config['dataset']['train_names']
    # i.e. corresponding to val-graph
    test_identifiers = dummy_config['dataset']['test_names']
    return train_identifiers, test_identifiers


def data_summary(models):
    s = ""
    train_ids, test_ids = get_used_datasets(models)
    s += "Models used for training (internal):\n"
    for train_id in train_ids:
        # s += map_summary(train_id)
        s += "\n"
    s += "Models used for validation (external):\n"
    for test_id in test_ids:
        # s += map_summary(test_id)
        s += "\n"
    return s


def tpr_cutoffs(model, split, cutoffs=None):
    if cutoffs is None:
        cutoffs = [0.25, 0.5, 0.75]
    fpr, tpr, t_opt, t_opt_ix = roc_thresh(model, split)
    return {
        # find fpr at (close) a given tpr
        # i.e. "if we want to receive {tpr}% of true positives, how many false positives do we get?"
        tpr_cutoff: fpr[
            # find index of largest tpr <= cutoff
            # assume tpr is in ascending order!
            len([r for r in tpr if r <= tpr_cutoff]) - 1
            ]
        for tpr_cutoff in cutoffs
    }


def tpr_cutoffs_str(model, split):
    s = ""
    cutoffs = tpr_cutoffs(model, split)
    s += "FPR at TPR cutoffs: \n"
    for tpr_cutoff, fpr in cutoffs.items():
        s += f"\t{tpr_cutoff:.2f}:\t {fpr:.3f}\n"
    return s


def combined_plots():
    global model, split, _, tresh
    # make combined plot: roc, confusion, loss
    fig, (ax_roc, ax_conf, ax_loss) = plt.subplots(3)
    fig.set_figheight(15)
    fig.set_figwidth(5)
    fig.suptitle("Performance on ext. val. split")
    model = model_to_inspect
    split = "val-graph"
    # ROC
    fpr, tpr, _, _ = roc_thresh(model, split)
    roc_auc = metrics.auc(fpr, tpr)
    ax_roc.set_title(f'Receiver Operating Characteristic ({split})')
    ax_roc.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    ax_roc.legend(loc='lower right')
    ax_roc.plot([0, 1], [0, 1], '--')
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1])
    ticks = np.append(np.arange(0, 1, step=0.25), 1)
    ax_roc.set_xticks(ticks)
    ax_roc.set_yticks(ticks)
    ax_roc.grid(b=True, linestyle="dotted")
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_xlabel('False Positive Rate')
    # confusion
    y_true, y_proba = get_prediction_and_truth(model, split)
    _, _, tresh, _ = roc_thresh(model, split)

    def decision_function(prob):
        return 0 if float(prob) < tresh else 1

    class_preds = [decision_function(prob) for prob in y_proba.values]
    conf_mat = metrics.confusion_matrix(y_true, class_preds)
    fig, ax = plot_confusion_matrix(conf_mat=conf_mat, axis=ax_conf)
    ax_conf.set_title(f"Confusion Matrix (t={tresh} on {split})")
    # loss
    try:
        targetpath = os.path.join(model['path'], "1", split, "stats.json")
        dictlist = json_to_dict_list(targetpath)
        # plt.figure(figsize=(6, 3), facecolor='lightgray')
        ax_loss.set_title(f"Loss ({split})")
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Loss")
        ax_loss.plot([epoch['loss'] for epoch in dictlist])
    except FileNotFoundError:
        pass
    plt.show()


def get_fav_mdls():
    """
    Get models to analyse results for.
    :return:
    """
    # if running config file directly
    if args.from_config:
        config_mdl = get_model_details(os.path.join(experiment_dir, "results", args.config))
        model_to_inspect = config_mdl
        return [config_mdl]
    else:  # assume ran a grid file
        models = read_model_results(grid_out_dir)
        fav_mdl, _ = sort_models("val-graph", 'auc', models)[0]
        model_to_inspect = fav_mdl
        return [fav_mdl]
    # manually selected models from grid results
    # selected_names = [
    #     'config-gnn-do=0.1',
    #     'config-gnn-do=0.2',
    #     'config-gnn-do=0.4',
    #     'config-gnn-lr_decay=0.0',
    #     'config-gnn-lr_decay=0.1',
    #     'config-gnn-weight_decay=0.0',
    #     'config-gnn-weight_decay=0.0005'
    # ]
    # fav_mdls = [mdl for mdl in models if mdl['name'] in selected_names]
    return fav_mdls

def save_summary(models, fav_models, target_dir=plot_out_dir):
    with open(os.path.join(target_dir, "summary.txt"), "w") as f:
        f.write(data_summary(models))
        f.write(top_k_all_splits(models, 'auc'))
        f.write(top_k_all_splits(models, 'accuracy'))
        for model_to_inspect in fav_models:
            f.write(f"additional info on: {model_to_inspect['name']}\n")
            f.write(str(tpr_cutoffs_str(model_to_inspect, 'val-graph')))


if __name__ == "__main__":

    if not os.path.exists(plot_out_dir):
        os.makedirs(plot_out_dir)

    fav_mdls = get_fav_mdls()
    for mdl in fav_mdls:
        for split in ["train", "val-graph"]:
            # TODO arrange these in subplots
            # NOTE this saves the plots in the subdir of the model results
            print(f"saving plots to {mdl['path']}")
            save_roc_plot(mdl, split=split, target_dir=mdl['path'])
            save_conf_mat_plot(mdl, split=split, target_dir=mdl['path'])
            save_conf_mat_aliases(mdl, target_dir=mdl['path'])
            save_loss_plot(mdl, split=split, target_dir=mdl['path'])
    if not args.from_config:
        models = read_model_results(grid_out_dir)
        save_summary(models, fav_mdls)