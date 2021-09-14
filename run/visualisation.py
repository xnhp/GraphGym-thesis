import os

import numpy as np
import pandas as pd
from graphgym.utils.io import json_to_dict_list
from matplotlib import pyplot as plt
from sklearn import metrics


def save_roc_plot(model, split="train", plot_random=True, target_dir=None):
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