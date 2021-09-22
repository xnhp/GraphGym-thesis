import os

import numpy as np
import pandas as pd
from graphgym.utils.io import json_to_dict_list
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scikitplot import metrics
from sklearn import metrics


def save_roc_plot(model, target_dir=None):
    fig, ax = init_fig()
    plot_roc_combined(ax, [model], ['train', 'val-graph'])
    fig.savefig(os.path.join(target_dir, "roc"))

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


readable_split_names = {
    'train': "Training",
    'val-graph': "Validation"
}


def init_fig(rows=1, cols=1):
    # return Figure (overall wrapper)
    # ax : `.axes.Axes` or array of Axes
    fig, ax = plt.subplots(rows, cols)
    return fig, ax


def plot_roc_combined(ax: Axes, models, splits):
    # list of models as given by get_model_details
    if len(splits) == 1:
        ax.set_title(f"ROC ({readable_split_names[splits[0]]})")
    else:
        ax.set_title(f"ROC")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot([0, 1], [0, 1], '--')  # 0.5 line
    ticks = np.append(np.arange(0, 1, step=0.25), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(b=True, linestyle="dotted")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    for model in models:
        for split in splits:
            plot_roc_onto(ax, model, split)
    ax.legend(loc='lower right')
    return ax


def guess_color(model_name):
    # list of colours (with actually displaying the colors)
    # https://stackoverflow.com/a/37232760/156884
    if "gnn" in model_name:
        return "darkorange"
    elif "gcn" in model_name:
        return "goldenrod"
    elif "gat" in model_name:
        return "darkkhaki"
    elif "svm" in model_name:
        return "cornflowerblue"
    else:
        return "r"


def plot_roc_onto(ax:Axes, model_details, split):
    # plot ROC of model_details onto the given ax
    fpr, tpr, _, _ = roc_thresh(model_details, split)
    roc_auc = metrics.auc(fpr, tpr)
    # TODO somehow describe model
    label_text = f"{model_details['name']}/{split} (AUC={roc_auc:1.2f})"
    ax.plot(fpr, tpr, guess_color(model_details['name']), label=label_text)
    ax.legend(loc='lower right')


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


def plot_tpr_cutoffs_onto(ax:Axes, model_details, split, group_ix, bar_sz):
    cutoffs = tpr_cutoffs(model_details, split)
    bar_x = [x + bar_sz * group_ix for x in np.arange(len(cutoffs.values()))]
    rects = ax.barh(bar_x, cutoffs.values(), height=bar_sz*0.9, color=guess_color(model_details['name']))
    return rects
    # TODO show absolute values for each bar ("rect") like here https://matplotlib.org/2.0.2/examples/api/barchart_demo.html
    # for rect in rects:
    #     width = rect.get_width()
    #     ax.text(rect.get_)


def plot_tpr_cutoffs_combined(ax: Axes, models, split):
    # TODO title etc
    bar_sz = 0.25
    ax: Axes
    ax.set_title("FPR values at TPR cutoffs")
    cutoffs_dummy = tpr_cutoffs(models[0], split)
    yticks_y = [r + bar_sz for r in range(len(cutoffs_dummy.values()))]
    ax.grid(b=True, linestyle="dotted", axis='x')
    ax.set_yticks(yticks_y)
    yticklabels = [f"{val:1.2f}" for val in cutoffs_dummy.keys()]
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("TPR cutoffs")
    legs_a = []
    legs_b = []
    for group_ix, model_details in enumerate(models):
        rects = plot_tpr_cutoffs_onto(ax, model_details, split, group_ix, bar_sz)
        legs_a.append(rects[0])
        legend = f"{model_details['name']}/{split}"
        legs_b.append(legend)
    ax.legend(legs_a, legs_b, loc="lower right")
    return ax