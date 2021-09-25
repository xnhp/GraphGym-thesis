import os

from run.visualisation import get_model_details, plot_tpr_cutoffs_combined, plot_roc_combined, init_fig


def get_experiment_dir(experiment_name):
    p = os.path.join("GraphGym/run/", experiment_name)
    return os.path.abspath(p)


def get_models_to_compare(experiment_id):
    models_to_compare = [
        get_model_details(os.path.join(get_experiment_dir(experiment_name), "results", config_name))
        for experiment_name, config_name in experiment_id
    ]
    return models_to_compare


def compare_roc_and_cutoffs(experiment_ids):
    models_to_compare = get_models_to_compare(experiment_ids)

    out_dir = os.path.join(models_to_compare[0]['path'], "../", "comparison")
    out_dir = os.path.abspath(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # TODO both in one fig like here
    # fig, ax = init_fig(rows=2, cols=1)
    # ax_roc = plot_roc_combined(ax[0], models_to_compare, ['val-graph'])
    # ax_cutoff = plot_tpr_cutoffs_combined(ax[1], models_to_compare, 'val-graph')
    # fig.tight_layout()
    # fig.show()

    fig, ax = init_fig(rows=1, cols=1)
    plot_roc_combined(ax, models_to_compare, ['val-graph'])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'roc'))
    fig.show()

    fig, ax = init_fig(rows=1, cols=1)
    plot_tpr_cutoffs_combined(ax, models_to_compare, 'val-graph')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'cutoffs'))
    fig.show()

    # TODO confusion matrices? but already doing this in compare-models...


if __name__ == '__main__':
    # compare_svm_repro()
    #
    # compare_roc_and_cutoffs([
    #     ('train-on-many', 'config-svm'),
    #     ('train-on-many', 'config-gcn'),
    #     ('train-on-many', 'config-gat'),
    #     ('train-on-many', 'config-gcn-weighted'),
    #     ('train-on-many', 'config-gat-weighted')
    # ])
    #
    # compare_roc_and_cutoffs([
    #     ('train-on-many', 'config-svm-ADonly'),
    #     ('train-on-many', 'config-gat-ADonly'),
    # ])

    compare_roc_and_cutoffs([
        ('feature-importance', 'config-gnn-degrees'),
        ('feature-importance', 'config-gnn-degrees-basic')
    ])
