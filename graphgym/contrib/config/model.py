from yacs.config import CfgNode as CN

from graphgym.register import register_config


def set_model_config(cfg):
    # index i is weight of class i. Used e.g. in SVM classifier
    # class 1 corresponds to duplicate
    cfg.model.class_weights = [1, 1]
    # parameters for sklearn's SVC
    #   ↝ https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    cfg.model.svm_gamma = 'scale' # or 'auto' or float; for rbf, poly and sigmoid
    # gamma = 1/sigma??
    cfg.model.svm_cost = 1.0  # a.k.a. `C`
    cfg.model.svm_kernel = 'rbf'

register_config('model_config', set_model_config)
