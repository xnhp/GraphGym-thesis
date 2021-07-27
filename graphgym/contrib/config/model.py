from yacs.config import CfgNode as CN

from graphgym.register import register_config


def set_model_config(cfg):
    # index i is weight of class i. Used e.g. in SVM classifier
    # class 1 corresponds to duplicate
    cfg.model.class_weights = 1
    # parameters for sklearn's SVC
    #   ↝ https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    # needs to be number, else we have type mismatch when values are supplied via grid file
    # but for sklear, this could also be 'scale' or 'auto'
    cfg.model.svm_gamma = 10.0 # or 'auto' or float; for rbf, poly and sigmoid
    # gamma = 1/sigma??
    cfg.model.svm_cost = 0.001  # a.k.a. `C`
    cfg.model.svm_kernel = 'rbf'

register_config('model_config', set_model_config)
