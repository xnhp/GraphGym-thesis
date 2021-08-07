import os

import pandas as pd
import torch


def save_labels(tens: torch.Tensor, filename, out_dir):
    df = pd.DataFrame(tens.numpy())
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, filename + ".csv"))
    # previous approach:
    # torch.save(Y_train, os.path.join(logger_train.out_dir, 'Y_train.pt'))