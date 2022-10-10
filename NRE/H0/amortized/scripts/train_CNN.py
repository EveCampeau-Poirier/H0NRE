# Librairies python
import argparse
import os
import numpy as np

# Librairies PyTorch
import torch
from torch.nn import MSELoss
from torch.optim import Adamax
from torch.optim.lr_scheduler import StepLR

# Functions
from lens_modeling import acc_fct, train_fn, learning_curves, evaluation

# Model
from networks import AccOverFeat96

# --- Execution ------------------------------------------------------------
if __name__ == "__main__":

    # --- Training ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train a classifier to be an estimator of the likelihood ratio of H_0")
    parser.add_argument("--path_in", type=str, default="", help="path to data")
    parser.add_argument("--path_out", type=str, default="", help="path to save the outputs")
    parser.add_argument("--path_hyper", type=str, default="", help="path to the hyperparameter file")
    parser.add_argument("--data_file", type=str, default="dataset.hdf5", help="filename of dataset")
    parser.add_argument("--sched", type=bool, default=False, help="True if a learning rate scheduler is needed")
    parser.add_argument("--anomaly", type=bool, default=False, help="True if detect_anomaly is needed")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--nepochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--std_noise", type=float, default=0.5, help="Noise standard deviation")

    args = parser.parse_args()

    # Path management
    if not os.path.exists(os.path.join(args.path_out, "models")):
        os.makedirs(os.path.join(args.path_out, "models"))
    if not os.path.exists(os.path.join(args.path_out, "plots")):
        os.makedirs(os.path.join(args.path_out, "plots"))

    # Hyperparameters
    if os.path.isfile(args.path_hyper + "/hyparams.txt"):
        freq, factor, thresh = np.loadtxt(args.path_hyper + "/hyparams.txt", unpack=True)
        p_drop, L2, rate, max_norm = 0., 0., 1e-4, None
    else:
        p_drop, L2, rate, max_norm, freq, factor, thresh = 0., 0., 1e-4, None, 100, .75, 1000


    nn = AccOverFeat96()
    loss_fct = MSELoss()
    opt = Adamax(nn.parameters(), lr=rate, weight_decay=L2)

    # Scheduler
    if args.sched:
        scheduler = StepLR(opt, step_size=freq, gamma=factor)
    else:
        scheduler = None

    mean, std = train_fn(model=nn,
                         file=args.data_file,
                         path_in=args.path_in,
                         path_out=args.path_out,
                         optimizer=opt,
                         loss_fn=loss_fct,
                         acc_fn=acc_fct,
                         threshold=thresh,
                         sched=scheduler,
                         grad_clip=max_norm,
                         anomaly_detection=args.anomaly,
                         batch_size=args.batch_size,
                         epochs=args.nepochs,
                         std_noise=args.std_noise
                         )

    # --- Results ----------------------------------------------------------

    # **Save model**
    torch.save(nn, args.path_out + "/models/" + "trained_model.pt")

    learning_curves(os.path.join(args.path_out, "logs.hdf5"), os.path.join(args.path_out, "plots"))

    evaluation(os.path.join(args.path_in, "keys.hdf5"), os.path.join(args.path_in, args.data_file),
               args.path_out + "/models/" + "trained_model.pt", os.path.join(args.path_out, "plots"),
               mean, std)