import time
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM

from simulator import training_set


def cal_pit(file_model, file_keys, file_data, oversamp_fact=1000, nom_miscov_lvl=.2):
    if torch.cuda.is_available():
        model = torch.load(file_model)
    else:
        model = torch.load(file_model, map_location="cpu")

    # import keys
    keys = h5py.File(file_keys, 'r')
    calib_keys = keys["calibration"][:]
    keys.close()

    nsamp = calib_keys.shape[0]
    pit_keys = np.arange(nsamp)
    train_keys = np.random.choice(pit_keys, size=int(.8 * nsamp), replace=False)
    pit_keys = np.setdiff1d(pit_keys, train_keys)
    valid_keys = np.random.choice(pit_keys, size=int(.1 * nsamp), replace=False)
    test_keys = np.setdiff1d(pit_keys, valid_keys)

    # Reading data
    dataset = h5py.File(file_data, 'r')
    dt = dataset["time_delays"][calib_keys]
    pot = dataset["Fermat_potential"][calib_keys]
    H0 = dataset["Hubble_cst"][calib_keys]
    dataset.close()
    samples = np.concatenate((dt[:, :, None], pot[:, :, None]), axis=2)

    x1_train, x2_train = samples[train_keys], H0[train_keys]
    x1_valid, x2_valid = samples[valid_keys], H0[valid_keys]
    x1_test, x2_test = samples[test_keys], H0[test_keys]

    grid = np.linspace(0, 1, 1000)

    gamma = np.random.uniform(0, 1, size=(nsamp, oversamp_fact))

