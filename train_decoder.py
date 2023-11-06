import os
import random as rn
from argparse import ArgumentParser
from os import mkdir
from os.path import exists

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau

seed = 0
os.environ["PYTHONHASHSEED"] = "0"

np.random.seed(seed)
rn.seed(seed)
tf.random.set_seed(seed)

from impl.architectures import simple_decoder
from impl.datagen import decoder_gen

if __name__ == "__main__":
    parser = ArgumentParser(description="DeepArUco v2 marker decoding model trainer.")
    parser.add_argument("source_dir", help="where to find source images")
    parser.add_argument("run_name", help="where to store the resulting model")
    args = parser.parse_args()

    # Control paramters

    batch_size = 32
    epochs = 1000
    patience = 20
    reduce_after = 10

    # Model

    model = simple_decoder()
    model.compile(loss="mae", optimizer="adam")

    # Load dataset

    train_csv = args.source_dir + "/train.csv"
    train_df = pd.read_csv(train_csv)
    train_generator = decoder_gen(train_df, args.source_dir, batch_size, True, True)

    valid_csv = args.source_dir + "/valid.csv"
    valid_df = pd.read_csv(valid_csv)
    valid_generator = decoder_gen(valid_df, args.source_dir, batch_size, True, True)

    # Callbacks

    stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=True,
        restore_best_weights=True,
        min_delta=1e-4,
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=reduce_after, factor=0.5)

    # Training

    if not exists("./models"):
        mkdir("./models")

    if not exists("./models/losses"):
        mkdir("./models/losses")

    model_name = "simple_decoder"
    csv_logger = CSVLogger(f"./models/losses/loss_{args.run_name}.csv")

    model.fit(
        train_generator,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        callbacks=[stop, reduce_lr, csv_logger],
        verbose=True,
    )
    model.save(f"./models/{args.run_name}.h5")
