from data_loader import *
from model_utils import *
from select_utils import *
import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from mrmr import mrmr_classif

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Fix keras error
from numpy.random import seed
seed(1995)


if __name__ == "__main__":

    # x1, x2, y = load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", predict_late=False, SPLIT_NUMBER=1, xer=True, keep_names=False, training="all")
    # x = pd.concat([x1, x2], axis=1)
    x, y = load_nop(WEIGHT="T2", LRMODE="aggregated", training="all", xer=True, impute=False)
    x_train, y_train = load_nop(WEIGHT="T2", LRMODE="aggregated", training=True, xer=True, impute=True)
    x_valid, y_valid = load_nop(WEIGHT="T2", LRMODE="aggregated", training=False, xer=True, impute=True)
    print(x.shape, x_train.shape, x_valid.shape)
    # n_neurons = 50
    k_fts = 5
    epochs = 250
    lr = 0.01       # default = 0.01
    thresh = 0.50   # threshold for final binary classification
    MODE = "image fts"
    # MODE = "time dose"
    # MODE = "image + td"
    td = ["time", "dose"]


    if MODE == "image fts":
        x_train, x_valid = x_train.drop(td, axis=1), x_valid.drop(td, axis=1)
        top_fts = mrmr_classif(x_train, y_train, K=k_fts)   # DROP time + ddis
    elif MODE == "time dose":
        top_fts = td
    elif MODE == "image + td":
        top_fts = mrmr_classif(x_train, y_train, K=k_fts-2) # save 2 slots for time + dose
        top_fts.append("time")
        top_fts.append("dose")
    else:
        pass

    x_train, x_valid = x_train[top_fts], x_valid[top_fts]
    Nsamp, Nfts = x_train.shape
    print(top_fts)

    # https://www.kaggle.com/code/carlmcbrideellis/very-simple-neural-network-for-classification/notebook   #inspiration
    model = Sequential()

    # NET 1
    # model.add(Dense(32, input_shape=(Nfts, )))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    # model.add(Dense(16, activation="relu"))    # normalize input?
    # model.add(Dense(1, activation="sigmoid"))   # binary classification layer (output)

    # NET 2:
    model.add(Dense(16, input_shape=(Nfts, ), activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    # model.add(Dense(4, activation="relu"))    # normalize input?
    # model.add(Dense(8, activation="relu"))    # normalize input?
    # model.add(Dense(8, activation="relu"))    # normalize input?
    # model.add(Dropout(.2))
    # model.add(Dense(8, activation="relu"))    # normalize input?
    model.add(Dense(1, activation="sigmoid"))   # binary classification layer (output)


    opt = Adam(learning_rate=lr)  # def = 0.01
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid))

    print(history.history.keys())

    y_pred_train = [v > thresh for v in model.predict(x_train)]
    y_pred_valid = [v > thresh for v in model.predict(x_valid)]
    print(y_pred_valid)
    acc_train = accuracy_score(y_train, y_pred_train)
    auc_train = roc_auc_score(y_train, y_pred_train)
    acc_valid = accuracy_score(y_valid, y_pred_valid)
    auc_valid = roc_auc_score(y_valid, y_pred_valid)


    # PLOT ACCURACY + LOSS
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 7))
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend([f'train (N={len(x_train)})', f'valid (N={len(x_valid)})'], loc='best')

    # Loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'valid'], loc='best')
    title = ""
    if MODE == "image fts":
        title += f"Top k image fts (mrmr selected): k = {k_fts}, training={x_train.shape}, valid={x_valid.shape}, params in model = {model.count_params()} ({len(model.weights)} weights)"
    if MODE == "time dose":
        title += f"Only using time & dose as input, training={x_train.shape}, valid={x_valid.shape}, params in model = {model.count_params()}"
    title += f"\nTrain acc={acc_train:.2f}, auc={auc_train:.2f}, valid acc={acc_valid:.2f}, auc={auc_valid:.2f}"
    fig.suptitle(title)
    plt.show()

    # y_pred = model.predict(x_final)
    # print(y_pred)

    # print(model.layers[0].get_weights())
    # print(x.iloc[0].shape)
    # print(model.predict(x))

    # model.add(layers.Embedding(input_dim=Nfts, output_dim=16))
    # model.add(layers.LSTM(32))
    # model.add(layers.Dense(8))
    # model.summary()