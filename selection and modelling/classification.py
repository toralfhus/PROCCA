import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, RepeatedStratifiedKFold, \
    RepeatedKFold, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
import sys
from mrmr import mrmr_classif

from select_utils import *
from data_loader import *
from model_utils import *
from feature_selection import load_delta_p_saliv_data, load_nop_saliv_data, load_saliv_data_reg_to_future, register_name_to_outcome
from DICOM_reader import find_folders
from name_dose_relation import load_naming_data
from endpoints import load_saliva

# PARAMS FOR PLOTTING RESULTS TABLES:
vmin_bs = 0.05
vmax_bs = 0.40


RF_PARAMS_CLASIF_LRSPLIT = {
    "acute":{
        "NO P T1":{
            15:{'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
        },
        "NO P T2":{
            15:{'criterion': 'entropy', 'max_depth': 16, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
        },
        "DELTA P":{
            15:{'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
        },
    },
    "baseline":{
        "DELTA":{
            15:{'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 500}
        }},
    "after irr":{
        "NO P T1":{
            15:{}
}}}
RF_PARAMS_CLASIF_LRAVG = {
    "acute":{
        "NO P T1":{
            15:{'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
        },
        "NO P T2":{
            15:{'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 16, 'n_estimators': 50}
        },
        "DELTA P":{
            15:{'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 16, 'n_estimators': 50}
        },
        "DELTA":{
            15:{'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 16, 'n_estimators': 100}
        }},
    "baseline":{
        "NO P T1":{
            15:{'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 500}
        },
        "NO P T2":{
            15:{'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 1000}
        },
        "DELTA P":{
            15:{'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 50}
        },
        "DELTA":{
            15:{'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 500}
        }},
    "after irr":{
        "NO P T1":{
            15:{'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
        },
        "NO P T2":{
            15:{'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}
        },
        "DELTA P":{
            15:{'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 500}
        },
        "DELTA":{
            15:{'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
}}}

RF_PARAMS_CLASIF_LRAGG = {
    "acute":{
        "NO P T1":{15:  {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 50}},
        "NO P T2":{15:  {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 16, 'n_estimators': 50}},
        "DELTA P":{15:  {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 50}}
    },
    "baseline":{
        "NO P T2":{15:  {'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 1000}},
        "NO P T1":{15:  {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 50}},
        "DELTA P":{15:  {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}}
    },
    "after irr":{
        "NO P T2":{15:  {'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 1000}},
        "NO P T1":{15:  {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}},
        "DELTA T2":{15: {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100}},
        "DELTA P":{15:  {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}},
        "DELTA T1":{15: {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}}
    }
}

RF_PARAMS_CLASIF_MAIN = {
    "split":RF_PARAMS_CLASIF_LRSPLIT,
    "average":RF_PARAMS_CLASIF_LRAVG,
    "aggregated":RF_PARAMS_CLASIF_LRAGG
}

def rf_hyperparamtuning(x_train, y_train, mode="classifier", lightmode=False, return_score=False):
    print("---- TUNING RANDOM FOREST PARAMETERS ----")
    import time
    t0 = time.perf_counter()
    if mode in ["classifier", "classification"]:    #https://www.kaggle.com/code/zlatankr/titanic-random-forest-82-78
        rf = RandomForestClassifier(max_features='sqrt', oob_score=True, random_state=1, n_jobs=-1)

        if not lightmode:
            param_grid = {"criterion": ["gini", "entropy"], "min_samples_leaf": [1, 5, 10],
                          "min_samples_split": [2, 4, 16], "n_estimators": [50, 500, 1000], "max_depth":[2, 8, 32]}
        else:
            param_grid = {"criterion": ["gini", "entropy"], "min_samples_leaf": [1, 3],
                          "min_samples_split": [2, 8], "n_estimators": [100, 500], "max_depth":[2, None]}
        # scoremode = "accuracy"
        scoremode = "roc_auc"

    elif mode == "regression":
        rf = RandomForestRegressor(max_features="auto", oob_score=True, random_state=1, n_jobs=-1)
        param_grid = {"criterion": ["squared_error", "absolute_error"], "min_samples_leaf":[1, 5, 10],
                      "min_samples_split": [2, 4, 8, 16], "n_estimators": [50, 100, 500, 1000], "max_depth":[2, 4, 8, 16, 32]}
        # scoremode = "explained_variance"
        scoremode = "r2"
    else:
        print("Mode", mode, "NOT VALID FOR HYPERPARAMTUNING")

    cv = RepeatedKFold(n_repeats=5, n_splits=2)
    gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scoremode, cv=cv, n_jobs=-1, verbose=1)
    gs = gs.fit(x_train, y_train)

    print(gs.best_score_)
    print(gs.best_params_)
    # print(gs.cv_results_)

    print(gs.best_estimator_)
    t1 = time.perf_counter()
    del time
    print(f"TIME OF TUNING: {t1-t0:.1f} seconds...")
    if return_score:
        return gs.best_params_, gs.best_score_
    else:
        return gs.best_params_

def compare_max_features(x_train, x_test, y_train, y_test, mode="OOB", RANDOMSTATE=None, title=""):
    # MODE: OOB (only train data) or ERROR (on test data)
    # p_fts = len(x_train.columns)    # TOTAL NUMBER OF PREDICTORS
    p_fts = x_train.shape[1]    # TOTAL NUMBER OF PREDICTORS
    n_train = len(x_train)
    n_test = len(x_test)

    print(p_fts, n_train)
    # mode = "OOB"
    # mode = "test error"
    n_repeats = 10
    N = 25
    nvals = np.linspace(50, 1500, N, dtype=int)
    print(len(nvals))
    # scores = [[], [], []]
    oob_scores = np.zeros((3, N))
    test_scores = np.zeros((3, N))
    # clfs = [RandomForestClassifier(max_features="sqrt", oob_score=True, warm_start=True, random_state=RANDOMSTATE),
    #         RandomForestClassifier(max_features="log2", oob_score=True, warm_start=True, random_state=RANDOMSTATE),
    #         RandomForestClassifier(max_features=None, oob_score=True, warm_start=True, random_state=RANDOMSTATE)]
    maxmodes = ["sqrt", "log2", None]
    # for i, rf in enumerate(clfs):
    for i, mode in enumerate(maxmodes):
        print("\n", i, mode)
        oob_temp = np.zeros(shape=(n_repeats, N))
        test_temp = np.zeros(shape=(n_repeats, N))
        for j in range(n_repeats):
            print("\tREPEAT", j)
            rf = RandomForestClassifier(max_features=mode, oob_score=True, warm_start=False, random_state=RANDOMSTATE)
            for ni, n in enumerate(nvals):
                # Warm start: re-use results from previous fit when adding estimators
                rf.set_params(n_estimators=n)
                # rf.fit(X, Y)
                rf.fit(x_train, y_train)
                # print(n, rf.oob_score_)
                # scores[i].append(1 - clf.oob_score_)
                # print(i, j, ni)
                oob_temp[j, ni] = 1 - rf.oob_score_
                # print(oob_temp)
                y_pred = rf.predict(x_test)
                err_test = 1 - accuracy_score(y_test, y_pred)
                test_temp[j, ni] = err_test
                print(f"MODE {mode} {i}/3, REPEAT {j}/{n_repeats}, {ni}/{N}: OOB err={1-rf.oob_score_:.3f}, Test err={err_test:.3f}")
        # scores[i].append(np.mean(oob_temp, axis=0))
        oob_scores[i] = np.mean(oob_temp, axis=0)
        test_scores[i] = np.mean(test_temp, axis=0)
        print(np.shape(oob_scores))
        print(np.shape(test_scores))

    plt.plot(nvals, oob_scores[0], label="OOB RF: $m=\sqrt{p}$", color="b", ls="--")
    plt.plot(nvals, oob_scores[1], label="OOB RF: $m=\log_2(p)$", color="orange", ls="--")
    plt.plot(nvals, oob_scores[2], label="OOB Bagged trees: $m=p$", color="g", ls="--")

    plt.plot(nvals, test_scores[0], label="Test RF: $m=\sqrt{p}$", color="b", ls="-")
    plt.plot(nvals, test_scores[1], label="Test RF: $m=\log_2(p)$", color="orange", ls="-")
    plt.plot(nvals, test_scores[2], label="Test Bagged trees: $m=p$", color="g", ls="-")

    plt.grid()
    plt.xlabel("Number of trees");  plt.ylabel("Error")
    plt.legend()
    if not title:
        # plt.title("Effect of forest size on OOB error estimates for different RF models,\n"
        #           f"evaluated on a constructed data set with {n_samples} samples having with {n_informative} of {p_fts} informative features.")
        # plt.title("Effect of forest size on OOB error estimates for different RF models,\n"
        #           f"evaluated on delta-p data with {n_samples} samples having {p_fts} features.")
        plt.title(f"Effect of forest size on {n_repeats}-repeated averaged OOB error estimates for different RF models,\n"
                  # f"evaluated on delta-p data with {n_samples} samples having {p_fts} features.")
                  f"evaluated on no-p data with {n_train} train / {n_test} test samples having p={p_fts} features.")
                  # f"evaluated on titanic data with {n_samples} samples having {p_fts} features.")
    else:
        plt.title(title)
    plt.show()
    return 1


# def bootstrapped_validation(x_train_original, y_train_original, x_valid_original, y_valid_original, mode="regression", params={}, Nboot=1000, savename=""):
def bootstrapped_validation(x_train_original, x_train_td_original, y_train_original, x_valid_original, x_valid_td_original, y_valid_original, mode="regression", params={}, Nboot=1000, savename=""):
    # rf: RF regressor trained + HP optimized on training data
    # Do 1000 bootstraps: fit new rf, test validation set, calculate
    # r2, MAE, MSE, MAPE, RMSE
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, explained_variance_score
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, jaccard_score
    savefolder = os.path.join(ClassifDir, "no coreg", LRMODE)
    savepath = os.path.join(ClassifDir, "no coreg", LRMODE, savename)
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    print(x_valid_original.shape, y_valid_original.shape, x_train_original.shape, x_valid_original.shape)
    if not mode in ["regression", "classification"]:
        print("INVALID MODE", mode)
        return 0

    # SCORES = [[], [], [], [], []]
    df = pd.DataFrame()
    for i in range(Nboot):
        rf = RandomForestRegressor(**params) if mode == "regression" else RandomForestClassifier(**params)
        rf.fit(x_train_original, y_train_original)

        rf_td = RandomForestClassifier()
        rf_td.fit(x_train_td_original, y_train_original)

        x_valid, y_valid = resample(x_valid_original, y_valid_original)
        x_valid_td = x_valid_td_original.loc[x_valid.index]
        y_pred = rf.predict(x_valid)
        y_pred_td = rf_td.predict(x_valid_td)

        print(f"BOOT {i} / {Nboot}: ", end="")
        if mode == "regression":
            r2 = r2_score(y_valid, y_pred)
            print(f"r2 = {r2:.3f}")
            mae = mean_absolute_error(y_valid, y_pred)
            mse = mean_squared_error(y_valid, y_pred)
            mape = mean_absolute_percentage_error(y_valid, y_pred)
            evs = explained_variance_score(y_valid, y_pred)
            # SCORES[0].append(r2);   SCORES[1].append(mae);  SCORES[2].append(mse);  SCORES[3].append(mape); SCORES[4].append(evs)
            df.loc[i, "r2"] = r2
            df.loc[i, "mae"] = mae
            df.loc[i, "mse"] = mse
            df.loc[i, "mape"] = mape
            df.loc[i, "evs"] = evs
        else:
            try:
                # roc_auc_score, accuracy_score, f1_score, precision_score, jaccard_score
                auc = roc_auc_score(y_valid, y_pred)
                acc = accuracy_score(y_valid, y_pred)
                f1 = f1_score(y_valid, y_pred)
                prec = precision_score(y_valid, y_pred)
                jac = jaccard_score(y_valid, y_pred)
                df.loc[i, "auc"] = auc
                print(f"auc = {auc:.3f}")
                df.loc[i, "acc"] = acc
                df.loc[i, "f1"] = f1
                df.loc[i, "prec"] = prec
                df.loc[i, "jac"] = jac

                auc_td = roc_auc_score(y_valid, y_pred_td)
                acc_td = accuracy_score(y_valid, y_pred_td)
                f1_td = f1_score(y_valid, y_pred_td)
                df.loc[i, "auc_td"] = auc_td
                df.loc[i, "acc_td"] = acc_td
                df.loc[i, "f1_td"] = f1_td

            except Exception as e:
                print(*e.args)
    # print(np.shape(SCORES))
    # print(np.mean(SCORES, axis=1))
    # print(np.std(SCORES, axis=1))
    # print(np.std(SCORES[0]))

    if savename:
        df.to_csv(savepath)
        print("SAVED DF")
    return 1


# def bootstrapped_validation_pairwise(x1_train_original, x2_train_original, y_train_original, x1_valid_original, x2_valid_original, y_valid_original, mode="classification", params1={}, params2={}, Nboot=1000, savename=""):
def bootstrapped_validation_pairwise(x1_train_original, x2_train_original, x12_train_original, td_train_original, y_train_original,
                                     x1_valid_original, x2_valid_original, x12_valid_original, td_valid_original, y_valid_original,
                                     mode="classification", params1={}, params2={}, params12={}, Nboot=1000, savename=""):
    # DO same as bootstrapped_validation, with co-registerred train / test df's 1 & 2
    # ASSUMING y_train / test is valid for BOTH 1 & 2 (with SAME indices!!)

    # from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, explained_variance_score
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, jaccard_score

    # print(x1_valid_original.shape, y_valid_original.shape, x1_train_original.shape, x2_valid_original.shape)
    if not mode in ["classification"]:
        print("INVALID MODE", mode)
        return 0

    # SCORES = [[], [], [], [], []]
    df = pd.DataFrame()
    for i in range(Nboot):
        rf1 = RandomForestClassifier(**params1)
        rf2 = RandomForestClassifier(**params2)
        rf12 = RandomForestClassifier(**params12)
        rftd = RandomForestClassifier()

        rf1.fit(x1_train_original, y_train_original)
        rf2.fit(x2_train_original, y_train_original)
        rf12.fit(x12_train_original, y_train_original)
        rftd.fit(td_train_original, y_train_original)

        x1_valid, y_valid = resample(x1_valid_original, y_valid_original)
        x2_valid = x2_valid_original.loc[x1_valid.index]
        x12_valid = x12_valid_original.loc[x1_valid.index]
        td_valid = td_valid_original.loc[x1_valid.index]

        y_pred1 = rf1.predict(x1_valid)
        y_pred2 = rf2.predict(x2_valid)
        y_pred12 = rf12.predict(x12_valid)
        y_predtd = rftd.predict(td_valid)

        print(f"BOOT {i} / {Nboot}: ", end="")

        try:
            # roc_auc_score, accuracy_score, f1_score, precision_score, jaccard_score
            auc1 = roc_auc_score(y_valid, y_pred1)
            auc2 = roc_auc_score(y_valid, y_pred2)
            auc12 = roc_auc_score(y_valid, y_pred12)
            auctd = roc_auc_score(y_valid, y_predtd)
            print(f"auc_1 = {auc1:.3f}, auc_2 = {auc2:.3f}, auc_comb = {auc12:.3f}, auc_td = {auctd:.3f}")
            df.loc[i, "auc_1"] = auc1
            df.loc[i, "auc_2"] = auc2
            df.loc[i, "auc_comb"] = auc12
            df.loc[i, "auc_td"] = auctd
            # acc = accuracy_score(y_valid, y_pred)
            # f1 = f1_score(y_valid, y_pred)
            # prec = precision_score(y_valid, y_pred)
            # jac = jaccard_score(y_valid, y_pred)
            # df.loc[i, "acc"] = acc
            # df.loc[i, "f1"] = f1
            # df.loc[i, "prec"] = prec
            # df.loc[i, "jac"] = jac
        except Exception as e:
            print(*e.args)

    if savename:
        savepath = os.path.join(ClassifDir, "pairwise T1T2 classif", savename)
        df.to_csv(savepath)
        print("SAVED DF")
    return 1


def visualize_model_performance(path, titlename=""):
    df = pd.read_csv(path, index_col=0)

    M = ["r2"] if "regression" in path else ["auc"]
    # for c in df.columns.values:
    for c in M:
        sns.histplot(data=df, x=c, kde=True)
        # sns.kdeplot(data=df, x=c, common_norm=False)
        vals = df[c].values
        avg = np.average(vals)
        sd = np.std(vals)
        title = ""
        if titlename:
            title += titlename + "\n"
        title += f"Metric {c} (mean +- sd): {avg:.3f} $\pm$ {sd:.3f}"
        plt.title(title)

        plt.show()
    # sns.heatmap(df.corr())
    # plt.show()
    return 1


def visualize_model_performance_pairwise(path, titlename=""):
    num_decimals = 2
    df = pd.read_csv(path, index_col=0)
    # print(df)
    avg_1, sd_1 = np.average(df["auc_1"].values), np.std(df["auc_1"].values)
    avg_2, sd_2 = np.average(df["auc_2"].values), np.std(df["auc_2"].values)
    avg12, sd12 = np.average(df["auc_comb"].values), np.std(df["auc_comb"].values)
    avgtd, sdtd = np.average(df["auc_td"].values), np.std(df["auc_td"].values)
    print(f"AUC1 = {avg_1:.3f} $\pm$ {sd_1:.3f}, AUC2 = {avg_2:.3f} $\pm$ {sd_2:.3f}, AUC12 = {avg12:.3f} $\pm$ {sd12:.3f}")
    # t, p = stats.ttest_ind(df["auc_1"], df["auc_2"])
    t, p = stats.ttest_ind(df["auc_1"], df["auc_2"], equal_var=False)

    print(f"Ind ttest p-value = {p:.5e}, t={t:.5f}")
    titlename += f"\n$AUC_{{T1}} = {avg_1:.{num_decimals}f} \pm {sd_1:.{num_decimals}f}$,\t $AUC_{{T2}} = {avg_2:.{num_decimals}f} \pm {sd_2:.{num_decimals}f}$," \
                 f"\n$AUC_{{T1 + T2}} = {avg12:.{num_decimals}f} \pm {sd12:.{num_decimals}f}$, \t$AUC_{{time + dose}} = {avgtd:.{num_decimals}f} \pm {sdtd:.{num_decimals}f}$" \
                 f"\nDifferent means T1 vs T2 t-test: t={t:.3f}, p={p:.3e}"
    # FIGSIZE = (14, 7)
    FIGSIZE = (8, 8)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.histplot(data=df, x="auc_1", kde=True, label="T1", ax=ax, color="skyblue")
    sns.histplot(data=df, x="auc_2", kde=True, label="T2", color="r", ax=ax)

    ylim = ax.get_ylim()
    ax.vlines([avg_1], *ylim, linestyles="dashed", colors="skyblue")
    ax.vlines([avg_1 - sd_1, avg_1 + sd_1], *ylim, linestyles="dotted", colors="skyblue")
    ax.vlines([avg_2], *ylim, linestyles="dashed", colors="r")
    ax.vlines([avg_2 - sd_2, avg_2 + sd_2], *ylim, linestyles="dotted", colors="r")

    ax.legend()
    ax.set_title(titlename)
    ax.set_xlabel("ROC AUC")
    plt.show()

    # DIFFERENCE PER BOOTSTRAP PLOT
    # fig2, ax2 = plt.subplots(figsize=FIGSIZE)
    # df.loc[:, "auc_diff"] = df["auc_1"].to_numpy() - df["auc_2"].to_numpy()
    # # print(df)
    # sns.histplot(data=df, x="auc_diff", kde=True, label="Diff", ax=ax2)
    # ax2.set_title(titlename)
    # ax2.set_xlabel("$AUC_{{T1}} - AUC_{{T2}}$")
    # plt.show()
    pass


def main_classification(MODE, LRMODE="split", WEIGHT="T2", predict_late=False, Nfts=15, num_bootstraps=10000,
                        do_hp_tuning=False, only_visualize=False, show_performance=False, params={}):
    # from mrmr import mrmr_classif
    if not(predict_late):
        savename = f"acute_RF_classification_validate_{'''-'''.join(MODE.split('''_'''))}_{WEIGHT}_LR{LRMODE}_Nfts={Nfts}.csv"
    elif MODE == "DELTA":
        savename = f"predict_late_with_DELTA_{WEIGHT}_LR{LRMODE}_RF_classification_validate_Nfts={Nfts}.csv"
    elif predict_late == "baseline":
        savename = f"predict_late_with_{'''-'''.join(MODE.split('''_'''))}_{WEIGHT}_LR{LRMODE}_baseline_RF_classification_validate_Nfts={Nfts}.csv"
    elif predict_late == "after irr":
        savename = f"predict_late_with_{'''-'''.join(MODE.split('''_'''))}_{WEIGHT}_LR{LRMODE}_after-irr_RF_classification_validate_Nfts={Nfts}.csv"
    else:
        print(">ERROR: predict_late value", predict_late, "invalid. Try: baseline or after irr")
        return 0

    # params = {}

    predict_mode = "acute" if not predict_late else predict_late
    if MODE == "NO P":
        if not any(params) and not do_hp_tuning:
            params = RF_PARAMS_CLASIF_LRSPLIT[predict_mode][MODE + " " + WEIGHT][Nfts] if LRMODE == "split" else \
                RF_PARAMS_CLASIF_LRAVG[predict_mode][MODE + " " + WEIGHT][Nfts]

        if not predict_late:
            x_train, y_train = load_nop(WEIGHT=WEIGHT, LRMODE=LRMODE, training=True, xer=True)
            x_valid, y_valid = load_nop(WEIGHT=WEIGHT, LRMODE=LRMODE, training=False, xer=True)
            # if WEIGHT == "T2":
            #     params = {'criterion': 'gini', 'max_depth': 16, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 50}  # Nfts = 15
            # else:
            #     params = {'criterion': 'gini', 'max_depth': 16, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}  # Nfts = 15
        else:
            baseline_bool = True if predict_late == "baseline" else False
            # x_train, y_train = load_predict_late(MODE="NO P", WEIGHT=WEIGHT, LRMODE=LRMODE, baseline=baseline_bool, training=True, xer=True, keep_times=True)
            # x_valid, y_valid = load_predict_late(MODE="NO P", WEIGHT=WEIGHT, LRMODE=LRMODE, baseline=baseline_bool, training=False, xer=True, keep_times=True)

            x_train, y_train = load_predict_late_not_delta(MODE="NO P", WEIGHT=WEIGHT, LRMODE=LRMODE, baseline=baseline_bool, training=True, xer=True, reset_y_index=True)
            x_valid, y_valid = load_predict_late_not_delta(MODE="NO P", WEIGHT=WEIGHT, LRMODE=LRMODE, baseline=baseline_bool, training=False, xer=True, reset_y_index=True)
            # x_train, x_valid = x_train.drop("name", axis=1), x_valid.drop("name", axis=1)
            # print(x_train.columns)
            # y_train.index = x_train.index
            # y_valid.index = x_valid.index


    elif MODE == "DELTA P":
        if not any(params) and not do_hp_tuning:
            params = RF_PARAMS_CLASIF_LRSPLIT[predict_mode][MODE][Nfts] if LRMODE == "split" else \
            RF_PARAMS_CLASIF_LRAVG[predict_mode][MODE][Nfts]

        if not predict_late:
            x_train, y_train = load_delta_P(WEIGHT="T2", LRMODE=LRMODE, training=True, xer=True)
            x_valid, y_valid = load_delta_P(WEIGHT="T2", LRMODE=LRMODE, training=False, xer=True)
            # params = {'criterion': 'entropy', 'max_depth': 16, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 50}
        else:
            baseline_bool = True if predict_late == "baseline" else False
            # params = {} if baseline_bool else {'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 500}

            # x_train, y_train = load_predict_late(MODE="DELTA P", WEIGHT=WEIGHT, LRMODE=LRMODE, baseline=baseline_bool, training=True, xer=True)
            # x_valid, y_valid = load_predict_late(MODE="DELTA P", WEIGHT=WEIGHT, LRMODE=LRMODE, baseline=baseline_bool, training=False, xer=True)
            x_train, y_train = load_predict_late_not_delta(MODE="DELTA P", WEIGHT=WEIGHT, LRMODE=LRMODE, baseline=baseline_bool, training=True, xer=True, reset_y_index=True)
            x_valid, y_valid = load_predict_late_not_delta(MODE="DELTA P", WEIGHT=WEIGHT, LRMODE=LRMODE, baseline=baseline_bool, training=False, xer=True, reset_y_index=True)



    elif MODE == "DELTA":
        if not any(params) and not do_hp_tuning:
            params = RF_PARAMS_CLASIF_LRSPLIT[predict_mode][MODE][Nfts] if LRMODE == "split" else \
            RF_PARAMS_CLASIF_LRAVG[predict_mode][MODE][Nfts]

        if not predict_late:
            print(">ERROR: DELTA only predicts late values.")
            return {}
        x_train, y_train = load_delta(LRMODE=LRMODE, WEIGHT=WEIGHT, training=True, xer=True, keep_time=True)
        x_valid, y_valid = load_delta(LRMODE=LRMODE, WEIGHT=WEIGHT, training=False, xer=True, keep_time=True)
        # print(x_train.filter(like="time", axis=1))
        x_train["time"] = [0]*len(x_train)  # function above returns time of saliva measurement, which is cheating
        x_valid["time"] = [0]*len(x_valid)
        # params = {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 500}
        # params = {'criterion': 'entropy', 'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

    else:
        print("INVALID MODE", MODE)
        return 0

    fts_td = ["time", "dose"]
    x_train_td = x_train[fts_td]
    x_valid_td = x_valid[fts_td]
    x_train, x_valid = x_train.drop(fts_td, axis=1), x_valid.drop(fts_td, axis=1)

    if Nfts != "all" and not only_visualize:
        selected_features = mrmr_classif(x_train, y_train, K=Nfts, return_scores=False)
        x_train, x_valid = x_train[selected_features], x_valid[selected_features]
        print(Nfts, "mrmr selected:", selected_features)

    img_title = f"{MODE} {WEIGHT} LR{LRMODE}" + " predict " + f"{'''late''' if predict_late else '''acute'''}" \
                                                  f"{'''''' if not predict_late else f''' using {predict_late}'''}"
    if do_hp_tuning:
        # ONLY USE TRAIN DATA FOR HP TUNING
        print("\n", MODE, WEIGHT)
        params = rf_hyperparamtuning(x_train, y_train, mode="classification")
        return params

    elif only_visualize:
        print(savename)
        visualize_model_performance(path=os.path.join(ClassifDir, "no coreg", savename), titlename=img_title)

    else:
        # bootstrapped_validation(x_train, y_train, x_valid, y_valid, params=params, mode="classification", Nboot=num_bootstraps, savename=savename)
        bootstrapped_validation(x_train, x_train_td, y_train, x_valid, x_valid_td, y_valid, params=params, mode="classification", Nboot=num_bootstraps, savename=savename)
        visualize_model_performance(path=os.path.join(ClassifDir, "no coreg", savename), titlename=img_title) if show_performance else 0
    return 0


def main_classification_pairwise_T1_T2(MODE, LRMODE, predict_late=False, do_hp_tuning=False, Nfts=15, num_bootstraps=10000,
                    only_visualize=False, show_performance=False, params_t1={}, params_t2={}, params_comb={}, SPLIT_NUMBER=1):
    from mrmr import mrmr_classif
    if not SPLIT_NUMBER in [1, 2, 3]:
        print("INVALID SPLIT NUMBER", SPLIT_NUMBER, "try: 1, 2, 3")
        return 0

    if not(predict_late):
        savename = f"pairwise_T1T2_LR{LRMODE}_acute_{'''-'''.join(MODE.split(''' '''))}_RF_classification_validate_Nfts={Nfts}_split{SPLIT_NUMBER}.csv"
    elif MODE == "DELTA":
        savename = f"pairwise_T1T2_LR{LRMODE}_predict_late_with_DELTA-time_RF_classification_validate_Nfts={Nfts}_split{SPLIT_NUMBER}.csv"
    elif predict_late in ["baseline", "after irr"]:
        savename = f"pairwise_T1T2_LR{LRMODE}_predict_late_{'''-'''.join(predict_late.split(''' '''))}_with_" \
                   f"{'''-'''.join(MODE.split(''' '''))}_RF_classification_validate_Nfts={Nfts}_split{SPLIT_NUMBER}.csv"
    else:
        print(">ERROR: predict_late value", predict_late, "invalid. Try: baseline or after irr")
        return 0
    print(savename)

    if not LRMODE in ["split", "average", "aggregated"]:
        print("LRmode", LRMODE, "not valid.")
        return 0

    predict_mode = "acute" if not predict_late else predict_late
    if MODE == "NO P":
        try:
            params_t1 = RF_PARAMS_CLASIF_MAIN[LRMODE][predict_mode]["NO P" + " " + "T1"][Nfts] if not params_t1 else params_t1
            params_t2 = RF_PARAMS_CLASIF_MAIN[LRMODE][predict_mode]["NO P" + " " + "T2"][Nfts] if not params_t2 else params_t2
            params_comb = {}
        except Exception as e:
            params_t1 = {}
            params_t2 = {}
            params_comb = {}

        t1_train, t2_train, y_train = load_T1T2_coreg(MODE="NO P", LRMODE=LRMODE, predict_late=predict_late,
                                                      training=True, xer=True, SPLIT_NUMBER=SPLIT_NUMBER)
        t1_valid, t2_valid, y_valid = load_T1T2_coreg(MODE="NO P", LRMODE=LRMODE, predict_late=predict_late,
                                                      training=False, xer=True, SPLIT_NUMBER=SPLIT_NUMBER)

        comb_train = aggregate_T1_T2_on_features(t1_train, t2_train)
        comb_valid = aggregate_T1_T2_on_features(t1_valid, t2_valid)

    else:
        print("MODE", MODE, "not valid.")
        return 0

    fts_td = ["time", "dose"]
    timedose_train, timedose_valid = t1_train[fts_td], t1_valid[fts_td]
    # print(t1_valid[fts_td])
    t1_train = t1_train.drop(fts_td, axis=1)
    t1_valid = t1_valid.drop(fts_td, axis=1)
    t2_train = t2_train.drop(fts_td, axis=1)
    t2_valid = t2_valid.drop(fts_td, axis=1)
    comb_train = comb_train.drop(fts_td, axis=1)
    comb_valid = comb_valid.drop(fts_td, axis=1)

    if str(Nfts).lower() != "all" and not only_visualize:
        fts_t1 = mrmr_classif(t1_train, y_train, K=Nfts, return_scores=False)
        t1_train, t1_valid = t1_train[fts_t1], t1_valid[fts_t1]

        fts_t2 = mrmr_classif(t2_train, y_train, K=Nfts, return_scores=False)
        t2_train, t2_valid = t2_train[fts_t2], t2_valid[fts_t2]

        fts_comb = mrmr_classif(comb_train, y_train, K=Nfts, return_scores=False)
        comb_train, comb_valid = comb_train[fts_comb], comb_valid[fts_comb]

        print("T1 fts:\t", fts_t1)
        print("T2 fts:\t", fts_t2)
        print("T1 + T2 fts:\t", fts_comb)
        print("Time + dose:\t", fts_td)

    img_title = f"T1 + T2 co-registerred validation (split {SPLIT_NUMBER}) for mode {MODE} LR{LRMODE} predict {predict_mode} using {Nfts} features" \
                f" $(N_{{samp, train}}={len(t1_train)}, N_{{samp, valid}}={len(t1_valid)})$"

    if do_hp_tuning:
        # ONLY TRAINING DATA FOR TUNING!!
        print("\n", MODE, LRMODE, "late=", predict_late)
        params_t1, score1 = rf_hyperparamtuning(t1_train, y_train, mode="classification")
        params_t2, score2 = rf_hyperparamtuning(t2_train, y_train, mode="classification")
        params_comb, score_comb = rf_hyperparamtuning(comb_train, y_train, mode="classification")
        return params_t1, score1, params_t2, score2, params_comb, score_comb

    elif only_visualize:
        print(savename)
        print("VISUALIZE NOT IMPLEMENTED")
        # visualize_model_performance(path=os.path.join(ModelDir, "RF", savename), titlename=img_title)
        visualize_model_performance_pairwise(path=os.path.join(ClassifDir, "pairwise T1T2 classif", savename), titlename=img_title)
        return 0

    else:
        # bootstrapped_validation_pairwise(t1_train, t2_train, y_train, t1_valid, t2_valid, y_valid, params1=params_t1, params2=params_t2, mode="classification", Nboot=num_bootstraps, savename=savename)
        bootstrapped_validation_pairwise(t1_train, t2_train, comb_train, timedose_train, y_train, t1_valid, t2_valid, comb_valid, timedose_valid, y_valid,
            params1=params_t1, params2=params_t2, params12=params_comb, mode="classification", Nboot=num_bootstraps, savename=savename)
        # visualize_model_performance_pairwise(path=os.path.join(ModelDir, "RF", savename),
        #                             titlename=img_title) if show_performance else 0
    return 1


def show_T12_coreg_classif_results(LRmode="aggregated", ftMode="NO-P"):
    PRED_MODES = ["acute", "baseline", "after-irr"]
    NUM_SPLITS = 3
    NUM_DECIMALS = 2  # number of decimal places in displayed text

    folder_path = os.path.join(ClassifDir, "pairwise T1T2 classif")
    files = [f for f in os.listdir(folder_path) if (LRmode in f) and (ftMode) in f]
    print(os.listdir(folder_path))
    print(files)

    print("Found", len(files), "csv files for LRmode", LRmode, "ftMode", ftMode)

    df_results = pd.DataFrame()
    df_disptext = pd.DataFrame()    # mean +- sd for display text
    for i, f in enumerate(files):
        # print(f)
        flist = f.split("_")
        pred_mode = [x for x in flist if x in PRED_MODES][0]
        print(pred_mode)
        if pred_mode == "acute":
            pred_mode = "simul"   # change name of this prediction mode

        # Nfts = int(flist[-2][5:])
        Nfts = flist[-2][5:]
        split_num = flist[-1][-5]
        # print(i, pred_mode, Nfts, split_num)
        df = pd.read_csv(os.path.join(folder_path, f), index_col=0)
        # print(df.columns.values)
        col = pred_mode + f" split {split_num}"
        auc_t1 = df["auc_1"]
        auc_t2 = df["auc_2"]
        auc_comb = df["auc_comb"]
        auc_td = df["auc_td"]

        row = "time + dose"
        df_results.loc[row, "num fts"] = 2
        df_disptext.loc[row, "num fts"] = 2

        df_results.loc[row, col] = auc_td.mean()
        df_disptext.loc[row, col] = f"{auc_td.mean():.{NUM_DECIMALS}f} $\pm$ {auc_td.std():.{NUM_DECIMALS}f}"

        for modelname, vals in zip(["T1", "T2", "T1 + T2"], [auc_t1, auc_t2, auc_comb]):
            row = f"{modelname} {Nfts} fts"
            if str(Nfts).lower() == "all":
                df_results.loc[row, "num fts"] = 10000
                df_disptext.loc[row, "num fts"] = 10000
            else:
                Nfts = int(Nfts)
                df_results.loc[row, "num fts"] = Nfts
                df_disptext.loc[row, "num fts"] = Nfts

            df_results.loc[row, col] = vals.mean()
            df_disptext.loc[row, col] = f"{vals.mean():.{NUM_DECIMALS}f} $\pm$ {vals.std():.{NUM_DECIMALS}f}"

    print(df_results)
    df_results = df_results.sort_values(by=["num fts"]).drop("num fts", axis=1)
    df_disptext = df_disptext.sort_values(by="num fts").drop("num fts", axis=1)
    print(df_results.shape, df_disptext.shape)
    # print(df_disptext)

    fig, ax = plt.subplots()
    FONTSIZE_LAB = 14
    FONTSIZE_ANNOT = 14
    sns.heatmap(data=df_results, annot=df_disptext.values, fmt="", cbar=False, cmap="RdYlGn", ax=ax, annot_kws={"fontsize":FONTSIZE_ANNOT})

    indsplits = [ind.split(" ") for ind in df_results.columns.values]
    ax.set_xticks(ax.get_xticks(), [f"{''' '''.join(x[0].split('''-''')).capitalize()}\n{''' '''.join(x[1:])}" for x in indsplits], rotation=0, fontsize=FONTSIZE_LAB)
    ax.xaxis.tick_top()

    ax.set_yticks(ax.get_yticks(), [x for x in df_disptext.index.values], fontsize=FONTSIZE_LAB, rotation=0)
    plt.show()
    plt.close()
    print(df_results.columns)

    for predmode in ["simul", "after-irr", "baseline"]:
        df_temp = df_results.filter(like=predmode, axis=1)
        df_disp_temp = df_disptext.filter(like=predmode, axis=1)
        # print(df_temp)
        # print(df_disptext.filter(like=predmode, axis=1))
        # df_results.loc[:, f"{predmode} avg"] = np.average(df_temp.values, axis=1)

        for row in list(df_temp.index):
            df_tempp = df_temp.loc[row]
            df_disp_tempp = df_disp_temp.loc[row]
            # print(df_tempp)
            muvals = df_tempp.values
            # print(df_disp_tempp)
            sdvals = [float(v.split(" ")[-1]) for v in df_disp_tempp.values]
            mu_comb, sd_comb = combine_means_sd(muvals, sdvals)
            df_results.loc[row, f"{predmode} avg"] = mu_comb
            df_disptext.loc[row, f"{predmode} avg"] = f"{mu_comb:.2f} $\pm$ {sd_comb:.2f}"

    # print(df_results)
    # print(df_disptext)

    fig, ax = plt.subplots()
    cols = ["simul avg", "after-irr avg", "baseline avg"]
    sns.heatmap(data=df_results[cols], annot=df_disptext[cols], fmt="", cbar=False, cmap="RdYlGn", ax=ax,
                annot_kws={"fontsize": FONTSIZE_ANNOT}, vmin=0.5, vmax=1)
    rows = list(df_results.index)
    print(rows)
    ax.set_yticks(ax.get_yticks(), rows, rotation=0)
    ax.set_xticks(ax.get_xticks(), ["simult.", "after irr", "baseline"], rotation=0)
    ax.set_title("Index averaged AUC LR" + LRmode)
    plt.show()
    pass


def main_classification_pairwise_T1_T2_loocv(MODE, LRMODE, predict_late=False, do_hp_tuning=False, NFTS=5, num_repeatRF=100):
    save_folder = os.path.join(ClassifDir, "pairwise T1T2 classif", "loocv")
    if not(predict_late):
        savename = f"pairwiseT12_loocv_LR{LRMODE}_simult_{MODE}_nfts={NFTS}.csv"
        savename_fts = f"pairwiseT12_loocv_LR{LRMODE}_simult_{MODE}_nfts={NFTS}_selectedfts.csv"
    elif MODE == "DELTA":
        savename = f"pairwiseT12_loocv_LR{LRMODE}_delta_{MODE}_nfts={NFTS}.csv"
        savename_fts = f"pairwiseT12_loocv_LR{LRMODE}_delta_{MODE}_nfts={NFTS}_selectedfts.csv"
    elif predict_late in ["baseline", "after irr"]:
        savename = f"pairwiseT12_loocv_LR{LRMODE}_{predict_late}_{MODE}_nfts={NFTS}.csv"
        savename_fts = f"pairwiseT12_loocv_LR{LRMODE}_{predict_late}_{MODE}_nfts={NFTS}_selectedfts.csv"
    else:
        print("TRY predict_late: False, baseline, after irr, delta")
    if do_hp_tuning:
        savename = savename.split(".")[0] + "_hptuned.csv"
        savename_fts = savename_fts.split(".")[0] + "_hptuned.csv"
    print(savename)
    print(savename_fts)
    # return 0

    predict_mode = "simult" if not(predict_late) else predict_late

    # if MODE == "NO P":
    df1, df2, y = load_T1T2_coreg(MODE, LRMODE=LRMODE, predict_late=predict_late, training="all", xer=True)
    df_comb = aggregate_T1_T2_on_features(df1, df2)

    if "time" in df1.columns and "dose" in df1.columns:
        df1 = df1.drop(["time", "dose"], axis=1)
        df2 = df2.drop(["time", "dose"], axis=1)

    else:
        df1 = df1.drop("dose", axis=1)
        df2 = df2.drop("dose", axis=1)

    df_td = df_comb[["time", "dose"]]
    df_comb = df_comb.drop(["time", "dose"], axis=1)

    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    df_comb = df_comb.reset_index(drop=True)
    df_td = df_td.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # print(df1.shape, df2.shape, df_comb.shape, df_td.shape, y.shape)
    DESCRIPTORS = ["T1", "T2", "T1 + T2", "td"]
    num_obs = len(y)
    print("\nLOOCV T12 coreg: ", MODE, LRMODE, "late =", predict_late, "num_obs =", num_obs)
    loo = LeaveOneOut()
    n = 0
    i = 0
    df_results = pd.DataFrame()
    df_selected = pd.DataFrame()
    # df_selected = pd.DataFrame(data=np.array([], shape=(num_obs, len(df_comb.columns))))
    # df_selected = pd.DataFrame(data=np.array([], shape=(num_obs, len(df_comb.columns))))

    # ft_index = get_feature_index_global()
    # fts_all = list(ft_index.keys())
    # df_selected = pd.DataFrame(index=fts_all, columns=DESCRIPTORS[:-1], data=np.zeros(shape=(len(fts_all), 3)))
    # print(df_selected)
    for idx_train, idx_test in loo.split(df1):
        n += 1
        print(f"\nLOOCV {predict_mode} {n} / {num_obs}")
        x1_train, x2_train, xcomb_train, xtd_train = df1.loc[idx_train], df2.loc[idx_train], df_comb.loc[idx_train], df_td.loc[idx_train]
        x1_test, x2_test, xcomb_test, xtd_test = df1.loc[idx_test], df2.loc[idx_test], df_comb.loc[idx_test], df_td.loc[idx_test]
        # print(x1_train.shape, x2_train.shape, xcomb_train.shape, xtd_train.shape)
        # print(x1_test.shape, x2_test.shape, xcomb_test.shape, xtd_test.shape)
        y_train, y_test = y.loc[idx_train], y[idx_test]
        # print(y_train.shape, y_test.shape)

        for c, xtrain, xtest in zip(DESCRIPTORS, [x1_train, x2_train, xcomb_train, xtd_train],
                                    [x1_test, x2_test, xcomb_test, xtd_test]):
            if not c == "td":
                fts = mrmr_classif(xtrain, y_train, K=NFTS)
                # fts = xtrain.columns.values[:5]
                df_selected.loc[n, c] = " ".join(fts)

            y_gt = y_test.ravel()[0]
            if do_hp_tuning:
                params_rf = rf_hyperparamtuning(xtrain, y_train, lightmode=True)
            else:
                params_rf = {}
            print(params_rf)
            i = (n-1) * num_repeat_RF
            for r in range(num_repeat_RF):
                print(".", end="")
                i += 1
                rf = RandomForestClassifier(**params_rf)
                rf.fit(xtrain, y_train)
                phat_rf = rf.predict_proba(xtest).T.ravel()[-1]

                lr = LogisticRegression()
                lr.fit(xtrain, y_train)
                phat_lr = lr.predict_proba(xtest).T.ravel()[-1]
                # print(c, n, r, y_gt, phat_rf, phat_lr)
                df_results.loc[i, "idx"] = n
                df_results.loc[i, "y_gt"] = y_gt
                df_results.loc[i, f"p_{c}_lr"] = phat_lr
                df_results.loc[i, f"p_{c}_rf"] = phat_rf
    print(df_results)
    print(df_selected)
    df_results.to_csv(os.path.join(save_folder, savename))
    df_selected.to_csv(os.path.join(save_folder, savename_fts))
    return 1


def show_T12coreg_loocv_results(hptuned=False, CLASSIFIERS="ALL", LRMODE="aggregated"):
    folder = os.path.join(ClassifDir, "pairwise T1T2 classif", "loocv")
    files = os.listdir(folder)
    CLASSIFIERS = CLASSIFIERS.upper()
    if not CLASSIFIERS in ["ALL", "LOGREG", "RF"]:
        print("Try CLASSIFIERS: ALL, LOGREG, RF")
        return 0
    if hptuned:
        files = list(filter(lambda f: "hptuned" in f, files))
    else:
        files = list(filter(lambda f: "hptuned" not in f, files))
    files = list(filter(lambda f: LRMODE in f, files))
    files_fts = list(filter(lambda f: "selectedfts" in f, files))
    files_results = list(filter(lambda f: "selectedfts" not in f, files))
    print("\n\n", len(files_fts), "+", len(files_results), "files found")
    print(files_fts)

    df_auc = pd.DataFrame()
    df_bs = pd.DataFrame()
    df_auc_annot = pd.DataFrame()
    df_bs_annot = pd.DataFrame()

    for f in files_results:
        ff = f.split("_")
        print(ff)
        lrmode, latemode, mode, nfts = ff[2:] if not hptuned else ff[2:-1]
        nfts = nfts.split("=")[1][:-4]
        print(lrmode, latemode, mode, nfts)

        df = pd.read_csv(os.path.join(folder, f), index_col=0)
        # df["y_gt"] = [1 if yi == "True" else 0 for yi in df["y_gt"]]
        df["y_gt"] = [int(yi) for yi in df["y_gt"]]
        df_means = df.groupby("idx").mean()
        df_sds = df.groupby("idx").std()

        y_true = df_means["y_gt"]
        print(f"{len(y_true[y_true == True])} of {len(y_true)} xer")

        cols_probs = df_means.drop("y_gt", axis=1).columns.values
        if not CLASSIFIERS == "ALL":
            cls = "lr" if CLASSIFIERS == "LOGREG" else "rf"
            cols_probs = list(filter(lambda c: cls in c, cols_probs))
            cols_new = [c.split("_")[1] for c in cols_probs]
        else:
            cols_new = [c.split("_")[1] + " " + "LOGREG" if c.split("_")[-1] == "lr" else c.split("_")[1] + " " + "RF"
                        for c in cols_probs]

        # print(cols_probs)
        # print(cols_new)
        PROBS = [df_means[c] for c in cols_probs]
        PROBS_SD = [df_sds[c] for c in cols_probs]
        PROBS_LOWER = [pmean - psd for pmean, psd in zip(PROBS, PROBS_SD)]
        PROBS_UPPER = [pmean + psd for pmean, psd in zip(PROBS, PROBS_SD)]

        AUCvals = [roc_auc_score(y_true, pvals) for pvals in PROBS]
        BSvals = [brier_score(y_true, pvals) for pvals in PROBS]

        AUC_LOWER = [roc_auc_score(y_true, pvals) for pvals in PROBS_LOWER]
        AUC_UPPER = [roc_auc_score(y_true, pvals) for pvals in PROBS_UPPER]
        AUC_LOWER_DIFF = np.abs(np.array(AUCvals) - np.array(AUC_LOWER))
        AUC_UPPER_DIFF = np.abs(np.array(AUCvals) - np.array(AUC_UPPER))
        AUC_UNCERTAINTY = np.max([AUC_LOWER_DIFF, AUC_UPPER_DIFF], axis=0)

        BS_LOWER = [brier_score(y_true, pvals) for pvals in PROBS_LOWER]
        BS_UPPER = [brier_score(y_true, pvals) for pvals in PROBS_UPPER]
        BS_LOWER_DIFF = np.abs(np.array(BSvals) - np.array(BS_LOWER))
        BS_UPPER_DIFF = np.abs(np.array(BSvals) - np.array(BS_UPPER))
        BS_UNCERTAINTY = np.max([BS_LOWER_DIFF, BS_UPPER_DIFF], axis=0)

        # for c, auc, bs in zip(cols_new, AUCvals, BSvals):
        for i, c in enumerate(cols_new):
            auc, bs = AUCvals[i], BSvals[i]

            auc_err, bs_err = AUC_UNCERTAINTY[i], BS_UNCERTAINTY[i]
            auc_str = f"{auc:.2f}$\pm${auc_err:.2f}"
            bs_str = f"{bs:.2f}$\pm${bs_err:.2f}"
            df_auc.loc[c, latemode] = auc
            df_bs.loc[c, latemode] = bs
            df_auc_annot.loc[c, latemode] = auc_str
            df_bs_annot.loc[c, latemode] = bs_str

            if c.split(" ")[0] == "td":
                sortval = 0
            else:
                sortval = 1
            df_auc.loc[c, "sortval"] = sortval
            df_bs.loc[c, "sortval"] = sortval
            df_auc_annot.loc[c, "sortval"] = sortval
            df_bs_annot.loc[c, "sortval"] = sortval

    print(df_auc)
    print(df_auc_annot)

    df_auc = df_auc.sort_values("sortval").drop("sortval", axis=1)
    df_bs = df_bs.sort_values("sortval").drop("sortval", axis=1)
    df_auc_annot = df_auc_annot.sort_values("sortval").drop("sortval", axis=1)
    df_bs_annot = df_bs_annot.sort_values("sortval").drop("sortval", axis=1)

    df_auc = df_auc[["simult", "baseline", "after irr"]]
    df_bs = df_bs[["simult", "baseline", "after irr"]]
    df_auc_annot = df_auc_annot[["simult", "baseline", "after irr"]]
    df_bs_annot = df_bs_annot[["simult", "baseline", "after irr"]]
    print(df_auc)

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    # sns.heatmap(df_auc, annot=True, fmt=".2f", cbar=False, cmap="RdYlGn", vmin=0, vmax=1, ax=ax1)
    sns.heatmap(df_auc, annot=df_auc_annot, fmt="", cbar=False, cmap="RdYlGn", vmin=0, vmax=1, ax=ax1)
    ax1.set_title("AUC")
    ax1.set_yticks(ax1.get_yticks(), [1, 2, 3, 4], rotation=0)

    # sns.heatmap(df_bs, annot=True, fmt=".2f", cbar=False, cmap="RdYlGn_r", vmin=vmin_bs, vmax=vmax_bs, ax=ax2)
    sns.heatmap(df_bs, annot=df_bs_annot, fmt="", cbar=False, cmap="RdYlGn_r", vmin=vmin_bs, vmax=vmax_bs, ax=ax2)
    ax2.set_title("BS")
    # ax2.set_xticks()
    plt.show()
    pass


def show_nocoreg_classif_results(metric="auc"):
    # metric should be auc, acc, or f1
    lrmodes = ["aggregated", "average"]
    NFTS_INCLUDE = [str(x) for x in [5, 10, 15, "all"]]
    # NFTS_INCLUDE = [str(x) for x in [5]]
    # lrmodes = ["aggregated"]
    metric_td = metric + "_td"
    mode_order = {"NO P T1":0, "NO P T2":1, "DELTA P T2":2, "DELTA T2":3} # for sorting
    df_results_acute = pd.DataFrame()
    df_disp_acute = pd.DataFrame()

    df_results_late = pd.DataFrame()
    df_disp_late = pd.DataFrame()

    df_results_delta = pd.DataFrame()
    df_disp_delta = pd.DataFrame()

    for LRMODE in lrmodes:
        folder = os.path.join(ClassifDir, "no coreg", LRMODE)
        files = os.listdir(folder)
        files_acute = list(filter(lambda f: "acute" in f, files))
        files_late = list(filter(lambda f: "late" in f, files))
        print("Found", len(files), f"csv files ({len(files_acute)} acute + {len(files_late)} late) for LRmode", LRMODE)
        for i, f in enumerate(files_acute):
            flist = f.split("_")
            mode = " ".join(flist[4:6])
            nfts = flist[-1].split("=")[-1][:-4]

            df = pd.read_csv(os.path.join(folder, f), index_col=0)

            vals = df[metric]
            vals_td = df[metric + "_td"]

            row = mode + " " + nfts + " fts"
            # col = metric + " " + LRMODE
            col = f"{metric.upper()} LR {LRMODE}"

            df_results_acute.loc[row, col] = vals.mean()
            df_disp_acute.loc[row, col] = f"{vals.mean():.2f} $\pm$ {vals.std():.2f}"

            df_results_acute.loc[row, "mode"] = mode_order[mode]
            df_disp_acute.loc[row, "mode"] = mode_order[mode]
            df_results_acute.loc[row, "num fts"] = int(nfts) if not nfts == "all" else 1000
            df_disp_acute.loc[row, "num fts"] = int(nfts) if not nfts == "all" else 1000

            row = mode + " " + "td"
            # col = metric + " " + LRMODE
            col = f"{metric.upper()} LR {LRMODE}"

            df_results_acute.loc[row, col] = vals_td.mean()
            df_disp_acute.loc[row, col] = f"{vals_td.mean():.2f} $\pm$ {vals_td.std():.2f}"

            df_results_acute.loc[row, "num fts"] = 2
            df_disp_acute.loc[row, "num fts"] = 2
            df_results_acute.loc[row, "mode"] = mode_order[mode]
            df_disp_acute.loc[row, "mode"] = mode_order[mode]

        for i, f in enumerate(files_late):
            flist = f.split("_")
            mode = " ".join(flist[3:5])
            nfts = flist[-1].split("=")[-1][:-4]
            latemode = " ".join(flist[6].split("-"))
            if nfts in NFTS_INCLUDE:
                print(mode, nfts, latemode)
                df = pd.read_csv(os.path.join(folder, f), index_col=0)

                vals = df[metric]
                vals_td = df[metric + "_td"]

                if latemode == "RF":    # Delta (not acute)
                    row = mode + " " + nfts + " fts"
                    col = f"{metric.upper()} {LRMODE}"

                    df_results_delta.loc[row, col] = vals.mean()
                    df_disp_delta.loc[row, col] = f"{vals.mean():.2f} $\pm$ {vals.std():.2f}"

                    df_results_delta.loc[row, "mode"] = mode_order[mode]
                    df_disp_delta.loc[row, "mode"] = mode_order[mode]
                    df_results_delta.loc[row, "num fts"] = int(nfts) if not nfts == "all" else 1000
                    df_disp_delta.loc[row, "num fts"] = int(nfts) if not nfts == "all" else 1000

                    if str(nfts) == "5":
                        row = mode + " " + "td"
                        col = f"{metric.upper()} {LRMODE}"
                        df_results_delta.loc[row, col] = vals_td.mean()
                        df_disp_delta.loc[row, col] = f"{vals_td.mean():.2f} $\pm$ {vals_td.std():.2f}"

                        df_results_delta.loc[row, "num fts"] = 2
                        df_disp_delta.loc[row, "num fts"] = 2
                        df_results_delta.loc[row, "mode"] = mode_order[mode]
                        df_disp_delta.loc[row, "mode"] = mode_order[mode]

                else:
                    row = mode + " " + nfts + " fts"
                    col = f"{metric.upper()} {LRMODE} {latemode}"
                    df_results_late.loc[row, col] = vals.mean()
                    df_disp_late.loc[row, col] = f"{vals.mean():.2f} $\pm$ {vals.std():.2f}"

                    df_results_late.loc[row, "mode"] = mode_order[mode]
                    df_disp_late.loc[row, "mode"] = mode_order[mode]
                    df_results_late.loc[row, "num fts"] = int(nfts) if not nfts == "all" else 1000
                    df_disp_late.loc[row, "num fts"] = int(nfts) if not nfts == "all" else 1000

                    row = mode + " " + "td"
                    col = f"{metric.upper()} {LRMODE} {latemode}"
                    df_results_late.loc[row, col] = vals_td.mean()
                    df_disp_late.loc[row, col] = f"{vals_td.mean():.2f} $\pm$ {vals_td.std():.2f}"

                    df_results_late.loc[row, "num fts"] = 2
                    df_disp_late.loc[row, "num fts"] = 2
                    df_results_late.loc[row, "mode"] = mode_order[mode]
                    df_disp_late.loc[row, "mode"] = mode_order[mode]

    df_results_acute = df_results_acute.sort_values(by=["mode", "num fts"], axis=0, ascending=True)
    df_results_acute = df_results_acute.drop(["mode", "num fts"], axis=1)
    df_disp_acute = df_disp_acute.sort_values(by=["mode", "num fts"], axis=0, ascending=True)
    df_disp_acute = df_disp_acute.drop(["mode", "num fts"], axis=1)

    df_results_late = df_results_late.sort_values(by=["mode", "num fts"], axis=0, ascending=True)
    df_results_late = df_results_late.drop(["mode", "num fts"], axis=1)
    df_disp_late = df_disp_late.sort_values(by=["mode", "num fts"], axis=0, ascending=True)
    df_disp_late = df_disp_late.drop(["mode", "num fts"], axis=1)

    df_results_delta = df_results_delta.sort_values(by=["mode", "num fts"], axis=0, ascending=True)
    df_results_delta = df_results_delta.drop(["mode", "num fts"], axis=1)
    df_disp_delta = df_disp_delta.sort_values(by=["mode", "num fts"], axis=0, ascending=True)
    df_disp_delta = df_disp_delta.drop(["mode", "num fts"], axis=1)

    fig, ax = plt.subplots(figsize=(11, 4)) # acute
    # sns.heatmap(data=df_results_acute, annot=True, fmt=".2f", cbar=False, cmap="RdYlGn", ax=ax)
    sns.heatmap(data=df_results_acute, annot=df_disp_acute, fmt="", cbar=False, cmap="RdYlGn", ax=ax)
    ax.set_title(f"No coreg acute {metric.upper()}")

    fig, ax = plt.subplots()    # late
    print(df_results_late.columns.values)
    # cols = df_results_late.columns.values
    cols_new = ['AUC aggregated baseline', 'AUC average baseline', 'AUC aggregated after irr', 'AUC average after irr'] # only changes ORDER of columns: no renaming
    df_results_late, df_disp_late = df_results_late[cols_new], df_disp_late[cols_new]

    sns.heatmap(data=df_results_late, annot=df_disp_late, fmt="", cbar=False, cmap="RdYlGn", ax=ax)
    # sns.heatmap(data=df_results_late, annot=True, fmt=".2f", cbar=False, cmap="RdYlGn", ax=ax)
    ax.set_title(f"No coreg late {metric.upper()}")
    print(ax.get_xticks())
    handles = ax.get_xticks()
    # labs = [f"{metric} {c.split(''' ''')[1]}\n{c.split(''' ''')[-1] if not c.split(''' ''')[-1] == '''RF''' else '''both'''}" for c in df_results_late.columns.values]
    labs = [f"{c.split(''' ''')[1]}\n{''' '''.join(c.split(''' ''')[2:]) if not c.split(''' ''')[-1] == '''RF''' else '''both'''}" for c in df_results_late.columns.values]
    ax.set_xticks(handles, labs, rotation=0)
    ax.xaxis.tick_top()
    # plt.show()

    fig, ax = plt.subplots()    # delta
    print(df_results_delta)
    sns.heatmap(data=df_results_delta, annot=df_disp_delta, fmt="", cbar=False, cmap="RdYlGn", ax=ax)
    ax.set_title(f"No coreg delta {metric.upper()}")
    handles = ax.get_xticks()
    # labs = [f"{c.split(''' ''')[1]}\n{c.split(''' ''')[-1] if not c.split(''' ''')[-1] == '''RF''' else '''both'''}" for
    #         c in df_results_delta.columns.values]
    labs = lrmodes
    print(labs)
    ax.set_xticks(handles, labs, rotation=0)
    # ax.xaxis.tick_top()

    plt.show()
    return 0


def main_loocv_late(MODE="NO P", WEIGHT="T2", LRMODE="aggregated", latemode="baseline",
                    do_hp_tuning=False, Nfts=5, classifier="LOGREG"):
    save_folder = os.path.join(ClassifDir, "no coreg\loocv late")
    if not classifier in ["LOGREG", "RF"]:
        print("Try classifier: LOGREG, RF")
        return 0

    if MODE == "DELTA":
        savename = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_nfts={Nfts}.csv"
        savename_features = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_nfts={Nfts}_selected features.csv"
        x_orig, y_orig = load_delta(WEIGHT="T2", LRMODE=LRMODE, training="all", xer=True, keep_time=False, keep_names=False)
        x_td_orig = x_orig["dose"]  # times does not make sense for delta-radiomics as the features are longitudinal
        x_orig = x_orig.drop("dose", axis=1)
    else:
        savename = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_{latemode}_nfts={Nfts}.csv"
        savename_features = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_{latemode}_nfts={Nfts}_selected features.csv"
        basebool = True if latemode == "baseline" else False
        x_orig, y_orig = load_predict_late_not_delta(MODE=MODE, WEIGHT=WEIGHT, LRMODE=LRMODE, training="all", baseline=basebool, xer=True)
        x_orig["time"] = [int(t[:-3]) for t in x_orig["time"].values]
        x_td_orig = x_orig[["time", "dose"]]
        x_orig = x_orig.drop(["time", "dose", "name", "id", "time_val"], axis=1)

    print("\nLOOCV LATE")
    print(classifier.upper(), MODE, WEIGHT, LRMODE, latemode, "hptuning=", do_hp_tuning, "nfts=", Nfts)
    print(x_orig.shape, y_orig.shape, x_td_orig.shape)

    df_results = pd.DataFrame()             # save y_true, y_pred, y_pred_probability (for final ROC curve)
    df_selected_features = pd.DataFrame()   # save NUMBER OF TIMES each feature were selected
    df_selected_features.loc["tot", "count"] = 0
    sample_size = len(x_orig)

    x_orig.index = list(range(sample_size))     # RESET INDEX
    x_td_orig.index = list(range(sample_size))
    y_orig.index = list(range(sample_size))
    # print(y_orig)

    # LOOCV
    for i in range(sample_size):
        print(f"{i+1} / {sample_size}", end="\t")
        x_test = x_orig.iloc[i]
        y_test = np.array([y_orig.iloc[i]])
        x_train = x_orig.drop(i)
        y_train = y_orig.drop(i)

        if MODE == "DELTA":
            x_td_test = np.array([x_td_orig.iloc[i]])
            x_td_train = x_td_orig.drop(i).values.reshape(-1, 1)
        else:
            x_td_test = np.array(x_td_orig.iloc[i])
            x_td_train = x_td_orig.drop(i).values

        # print(x_train.shape, y_train.shape)
        # print(x_train)
        if not str(Nfts).upper() == "ALL":
            top_fts = mrmr_classif(X=x_train, y=y_train, K=Nfts)
        else:
            top_fts = x_train.columns.values
        # top_fts = x_train.columns.values[:3]

        # print(top_fts)
        x_train = x_train[top_fts].values
        # x_td_train = x_td_train.values

        x_test = x_test[top_fts].values.reshape(1, -1)
        x_td_test = x_td_test.reshape(1, -1)

        # print(x_train.shape, x_td_train.shape, y_train.shape)
        # print(x_test.shape, x_td_test.shape, y_test.shape)

        if do_hp_tuning:
            # print("Finding HPs")
            if classifier == "RF":
                model = RandomForestClassifier()
                param_grid = {"criterion": ["gini", "entropy"], "min_samples_leaf": [1, 10],
                              "min_samples_split": [2], "n_estimators": [50, 1000],
                              "max_depth": [8, None]}
            elif classifier == "LOGREG":
                model = LogisticRegression(max_iter=250)
                param_grid = {"penalty": ['none', 'l2'], "C":np.logspace(-2,2,5)}
            else:
                print(classifier, "not implemented")
                sys.exit()

            # scoremode = "f1"
            scoremode = "roc_auc"
            cv = RepeatedKFold(n_repeats=2, n_splits=2)
            gs = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoremode, cv=cv, verbose=0, n_jobs=-1)
            gs.fit(x_train, y_train)
            print(gs.best_score_)
            print(gs.best_params_)
            params = gs.best_params_
            # sys.exit()
        else:
            params = {'C': 0.01, 'penalty': 'l2'} if classifier == "LOGREG" else {}

        model = RandomForestClassifier(**params) if classifier == "RF" else LogisticRegression(**params)
        model_td = RandomForestClassifier() if classifier == "RF" else LogisticRegression()

        model.fit(X=x_train, y=y_train)
        model_td.fit(X=x_td_train, y=y_train)

        y_pred = model.predict(X=x_test)[0]
        y_td_pred = model_td.predict(X=x_td_test)[0]
        y_true = y_test[0]

        prob = model.predict_proba(X=x_test)[0, 1]  # PROBABILITY OF TRUE (normal treshold = 0.5?)
        prob_td = model_td.predict_proba(X=x_td_test)[0, 1]

        # print(y_true, y_pred, y_td_pred)
        # print(prob, prob_td)

        df_results.loc[i, "y_true"] = y_true
        df_results.loc[i, "y_pred"] = y_pred
        df_results.loc[i, "y_td_pred"] = y_td_pred
        df_results.loc[i, "prob"] = prob
        df_results.loc[i, "prob_td"] = prob_td

        df_selected_features.loc["tot", "count"] += 1
        for ft in top_fts:
            if not ft in df_selected_features.index:
                df_selected_features.loc[ft, "count"] = 1
            else:
                df_selected_features.loc[ft, "count"] += 1

    print(df_results)
    print(df_selected_features)
    df_results.to_csv(os.path.join(save_folder, savename))
    df_selected_features.to_csv(os.path.join(save_folder, savename_features))
    return 1

def main_loocv_simul(MODE="NO P", WEIGHT="T2", LRMODE="aggregated", do_hp_tuning=False, Nfts=5, classifier="LOGREG"):
    save_folder = os.path.join(ClassifDir, "loocv simult")
    if not classifier in ["LOGREG", "RF"]:
        print("Try classifier: LOGREG, RF")
        return 0

    savename = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_simult_nfts={Nfts}.csv"
    savename_features = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_simult_nfts={Nfts}_selected features.csv"

    if MODE == "NO P":
        x_orig, y_orig = load_nop(WEIGHT=WEIGHT, LRMODE=LRMODE, training="all", xer=True, impute=False, keep_names=False)
    elif MODE == "DELTA P":
        x_orig, y_orig = load_delta_P(WEIGHT=WEIGHT, LRMODE=LRMODE, training="all", xer=True, keep_names=False)
        pass
    else:
        print("Try MODE: NO P, DELTA P")
        return 0

    x_td_orig = x_orig[["time", "dose"]]
    x_orig = x_orig.drop(["time", "dose"], axis=1)
    # print(x_orig)
    print(x_orig.shape, x_td_orig.shape, y_orig.shape)

    df_results = pd.DataFrame()
    df_selected_features = pd.DataFrame()
    df_selected_features.loc["tot", "count"] = 0
    sample_size = len(x_orig)

    x_orig.index = list(range(sample_size))     # RESET INDEX
    x_td_orig.index = list(range(sample_size))
    y_orig.index = list(range(sample_size))

    # loocv
    for i in range(sample_size):
        print(f"{i+1} / {sample_size}")#, end="\t")
        x_test = x_orig.iloc[i]
        y_test = np.array([y_orig.iloc[i]])
        x_train = x_orig.drop(i)
        y_train = y_orig.drop(i)

        x_td_test = np.array(x_td_orig.iloc[i])
        x_td_train = x_td_orig.drop(i).values

        if not str(Nfts).upper() == "ALL":
            top_fts = mrmr_classif(X=x_train, y=y_train, K=Nfts)
        else:
            top_fts = x_train.columns.values

        x_train = x_train[top_fts].values
        x_test = x_test[top_fts].values.reshape(1, -1)
        x_td_test = x_td_test.reshape(1, -1)
        # print(x_train.shape, x_td_train.shape, y_train.shape)
        # print(x_test.shape, x_td_test.shape, y_test.shape)

        if do_hp_tuning:
            # print("Finding HPs")
            if classifier == "RF":
                model = RandomForestClassifier()
                param_grid = {"criterion": ["gini", "entropy"], "min_samples_leaf": [1, 10],
                              "min_samples_split": [2], "n_estimators": [50, 1000],
                              "max_depth": [8, None]}
            elif classifier == "LOGREG":
                model = LogisticRegression(max_iter=250)
                param_grid = {"penalty": ['none', 'l2'], "C":np.logspace(-2,2,5)}
            else:
                print(classifier, "not implemented")
                return 0
            # scoremode = "f1"
            scoremode = "roc_auc"
            cv = RepeatedKFold(n_repeats=2, n_splits=2)
            gs = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoremode, cv=cv, verbose=0, n_jobs=-1)
            gs.fit(x_train, y_train)
            print(gs.best_score_)
            print(gs.best_params_)
            params = gs.best_params_
        else:
            params = {'C': 0.01, 'penalty': 'l2'} if classifier == "LOGREG" else {}

        model = RandomForestClassifier(**params) if classifier == "RF" else LogisticRegression(**params)
        model_td = RandomForestClassifier() if classifier == "RF" else LogisticRegression()

        model.fit(X=x_train, y=y_train)
        model_td.fit(X=x_td_train, y=y_train)

        y_pred = model.predict(X=x_test)[0]
        y_td_pred = model_td.predict(X=x_td_test)[0]
        y_true = y_test[0]

        prob = model.predict_proba(X=x_test)[0, 1]  # PROBABILITY OF TRUE (normal treshold = 0.5?)
        prob_td = model_td.predict_proba(X=x_td_test)[0, 1]

        df_results.loc[i, "y_true"] = y_true
        df_results.loc[i, "y_pred"] = y_pred
        df_results.loc[i, "y_td_pred"] = y_td_pred
        df_results.loc[i, "prob"] = prob
        df_results.loc[i, "prob_td"] = prob_td

        df_selected_features.loc["tot", "count"] += 1
        for ft in top_fts:
            if not ft in df_selected_features.index:
                df_selected_features.loc[ft, "count"] = 1
            else:
                df_selected_features.loc[ft, "count"] += 1

    print(df_results)
    print(df_selected_features)
    df_results.to_csv(os.path.join(save_folder, savename))
    df_selected_features.to_csv(os.path.join(save_folder, savename_features))
    return 1


def main_loocv_bootstrapped(PREDMODE="simul", MODE="NO P", WEIGHT="T1", LRMODE="aggregated", do_hp_tuning=False, Nfts=5,
                            classifier="LOGREG", NUM_BOOT=1000):
    if not PREDMODE in ["simul", "after irr", "baseline", "delta"]:
        print("PREDMODE", PREDMODE, "not valid. Try: simul, after irr, baseline")
        return 0
    if not classifier in ["LOGREG", "RF"]:
        print("Try classifier: LOGREG, RF")
        return 0

    if PREDMODE == "simul":
        save_folder = os.path.join(ClassifDir, "loocv simult", "boot")
        savename = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_simult_nfts={Nfts}_bootmatrix.csv"
        savename_td = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_simult_nfts={Nfts}_bootmatrix_td.csv"
        savename_features = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_simult_nfts={Nfts}_featurematrix.csv"
        savename_truth = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_simult_nfts={Nfts}_truthvalues.csv"

        if MODE == "NO P":
            x_orig, y_orig = load_nop(WEIGHT=WEIGHT, LRMODE=LRMODE, training="all", xer=True, impute=False,
                                      keep_names=False)
        elif MODE == "DELTA P":
            x_orig, y_orig = load_delta_P(WEIGHT=WEIGHT, LRMODE=LRMODE, training="all", xer=True, keep_names=False)
            pass
        else:
            print("Try MODE: NO P, DELTA P")
            return 0

        xtd_orig = x_orig[["time", "dose"]]
        x_orig = x_orig.drop(["time", "dose"], axis=1)


    elif PREDMODE.lower() in ["after irr", "baseline", "delta"]:
        save_folder = os.path.join(ClassifDir, "no coreg", "loocv late", "boot")
        if MODE == "DELTA":
            savename = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_nfts={Nfts}_bootmatrix.csv"
            savename_td = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_nfts={Nfts}_bootmatrix_td.csv"
            savename_features = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_nfts={Nfts}_featurematrix.csv"
            savename_truth = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_nfts={Nfts}_truthvalues.csv"
            x_orig, y_orig = load_delta(WEIGHT="T2", LRMODE=LRMODE, training="all", xer=True, keep_time=False,
                                        keep_names=False)
            xtd_orig = x_orig["dose"]  # times does not make sense for delta-radiomics as the features are longitudinal
            x_orig = x_orig.drop("dose", axis=1)

        else:
            savename = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_{PREDMODE}_nfts={Nfts}_bootmatrix.csv"
            savename_td = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_{PREDMODE}_nfts={Nfts}_bootmatrix_td.csv"
            savename_features = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_{PREDMODE}_nfts={Nfts}_featurematrix.csv"
            savename_truth = f"{LRMODE}_{classifier}_{MODE}_{WEIGHT}_{PREDMODE}_nfts={Nfts}_truthvalues.csv"

            x_orig, y_orig = load_predict_late_not_delta(MODE=MODE, WEIGHT=WEIGHT, LRMODE=LRMODE, training="all",
                                                         baseline=PREDMODE, xer=True)
            print(x_orig["time"].dtype)
            if not x_orig["time"].dtype == "int64":
                x_orig["time"] = [int(t[:-3]) for t in x_orig["time"].values]

            xtd_orig = x_orig[["time", "dose"]]
            x_orig = x_orig.drop(["time", "dose"], axis=1)
            print(x_orig.columns)

    else:
        return 0
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    sample_size = len(x_orig)
    x_orig.index = list(range(sample_size))     # RESET INDEX
    xtd_orig.index = list(range(sample_size))
    y_orig.index = list(range(sample_size))

    print(f"\n----- LOADED FOR {NUM_BOOT} BOOTSTRAPPED LOOCV -----")
    descriptor = f"\tpredmode = {PREDMODE}, featurespace = {MODE} {WEIGHT}, LR = {LRMODE}, Nfts = {Nfts}, classifier = {classifier}, num_boot = {NUM_BOOT}"
    print(descriptor)
    print(x_orig.shape, xtd_orig.shape, y_orig.shape)

    df_probabilities = pd.DataFrame(index=list(range(sample_size)), columns=list(range(NUM_BOOT)))   # ALL estimated LOOCV class probabilities (columns) for each bootstrap (rows)
    df_probabilities_td = pd.DataFrame(index=list(range(sample_size)), columns=list(range(NUM_BOOT)))
    df_groundtruth_y = pd.DataFrame(index=list(range(sample_size)), columns=list(range(NUM_BOOT)))  # save all bootstrapped y-vectors
    # print(df_boot_y)

    all_fts = x_orig.columns.values
    df_selected_features_matrix = pd.DataFrame(index=all_fts, columns=list(range(NUM_BOOT)), data=np.zeros(shape=(len(all_fts), NUM_BOOT)), dtype=int)
    # print(df_selected_features_matrix)


    # # DO LOOCV WITH NESTED BOOTSTRAPP
    for i in range(sample_size):
        y_train_loo = y_orig.drop(i)
        x_train_loo = x_orig.drop(i)
        xtd_train_loo = xtd_orig.drop(i)

        y_test = np.array([y_orig.loc[i]])
        x_test_loo = x_orig.loc[i]
        if np.ndim(xtd_orig) == 1:
            xtd_test_loo = np.array([xtd_orig.loc[i]])
            # xtd_train_loo = xtd_train_loo.values.reshape(-1, 1)
        else:
            xtd_test_loo = np.array(xtd_orig.loc[i])

        # print(y_test.shape, x_test_loo.shape, xtd_test_loo.shape)
        # print(descriptor)

        # BOOTSTRAP TRAINING SET
        for j in range(NUM_BOOT):
            print(f"LOOCV {i} / {sample_size} BOOT {j} / {NUM_BOOT}\t", descriptor)
            x_train, y_train = resample(x_train_loo, y_train_loo)
            xtd_train = xtd_train_loo.loc[x_train.index].values

            if np.ndim(xtd_train) == 1:
                xtd_train = xtd_train.reshape(-1, 1)

            if str(Nfts).upper() == "ALL":
                top_fts = x_train.columns.values
            elif str(Nfts).upper() == "DUMMY":
                top_fts = x_train.columns.values[:5]
            else:
                top_fts = mrmr_classif(X=x_train, y=y_train, K=Nfts, show_progress=False)

            x_train = x_train[top_fts].values
            x_test = x_test_loo[top_fts].values.reshape(1, -1)
            xtd_test = xtd_test_loo.reshape(1, -1)

            # print(x_train.shape, xtd_train.shape, y_train.shape)
            # print(x_test.shape, xtd_test.shape, y_test.shape)

            if do_hp_tuning:
                print("HP TUNING NOT IMPLEMENTED")
                return 0
            else:
                params = {'C': 0.01, 'penalty': 'l2'} if classifier == "LOGREG" else {}

            model = RandomForestClassifier(**params) if classifier == "RF" else LogisticRegression(**params)
            modeltd = RandomForestClassifier() if classifier == "RF" else LogisticRegression()

            model.fit(X=x_train, y=y_train)
            modeltd.fit(X=xtd_train, y=y_train)

            p = model.predict_proba(X=x_test)[0, 1]  # 1 sample, predicted proba for class 1 (True) --> [0, 1]
            ptd = modeltd.predict_proba(X=xtd_test)[0, 1]
            # print(p1, p2, y_test[0])

            df_groundtruth_y.loc[i, j] = y_test
            df_probabilities.loc[i, j] = p
            df_probabilities_td.loc[i, j] = ptd
            for ft in top_fts:
                df_selected_features_matrix.loc[ft, j] += 1
        # break
    # for j in range(NUM_BOOT):
    #     print(f"\nNEW BOOT: {j}")
    #     x_boot, y_boot = resample(x_orig, y_orig)
    #     xtd_boot = x_td_orig.loc[x_boot.index]
    #
    #     df_groundtruth_y.loc[:, j] = list(y_boot)  # save bootstrapped ground truth
    #     print(x_boot.shape, xtd_boot.shape, y_boot.shape)
    #     p_vec, ptd_vec, fts_vec = loocv_classif(x_boot, xtd_boot, y_boot, Nfts=Nfts, classifier=classifier, j=j, do_hp_tuning=do_hp_tuning)
    #
    #     df_probabilities.loc[:, j] = p_vec
    #     df_probabilities_td.loc[:, j] = ptd_vec
    #     df_selected_features_matrix.loc[:, j] = fts_vec

    print(df_groundtruth_y)
    print(df_probabilities)
    print(df_probabilities_td)
    print(df_selected_features_matrix)

    savepath = os.path.normpath(os.path.join(save_folder, savename))
    df_probabilities.to_csv(savepath)
    savepath = os.path.normpath(os.path.join(save_folder, savename_td))
    df_probabilities_td.to_csv(savepath)
    savepath = os.path.normpath(os.path.join(save_folder, savename_truth))
    df_groundtruth_y.to_csv(savepath)
    savepath = os.path.normpath(os.path.join(save_folder, savename_features))
    df_selected_features_matrix.to_csv(savepath)
    return 1


def loocv_classif(x1, x2, y, Nfts=5, classifier="LOGREG", j=-1, do_hp_tuning=False):
    # LOOCV
    # for all i in len(y):
    #       train models m1, m2 on x1.drop(i), x2.drop(i) with y.drop(i) as outcome
    #       estimate class probabilities p1i, p2i using m2, m2 on yi
    df_p1 = pd.Series()
    df_p2 = pd.Series()
    df_ft_count = pd.Series(data=np.zeros(len(x1.columns)), index=x1.columns)

    Nobs = len(y)  # sample size
    x1.index = list(range(Nobs))    # RESET INDEX FOR LOOCV
    x2.index = list(range(Nobs))
    y.index = list(range(Nobs))
    for i in range(Nobs):
        print(f"BOOT {j} loocv {i} / {Nobs}") if not j == -1 else print(f"loocv {i} / {Nobs}")
        x1train = x1.drop(i)
        x1test = x1.iloc[i]

        ytrain = np.array(y.drop(i).values)
        ytest = np.array([y.iloc[i]])

        x2train = x2.drop(i)
        if np.ndim(x2) == 1:
            x2test = np.array([x2.iloc[i]])
            x2train = x2train.values.reshape(-1, 1)
        else:
            x2test = np.array(x2.iloc[i])

        # feature selection
        if str(Nfts).upper() == "ALL":
            top_fts = x1train.columns.values
        elif str(Nfts).upper() == "DUMMY":
            top_fts = x1train.columns.values[:5]
        else:
            top_fts = mrmr_classif(X=x1train, y=ytrain, K=Nfts)

        x1train = x1train[top_fts].values
        x1test = x1test[top_fts].values.reshape(1, -1)
        x2test = x2test.reshape(1, -1)

        if do_hp_tuning:
            # print("Finding HPs")
            if classifier == "RF":
                model = RandomForestClassifier()
                param_grid = {"criterion": ["gini", "entropy"], "min_samples_leaf": [1, 10],
                              "min_samples_split": [2], "n_estimators": [50, 1000],
                              "max_depth": [8, None]}
            elif classifier == "LOGREG":
                model = LogisticRegression(max_iter=250)
                param_grid = {"penalty": ['none', 'l2'], "C":np.logspace(-2,2,5)}
            else:
                print(classifier, "not implemented")
                return 0
            # scoremode = "f1"
            scoremode = "roc_auc"
            cv = RepeatedKFold(n_repeats=2, n_splits=2)
            gs = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoremode, cv=cv, verbose=0, n_jobs=-1)
            gs.fit(x_train, y_train)
            print(gs.best_score_)
            print(gs.best_params_)
            params = gs.best_params_
        else:
            params = {'C': 0.01, 'penalty': 'l2'} if classifier == "LOGREG" else {}

        model1 = RandomForestClassifier(**params) if classifier == "RF" else LogisticRegression(**params)
        model2 = RandomForestClassifier() if classifier == "RF" else LogisticRegression()

        model1.fit(X=x1train, y=ytrain)
        model2.fit(X=x2train, y=ytrain)

        p1 = model1.predict_proba(X=x1test)[0, 1]   # 1 sample, predicted proba for class 1 (True) --> [0, 1]
        p2 = model2.predict_proba(X=x2test)[0, 1]

        df_p1.loc[i] = p1
        df_p2.loc[i] = p2

        for ft in top_fts:
            df_ft_count.loc[ft] += 1

    # print(df_p1)
    # print(df_p2)
    # print(df_ft_count)
    return df_p1, df_p2, df_ft_count


def show_loocv_results(PREDMODE="late", plot_roc_curves=False, show_models=["LOGREG", "RF"], num_dec=3):
    if PREDMODE == "late":
        folder = os.path.join(ClassifDir, "no coreg", "loocv late")
    elif PREDMODE in ["acute", "simult", "simultaneous"]:
        folder = os.path.join(ClassifDir, "loocv simult")
    else:
        print("predmode", PREDMODE, "not valid. Try: late, simult")
    # NFTS_INCLUDE = [str(x) for x in [5, 10, 15, "all"]]
    NFTS_INCLUDE = [str(x) for x in [5]]
    # bs_min = 0.05   # lower / upper limit for brier score cell colors
    # bs_max = 0.40

    # files_all = list(filter(lambda f: ".csv" in f and LRMODE in f, os.listdir(folder)))
    files_all = list(filter(lambda f: ".csv" in f, os.listdir(folder)))
    files_features = list(filter(lambda f: "selected features" in f, files_all))
    files_results = list(filter(lambda f: f not in files_features, files_all))
    print("FOUND", len(files_results), "result files,", len(files_features), "feature files")
    df_results_auc = pd.DataFrame()
    df_results_brier = pd.DataFrame()
    for f in files_results:
        # print(f, end="\t")
        ff = f.split("_")
        lrmode = ff[0]
        classifier = ff[1]

        mode = ff[2]
        weight = ff[3]
        predmode = ff[4] if not mode == "DELTA" else "delta"
        num_fts = ff[-1].split("=")[-1][:-4]

        # print(lrmode, classifier, mode, weight, predmode, num_fts)
        if classifier.upper() in show_models and str(num_fts) in NFTS_INCLUDE:
        # if classifier.upper() in show_models:
            # num_fts = int(ff[-1].split("=")[-1][:-4])

            # print(ff)
            # print(lrmode, classifier, mode, weight, predmode, num_fts)

            df = pd.read_csv(os.path.join(folder, f), index_col=0)
            # print(df.columns.values)

            ytrue = df["y_true"]
            probs = df["prob"]
            probs_td = df["prob_td"]

            auc, roc = make_roc_curve_from_probs(ytrue, probs, plot=False)
            auc_td, roc_td = make_roc_curve_from_probs(ytrue, probs_td, plot=False)
            bs = brier_score(ytrue, probs)
            bs_td = brier_score(ytrue, probs_td)
            print()
            print(mode, weight, lrmode, predmode, num_fts, classifier, end="\n")
            print(f"auc={auc:.2f}, auc_td={auc_td:.2f}", end="\t")

            aucskl = roc_auc_score(ytrue, df["y_pred"])
            aucskl_td = roc_auc_score(ytrue, df["y_td_pred"])
            print(f"sklearn: auc={aucskl:.2f}, auc_td={aucskl_td:.2f}")
            print(f"bs={bs:.2f}, bs_td={bs_td:.2f}")

            # col = predmode
            if not mode == "DELTA":
                # col = f"{lrmode} {predmode} {classifier}"
                col = f"{predmode} {lrmode} {classifier}"
                row = f"{mode} {weight} {num_fts} fts"
                numfts = int(num_fts) if not str(num_fts).lower() == "all" else 1000
                df_results_auc.loc[row, col] = auc
                df_results_auc.loc[row, "num fts"] = numfts
                df_results_brier.loc[row, col] = bs
                df_results_brier.loc[row, "num fts"] = numfts

                row = f"{mode} {weight} td"
                df_results_auc.loc[row, col] = auc_td
                df_results_auc.loc[row, "num fts"] = 2
                df_results_brier.loc[row, col] = bs_td
                df_results_brier.loc[row, "num fts"] = 2
                # print(col)
            else:
                col = f"after irr {lrmode} {classifier}"
                row = f"{mode} {weight} {num_fts} fts"
                numfts = int(num_fts) if not str(num_fts).lower() == "all" else 1000
                df_results_auc.loc[row, col] = auc
                df_results_auc.loc[row, "num fts"] = numfts
                df_results_brier.loc[row, col] = bs
                df_results_brier.loc[row, "num fts"] = numfts

                row = f"{mode} {weight} td"
                df_results_auc.loc[row, col] = auc_td
                df_results_auc.loc[row, "num fts"] = 2
                df_results_brier.loc[row, col] = bs_td
                df_results_brier.loc[row, "num fts"] = 2


                col = f"baseline {lrmode} {classifier}"
                row = f"{mode} {weight} {num_fts} fts"
                numfts = int(num_fts) if not str(num_fts).lower() == "all" else 1000
                df_results_auc.loc[row, col] = auc
                df_results_auc.loc[row, "num fts"] = numfts
                df_results_brier.loc[row, col] = bs
                df_results_brier.loc[row, "num fts"] = numfts

                row = f"{mode} {weight} td"
                df_results_auc.loc[row, col] = auc_td
                df_results_auc.loc[row, "num fts"] = 2
                df_results_brier.loc[row, col] = bs_td
                df_results_brier.loc[row, "num fts"] = 2

            # if plot_roc_curves:
            # if plot_roc_curves and auc > 0.70:
            if plot_roc_curves and mode == "NO P" and weight == "T1":
            #     print(f"({roc[0][0]}, {roc[0][1]}), ({roc[0][-1]}, {roc[1][-1]})")
            #     print(f"({roc_td[0][0]}, {roc_td[0][1]}), ({roc_td[0][-1]}, {roc_td[1][-1]})")
            #     print("fpr=", [f"{x:.1f}" for x in roc_td[0]])
            #     print("tpr=", [f"{x:.1f}" for x in roc_td[1]])

                plt.plot(roc[0], roc[1], "x-", label=f"auc={auc:.2f}")
                plt.plot(roc_td[0], roc_td[1], "x-", label=f"td auc={auc_td:.2f}")
                plt.plot([0, 1], [0, 1], "--", color="black")
                plt.legend()
                plt.title(f"{mode} {weight} {predmode} {num_fts} fts LR{lrmode} {classifier}")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.show()

    if PREDMODE == "late":
        df_results_auc = df_results_auc.sort_values(by="num fts", axis=0).drop("num fts", axis=1)
        df_results_brier = df_results_brier.sort_values(by="num fts", axis=0).drop("num fts", axis=1)

        df_auc_delta = df_results_auc.filter(like="DELTA T2", axis=0)
        df_auc_nodelta = df_results_auc.drop(df_auc_delta.index, axis=0)
        df_brier_delta = df_results_brier.filter(like="DELTA T2", axis=0)
        df_brier_nodelta = df_results_brier.drop(df_brier_delta.index, axis=0)

        # cols = df_results_auc.columns.values
        # print(df_results)   # have to print results for sorting to work?????? nå må jeg gå hjem tror jeg...
        # cols.sort()
        # df_results = df_results[cols]

        df_auc_delta = df_auc_delta.drop(list(filter(lambda c:"after irr" in c, df_auc_delta.columns.values)), axis=1)  # drop after irr as equal to baseline
        df_auc_delta.columns = [f"delta {''' '''.join(c.split(''' ''')[1:])}" for c in list(df_auc_delta.columns)]
        df_brier_delta = df_brier_delta.drop(list(filter(lambda c:"after irr" in c, df_brier_delta.columns.values)), axis=1)
        df_brier_delta.columns = [f"delta {''' '''.join(c.split(''' ''')[1:])}" for c in list(df_brier_delta.columns)]

        # plot all in same
        # fig, ax = plt.subplots()
        # sns.heatmap(df_results, annot=True, fmt=".2f", cbar=False, cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
        # print(ax.get_xticks())

        # ax.set_xticks(ax.get_xticks(), cols, rotation=0)
        # plt.show()
        # plt.close()

        cols = list(df_auc_nodelta.columns)
        cols.sort()
        df_auc_nodelta = df_auc_nodelta[cols]
        print(cols)

        fig, ax = plt.subplots()
        sns.heatmap(df_auc_nodelta, annot=True, fmt=f".{num_dec}f", cbar=False, cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
        cols = [f"{''' '''.join(c.split(''' ''')[:2])}\n{''' '''.join(c.split(''' ''')[2:])}" for c in cols]
        cols = [f"{''' '''.join(c.split(''' ''')[:-1])}\n{c.split(''' ''')[-1]}" for c in cols]     # formatting
        # print(cols)
        ax.set_xticks(ax.get_xticks(), cols, rotation=0)

        fig, ax = plt.subplots()
        sns.heatmap(df_auc_delta, annot=True, fmt=f".{num_dec}f", cbar=False, cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
        cols = df_auc_delta.columns.values
        cols = [f"{c.split(''' ''')[0]}\n{''' '''.join(c.split(''' ''')[1:])}" for c in cols]
        rows = df_auc_delta.index.values
        ax.set_xticks(ax.get_xticks(), cols, rotation=0)
        ax.set_yticks(ax.get_yticks(), rows, rotation=0)

        cols = list(df_brier_nodelta.columns)
        cols.sort()
        df_brier_nodelta = df_brier_nodelta[cols]
        cols = [f"{''' '''.join(c.split(''' ''')[:2])}\n{''' '''.join(c.split(''' ''')[2:])}" for c in cols]
        cols = [f"{''' '''.join(c.split(''' ''')[:-1])}\n{c.split(''' ''')[-1]}" for c in cols]     # formatting

        fig, ax = plt.subplots()
        sns.heatmap(df_brier_nodelta, annot=True, fmt=f".{num_dec}f", cbar=False, cmap="RdYlGn_r", ax=ax, vmin=vmin_bs, vmax=vmax_bs)
        ax.set_xticks(ax.get_xticks(), cols, rotation=0)

        cols = df_brier_delta.columns
        cols = [f"{c.split(''' ''')[0]}\n{''' '''.join(c.split(''' ''')[1:])}" for c in cols]

        fig, ax = plt.subplots()
        sns.heatmap(df_brier_delta, annot=True, fmt=f".{num_dec}f", cbar=False, cmap="RdYlGn_r", ax=ax, vmin=vmin_bs, vmax=vmax_bs)
        ax.set_xticks(ax.get_xticks(), cols, rotation=0)
        plt.show()

    else:
        print(df_results_auc)
        df_results_auc = df_results_auc.sort_values(by="num fts", axis=0)
        print(df_results_auc)
        df_results_auc = df_results_auc.drop("num fts", axis=1)
        df_results_brier = df_results_brier.sort_values(by="num fts", axis=0).drop("num fts", axis=1)

        fig, ax = plt.subplots()
        sns.heatmap(df_results_auc, annot=True, fmt=f".{num_dec}f", cbar=False, cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
        cols = list(df_results_auc.columns)
        cols = [f"AUC\n{''' '''.join(c.split(''' ''')[1:])}" for c in cols]
        ax.set_xticks(ax.get_xticks(), cols, rotation=0)

        fig, ax = plt.subplots()
        sns.heatmap(df_results_brier, annot=True, fmt=f".{num_dec}f", cbar=False, cmap="RdYlGn_r", ax=ax, vmin=vmin_bs, vmax=vmax_bs)
        cols = list(df_results_brier.columns)
        cols = [f"BS\n{''' '''.join(c.split(''' ''')[1:])}" for c in cols]
        ax.set_xticks(ax.get_xticks(), cols, rotation=0)
        plt.show()
    pass


def show_loocv_boot(PREDMODE="simult", plot_roc_curves=True):
    if PREDMODE == "simult":
        folder = os.path.join(ClassifDir, "loocv simult", "boot")
    elif PREDMODE == "late":
        folder = os.path.join(ClassifDir, "no coreg", "loocv late", "boot")
    else:
        print("Try predmode: simult, late")
    files = os.listdir(folder)
    files = list(filter(lambda f: "bootmatrix" in f, files))
    files = list(filter(lambda f: "_td" not in f, files))
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', "{:,.2f}".format)
    roc_alpha = 0.50    # alpha value for each of the nboot ROC curves

    # files = list(filter(lambda f: "DELTA_" in f, files))
    df_auc_results = pd.DataFrame()
    df_auc_annot = pd.DataFrame()
    df_auc_noboot = pd.DataFrame()

    df_bs_results = pd.DataFrame()
    df_bs_annot = pd.DataFrame()
    df_bs_noboot = pd.DataFrame()

    for f in files:
        # print(f)
        ff = f.split("_")
        print(ff)
        lrmode, classifier, mode, weight, predmode = ff[:5]
        if mode.upper() == "DELTA":
            weight, predmode = ff[-3], mode
            nfts = ff[-2].split("=")[-1]

        else:
            weight, predmode = ff[3:5]
            nfts = ff[-2].split("=")[-1]

        # print(lrmode, mode, weight, nfts, predmode, classifier)

        f_td = f[:-4] + "_td" + ".csv"
        f_gt = "_".join(ff[:-1]) + "_truthvalues.csv"
        f_fts = "_".join(ff[:-1]) + "_featurematrix.csv"

        P = pd.read_csv(os.path.join(folder, f), index_col=0)
        n_obs, n_boot = P.shape
        descriptor = f"LOOCV {mode} {weight} {predmode} ($N_{{obs}}$={n_obs}) LR{lrmode} nfts={nfts} {classifier} {n_boot}*boot training"
        print("\n", descriptor)

        Ptd = pd.read_csv(os.path.join(folder, f_td), index_col=0)
        Ygt = pd.read_csv(os.path.join(folder, f_gt), index_col=0)
        F = pd.read_csv(os.path.join(folder, f_fts), index_col=0)


        # print(P)
        print("nobs, nboot =", P.shape)

        AUC_SCORES = []
        AUC_TD_SCORES = []
        BS_SCORES = []
        BS_TD_SCORES = []

        if plot_roc_curves:
            fig, (ax1, ax2) = plt.subplots(ncols=2,  figsize=(14, 6))

        # loop over all bootstraps
        for j in range(n_boot):
            j = str(j)
            # ygt = Ygt.loc[:, j]
            ygt = np.array([em == "[ True]" for em in list(Ygt.loc[:, j])])
            # print(ygt)

            pj = P.loc[:, j]
            ptdj = Ptd.loc[:, j]
            # print(ygt.shape, pj.shape, ptdj.shape)
            # print(np.shape(rocj))
            # print(pj)
            print(".", end="") if not int(j) % 2 else 0
            aucj, rocj = make_roc_curve_from_probs(ytrue=ygt, probs=pj, plot=False)
            auctdj, roctdj = make_roc_curve_from_probs(ygt, ptdj, plot=False)
            AUC_SCORES.append(aucj)
            AUC_TD_SCORES.append(auctdj)

            bsj = brier_score(ygt, pj)
            bstdj = brier_score(ygt, ptdj)
            BS_SCORES.append(bsj)
            BS_TD_SCORES.append(bstdj)

            if plot_roc_curves:
                ax1.plot(rocj[0], rocj[1], ":", c="black", alpha=roc_alpha, zorder=1)
                ax2.plot(roctdj[0], roctdj[1], ":", c="black", alpha=roc_alpha, zorder=1)
        print()

        # collect loocv no boot results
        y_orig, p_orig, ptd_orig = get_loocv_vals_noboot(predmode, lrmode, classifier, mode, weight, nfts)
        auc_orig, roc_orig = make_roc_curve_from_probs(y_orig, p_orig)
        auc_td_orig, roc_td_orig = make_roc_curve_from_probs(y_orig, ptd_orig)
        bs_orig = brier_score(y_orig, p_orig)
        bs_td_orig = brier_score(y_orig, ptd_orig)

        auc = np.mean(AUC_SCORES)
        auc_td = np.mean(AUC_TD_SCORES)
        # auc_str = f"{auc:.2f} $\pm$ {np.std(AUC_SCORES):.2f}"
        # auc_td_str = f"{auc_td:.2f} $\pm$ {np.std(AUC_TD_SCORES):.2f}"
        auc_str = f"{auc:.2f} $\pm$ {np.std(AUC_SCORES):.2f} ({auc_orig:.2f})"
        auc_td_str = f"{auc_td:.2f} $\pm$ {np.std(AUC_TD_SCORES):.2f} ({auc_td_orig:.2f})"


        bs = np.mean(BS_SCORES)
        bs_td = np.mean(BS_TD_SCORES)
        bs_str = f"{bs:.2f} $\pm$ {np.std(BS_SCORES):.2f}"
        bs_td_str = f"{bs_td:.2f} $\pm$ {np.std(BS_TD_SCORES):.2f}"

        print(f"AUC = {auc:.2f} +- {np.std(AUC_SCORES):.2f},\tAUCtd = {auc_td:.2f} +- {np.std(AUC_TD_SCORES):.2f}")
        print(f"BS = {bs:.2f} +- {np.std(BS_SCORES):.2f},\tBStd = {bs_td:.2f} +- {np.std(BS_TD_SCORES):.2f}")


        row = f"{predmode} {mode} {weight}"
        col = f"LR{lrmode} {classifier}"
        df_auc_results.loc[row, col] = auc
        df_auc_annot.loc[row, col] = auc_str
        df_bs_results.loc[row, col] = bs
        df_bs_annot.loc[row, col] = bs_str
        # df_auc_noboot.loc[row, col] = auc_orig
        # df_bs_noboot.loc[row, col] = bs_orig
        for df in [df_auc_results, df_auc_annot, df_bs_results, df_bs_annot]:
            df.loc[row, "num fts"] = int(nfts) if not nfts.lower() == "all" else 999

        row = f"{predmode} {mode} {weight} td"
        df_auc_results.loc[row, col] = auc_td
        df_auc_annot.loc[row, col] = auc_td_str
        df_bs_results.loc[row, col] = bs_td
        df_bs_annot.loc[row, col] = bs_td_str
        # df_auc_noboot.loc[row, col] = auc_td_orig
        # df_bs_noboot.loc[row, col] = bs_td_orig
        for df in [df_auc_results, df_auc_annot, df_bs_results, df_bs_annot]:
            df.loc[row, "num fts"] = 2

        if plot_roc_curves:
            # y_orig, p_orig, ptd_orig = get_loocv_vals_noboot(predmode, lrmode, classifier, mode, weight, nfts)
            # auc_orig, roc_orig = make_roc_curve_from_probs(y_orig, p_orig)
            # auc_td_orig, roc_td_orig = make_roc_curve_from_probs(y_orig, ptd_orig)
            ax1.plot(roc_orig[0], roc_orig[1], "--", color="red", label=f"loocv no boot: auc={auc_orig:.2f}", lw=3, zorder=10)
            ax2.plot(roc_td_orig[0], roc_td_orig[1], "--", color="red", label=f"loocv no boot: auc={auc_td_orig:.2f}", lw=3, zorder=10)

            for ax in [ax1, ax2]:
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.plot([0, 1], [0, 1], "--", color="blue", lw=2, zorder=5)
                ax.grid(True, zorder=0)
            ax1.set_title(f"{nfts} fts, auc={auc_str}, bs={bs_str}")
            ax2.set_title(f"time + dose, auc={auc_td_str}, bs={bs_td_str}")
            fig.suptitle(descriptor)
            ax1.legend(loc="lower right")
            ax2.legend(loc="lower right")
            plt.show()

        fts_stats = True
        if fts_stats:
            F = F.loc[F.mean(axis=1).sort_values(ascending=False).index]
            print(F.head(10))
            print(F.head(5).T.describe(percentiles=[.025, .25, .50, .75, .975]).T)

    print(df_auc_results)

    df_auc_results = df_auc_results.sort_values(by="num fts").drop("num fts", axis=1)
    df_auc_annot = df_auc_annot.sort_values(by="num fts").drop("num fts", axis=1)
    df_bs_results = df_bs_results.sort_values(by="num fts").drop("num fts", axis=1)
    df_bs_annot = df_bs_annot.sort_values(by="num fts").drop("num fts", axis=1)

    cols = df_auc_results.columns.values

    fig, ax = plt.subplots()
    sns.heatmap(data=df_auc_results, annot=df_auc_annot, fmt="", cbar=False, vmin=0.0, vmax=1.0, cmap="RdYlGn")

    cols = [f"AUC\n{c}" for c in cols]
    ax.set_xticks(ax.get_xticks(), cols, rotation=0)
    fig.suptitle(descriptor)
    plt.show()

    pass


def get_loocv_vals_noboot(predmode, lrmode, classifier, mode, weight, nfts):
    # helper function: link loocv no boot results to boot loocv results
    if predmode.lower() in ["after irr", "baseline", "delta"]:
        folder = os.path.join(ClassifDir, "no coreg", "loocv late")
        if mode.upper() == "DELTA":
            filename = f"{lrmode}_{classifier}_{mode}_{weight}_nfts={nfts}.csv"
        else:
            filename = f"{lrmode}_{classifier}_{mode}_{weight}_{predmode}_nfts={nfts}.csv"
    elif predmode.lower() in ["acute", "simult", "simultaneous"]:
        folder = os.path.join(ClassifDir, "loocv simult")
        filename = f"{lrmode}_{classifier}_{mode}_{weight}_simult_nfts={nfts}.csv"
    else:
        print("predmode", predmode, "not valid. Try: simult, after irr, baseline, delta")
    df = pd.read_csv(os.path.join(folder, filename), index_col=0)
    # print(df)
    y = df["y_true"]
    p = df["prob"]
    ptd = df["prob_td"]
    return y, p, ptd


def find_optimal_number_of_fts(model="LOGREG", MIN_FTS=2):
    from sklearn.feature_selection import RFECV, RFE

    x, y = load_nop(WEIGHT="T1", LRMODE="aggregated", xer=True, training="all")
    print(x.shape, y.shape)

    if model.upper() == "LOGREG":
        m = LogisticRegression()
    elif model.upper() == "RF":
        m = RandomForestClassifier()
    else:
        print("TRY model RF, LOGREG")
        return 0

    # rfe = RFE(estimator=m, step=1, verbose=1)
    # rfe = RFECV(estimator=m, step=1, scoring="accuracy", cv=RepeatedKFold(n_repeats=1, n_splits=2), verbose=1, min_features_to_select=MIN_FTS)
    # rfe = RFECV(estimator=m, step=1, scoring="accuracy", cv=2, verbose=1, min_features_to_select=MIN_FTS)
    rfe = RFECV(estimator=m, step=1, scoring="roc_auc", cv=2, verbose=1, min_features_to_select=MIN_FTS)
    rfe.fit(x, y)
    print(rfe.ranking_)
    # xred = rfe.transform(x)
    # print(xred)
    print(rfe.get_feature_names_out())
    print("Nfts selected =", rfe.n_features_)

    fig, ax = plt.subplots()
    ax.set_xlabel("Num fts")
    ax.set_ylabel("CV acc")
    ax.plot(range(MIN_FTS, len(rfe.grid_scores_) + MIN_FTS), rfe.grid_scores_, "x-")
    plt.show()
    return 1


def calibration_curve(ytrue, probs, n_bins=3, plot=False):
    from sklearn.calibration import calibration_curve
    ptrue, ppred = calibration_curve(ytrue, probs, n_bins=n_bins)
    # print(ytrue)
    # print(probs)
    # print(ptrue, ppred)

    if plot:
        plt.plot([0, 1], [0, 1], "--")
        plt.plot(ppred, ptrue, marker=".")
        plt.show()

    return ptrue, ppred


def classification_xer_timedose_all(NREP=1, do_hp_tuning=False, do_loocv=True):
    # use all 347 saliva data points --> xer thresh = y
    # use time, dose to classify / predict prob of y = True
    # import statsmodels.api as sm
    import statsmodels.formula.api as sm

    saliv = load_saliva(melt=True).reset_index(drop=True)
    saliv.loc[:, "dose"] = [dose_to_name(nm, time, ignore_exp=True) for nm, time in zip(saliv["name"].values, saliv["time"].values)]
    saliv.loc[:, "xer"] = binary_thresh_xerostomia(saliv)
    print(saliv)
    x_orig = saliv[["time", "dose"]]
    y_xer = saliv["xer"]
    y_ctr = saliv["ctr"]
    num_obs = len(y_xer)

    saliv.loc[:, "xerval"] = [int(x) for x in saliv["xer"]]    # statsmodels needs this
    log_sm = sm.logit(formula='xerval ~ time + dose', data=saliv).fit()
    print(log_sm.summary())
    log_sm = sm.logit(formula='xerval ~ time + dose + time:dose', data=saliv).fit()
    print(log_sm.summary())
    # log_sm = sm.logit(formula='xerval ~ dose + time:dose', data=saliv).fit()
    # print(log_sm.summary())

    # saliv.loc[:, "ctrval"] = [int(x) for x in saliv["ctr"]]
    # log_sm = sm.logit(formula='ctrval ~ time + dose + time:dose', data=saliv).fit()
    # print(log_sm.summary())

    print("\nxer ~ logreg(time + dose)")
    log_skl = LogisticRegression()
    log_skl.fit(x_orig, y_xer)
    probs = log_skl.predict_proba(x_orig)[:, -1]
    auc, _ = make_roc_curve_from_probs(y_xer, probs)
    print(log_skl.feature_names_in_)
    print(f"intercept = {log_skl.intercept_.flatten()[0]:.2f}", "coefs =", "\t".join("{:.2e}".format(cf) for cf in log_skl.coef_.flatten()))
    print(f"Logreg sklearn acc={log_skl.score(x_orig, y_xer):.2f}, auc={auc:.2f}, bs={brier_score(y_xer, probs):.2f}:\t"
          f"{log_skl.intercept_[0]:.2f} + {log_skl.coef_[0, 0]:.2e} {log_skl.feature_names_in_[0]} + {log_skl.coef_[0, 1]:.2e} {log_skl.feature_names_in_[1]}")
    # print(roc_auc_score(y_xer, probs))

    print("\nxer ~ RF(time + dose)")
    rf = RandomForestClassifier(oob_score=True)
    rf.fit(x_orig, y_xer)
    probs = rf.predict_proba(x_orig)[:, -1]
    acc = rf.score(x_orig, y_xer)
    auc, _ = make_roc_curve_from_probs(y_xer, probs)
    bs = brier_score(y_xer, probs)
    print(f"acc={acc:.2f}, acc_oob={rf.oob_score_:.2f}, auc={auc:.2f}, bs={bs:.2f}")
    print("importances:", rf.feature_importances_)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    x_poly = poly.fit_transform(x_orig)
    loginter_skl = LogisticRegression()
    # print(x_poly)
    loginter_skl.fit(x_poly, y_xer)
    print("\nxer ~ logreg(time + dose + time:dose)")
    print(f"intercept = {loginter_skl.intercept_.flatten()[0]:.2f}", "coefs =", "\t".join("{:.2e}".format(cf) for cf in loginter_skl.coef_.flatten()))
    acc = accuracy_score(y_xer, loginter_skl.predict(x_poly))
    probs = loginter_skl.predict_proba(x_poly)[:, -1]
    auc, _ = make_roc_curve_from_probs(y_xer, probs)
    bs = brier_score(y_xer, probs)
    print(f"acc={acc:.2f}, auc={auc:.2f}, bs={bs:.2f}")

    print("\nxer ~ RF(time + dose + time:dose")
    rf = RandomForestClassifier(oob_score=True)
    rf.fit(x_poly, y_xer)
    probs = rf.predict_proba(x_poly)[:, -1]
    acc = rf.score(x_poly, y_xer)
    auc, _ = make_roc_curve_from_probs(y_xer, probs)
    bs = brier_score(y_xer, probs)
    print(f"acc={acc:.2f}, acc_oob={rf.oob_score_:.2f}, auc={auc:.2f}, bs={bs:.2f}")
    print("importance:", rf.feature_importances_)

    if not do_loocv:
        return 1
    #LOOCV
    print("\nLOOCV")
    i = 0
    df_loocv = pd.DataFrame()
    for n in range(num_obs):
        print(f"LOOCV {n} / {num_obs}")
        x_train = x_orig.drop(n)
        x_test = np.reshape(x_orig.loc[n].values, (1, -1))

        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        x_inter_train = poly.fit_transform(x_train)
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        x_inter_test = poly.fit_transform(x_test)

        y_train = y_xer.drop(n)
        y_true = y_xer.loc[n]   # ground truth

        # print(x_train.shape, x_test.shape)
        # print(x_inter_train.shape, x_inter_test.shape)
        # print(y_train.shape, y_true)
        if do_hp_tuning:
            params_rf = rf_hyperparamtuning(x_train, y_train)
            params_inter_rf = rf_hyperparamtuning(x_inter_train, y_train)
        else:
            params_rf = {}
            params_inter_rf = {}

        for r in range(NREP):
            lr = LogisticRegression()
            lr_inter = LogisticRegression()
            rf = RandomForestClassifier(**params_rf)
            rf_inter = RandomForestClassifier(**params_inter_rf)

            lr.fit(x_train, y_train)
            rf.fit(x_train, y_train)
            prob_lr = lr.predict_proba(x_test)[0, -1]
            prob_rf = rf.predict_proba(x_test)[0, -1]
            # print(prob_lr, prob_rf)

            lr_inter.fit(x_inter_train, y_train)
            rf_inter.fit(x_inter_train, y_train)
            prob_lr_inter = lr_inter.predict_proba(x_inter_test)[0, -1]
            prob_rf_inter = rf_inter.predict_proba(x_inter_test)[0, -1]
            # print(prob_lr_inter, prob_rf_inter)
            # print(lr_inter.classes_)

            df_loocv.loc[i, "idx"] = n
            df_loocv.loc[i, "rep"] = r
            df_loocv.loc[i, ["y_gt", "p_lr", "p_rf", "p_inter_lr", "p_inter_rf"]] = [y_true, prob_lr, prob_rf, prob_lr_inter, prob_rf_inter]
            i += 1

        # print(df_loocv)
    savename = f"loocv_probs_timedose_nrep={NREP}_hptune={do_hp_tuning}.csv"
    df_loocv.to_csv(os.path.join(ClassifDir, "timedose all", savename))
    pass


def show_loocv_timedose(nrep=100, hptune=False):
    path = os.path.join(ClassifDir, "timedose all", f"loocv_probs_timedose_nrep={nrep}_hptune={hptune}.csv")
    df_loaded = pd.read_csv(path, index_col=0)
    df_mean = df_loaded.groupby("idx").mean()
    df_sd = df_loaded.groupby("idx").std()
    df_pc5 = df_loaded.groupby("idx").quantile(q=0.05)  # 5'th percentile
    df_pc25 = df_loaded.groupby("idx").quantile(q=0.25)  # 95'th percentile
    df_pc75 = df_loaded.groupby("idx").quantile(q=0.75)  # 95'th percentile
    df_pc95 = df_loaded.groupby("idx").quantile(q=0.95)  # 95'th percentile
    num_obs = len(df_mean)
    num_xer = len(df_mean[df_mean["y_gt"] == 1])
    print(f"\nLOOCV\nHaving {num_xer} of {num_obs} xer")
    print(df_loaded.columns.values)
    y_true = df_mean["y_gt"]

    df_scores = pd.DataFrame()
    df_annot = pd.DataFrame()
    rownames = ["Linear reg", "Linear reg w/ inter", "RF reg", "RF reg w/ inter"]
    # for probs, row in zip([probs_lr, probs_lr_inter, probs_rf, probs_rf_inter], rownames):
    for p_col, row in zip(["p_lr", "p_inter_lr", "p_rf", "p_inter_rf"], rownames):
        probs = df_mean[p_col]
        # probs_sd = df_sd[p_col]
        # probs_upper = probs + probs_sd
        # probs_lower = probs - probs_sd
        # probs_lower = df_pc5[p_col]
        # probs_upper = df_pc95[p_col]
        probs_lower = df_pc25[p_col]
        probs_upper = df_pc75[p_col]

        y_pred = [p >= .5 for p in probs]
        y_pred_lower = [p >= .5 for p in probs_lower]
        y_pred_upper = [p >= .5 for p in probs_upper]
        acc = accuracy_score(y_true, y_pred)
        acc_lw = accuracy_score(y_true, y_pred_lower)
        acc_up = accuracy_score(y_true, y_pred_upper)
        print(f"{row}: acc={acc:.2f} ({acc_lw:.2f}, {acc_up:.2f})")
        auc = roc_auc_score(y_true, probs)
        auc_lw = roc_auc_score(y_true, probs_lower)
        auc_up = roc_auc_score(y_true, probs_upper)
        print(f"\t\tauc={auc:.2f} ({auc_lw:.2f}, {auc_up:.2f})")
        bs = brier_score(y_true, probs)
        bs_lw = brier_score(y_true, probs_lower)
        bs_up = brier_score(y_true, probs_upper)
        print(f"\t\tbs={bs:.2f} ({bs_lw:.2f}, {bs_up:.2f})")

        df_scores.loc[row, "Acc"] = acc
        df_scores.loc[row, "AUC"] = auc
        df_scores.loc[row, "BS"] = bs
    sns.heatmap(df_scores, annot=True, fmt=f".{num_dec}f", cbar=False, cmap="RdYlGn", vmin=0, vmax=1)
    plt.show()
    pass


def get_best_fts_loocv(MODE, WEIGHT, LRMODE, NFTS, CLASSIFIER, predict_late=False, pick_thresh=0.5, plot=True):
    if not predict_late:
        folder = os.path.join(ClassifDir, "loocv simult")
        filename = "_".join([LRMODE, CLASSIFIER, MODE, WEIGHT]) + f"_simult_nfts={NFTS}_selected features.csv"
    else:
        folder = os.path.join(ClassifDir, "no coreg", "loocv late")
        if not str(predict_late) in ["baseline", "after irr", "delta"]:
            print("Try predict_late: baseline, after irr, delta")
        if MODE == "DELTA":
            filename = f"{LRMODE}_{CLASSIFIER}_{MODE}_{WEIGHT}_nfts={NFTS}_selected features.csv"
        else:
           filename = f"{LRMODE}_{CLASSIFIER}_{MODE}_{WEIGHT}_{predict_late}_nfts={NFTS}_selected features.csv"
    df = pd.read_csv(os.path.join(folder, filename), index_col=0)
    # print("LOADED:", df.shape)
    N_picks_tot = df.loc["tot", "count"]
    df = df.drop("tot")

    index_dict = get_feature_index_fsps(LRMODE, MODE, WEIGHT)
    df.loc[:, "idx"] = [index_dict[ft] for ft in df.index]
    # print(df)
    # print(list(index_dict.values()))
    df = df.sort_values(by="idx")
    Nfts_tot = len(df)
    Nfts_picked = len(df[df["count"] > 0])
    # print(N_picks_tot)
    df.loc[:, "picks rel"] = np.array(df["count"].values) / N_picks_tot
    df_red = df[df["picks rel"] >= pick_thresh]
    N_above_thresh = len(df_red)
    # print(df_red)

    fts_all = df.index.values
    fts_red = df_red.index.values
    if plot:
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        x1 = list(index_dict.values())
        # x1 = list(range(len(fts_all)))
        y1 = []
        for ft in index_dict.keys():
            if ft in df.index:
                y1.append(df.loc[ft, "picks rel"])
            else:
                y1.append(0)
        # y1 = list(df["picks rel"])
        ax1.vlines(x1, ymin=0, ymax=y1, linestyles="-", linewidths=1, colors="orange")
        ax1.plot(x1, y1, "o", color="black", markersize=1.0)
        # ax1.hlines([pick_thresh], xmin=0, xmax=np.max(x1), linestyles="--", linewidths=0.5, colors="black")
        ax1.plot(x1, np.repeat(pick_thresh, len(x1)), "--", color="black")
        ax1.set_ylabel("Fraction of times a feature was selected")
        ax1.set_xlabel("Feature number")

        # x2 = fts_red
        x2 = np.argwhere([ft in fts_red for ft in df.index.values]).reshape(-1)

        y2 = list(df_red["picks rel"])
        ax2.vlines(x2, ymin=0, ymax=y2, linestyles="-", linewidths=1.0, colors="orange")
        ax2.plot(x2, y2, "o", color="black", markersize=1.0)
        # ax2.set_xticks(ax2.get_xticks(), x2, rotation=10)
        # ax2.set_xticks(x2, x2, rotation=10)
        ax2.set_xlabel("Feature number")
        fig.suptitle(f"{filename}\nTot fts: {Nfts_tot}, picked: {Nfts_picked}, picked > {pick_thresh} for {N_picks_tot} models: {N_above_thresh}")
        plt.show()
    return fts_red


def loocv_combined_xer(MODE="NO P", WEIGHT="T2", LRMODE="aggregated", CLASSIFIER="LOGREG", NREP=100, latemode=False, do_hp_tuning=False):
    savefolder = os.path.join(ClassifDir, "td fts combined")
    if not latemode:
        savename = f"loocv_comb_simult_{MODE}_{WEIGHT}_{LRMODE}_{CLASSIFIER}_nrep={NREP}.csv"
        if MODE == "NO P":
            x_orig, y_orig = load_nop(WEIGHT=WEIGHT, LRMODE=LRMODE, training="all", xer=True, impute=False,
                                      keep_names=False)
        elif MODE == "DELTA P":
            x_orig, y_orig = load_delta_P(WEIGHT=WEIGHT, LRMODE=LRMODE, training="all", xer=True, keep_names=False)
            pass
        else:
            print("Try MODE: NO P, DELTA P")
            return 0
        # x_td_orig = x_orig[["time", "dose"]]
        # x_orig = x_orig.drop(["time", "dose"], axis=1)
    else:
        savename = f"loocv_{latemode}_{MODE}_{WEIGHT}_{LRMODE}_{CLASSIFIER}_nrep={NREP}.csv"

        if MODE == "DELTA":
            x_orig, y_orig = load_delta(WEIGHT="T2", LRMODE=LRMODE, training="all", xer=True, keep_time=False,
                                        keep_names=False)
            # x_td_orig = x_orig["dose"]  # times does not make sense for delta-radiomics as the features are longitudinal
            # x_orig = x_orig.drop("dose", axis=1)
        else:
            basebool = True if latemode == "baseline" else False
            x_orig, y_orig = load_predict_late_not_delta(MODE=MODE, WEIGHT=WEIGHT, LRMODE=LRMODE, training="all",
                                                         baseline=basebool, xer=True)

            if x_orig["time"].dtype != "int64":
                x_orig["time"] = [int(t[:-3]) for t in x_orig["time"].values]

            # x_td_orig = x_orig[["time", "dose"]]
            # x_orig = x_orig.drop(["time", "dose", "name", "id", "time_val"], axis=1)
            if "name" in x_orig.columns:
               x_orig = x_orig.drop(["name", "id", "time_val"], axis=1)

    x_orig = x_orig.reset_index(drop=True)
    y_orig = y_orig.reset_index(drop=True)

    print(CLASSIFIER.upper(), MODE, WEIGHT, LRMODE, latemode, "hptuning=", do_hp_tuning)
    fts_best = get_best_fts_loocv(MODE, WEIGHT, LRMODE, 5, CLASSIFIER, predict_late=latemode, pick_thresh=0.5, plot=False)
    if len(fts_best) == 0:
        print("NO FTS FOUND ABOVE THRESH 0.5...:")
        return 0

    print("BEST FTS = ", *fts_best)

    ft_index_dict = get_feature_index_fsps(LRMODE, MODE, WEIGHT)
    fts_idx = [ft_index_dict[ft] for ft in fts_best]
    print("\tindices=", *fts_idx)
    # print(x_orig)
    print(x_orig.shape, y_orig.shape)

    td = ["time", "dose"] if not MODE == "DELTA" else ["dose"]
    # print(x_orig.loc[:, td].shape)

    XLIST = [x_orig.loc[:, td], x_orig.loc[:, fts_best]]
    for ft in fts_best:
        XLIST.append(x_orig[[*td, ft]])
    DESCRIPTORS = ["td", "fts", *[f"td+ft{k}" for k in fts_idx]]

    df_results = pd.DataFrame()
    df_results.loc[-1, "idx"] = -1
    df_results.loc[-1, "y_gt"] = -1
    for c, ft in zip(DESCRIPTORS, ["td", "all fts", *fts_best]):
        df_results.loc[-1, c] = ft
    print(df_results)

    num_obs = len(y_orig)
    i = 0
    for n in range(num_obs):
        print(f"\nLOOCV {n} / {num_obs}")
        y_train, y_test = y_orig.drop(n), np.reshape([y_orig.loc[n]], (-1, 1))
        XTRAIN = [X.drop(n) for X in XLIST]
        XTEST = [np.reshape([X.loc[n]], (1, -1)) for X in XLIST]

        if do_hp_tuning and CLASSIFIER == "RF":
            print("NOT IMPLEMENTED")
            return 0
        else:
            params = {}

        for r in range(NREP):
            predictions = []
            coefs = []
            print(".", end="")
            for x_train, x_test in zip(XTRAIN, XTEST):
                md = RandomForestClassifier(**params) if CLASSIFIER == "RF" else LogisticRegression()
                md.fit(x_train, y_train)
                comb_length = 2 if MODE == "DELTA" else 3
                if CLASSIFIER == "LOGREG" and len(x_train.T) == comb_length:
                    coef_fti = md.coef_.reshape(-1)[-1]
                    # print(md.coef_)
                    # print(md.feature_names_in_)
                    # print(coef_fti)
                    coefs.append(coef_fti)
                # pred = md.predict(x_test)
                # print(md.classes_)
                pred = md.predict_proba(x_test).reshape(-1)[-1]
                # print(pred)
                predictions.append(pred)
            y_gt = y_test.reshape(-1)[0]

            df_results.loc[i, "idx"] = n
            df_results.loc[i, "y_gt"] = y_gt

            # print(coefs)
            k = 0
            for c, pred in zip(DESCRIPTORS, predictions):
                df_results.loc[i, c] = pred
                if CLASSIFIER == "LOGREG" and c not in ["td", "fts"]:
                    df_results.loc[i, f"{c} coef"] = coefs[k]
            i += 1
    print(df_results)
    df_results.to_csv(os.path.join(savefolder, savename))
    pass


def show_combined_loocv_results(simult=True, ttest_thresh=.05, rel_diff_thresh=-0.01):
    from scipy.stats import ttest_rel
    folder = os.path.join(ClassifDir, "td fts combined")
    files = os.listdir(folder)
    print(files)
    files = list(filter(lambda f: ".csv" in f, files))
    files_simult = list(filter(lambda f: "simult" in f, files))
    files_late = list(filter(lambda f: "simult" not in f, files))
    print(f"Found {len(files)} files: {len(files_simult)} simult + {len(files_late)} late")

    files = files_simult if simult else files_late
    tot_tests = 0
    df_sign_models = pd.DataFrame(index=files, columns=["num sign"], data=np.zeros(len(files)))
    for f in files:
        print("\n", f)
        ff = f.split("_")
        if simult:
            mode, weight, lrmode, classifier, nrep = ff[3:]
            print(mode, weight, lrmode, classifier, nrep)
        else:
            # print(ff)
            latemode, mode, weight, lrmode, classifier, nrep = ff[1:]
            print(latemode, mode, weight, lrmode, classifier, nrep)
        # print(ff)
        df = pd.read_csv(os.path.join(folder, f), index_col=0)
        fts_best = df.drop(["idx", "y_gt", "td", "fts"], axis=1).loc[-1].dropna()
        fts_best = fts_best.to_frame(name="ft")
        print(f"\tnum best fts k={len(fts_best)}")
        ft_index = get_feature_index_global()
        for ft, row in zip(fts_best["ft"], fts_best.index):
            ft_glb = "_".join(ft.split("_")[:-1]) if lrmode == "aggregated" else ft
            idx_global = ft_index[ft_glb]
            fts_best.loc[row, "idx global"] = int(idx_global)

        print(fts_best)
        df = df.drop(-1)
        df.loc[:, "y_gt"] = [int(yi == "True") for yi in df["y_gt"].values]
        df = df.astype("float64")

        num_fits = len(df)
        df_means = df.groupby("idx").mean()
        df_sd = df.groupby("idx").std()
        num_obs = len(df_means)
        y_true = df_means["y_gt"]

        cols_probs = list(filter(lambda c: "coef" not in c, df_means.columns.values))
        cols_coefs = list(filter(lambda c: "coef" in c, df_means.columns.values))
        df_squared_error = pd.DataFrame(columns=cols_probs)
        for c in cols_probs:
            probs = df_means[c]
            df_squared_error[c] = [(ygt - pi)**2 for ygt, pi in zip(y_true, probs)]

        # print(df_squared_error)
        df_stats = pd.DataFrame(columns=["BS"], data=df_squared_error.sum() / num_obs)
        df_stats.loc[:, "AUC"] = [roc_auc_score(y_true, df_means[c]) for c in cols_probs]
        ft_index_rows = list(filter(lambda c: c not in ["y_gt", "td"], cols_probs))

        # print(cols_coefs)
        for c in cols_coefs:
            cfs = df[c]
            sign_fraction = np.sum(np.sign(cfs)) / num_obs
            row_ft = c.split(" ")[0]
            df_stats.loc[row_ft, "rel sign"] = sign_fraction
        # print(df_stats)

        td_sqe = df_squared_error["td"]
        # print(ft_index_rows)

        bs_td = df_stats.loc["td", "BS"]
        for c in ft_index_rows:
            # print(c)
            tot_tests += 1 if not c == "fts" else 0
            # tot_tests += 1
            sqe_vals = df_squared_error[c].values
            t, p = ttest_rel(td_sqe, sqe_vals)
            df_stats.loc[c, "p-val equal"] = p
            t, p = ttest_rel(td_sqe, sqe_vals, alternative="greater")
            df_stats.loc[c, "p-val greater"] = p

            bs_c = df_stats.loc[c, "BS"]
            bs_reldiff = (bs_c - bs_td) / bs_td
            df_stats.loc[c, "BS rel td"] = bs_reldiff

        # df_stats_sign = df_stats[df_stats["p-val greater"] < ttest_thresh]
        df_stats_sign = df_stats[(df_stats["p-val greater"] < ttest_thresh) & (df_stats["BS rel td"] < rel_diff_thresh)]
        num_sign = len(df_stats_sign)
        if "fts" in df_stats_sign.index:
            num_sign -= 1


        df_sign_models.loc[f, "num sign"] = num_sign

        df_stats_sign = pd.concat([df_stats.loc[["td", "fts"], :], df_stats_sign], axis=0)
        if num_sign > 0:
            # print(df_stats)
            print(df_stats_sign)
        # break

    print(df_sign_models)
    print(df_sign_models[(df_sign_models.T != 0).any()])
    print("Num tests run:", tot_tests)
    print("TOT sign:", np.sum(df_sign_models["num sign"].values))
    if not simult:
        df_sign_models_baseline = df_sign_models.filter(like="baseline", axis=0)
        df_sign_models_afterirr = df_sign_models.filter(like="after irr", axis=0)
        df_sign_models_delta = df_sign_models.filter(like="delta", axis=0)
        print("Sign baseline:", np.sum(df_sign_models_baseline["num sign"].values))
        print("Sign after irr:", np.sum(df_sign_models_afterirr["num sign"].values))
        print("Sign delta:", np.sum(df_sign_models_delta["num sign"].values))
    return 1


def univariate_loocv_logreg(FTS = [], MODE="NO P", WEIGHT="T2", LRMODE="aggregated", latemode="simult", do_hp_tuning=False):
    if not any(FTS):
        print("PLEASE PROVIDE FEATURES (ft indices) FOR UNIVARIATE ANALYSIS!!")
        return 0
    savefolder = os.path.join(ClassifDir, "no coreg", "univar loocv")
    if latemode == "simult":
        savename = f"loocv_simult_{MODE}_{WEIGHT}_{LRMODE}_LOGREG_hptune={do_hp_tuning}.csv"
        if MODE == "NO P":
            x_orig, y_orig = load_nop(WEIGHT=WEIGHT, LRMODE=LRMODE, training="all", xer=True, impute=False,
                                      keep_names=False)
        elif MODE == "DELTA P":
            x_orig, y_orig = load_delta_P(WEIGHT=WEIGHT, LRMODE=LRMODE, training="all", xer=True, keep_names=False)
            pass
        else:
            print("Try MODE: NO P, DELTA P")
            return 0
    elif latemode in ["baseline", "after irr", "delta"]:
        savename = f"loocv_{latemode}_{MODE}_{WEIGHT}_{LRMODE}_LOGREG_hptune={do_hp_tuning}.csv"

        if MODE == "DELTA":
            x_orig, y_orig = load_delta(WEIGHT="T2", LRMODE=LRMODE, training="all", xer=True, keep_time=False,
                                        keep_names=False)
            # x_td_orig = x_orig["dose"]  # times does not make sense for delta-radiomics as the features are longitudinal
            # x_orig = x_orig.drop("dose", axis=1)
        else:
            basebool = True if latemode == "baseline" else False
            x_orig, y_orig = load_predict_late_not_delta(MODE=MODE, WEIGHT=WEIGHT, LRMODE=LRMODE, training="all",
                                                         baseline=basebool, xer=True)

            if x_orig["time"].dtype != "int64":
                x_orig["time"] = [int(t[:-3]) for t in x_orig["time"].values]

            # x_td_orig = x_orig[["time", "dose"]]
            # x_orig = x_orig.drop(["time", "dose", "name", "id", "time_val"], axis=1)
            if "name" in x_orig.columns:
                x_orig = x_orig.drop(["name", "id", "time_val"], axis=1)
    else:
        print("latemode", latemode, "not valid. Try: simult, baseline, after irr, delta")
        return 0

    x_orig = x_orig.reset_index(drop=True)
    y_orig = y_orig.reset_index(drop=True)

    XLIST = [x_orig.loc[:, ft] for ft in FTS]
    for x in XLIST:
        print(x)
    ft_index = get_feature_index_global()
    if not LRMODE == "aggregated":
        DESCRIPTORS = [f"Ft{ft_index[ft]}" for ft in FTS]
    else:
        FTTS = ["_".join(ft.split("_")[:-1]) for ft in FTS]
        LR = [ft.split("_")[-1] for ft in FTS]
        DESCRIPTORS = [f"Ft{ft_index[ft]}{lr}" for ft, lr in zip(FTTS, LR)]

    df_results = pd.DataFrame()

    num_obs = len(y_orig)
    for n in range(num_obs):
        print(f"\nLOOCV {n} / {num_obs}")
        y_train, y_test = y_orig.drop(n), np.reshape([y_orig.loc[n]], (-1, 1))
        XTRAIN = [np.reshape(X.drop(n).values, (-1, 1)) for X in XLIST]
        XTEST = [np.reshape([X.loc[n]], (1, -1)) for X in XLIST]

        y_gt = y_test.reshape(-1)[0]
        df_results.loc[n, "y_true"] = y_gt

        for x_train, x_test, c in zip(XTRAIN, XTEST, DESCRIPTORS):
            # print(x_train.shape, x_test.shape)
            if do_hp_tuning:
                print("NOT IMPLEMENTED")
                param_grid = {}
                md = LogisticRegression()
                cv = LeaveOneOut()
                gs = GridSearchCV(estimator=md, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1)
                gs = gs.fit(x_train, y_train)
                print(gs.best_score_)
                print(gs.best_params_)
                params = gs.best_params_
            else:
                params = {}

            md = LogisticRegression(**params)
            md.fit(x_train, y_train)
            coef_ft = md.coef_.reshape(-1)[-1]
            phat = md.predict_proba(x_test).reshape(-1)[-1]
            df_results.loc[n, c] = phat
            ccoef = f"{c} beta"
            df_results.loc[n, ccoef] = coef_ft
    print(df_results)
    df_results.to_csv(os.path.join(savefolder, savename))
    print("---DF SAVED---")
    return 1


def show_univariate_loocv_results(hptuned=False):
    loadfolder = os.path.join(ClassifDir, "no coreg", "univar loocv")
    files = os.listdir(loadfolder)
    files = list(filter(lambda f: "hptune=True" in f if hptuned else "hptune=False" in f, files))
    print("FOUND", len(files), "UNIVARIATE LOOCV FILES")

    df_results = pd.DataFrame()
    for f in files:
        ff = f.split("_")
        # print(ff)
        predmode, mode, weight, lrmode, classifier = ff[1:6]
        print(predmode, mode, weight, lrmode, classifier)
        path = os.path.join(loadfolder, f)
        df = pd.read_csv(path, index_col=0)
        # print(df)
        y_true = df["y_true"].values
        num_obs = len(y_true)
        fts = df.columns.values
        fts = list(filter(lambda c: "Ft" in c, fts))
        fts = list(filter(lambda c: "beta" not in c, fts))
        print(fts)
        for ft in fts:
            phats = df[ft]
            betas = df[f"{ft} beta"]
            print(f"\t{ft} average beta = {np.average(betas):.2f}, sign perc = {np.sum(np.sign(betas)) / num_obs * 100:.0f}%", end=" ")
            auc, _ = make_roc_curve_from_probs(y_true, phats)
            print(f"auc={auc:.2f}")#, {roc_auc_score(y_true, phats):.2f}")
            bs = brier_score(y_true, phats)
            row = f"{predmode} {mode} {weight} {ft}"
            df_results.loc[row, "AUC"] = auc
            df_results.loc[row, "BS"] = bs

    print(df_results)
    sns.heatmap(df_results, annot=True, fmt=".2f", cbar=False, cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("UNIVARIATE PREDICTIONS USING BEST FEATURES")
    plt.show()
    pass

if __name__ == "__main__":

    # show_loocv_results(PREDMODE="simult")
    # show_loocv_results(PREDMODE="late")
    # show_combined_loocv_results(simult=True, rel_diff_thresh=-0.01, ttest_thresh=.05)
    # show_combined_loocv_results(simult=False, rel_diff_thresh=-0.01, ttest_thresh=.05)
    # show_combined_loocv_results(simult=True, rel_diff_thresh=-0.01, ttest_thresh=.00109)    # bonferroni thresh
    # show_combined_loocv_results(simult=False, rel_diff_thresh=-0.01, ttest_thresh=.00058)    # bonferroni

    # UNIVARIATE PREDICTIONS USING TOP PERFORMING FTS
    # FTS = ["lbp-2D_firstorder_RobustMeanAbsoluteDeviation_R"]
    # univariate_loocv_logreg(FTS=FTS, MODE="DELTA P", WEIGHT="T2", LRMODE="aggregated", latemode="simult", do_hp_tuning=True)
    # FTS = ["square_glrlm_RunVariance_R"]
    # univariate_loocv_logreg(FTS=FTS, MODE="NO P", WEIGHT="T1", LRMODE="aggregated", latemode="simult", do_hp_tuning=True)

    # FTS = ["original_shape2D_MajorAxisLength"]
    # univariate_loocv_logreg(FTS=FTS, MODE="DELTA P", WEIGHT="T2", LRMODE="average", latemode="baseline", do_hp_tuning=True)
    # FTS = ["wavelet-H_gldm_LargeDependenceEmphasis"]
    # univariate_loocv_logreg(FTS=FTS, MODE="NO P", WEIGHT="T1", LRMODE="average", latemode="baseline", do_hp_tuning=True)

    # FTS = ["logarithm_glcm_JointAverage_R"]
    # univariate_loocv_logreg(FTS=FTS, MODE="DELTA P", WEIGHT="T2", LRMODE="aggregated", latemode="after irr", do_hp_tuning=True)
    # FTS = ["logarithm_gldm_SmallDependenceLowGrayLevelEmphasis_R"]
    # univariate_loocv_logreg(FTS=FTS, MODE="NO P", WEIGHT="T2", LRMODE="aggregated", latemode="after irr", do_hp_tuning=True)
    # FTS = ["original_glrlm_LongRunHighGrayLevelEmphasis", "gradient_glcm_Imc1"]
    # univariate_loocv_logreg(FTS=FTS, MODE="NO P", WEIGHT="T2", LRMODE="average", latemode="after irr", do_hp_tuning=True)

    # FTS = ["gradient_glcm_Correlation_R"]
    # univariate_loocv_logreg(FTS=FTS, MODE="DELTA", WEIGHT="T2", LRMODE="aggregated", latemode="delta", do_hp_tuning=True)
    FTS = ["original_shape2D_Elongation"]
    univariate_loocv_logreg(FTS=FTS, MODE="DELTA", WEIGHT="T2", LRMODE="average", latemode="delta", do_hp_tuning=True)
    # show_univariate_loocv_results(hptuned=False)

    sys.exit()

    # loocv_combined_xer()

    show_nocoreg_classif_results(metric="auc")
    # show_T12_coreg_classif_results(LRmode="aggregated")
    # show_loocv_results(PREDMODE="simult", plot_roc_curves=False, show_models=["RF", "LOGREG"])
    # show_loocv_results(PREDMODE="late", plot_roc_curves=False, show_models=["RF", "LOGREG"])

    # find_optimal_number_of_fts()
    # show_loocv_boot(PREDMODE="simult", plot_roc_curves=False)
    # show_loocv_boot(PREDMODE="late", plot_roc_curves=False)

    # classification_xer_timedose_all(NREP=100, do_hp_tuning=False, do_loocv=False)
    # classification_xer_timedose_all(NREP=100, do_hp_tuning=True)
    # show_loocv_timedose()

    # get_best_fts_loocv("NO P", "T2", "aggregated", 5, "LOGREG", predict_late=False)
    # get_best_fts_loocv("NO P", "T2", "aggregated", 5, "LOGREG", predict_late="baseline")
    # get_best_fts_loocv("NO P", "T2", "aggregated", 5, "LOGREG", predict_late="after irr")

    # for lrmode in ["average", "aggregated", "split"]:
    #     get_fsps_best_norm(lrmode, "T2", 0.05)
    #     t2_nop = load_fsps_data(WEIGHT="T2", MODE="NO P", LRMODE=lrmode, TRAIN_SET="all", verbose=False)
    #     t2_deltap = load_fsps_data(WEIGHT="T2", MODE="DELTA P", LRMODE=lrmode, TRAIN_SET="all", verbose=False)
    #     t2_delta = load_fsps_data(WEIGHT="T2", MODE="DELTA", LRMODE=lrmode, TRAIN_SET="all", verbose=False)
    #     print("FSPS no-p:", t2_nop.shape, "delta-p:", t2_deltap.shape, "delta:", t2_delta.shape)
    #
    #     get_fsps_best_norm(lrmode, "T1", 0.15)
    #     t1 = load_fsps_data(WEIGHT="T1", MODE="NO P", LRMODE=lrmode, TRAIN_SET="all", verbose=False)
    #     print("FSPS DATA T1:", t1.shape)

    # show_T12coreg_loocv_results(hptuned=True, CLASSIFIERS="RF")
    sys.exit()
    # METHOD = "classif T12 coreg"
    # METHOD = "loocv no coreg"
    # METHOD = "test new stuff"
    # METHOD = "classif no coreg"
    # METHOD = "loocv acute"
    # METHOD = "loocv late"
    # METHOD = "loocv boot"
    # METHOD = "loocv combined"
    # METHOD = "reliability diagram"
    METHOD = "classif T12 loocv"
    # PREDMODE = "simul"
    # LRMODE = "aggregated"
    # LRMODE = "average"
    print("METHOD", METHOD)
    ONLY_VIS = False
    tune_params = False
    # MODE = "NO P"
    # num_boot = 1000
    num_boot = 5

    if METHOD == "classif T12 coreg":
        # CO- REGISTERRED T1 + T2 DATA: XEROSTOMIA BINARY CLASSIFICATION on acute, baseline, after irr (ONLY NO-P DATA)
        #         Nboot = 1000
        #         MODE = "NO P"
        #         params_t1, params_t2, params_comb = {}, {}, {}
        #         params_dict = {}
        #         # for NFTS in [5, 10, 15, "all"]:
        for LRMODE in ["aggregated", "average"]:
            NFTS = 5
            LATELIST = ["after irr"] if NFTS == 10 else [False, "baseline", "after irr"]
            for LATEMODE in LATELIST:
                # SPLITLIST = [3] if (LATEMODE == "after irr" and NFTS == 10) else [1, 2, 3]
                SPLITLIST = [1, 2, 3]
                for SPLIT_NUM in SPLITLIST:
                    latename = "acute" if LATEMODE == False else LATEMODE
                    print("NFTS=", NFTS, "PREDICT", latename, "SPLT=", SPLIT_NUM)
                    if not ONLY_VIS and tune_params:
                        params_t1, score_t1, params_t2, score_t2, params_comb, score_comb = main_classification_pairwise_T1_T2(MODE=MODE,
                                    LRMODE=LRMODE, predict_late=LATEMODE, Nfts=NFTS, num_bootstraps=Nboot, do_hp_tuning=True,
                                    only_visualize=False, show_performance=False, SPLIT_NUMBER=SPLIT_NUM)
                        params_dict[MODE + " LR" + LRMODE + " " + latename + f" SPLITNUM {SPLIT_NUM} " + "T1"] = params_t1
                        params_dict[MODE + " LR" + LRMODE + " " + latename + f" SPLITNUM {SPLIT_NUM} " + "T2"] = params_t2
                        params_dict[MODE + " LR" + LRMODE + " " + latename + f" SPLITNUM {SPLIT_NUM} " + "COMB"] = params_comb
                    #
                    main_classification_pairwise_T1_T2(MODE=MODE, LRMODE=LRMODE, predict_late=LATEMODE, Nfts=NFTS, num_bootstraps=Nboot,
                                                       do_hp_tuning=False, only_visualize=ONLY_VIS, show_performance=False, params_t1=params_t1,
                                                       params_t2=params_t2, params_comb=params_comb, SPLIT_NUMBER=SPLIT_NUM)

        print(MODE, LRMODE, f"T1 / T2 co-reg split num {SPLIT_NUM} params:")
        for it in params_dict.items():
            print(it)
        # print("T1:", params_t1)
        # print("T2:", params_t2)
        sys.exit()

    elif METHOD == "classif T12 loocv":
        MODE = "NO P"  # only valid mode
        num_repeat_RF = 100
        HPTUNE = True
        # HPTUNE = False
        for LRMODE in ["aggregated", "average"]:
            main_classification_pairwise_T1_T2_loocv(MODE, LRMODE, predict_late=False, do_hp_tuning=HPTUNE, NFTS=5, num_repeatRF=num_repeat_RF)
            main_classification_pairwise_T1_T2_loocv(MODE, LRMODE, predict_late="baseline", do_hp_tuning=HPTUNE, NFTS=5, num_repeatRF=num_repeat_RF)
            main_classification_pairwise_T1_T2_loocv(MODE, LRMODE, predict_late="after irr", do_hp_tuning=HPTUNE, NFTS=5, num_repeatRF=num_repeat_RF)
        pass

    elif METHOD == "classif no coreg":
        #todo: sort results into folders by LRMODE
        params = {'criterion': 'entropy'}
        # for Nfts in [5, 10, 15, "all":
        for Nfts in [5]:
            # for MODE in ["NO P", "DELTA P", "DELTA"]:
            for MODE in ["DELTA"]:
                WEIGHTS = ["T1", "T2"] if MODE == "NO P" else ["T2"]
                for W in WEIGHTS:
                # for Nfts in [5, 10, 15, "all"]:
                    if not MODE == "DELTA":
                        print("\n\n-----", LRMODE, MODE, W, Nfts, "fts acute------")
                        if not ONLY_VIS and tune_params:
                            params, score = main_classification(MODE, LRMODE=LRMODE, WEIGHT=W, do_hp_tuning=True, only_visualize=False, predict_late=False, Nfts=Nfts, num_bootstraps=num_boot) if not ONLY_VIS and tune_params else {'criterion': 'entropy'}
                            print(params)
                        main_classification(MODE, WEIGHT=W, LRMODE=LRMODE, do_hp_tuning=False, only_visualize=ONLY_VIS,
                                            predict_late=False, params=params, Nfts=Nfts, num_bootstraps=num_boot)
                        print("\n\n-----", LRMODE, MODE, W, Nfts, "fts baseline ------")
                        if not ONLY_VIS and tune_params:
                            params, score = main_classification(MODE, LRMODE=LRMODE, WEIGHT=W, do_hp_tuning=True, only_visualize=False, predict_late="baseline", Nfts=Nfts, num_bootstraps=num_boot) if not ONLY_VIS and tune_params else {'criterion': 'entropy'}
                        main_classification(MODE, WEIGHT=W, LRMODE=LRMODE, do_hp_tuning=False, only_visualize=ONLY_VIS,
                                            predict_late="baseline", params=params, Nfts=Nfts, num_bootstraps=num_boot)
                    print("\n\n-----", LRMODE, MODE, W, Nfts, "fts after irr------")
                    if not ONLY_VIS and tune_params:
                        params, score = main_classification(MODE, LRMODE=LRMODE, WEIGHT=W, do_hp_tuning=True, only_visualize=False, predict_late="after irr", Nfts=Nfts, num_bootstraps=num_boot)
                        # params_dict["after irr:" + MODE + ":" + W] = params
                    main_classification(MODE, WEIGHT=W, LRMODE=LRMODE, do_hp_tuning=False, only_visualize=ONLY_VIS,
                                        predict_late="after irr", params=params, Nfts=Nfts, num_bootstraps=num_boot)
                    # print(params_dict)
        # print("\nOPTIMAL PARAMS RF CLASSIFY LR aggregated:")
        # for p in params_dict.items():
        #     print(p)
        # sys.exit()

    elif METHOD == "test new stuff":
        x1train, x2train, ytrain = load_T1T2_coreg()
        print(x1train)

        # logr = LogisticRegression()

    elif METHOD == "loocv acute":
        # main_loocv_simul(MODE="NO P", WEIGHT="T2", LRMODE="average", do_hp_tuning=False, Nfts=5, classifier="RF")
        # main_loocv_simul(MODE="DELTA P", WEIGHT="T2", LRMODE="average", do_hp_tuning=True, Nfts=5, classifier="RF")
        # for LRMODE in ["aggregated", "average"]:
        #     for classifier in ["LOGREG", "RF"]:
        #         main_loocv_simul(MODE="NO P", WEIGHT="T1", LRMODE=LRMODE, do_hp_tuning=tune_params, Nfts=5, classifier=classifier)
        #         main_loocv_simul(MODE="NO P", WEIGHT="T2", LRMODE=LRMODE, do_hp_tuning=tune_params, Nfts=5, classifier=classifier)
        #         main_loocv_simul(MODE="DELTA P", WEIGHT="T2", LRMODE=LRMODE, do_hp_tuning=tune_params, Nfts=5, classifier=classifier)
        pass

    elif METHOD == "loocv late":
        for LRMODE in ["aggregated", "average"]:
            for n_fts in ["all", 5, 10]:
                # for CLASSIF in ["LOGREG"]:#, "RF"]:
                for CLASSIF in ["RF"]:
                    main_loocv_late(LRMODE=LRMODE, MODE="DELTA", WEIGHT="T2", do_hp_tuning=True, classifier=CLASSIF, Nfts=n_fts)

                    main_loocv_late(LRMODE=LRMODE, MODE="NO P", WEIGHT="T1", latemode="baseline", do_hp_tuning=True, classifier=CLASSIF, Nfts=n_fts)
                    main_loocv_late(LRMODE=LRMODE, MODE="NO P", WEIGHT="T2", latemode="baseline", do_hp_tuning=True, classifier=CLASSIF, Nfts=n_fts)
                    main_loocv_late(LRMODE=LRMODE, MODE="DELTA P", WEIGHT="T2", latemode="baseline", do_hp_tuning=True, classifier=CLASSIF, Nfts=n_fts)

                    main_loocv_late(LRMODE=LRMODE, MODE="NO P", WEIGHT="T1", latemode="after irr", do_hp_tuning=True, classifier=CLASSIF, Nfts=n_fts)
                    main_loocv_late(LRMODE=LRMODE, MODE="NO P", WEIGHT="T2", latemode="after irr", do_hp_tuning=True, classifier=CLASSIF, Nfts=n_fts)
                    main_loocv_late(LRMODE=LRMODE, MODE="DELTA P", WEIGHT="T2", latemode="after irr", do_hp_tuning=True, classifier=CLASSIF, Nfts=n_fts)
        main_classification_pairwise_T1_T2(MODE="NO P", LRMODE="aggregated", predict_late="after irr", Nfts=5, num_bootstraps=5,
                                           do_hp_tuning=False, only_visualize=False, show_performance=False, SPLIT_NUMBER=1)
        sys.exit()

    elif METHOD == "loocv boot":
        # NFTS = "DUMMY"    # sselect 5 first features for speed
        # NBOOT = 5
        NFTS = 5
        NBOOT = 100
        LRMODE = "average"
        tune_hp = False

        for CLASSIFIER in ["LOGREG", "RF"]:
        # for CLASSIFIER in ["RF"]:
            main_loocv_bootstrapped(PREDMODE="simul", MODE="NO P", WEIGHT="T1", LRMODE=LRMODE, do_hp_tuning=tune_hp, Nfts=NFTS, classifier=CLASSIFIER, NUM_BOOT=NBOOT)
            #
            main_loocv_bootstrapped(PREDMODE="simul", MODE="NO P", WEIGHT="T2", LRMODE=LRMODE, do_hp_tuning=tune_hp, Nfts=NFTS, classifier=CLASSIFIER, NUM_BOOT=NBOOT)
            main_loocv_bootstrapped(PREDMODE="simul", MODE="DELTA P", WEIGHT="T2", LRMODE=LRMODE, do_hp_tuning=tune_hp, Nfts=NFTS, classifier=CLASSIFIER, NUM_BOOT=NBOOT)
            #
            main_loocv_bootstrapped(PREDMODE="baseline", MODE="DELTA P", WEIGHT="T2", LRMODE=LRMODE, do_hp_tuning=tune_hp, Nfts=NFTS, classifier=CLASSIFIER, NUM_BOOT=NBOOT)
            main_loocv_bootstrapped(PREDMODE="baseline", MODE="NO P", WEIGHT="T1", LRMODE=LRMODE, do_hp_tuning=tune_hp, Nfts=NFTS, classifier=CLASSIFIER, NUM_BOOT=NBOOT)
            main_loocv_bootstrapped(PREDMODE="baseline", MODE="NO P", WEIGHT="T2", LRMODE=LRMODE, do_hp_tuning=tune_hp, Nfts=NFTS, classifier=CLASSIFIER, NUM_BOOT=NBOOT)
            #
            main_loocv_bootstrapped(PREDMODE="after irr", MODE="DELTA P", WEIGHT="T2", LRMODE=LRMODE, do_hp_tuning=tune_hp, Nfts=NFTS, classifier=CLASSIFIER, NUM_BOOT=NBOOT)
            main_loocv_bootstrapped(PREDMODE="after irr", MODE="NO P", WEIGHT="T1", LRMODE=LRMODE, do_hp_tuning=tune_hp, Nfts=NFTS, classifier=CLASSIFIER, NUM_BOOT=NBOOT)
            main_loocv_bootstrapped(PREDMODE="after irr", MODE="NO P", WEIGHT="T2", LRMODE=LRMODE, do_hp_tuning=tune_hp, Nfts=NFTS, classifier=CLASSIFIER, NUM_BOOT=NBOOT)

            main_loocv_bootstrapped(PREDMODE="delta", MODE="DELTA", WEIGHT="T2", LRMODE=LRMODE, do_hp_tuning=tune_hp, Nfts=NFTS, classifier=CLASSIFIER, NUM_BOOT=NBOOT)

    elif METHOD == "loocv combined":
        for lrmode in ["aggregated", "average"]:
            for wgtmode in ["NO P:T1", "NO P:T2", "DELTA P:T2", "DELTA:T2"]:
                # for classifier in ["LOGREG", "RF"]:
                for classifier in ["RF"]:
                    mode, weight = wgtmode.split(":")
                    # nrep = 100 if classifier == "RF" else 1
                    # nrep = 25 if classifier == "RF" else 1
                    nrep = 50 if classifier == "RF" else 1
                    print(mode, weight, lrmode)
                    if not mode == "DELTA":
                        # MODE="NO P", WEIGHT="T2", LRMODE="aggregated", CLASSIFIER="LOGREG", NREP=100, latemode=False
                        loocv_combined_xer(mode, weight, lrmode, classifier, nrep, latemode=False)
                        loocv_combined_xer(mode, weight, lrmode, classifier, nrep, latemode="baseline")
                        loocv_combined_xer(mode, weight, lrmode, classifier, nrep, latemode="after irr")
                    else:
                        loocv_combined_xer(mode, weight, lrmode, classifier, nrep, latemode="delta")
        sys.exit()

    elif METHOD == "reliability diagram":
        # PLOT CALIBRATION / RELIABILITY CURVES
        fig, ax = plt.subplots()
        predmode = "simult"
        # classif = "RF"
        classif = "LOGREG"
        lrmode = "aggregated"
        for mode, wgt in zip(["NO P", "NO P", "DELTA P"], ["T1", "T2", "T2"]):
            df, df_fts = load_classif_results(PREDMODE=predmode, WEIGHT=wgt, MODE=mode, CLASSIFIER=classif,
                                              LRMODE=lrmode, loocv=True, boot=False)

            ytrue = df["y_true"]
            probs = df["prob"]
            probs_td = df["prob_td"]

            ptrue, ppred = calibration_curve(ytrue, probs, n_bins=6)
            ax.plot(ppred, ptrue, marker=".", label=f"{mode} {wgt}", zorder=5)

        ax.plot([0, 1], [0, 1], "--", zorder=1)
        ax.legend()
        ax.grid(zorder=0)
        ax.set_xlabel("Observed xer frequency")
        ax.set_ylabel("Predicted xer probability")
        fig.suptitle(f"reliability curves {predmode} {classif} {lrmode}")
        plt.show()
        sys.exit()

    else:
        print("NO METHOD", METHOD, "FOUND.....")

    sys.exit()
    # main_classification("NO P", "T1")
    # main_classification("DELTA P")
    # main_classification("DELTA")
    # main_classification("NO P", "T1", do="hp tuning")
    # main_classification("DELTA P", "T1", do="hp tuning")
    from mrmr import mrmr_regression
    # Nfts = "all"
    Nfts = 15
    params = {}

    # x_train, y_train = load_delta(training=True)
    # x_valid, y_valid = load_delta(training=False)
    # savename = f"bootstrapped_RF_regression_validate_Delta-time_Nfts={Nfts}.csv"
    # savename = f"bootstrapped_RF_regression_validate_Delta-time_Nfts={Nfts}_notuning.csv"
    W = "T1"
    x_train, y_train = load_nop(WEIGHT=W, training=True)
    x_valid, y_valid = load_nop(WEIGHT=W, training=False)
    savename = f"bootstrapped_RF_regression_validate_NOP_{W}_Nfts={Nfts}_notuning.csv"
    # savename = f"bootstrapped_RF_regression_validate_NOP_T2_Nfts={Nfts}_notuning.csv"
    # x_train, y_train = load_delta_P(WEIGHT="T2", training=True)
    # x_valid, y_valid = load_delta_P(WEIGHT="T2", training=False)
    # savename = f"bootstrapped_RF_regression_validate_Delta-P_Nfts={Nfts}.csv"

    if Nfts != "all":
        selected_features = mrmr_regression(x_train, y_train, K=Nfts, return_scores=False)
        x_train, x_valid = x_train[selected_features], x_valid[selected_features]

    # rf_hyperparamtuning(x_train, y_train, mode="classification")
    # rf_hyperparamtuning(x_train, y_train, mode="regression")
    # params = {'criterion': 'squared_error', 'max_depth': 5, 'min_samples_leaf': 10, 'min_samples_split': 4, 'n_estimators': 400}  # NO P T2 n_fts=15
    # params = {'criterion': 'absolute_error', 'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 100}  # DELTA-P n_fts=15
    # params = {'criterion': 'squared_error', 'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 50}      # DELTA (time) n_fts=15

    # bootstrapped_validation(x_train, y_train, x_valid, y_valid, params=params, mode="classification", Nboot=1000, savename=savename)
    bootstrapped_validation(x_train, y_train, x_valid, y_valid, params=params, mode="regression", Nboot=1000, savename=savename)
    visualize_model_performance(path=os.path.join(ClassifDir, savename))

    sys.exit()


    # COMPARE NUMBER OF SELECTED FEATURES (with mrmr) TO R2 USING RF REGRESSION + REPEATED CV
    STATE = 42

    # WEIGHT, MODE = "T1", "NO P"
    WEIGHT, MODE = "T2", "DELTA P"
    # WEIGHT, MODE = "T2", "DELTA"
    Nrep = 5
    k_splits = 5
    df = load_fsps_data(WEIGHT, MODE, TRAIN_SET=True)
    if MODE == "DELTA":
        # print(df.T.head(5))
        df_y = df["saliv late"]
        df_red = df.drop(["name", "saliv late"], axis=1)
    else:
        df_y = load_saliva(melt=False)
        df_y = register_name_to_outcome(df, df_y, melt=True, make_70_exception=True)
        df_y = df_y["val"]
        df_red = df.loc[df_y.index.values].drop(["name"], axis=1)
        df_red["time"] = [int(x[:-3]) for x in df_red["time"].values]
    y_train = df_y
    N_samp = len(df_red)
    kvals = [1, 2, 5, 10, 15, 20, 25, 50, 100]
    # kvals = [2, *np.arange(5, 105, 5)]
    # kvals = [2, 3]
    scores_means = []
    scores_sds = []
    oob_scores = []
    for k in kvals:
        fts = mrmr_regression(df_red, y_train, K=k, return_scores=False)
        x_train = df_red[fts]

        rf = RandomForestRegressor(random_state=None, oob_score=True)
        rf.fit(x_train, y_train)
        # r2 = rf.score(x_train, y_train)
        oob_scores.append(rf.oob_score_)
        # print(k, r2)
        # scores.append(r2)

        cvrep = RepeatedKFold(n_repeats=Nrep, n_splits=k_splits)
        acc_scores = cross_val_score(rf, x_train, df_y, cv=cvrep, scoring="r2")
        print("k=", k, f"{Nrep}-repeated {k_splits}-fold: r2 mean / sd = {np.mean(acc_scores):.3f} / {np.std(acc_scores):.3f}", end="")
        print(fts)
        scores_means.append(np.mean(acc_scores))
        scores_sds.append(np.std(acc_scores))

    # plt.plot(kvals, scores_means, label="Avg R2")
    plt.plot(kvals, oob_scores, label="OOB scores")
    plt.errorbar(kvals, scores_means, yerr=scores_sds, capsize=5, capthick=2, label="Avg R2")
    plt.title(f"R2 scores for RF regression on training data (N={N_samp}) with {Nrep}-repeated {k_splits}-fold CV\n"
              f"for {MODE} features from {WEIGHT}-w images.")
    plt.grid()
    plt.xlabel("# fts")
    plt.ylabel("$R^2$")
    plt.legend(loc="best")
    plt.show()
    sys.exit()

    # split_state = 42
    # RANDOMSTATE = 42
    # split_state = None
    # X, Y = load_nop_saliv_data(univar_tresh=0.80, mode="cont")
    # X, Y = load_nop_saliv_data(univar_tresh=0.80)       ;featureset="No-P"
    # X, Y = load_delta_p_saliv_data(univar_tresh=0.80, impute_saliva=False)     ;featureset="Delta-P"
    # X, Y = load_saliv_data_reg_to_future(mode="no-p")    # mode: delta-p or no-p
    # X, Y = load_titanic_data()


    # p_fts = 75
    # n_samples = 500
    # n_informative = 5
    # X, Y = make_classification(n_samples=n_samples, n_features=p_fts, n_clusters_per_class=1, n_informative=n_informative, random_state=RANDOMSTATE)    #make up some data :)
    # print(X)
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=split_state)  # SPLIT (MAIN) TEST SET INTO MODEL TRAIN / TEST
    # print(x_train.shape, y_test.shape)
    # compare_max_features(x_train, x_test, y_train, y_test, RANDOMSTATE=RANDOMSTATE)#, title="Effect of forest size on OOB and test error on RF classification, "
    #      f"\nevaluated on a constructed data set with {len(x_train)} train / {len(x_test)} samples having {n_informative} of p={p_fts} informative features.")
    # rf = RandomForestClassifier()



    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=split_state)  # SPLIT (MAIN) TEST SET INTO MODEL TRAIN / TEST
    # print("TRAIN:", x_train.shape, "TEST:", x_test.shape)
    # # rf_hyperparamtuning(x_train, y_train)
    # accs = []
    # Nruns = 2500
    # for i in range(Nruns):
    #     rf = RandomForestClassifier(criterion="entropy", max_depth=8, min_samples_leaf=1, min_samples_split=4, n_estimators=100, n_jobs=-1)    # DELTA-P
    #     # rf = RandomForestClassifier(criterion="gini", max_depth=4, min_samples_leaf=1, min_samples_split=2, n_estimators=50)    # NO-P
    #     rf.fit(x_train, y_train)
    #     y_pred = rf.predict(x_test)
    #     acc = accuracy_score(y_test, y_pred)
    #     accs.append(acc)
    #     auc = roc_auc_score(y_test, y_pred)
    #     print(i, "RF acc=%.3f, auc=%.3f" % (acc, auc))
    # plt.title(f"RF test accuracy for {Nruns} runs after single train / test split of {featureset} data\n"
    #           f"having mean={np.mean(accs):.3f}, sd={np.std(accs):.3f}")
    # # plt.hist(accs, density=1, color="green", alpha=0.8, bins=10)
    # sns.histplot(accs, stat="density", kde=True)
    # # sns.kdeplot(accs, common_norm=False, color="r")
    # plt.grid()
    # plt.show()
    # sys.exit()

    # REPEATED TEST TRAIN SPLIT WITH ACC / AUC EVALUATIONS FOR NO-P AND DELTA-P FEATURE SPACES
    # Nrep = 5000   # NUMBER OF TIMES TO REPEAT TRAIN / TEST SPLIT
    # X, Y = load_nop_saliv_data(univar_tresh=0.80)
    # print(X.shape, Y.shape)
    # acc_vals_nop = []
    # auc_vals_nop = []
    # feature_importance_accumulated_nop = np.zeros(shape=(X.shape[1], ))
    # for n in range(Nrep):
    #     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=None)  # SPLIT (MAIN) TEST SET INTO MODEL TRAIN / TEST
    #     rf = RandomForestClassifier(criterion="gini", n_estimators=50, random_state=None, oob_score=True, warm_start=True)
    #     rf.fit(x_train, y_train)
    #     y_pred = rf.predict(x_test)
    #     acc_nop = accuracy_score(y_test, y_pred)
    #     auc_nop = roc_auc_score(y_test, y_pred)
    #     # print("OOB score = ", rf.oob_score_)
    #     n_train_nop = len(x_train)
    #     n_test_nop = len(x_test)
    #     print(f"{n}/{Nrep} NO-P Single split Ntrain / Ntest = ({n_train_nop}, {n_test_nop}): Acc={acc_nop:.3f}, ROC_AUC={auc_nop:.3f}")
    #     acc_vals_nop.append(acc_nop)
    #     auc_vals_nop.append(auc_nop)
    #     feature_importance_accumulated_nop += rf.feature_importances_
    # sortidx = feature_importance_accumulated_nop.argsort()
    # feature_importance_accumulated_nop_sorted = feature_importance_accumulated_nop[sortidx[::-1]]
    # feature_names_sorted_nop = rf.feature_names_in_[sortidx[::-1]]
    #
    # X, Y = load_delta_p_saliv_data(univar_tresh=0.80)
    # feature_importance_accumulated_deltap = np.zeros(shape=(X.shape[1],))
    # acc_vals_deltap = []
    # auc_vals_deltap = []
    # for n in range(Nrep):
    #     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=None)
    #     rf = RandomForestClassifier(criterion="gini", n_estimators=50, random_state=None, oob_score=True, warm_start=True)
    #     rf.fit(x_train, y_train)
    #     y_pred = rf.predict(x_test)
    #     acc_deltap = accuracy_score(y_test, y_pred)
    #     auc_deltap = roc_auc_score(y_test, y_pred)
    #     print(f"{n}/{Nrep} DELTA-P split Ntrain / Ntest = ({len(x_train)}, {len(x_test)}): Acc={acc_deltap:.3f}, ROC_AUC={auc_deltap:.3f}")
    #     acc_vals_deltap.append(acc_deltap)
    #     auc_vals_deltap.append(auc_deltap)
    #     feature_importance_accumulated_deltap += rf.feature_importances_
    # sortidx = feature_importance_accumulated_deltap.argsort()
    # feature_importance_accumulated_deltap_sorted = feature_importance_accumulated_deltap[sortidx[::-1]]
    # feature_names_sorted_deltap = rf.feature_names_in_[sortidx[::-1]]
    #
    # print(f"\n\nNO-P {Nrep} splits Ntrain / Ntest = ({n_train_nop}, {n_test_nop}): Acc={np.mean(acc_vals_nop):.3f} ({np.std(acc_vals_nop):.3f}), ROC_AUC={np.mean(auc_vals_nop):.3f} ({np.std(auc_vals_nop):.3f})")
    # print(f"DELTA-P {Nrep} splits Ntrain / Ntest = ({len(x_train)}, {len(x_test)}): Acc={np.mean(acc_vals_deltap):.3f} ({np.std(acc_vals_deltap):.3f}), ROC_AUC={np.mean(auc_vals_deltap):.3f} ({np.std(auc_vals_deltap):.3f})")
    # # TODO: PLOT HISTOGRAM OF ACCURACY / AUC DISTRIBUTIONS?
    #
    # i = 0   # TOP TEN FEATURES RANKED BY IMPORTANCE
    # for ft1, ft2, val1, val2 in zip(feature_names_sorted_nop, feature_names_sorted_deltap, feature_importance_accumulated_nop_sorted, feature_importance_accumulated_deltap_sorted):
    #     print("No-P:", ft1, f"importance={val1/Nrep:.3f}", "\t\tDelta-P:", ft2, f"importance={val2/Nrep:.3f}")
    #     i+= 1
    #     if i>10:
    #         break
    # fig, ax = plt.subplots(ncols=2)
    # ax[0].hist(acc_vals_nop);       ax[0].set_title("No-P data: mean=%.3f, sd=%.3f" % (np.mean(acc_vals_nop), np.std(acc_vals_nop)))
    # ax[1].hist(acc_vals_deltap);    ax[1].set_title("Delta-P data: mean=%.3f, sd=%.3f" % (np.mean(acc_vals_deltap), np.std(acc_vals_deltap)))
    # fig.suptitle(f"Accuracy distributions after {Nrep} train / test splits")
    # plt.show()

    # cvrep = RepeatedKFold(n_repeats=5, n_splits=5)
    # acc_scores = cross_val_score(rf, x_train, y_train, cv=cvrep, scoring="accuracy")
    # print("5 repeated 5-fold: Mean acc (sd) = %.3f (%.3f)" % (np.mean(acc_scores), np.std(acc_scores)))


    # PLOTTING ACCURACY VS NUMBER OF TREES IN RF, USING REPEATED K-FOLD EVALUATION
    # fig, ax = plt.subplots(figsize=(12, 8))
    # # nvals = np.linspace(20, 1000, 10, dtype=int)
    # Nsteps = 150
    # nvals = np.linspace(20, 1500, Nsteps, dtype=int)
    # Nrep = 3
    # kval = 3
    # scores = []
    # scores_sd = []
    # scores_auc = []
    # scores_auc_sd = []
    # for n in nvals:
    #     print("n=", n)
    #     rf = RandomForestClassifier(criterion="gini", n_estimators=50, random_state=None, oob_score=True,
    #                                 warm_start=True)
    #     cvrep = RepeatedKFold(n_repeats=Nrep, n_splits=kval)
    #     # rf.fit(x_train, y_train)
    #     acc_scores = cross_val_score(rf, x_train, y_train, cv=cvrep, scoring="accuracy")
    #     auc_scores = cross_val_score(rf, x_train, y_train, cv=cvrep, scoring="roc_auc")
    #     print(f"{Nrep} repeated {kval}-fold: Mean acc (sd) = ", np.mean(acc_scores), np.std(acc_scores), acc_scores.shape)
    #     scores.append(np.mean(acc_scores))
    #     scores_sd.append(np.std(acc_scores))
    #     scores_auc.append(np.mean(auc_scores))
    #     scores_auc_sd.append(np.std(auc_scores))
    #     # ax.errorbar(n, np.mean(acc_scores), yerr=np.std(acc_scores))
    #     # ax.plot(n, np.mean(acc_scores), "x")
    # # plt.errorbar(nvals, scores, yerr=scores_sd, c="b", ls=":", alpha=1, capsize=5)
    # plt.plot(nvals, scores, "-", label="Accuracy", c="b")
    # # plt.errorbar(nvals, scores_auc, yerr=scores_auc_sd, c="r", ls=":", alpha=0.7, capsize=5)
    # plt.plot(nvals, scores_auc, "-", label="AUC", c="r")
    # plt.legend()
    # plt.grid()
    # # plt.title(f"{Nrep} repeated {kval}-fold: varying number of estimators (trees) for no-p training data")
    # plt.title(f"{Nrep} repeated {kval}-fold: varying number of estimators (trees) for titanic training data")
    # plt.ylabel("Score")
    # plt.xlabel("Number of estimators (trees)")
    # plt.savefig(os.path.join(FigDir, "modelling", "random forest", f"rep({Nrep})k({kval})foldscores_varying_numtrees.png"))
    # plt.show()

    # rf_hyperparamtuning(x_train, y_train, "classifier")


    # K-fold CV
    # k_splits = 10
    # Nrep = 5
    # kvals = np.arange(2, 11, 1)
    # rf = RandomForestClassifier(random_state=None)
    # cvrepeat = RepeatedKFold(n_splits=k_splits, n_repeats=Nrep, random_state=None)
    # acc_scores = cross_val_score(rf, X, Y, cv=cvrepeat, scoring="accuracy")
    # cvrepeatstrat = RepeatedStratifiedKFold(n_splits=k_splits, n_repeats=Nrep, random_state=None)
    # print(f"{Nrep} TIMES REPEATED K-fold: k={k_splits}")
    # print("acc =\t %.3f (%.3f)" % (np.mean(acc_scores), np.std(acc_scores)))
    # acc_scores = cross_val_score(rf, X, Y, cv=cvrepeatstrat, scoring="accuracy")
    # print("strat acc =\t %.3f (%.3f)" % (np.mean(acc_scores), np.std(acc_scores)))


    # COMPARE OOB FOR DIFFERENT RF MODES (max_features)
    # plt.close()
    # # INSP: https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html#sphx-glr-auto-examples-ensemble-plot-ensemble-oob-py
    # X, Y = load_delta_p_saliv_data(mode="disc", univar_tresh=0.20)
    # X, Y = load_titanic_data()
    # p_fts = len(X.columns)    # TOTAL NUMBER OF PREDICTORS
    # n_samples = len(X)
    # print(p_fts, n_samples)
    # mode = "OOB"
    # mode = "test error"
    # # p_fts = 50
    # # n_samples = 500
    # n_informative = 25
    # RANDOMSTATE = None
    # # X, Y = make_classification(n_samples=n_samples, n_features=p_fts, n_clusters_per_class=1, n_informative=n_informative, random_state=RANDOMSTATE)    #make up some data :)
    # n_repeats = 10
    # N = 250
    # nvals = np.linspace(50, 1500, N, dtype=int)
    # print(len(nvals))
    # # scores = [[], [], []]
    # scores = np.zeros((3, N))
    # clfs = [RandomForestClassifier(max_features="sqrt", oob_score=True, warm_start=True, random_state=RANDOMSTATE),
    #         RandomForestClassifier(max_features="log2", oob_score=True, warm_start=True, random_state=RANDOMSTATE),
    #         RandomForestClassifier(max_features=None, oob_score=True, warm_start=True, random_state=RANDOMSTATE)]
    # maxmodes = ["sqrt", "log2", None]
    # # for i, rf in enumerate(clfs):
    # for i, mode in enumerate(maxmodes):
    #     print("\n", i, mode)
    #     scores_temp = np.zeros(shape=(n_repeats, N))
    #     for j in range(n_repeats):
    #         print("\tREPEAT", j)
    #         rf = RandomForestClassifier(max_features=mode, oob_score=True, warm_start=True, random_state=RANDOMSTATE)
    #         for ni, n in enumerate(nvals):
    #             # Warm start: re-use results from previous fit when adding estimators
    #             rf.set_params(n_estimators=n)
    #             rf.fit(X, Y)
    #             # print(n, rf.oob_score_)
    #             # scores[i].append(1 - clf.oob_score_)
    #             # print(i, j, ni)
    #             print(f"MODE {mode} {i}/3, REPEAT {j}/{n_repeats}, {ni}/{N}: OOB score={rf.oob_score_}")
    #             scores_temp[j, ni] = 1 - rf.oob_score_
    #             # print(scores_temp)
    #     # scores[i].append(np.mean(scores_temp, axis=0))
    #     scores[i] = np.mean(scores_temp, axis=0)
    #     print(np.shape(scores))
    #
    # plt.plot(nvals, scores[0], label="RF: $m=\sqrt{p}$")
    # plt.plot(nvals, scores[1], label="RF: $m=\log_2(p)$")
    # plt.plot(nvals, scores[2], label="Bagged trees: $m=p$")
    # plt.grid()
    # plt.xlabel("Number of trees");  plt.ylabel("OOB Error")
    # plt.legend()
    # # plt.title("Effect of forest size on OOB error estimates for different RF models,\n"
    # #           f"evaluated on a constructed data set with {n_samples} samples having with {n_informative} of {p_fts} informative features.")
    # # plt.title("Effect of forest size on OOB error estimates for different RF models,\n"
    # #           f"evaluated on delta-p data with {n_samples} samples having {p_fts} features.")
    # plt.title(f"Effect of forest size on {n_repeats}-repeated averaged OOB error estimates for different RF models,\n"
    #           # f"evaluated on delta-p data with {n_samples} samples having {p_fts} features.")
    #           f"evaluated on titanic data with {n_samples} samples having {p_fts} features.")
    # plt.show()


    # for i in range(9):
    #     accvals = []
    #     accstrat = []
    #     for k_splits in kvals:
    #         cv = KFold(n_splits=k_splits, random_state=None, shuffle=True)
    #         cvstrat = StratifiedKFold(n_splits=k_splits, random_state=None, shuffle=True)
    #         # for train, test in cv.split(X, Y):
    #         #     print(train.shape, test.shape)
    #         acc_scores = cross_val_score(rf, X, Y, cv=cv, n_jobs=-1, scoring="accuracy")
    #         auc_scores = cross_val_score(rf, X, Y, cv=cv, n_jobs=-1, scoring="roc_auc")
    #         print(f"{i}: K-fold CV with k={k_splits} having means: acc={np.mean(acc_scores):.3f}, auc={np.mean(auc_scores):.3f}")
    #         accvals.append(np.mean(acc_scores))
    #         # plt.plot(k_splits, np.mean(acc_scores), "x", c="b")
    #         # plt.errorbar(k_splits, np.mean(acc_scores), yerr=np.std(acc_scores), c="b")
    #         accstrat.append(np.mean(cross_val_score(rf, X, Y, cv=cvstrat, n_jobs=-1, scoring="accuracy")))
    #     plt.plot(kvals, accvals, "x-", c="r")
    #     plt.plot(kvals, accstrat, "o-", c="b")
    # plt.show()