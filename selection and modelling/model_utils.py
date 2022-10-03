from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import os
import sys
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 75)
pd.set_option('display.width', None)
pd.set_option('precision', 3)

ClassifDir = os.path.normpath(os.path.join(os.getcwd(), "..", "..", "Radiomic Modelling", "clsf"))
RegressDir = os.path.normpath(os.path.join(os.getcwd(), "..", "..", "Radiomic Modelling", "Regression"))


def show_decision_boundary_classif(rf, x_train, y_train, x_test, y_test, Nsteps=10, show_train=True, show_test=True,
                           figsize=(7, 4), title="", xlabel="", ylabel="", showplot=True, figax=[]):
    # rf: trained RF classifier model
    # X: two first columns ft1, ft2 area feature space for plotting decision boundary in
    # Y: corresponding y-values
    # Nsteps: number of step between values in ft1, ft2
    print("----- PLOTTING decision boundary for fts", x_train.columns.values, "-----")

    if not figax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    y_pred = rf.predict(x_test)
    # auc = roc_auc_score(y_test, y_pred)
    # acc = accuracy_score(y_test, y_pred)
    # print(f"auc={auc:.2f}")

    ft1, ft2 = x_train.columns.values[:2]
    print(ft1, ft2)
    if show_test and show_train:
        ft1_min, ft1_max = np.min([*x_train.values[:, 0], *x_test.values[:, 0]]), np.max([*x_train.values[:, 0], *x_test.values[:, 0]])
        ft2_min, ft2_max = np.min([*x_train.values[:, 1], *x_test.values[:, 1]]), np.max([*x_train.values[:, 1], *x_test.values[:, 1]])
        xticks = np.unique([*x_train.values[:, 0], *x_test.values[:, 0]])
        yticks = np.unique([*x_train.values[:, 1], *x_test.values[:, 1]])
    elif show_test:
        ft1_min, ft1_max = np.min(x_test.values[:, 0]), np.max(x_test.values[:, 0])
        ft2_min, ft2_max = np.min(x_test.values[:, 1]), np.max(x_test.values[:, 1])
        xticks = np.unique(x_test.values[:, 0])
        yticks = np.unique(x_test.values[:, 1])
    elif show_train:
        ft1_min, ft1_max = np.min(x_train.values[:, 0]), np.max(x_train.values[:, 0])
        ft2_min, ft2_max = np.min(x_train.values[:, 1]), np.max(x_train.values[:, 1])
        xticks = np.unique(x_train.values[:, 0])
        yticks = np.unique(x_train.values[:, 1])
    else:
        print("Either show_test or show_train has to be true...")
        return 0

    print("Ft 1 vals:", xticks)
    print("Ft 2 vals:", yticks)
    print(f"Ft 1 min / max: {ft1_min:.2f} / {ft1_max}, ft2 min / max: {ft2_min:.2f} / {ft2_max:.2f}")

    x_train_xer = x_train.loc[y_train[y_train == True].index]
    x_train_noxer = x_train.loc[y_train[y_train == False].index]
    x_test_xer = x_test.loc[y_test[y_test == True].index]
    x_test_noxer = x_test.loc[y_test[y_test == False].index]
    train_xer_vals, train_noxer_vals = x_train_xer.T.values, x_train_noxer.T.values
    test_xer_vals, test_noxer_vals = x_test_xer.T.values, x_test_noxer.T.values


    pd = 0.1    # buffer to min / max feature values for decision surface plotting
    ft1_max += pd + abs(ft1_max * pd)
    ft1_min -= pd + abs(ft1_min * pd)
    ft2_max += pd + abs(ft2_max * pd)
    ft2_min -= pd + abs(ft2_min * pd)
    print(f"Ft 1 min / max: {ft1_min:.2f} / {ft1_max}, ft2 min / max: {ft2_min:.2f} / {ft2_max:.2f}")

    xx, yy = np.meshgrid(np.linspace(ft1_min, ft1_max, Nsteps), np.linspace(ft2_min, ft2_max, Nsteps))
    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    sep_constant = 0.05
    horizontal_shift = (ft1_max - ft1_min) * sep_constant
    vertical_shift = (ft2_max - ft2_min) * sep_constant   # shift points vertically for separability

    cs = ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu_r, alpha=0.6, zorder=0)
    lw = 0.5
    edgc = "white"
    if show_train and show_test:
        ax.scatter(train_xer_vals[0] - 1.0 * horizontal_shift,
                   [x + s for x, s in zip(train_xer_vals[1], np.linspace(-vertical_shift, vertical_shift, len(train_xer_vals[1])))],
                   marker="X", c="r", label=f"Train xer (N={len(x_train_xer)})", edgecolors=edgc, linewidths=lw, zorder=5)
        ax.scatter(train_noxer_vals[0] + 1 / 3 * horizontal_shift,
                   [x + s for x, s in zip(train_noxer_vals[1], np.linspace(-vertical_shift, vertical_shift, len(train_noxer_vals[1])))],
                   marker="X", c="b", label=f"Train not xer (N={len(x_train_noxer)})", edgecolors=edgc, linewidths=lw, zorder=5)

        ax.scatter(test_xer_vals[0] - 1 / 3 * horizontal_shift,
                   [x + s for x, s in zip(test_xer_vals[1], np.linspace(-vertical_shift, vertical_shift, len(test_xer_vals[1])))],
                   marker="o", c="r", label=f"Test xer (N={len(x_test_xer)})", edgecolors=edgc, linewidths=lw, zorder=5)
        ax.scatter(test_noxer_vals[0] + 1.0 * horizontal_shift,
                   [x + s for x, s in zip(test_noxer_vals[1], np.linspace(-vertical_shift, vertical_shift, len(test_noxer_vals[1])))],
                   marker="o", c="b", label=f"Test not xer (N={len(x_test_noxer)})", edgecolors=edgc, linewidths=lw, zorder=5)
    elif show_test:
        ax.scatter(test_xer_vals[0] + 0.5 * horizontal_shift,
                   [x + s for x, s in zip(test_xer_vals[1], np.linspace(-vertical_shift, vertical_shift, len(test_xer_vals[1])))],
                   marker="o", c="r", label=f"Test xer (N={len(x_test_xer)})")
        ax.scatter(test_noxer_vals[0] - 0.5 * horizontal_shift,
                   [x + s for x, s in zip(test_noxer_vals[1], np.linspace(-vertical_shift, vertical_shift, len(test_noxer_vals[1])))],
                   marker="o", c="b", label=f"Test not xer (N={len(x_test_noxer)})")
    elif show_train:
        ax.scatter(train_xer_vals[0] + 0.5 * horizontal_shift,
                   [x + s for x, s in zip(train_xer_vals[1], np.linspace(-vertical_shift, vertical_shift, len(train_xer_vals[1])))],
                   marker="x", c="r", label=f"Train xer (N={len(x_train_xer)})")
        ax.scatter(train_noxer_vals[0] - 0.5 * horizontal_shift,
                   [x + s for x, s in zip(train_noxer_vals[1], np.linspace(-vertical_shift, vertical_shift, len(train_noxer_vals[1])))],
                   marker="x", c="b", label=f"Train not xer (N={len(x_train_noxer)})")
    else:
        pass

    ax.set_xlabel(ft1)
    ax.set_ylabel(ft2)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid(True, zorder=1, color="white")
    ax.set_xlabel(xlabel) if xlabel else 0
    ax.set_ylabel(ylabel) if ylabel else 0
    # ax.set_title(f"{title + ''', ''' if title else ''''''}auc={auc:.2f}, acc={acc:.2f}")
    ax.set_title(title)
    ax.legend()
    plt.show() if showplot else 0
    pass


def show_decision_boundary_regression(rf, x_train, y_train, x_test, y_test, Nsteps=10, title="", xlab="", ylab=""):
    if not any([xlab, ylab]):
        xlab = "Time [Day]"
        ylab = "Dose [Gy]"

    ft1_min, ft1_max = np.min([*x_train.values[:, 0], *x_test.values[:, 0]]), np.max(
        [*x_train.values[:, 0], *x_test.values[:, 0]])

    ft2_min, ft2_max = np.min([*x_train.values[:, 1], *x_test.values[:, 1]]), np.max(
        [*x_train.values[:, 1], *x_test.values[:, 1]])
    pd = 0.1
    ft1_max += pd + abs(ft1_max * pd)
    ft1_min -= pd + abs(ft1_min * pd)
    ft2_max += pd + abs(ft2_max * pd)
    ft2_min -= pd + abs(ft2_min * pd)

    xx, yy = np.meshgrid(np.linspace(ft1_min, ft1_max, Nsteps), np.linspace(ft2_min, ft2_max, Nsteps))
    Zpred = rf.predict(np.c_[xx.ravel(), yy.ravel()])
    Zpred = Zpred.reshape(xx.shape)

    points = x_train.values   # x, y coordinates
    values = y_train.values   # corresponding values at (x, y)
    Zinterp = griddata(points, values, (xx, yy), method="nearest")

    fig, ax = plt.subplots(ncols=2)
    xticks = np.unique([*x_train.values[:, 0], *x_test.values[:, 0]])
    yticks = np.unique([*x_train.values[:, 1], *x_test.values[:, 1]])

    cs0 = ax[0].contourf(xx, yy, Zinterp, cmap=plt.cm.RdYlBu_r, alpha=0.6, zorder=0)
    # cbar = fig.colorbar(cs0, ax=ax[0])
    cs1 = ax[1].contourf(xx, yy, Zpred, cmap=plt.cm.RdYlBu_r, alpha=0.6, zorder=0)  # Predicted surface
    cbar = fig.colorbar(cs1, ax=ax[1])
    cbar.ax.set_ylabel("Saliva amount [$\mu$L]")

    shift_val = 0.125 # add shift to values to show multiple instances at same (time, dose)
    ax[0].scatter(*[x + np.random.normal(loc=0, scale=shift_val, size=(len(x))) for x in x_train.values.T], marker="x", label=f"Training (N={len(x_train)})", alpha=1, zorder=5)
    ax[0].scatter(*[x + np.random.normal(loc=0, scale=shift_val, size=(len(x))) for x in x_test.values.T], marker="x", label=f"Test (N={len(x_test)})", alpha=1, zorder=5)

    ax[1].scatter(*[x + np.random.normal(loc=0, scale=shift_val, size=(len(x))) for x in x_train.values.T], marker="x", label=f"Training (N={len(x_train)})", alpha=1, zorder=5)
    ax[1].scatter(*[x + np.random.normal(loc=0, scale=shift_val, size=(len(x))) for x in x_test.values.T], marker="x", label=f"Test (N={len(x_test)})", alpha=1, zorder=5)

    yticks = list(filter(lambda x: x != 74.0, yticks))

    ax[0].set_xticks(xticks)
    ax[0].set_yticks(yticks)
    ax[0].grid(1, zorder=1)
    ax[0].legend()
    ax[0].set_xlabel(xlab)
    ax[0].set_ylabel(ylab)
    ax[0].set_title("Interpolated surface from training data")

    ax[1].set_xticks(xticks)
    ax[1].set_yticks(yticks)
    ax[1].grid(1, zorder=1)
    ax[1].legend()
    ax[1].set_xlabel(xlab)
    ax[1].set_ylabel(ylab)
    ax[1].set_title(f"Predicted with regression model (R2={rf.score(x_test, y_test):.2f})")
    fig.suptitle(title)
    plt.show()
    pass


def show_interpolated_surface(x1, x2, y, y_sep=[], sep_names=[], xlab="", ylab="", cbarlab="", title="", figax=[],
                              Nsteps=10, interp_method="linear", make_3d=True):
    # sep_names = [a, b]: label x1, x2 where y_sep is TRUE as a, FALSE as b
    if not interp_method in ["nearest", "linear", "cubic"]:
        print("INTERPOLATION METHOD", interp_method, "INVALID. try: nearest, linear, cubic")
        return 0

    if not figax:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax

    ft1_min, ft1_max = np.min(x1), np.max(x1)
    ft2_min, ft2_max = np.min(x2), np.max(x2)
    print(f"{'''ft1''' if not xlab else xlab} min/ max:", ft1_min, ft1_max)
    print(f"{'''ft2''' if not ylab else ylab} min/ max:", ft2_min, ft2_max)

    points = np.array([x1, x2]).T   # x, y coordinates in shape n, D
    # print(points.shape, y.shape)    # y: shape n,

    xx, yy = np.meshgrid(np.linspace(ft1_min, ft1_max, Nsteps), np.linspace(ft2_min, ft2_max, Nsteps))
    Zinterp = griddata(points, y, (xx, yy), method=interp_method)
    surf = ax.contourf(xx, yy, Zinterp, zorder=1, cmap=plt.cm.RdYlBu_r)
    cbar = fig.colorbar(surf)
    cbar.ax.set_ylabel(cbarlab)

    shift_val = 0.125
    if not any(y_sep):
        ax.scatter(x1, x2, marker="x", label=f"", alpha=1, zorder=5)
    else:
        x1_1 = x1[y_sep == True]
        x1_2 = x1[y_sep == False]
        x2_1 = x2[y_sep == True]
        x2_2 = x2[y_sep == False]
        ax.scatter(x1_1, x2_1, marker="x", label=f"{sep_names[0]} (N={len(x1_1)})", zorder=5)
        ax.scatter(x1_2, x2_2, marker="x", label=f"{sep_names[1]} (N={len(x1_2)})", zorder=5)

    xticks = np.unique(x1)
    yticks = list(filter(lambda x: x != 74.0, np.unique(x2)))

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim([ft1_min - 1, ft1_max])
    ax.set_ylim([ft2_min - 1, ft2_max])
    fig.suptitle(title + f"\nInterpolation method: {interp_method}")
    ax.legend()
    ax.grid(1, zorder=1)

    if make_3d:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(x1, x2, y)
        ax2.set_xlabel(xlab)
        ax2.set_ylabel(ylab)
        ax2.set_zlabel(cbarlab)
        ax2.plot_wireframe(xx, yy, Zinterp, color="black", alpha=0.25)
        fig2.suptitle(interp_method)

    plt.show()
    return fig, ax


def make_roc_curve_from_probs(ytrue, probs, plot=False):
    from scipy import integrate

    num_positive = len(ytrue[ytrue == True])    # true positives
    num_negative = len(ytrue[ytrue == False])   # true negatives
    # print("Having", num_positive, "true positive,", num_negative, "true negative")

    tpr_vals = [0]
    fpr_vals = [0]
    # pvals = [0, *probs.sort_values(), 1]
    pvals = [0, *np.sort(probs), 1]
    pvals.reverse()
    # print(pvals)
    for pthresh in pvals:
        ypred = [pr >= pthresh for pr in probs]
        # print(list(ytrue))
        # print(ypred)
        true_positives = np.count_nonzero(np.logical_and(ypred, ytrue))
        false_positives = np.count_nonzero(np.logical_and(ypred, np.logical_not(ytrue)))    # type 1: predicted True & actual False
        tpr = true_positives / num_positive
        fpr = false_positives / num_negative
        # print(f"{pthresh:.2f}: fpr={fpr:.2f}, tpr={tpr:.2f}")
        tpr_vals.append(tpr)
        fpr_vals.append(fpr)
        # print("thresh=", pthresh, "TPR=", tpr, "FPR=", fpr)
    if not (fpr_vals[-1] == 1.0 and tpr_vals[-1] == 1.0):
        fpr_vals.append(1.0)
        tpr_vals.append(1.0)

    roc_curve = [fpr_vals, tpr_vals]
    auc = integrate.trapz(tpr_vals, fpr_vals)

    if plot:
        plt.plot(fpr_vals, tpr_vals, "x-", label=f"auc={auc:.2f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()
    return auc, roc_curve


def brier_score(ytrue, probs):
    # strictly proper scoring rule,
    # analogous to MSE for binary outcomes (ytrue) given estimated class probabilities (probs)
    # always between 0 and 1 (lower is better)
    N = len(ytrue)
    diff_squared = [(yi - pi)**2 for yi, pi in zip(ytrue, probs)]
    score = np.sum(diff_squared) / N
    return score


def combine_means_sd(muvec, sdvec, nvec=[]):
    # assume equal amount of observations N for each mu, sd in the vectors if nvec=[]
    if not nvec:
        nvec = [10]*len(muvec)
    gxvec = [mui * ni for mui, ni in zip(muvec, nvec)] # helper sum for each group
    gxxvec = [si**2 * (ni-1) + (gxi**2 / ni) for si, ni, gxi in zip(sdvec, nvec, gxvec)]

    tn = np.sum(nvec)
    tx = np.sum(gxvec)
    txx = np.sum(gxxvec)
    # print(tn, tx, txx)

    # mu_new = np.average(muvec, weights=nvec)
    mu_new = tx / tn
    sd_new = np.sqrt((txx - tx**2 / tn) / (tn - 1))
    # print(sd_new)
    return mu_new, sd_new


def find_overlap_values(df1, df2, col_to_save="xer", pick_latest_value=True):
    # find overlapping IDs in ["name"] column between df1, df2
    # save value col_to_save from both in new df
    # if multiple avaliable: pick_latest_value=True picks latest value based on ["time"] integer
    #   else pick earliest value

    id_overlap = set(df1["name"].values).intersection(set(df2["name"].values))
    df_overlap = pd.DataFrame()
    for nm in id_overlap:
        df1_nm = df1[df1["name"] == nm].sort_values(by="time", axis=0, ascending=True)
        df2_nm = df2[df2["name"] == nm].sort_values(by="time", axis=0, ascending=True)
        print(".", end="")
        # print(df1_nm)
        # print(df2_nm)
        # print(df2_nm[col_to_save].values)
        val1 = df1_nm[col_to_save].values[-1] if pick_latest_value else df1_nm[col_to_save].values[0]
        val2 = df2_nm[col_to_save].values[-1] if pick_latest_value else df2_nm[col_to_save].values[0]
        # print(val1, val2)
        df_overlap.loc[nm, [col_to_save + "1", col_to_save + "2"]] = [val1, val2]
        df_overlap.loc[nm, "ctr"] = df2_nm["ctr"].values[0]
    print()
    return df_overlap


def create_contingency_table(df, c1, c2):
    # df: binary outcome values in columns c1, c2
    # create 2x2 contingency table (e.g. for McNemar test)
    N = len(df)
    a = len(df[(df[c1] == True) & (df[c2] == True)])
    b = len(df[(df[c1] == False) & (df[c2] == True)])
    c = len(df[(df[c1] == True) & (df[c2] == False)])
    d = len(df[(df[c1] == False) & (df[c2] == False)])
    # print(N, a)
    if not (a + b + c + d) == N:
        print("ERR: could not create contingency table..")
        return 0
    return [[a, b], [c, d]]


if __name__ == "__main__":

    # df = pd.read_csv(os.path.join(ClassifDir, "no coreg", "loocv late", "aggregated_RF_NO P_T2_baseline_nfts=5.csv"), index_col=0)
    # df = pd.read_csv(os.path.join(ClassifDir, "no coreg", "loocv late", "aggregated_LOGREG_DELTA P_T2_baseline_nfts=5.csv"), index_col=0)
    df = pd.read_csv(os.path.join(ClassifDir, "no coreg", "loocv late", "average_LOGREG_NO P_T1_baseline_nfts=5.csv"), index_col=0)
    # print(df)

    ytrue = df["y_true"].values
    sample_size = len(ytrue)
    num_true = len(ytrue[ytrue==True])
    print(f"{num_true} of {sample_size} True ({num_true / sample_size * 100 :.1f}%)")
    probs = df["prob"]
    probs_td = df["prob_td"]
    # probs_h0 = [p / 10 for p in probs]
    probs_h0 = [np.random.randint(2) for p in probs]

    # z = 1.96        #95%ci
    z = 10
    probs_upper = [p + z * np.sqrt((p * (1 - p) / sample_size)) for p in probs]
    probs_upper = [1 if p > 1 else p for p in probs_upper]

    probs_lower = [p - z * np.sqrt((p * (1 - p) / sample_size)) for p in probs]
    probs_lower = [0 if p < 0 else p for p in probs_lower]

    print(list(probs))
    print(probs_lower)

    auc, roc = make_roc_curve_from_probs(ytrue, probs)
    auc_up, roc_up = make_roc_curve_from_probs(ytrue, probs_upper)
    auc_low, roc_low = make_roc_curve_from_probs(ytrue, probs_lower)

    auc_td, roc_td = make_roc_curve_from_probs(ytrue, probs_td)
    auc_h0, roc_h0 = make_roc_curve_from_probs(ytrue, probs_h0)

    bri = brier_score(ytrue, probs)
    bri_low, bri_upp = brier_score(ytrue, probs_lower), brier_score(ytrue, probs_upper)
    bri_td = brier_score(ytrue, probs_td)
    bri_h0 = brier_score(ytrue, probs_h0)

    print(f"ROC AUC: \t\tfts={auc:.2f}, td={auc_td:.2f}, h0={auc_h0:.2f}")
    print(f"\tAUCupper = {auc_up:.2f}, AUClow = {auc_low:.2f}")
    print(f"Brier score: \tfts={bri:.2f}, td={bri_td:.2f}, h0={bri_h0:.2f}")
    print(f"\tbrier low = {bri_low:.2f}, upp = {bri_upp:.2f}")
    # print("fpr=", [f"{x:.1f}" for x in roc[0]])
    # print("tpr=", [f"{x:.1f}" for x in roc[1]])

    # combine_means_sd([0.49, 0.66, 0.73], [0.11, 0.10, 0.10])
    # combine_means_sd([11.8, 15.3, 8.4], [2.4, 3.2, 4.1], [10, 20, 15])