from select_utils import *
from data_loader import *
import hoggorm as ho
import hoggormplot as hoplt
from mrmr import mrmr_classif
from matplotlib import pyplot as plt


def plot_scores(model, idvals, xervals, dosevals, figax=[]):
    if not figax:
        fig, ax = plt.subplots()
        showbool = True
    else:
        fig, ax = figax
        showbool = False
    try:
        scores = model.X_scores()
        exp_var = model.X_calExplVar()

    except Exception as e:
        print(*e.args)
        return 0

    lns = [0, 0, 0, 0]  # For legend: control xer / no xer, irr xer / no xer
    for sc, id, xer, d in zip(scores, idvals, xervals, dosevals):
        ctr = is_control(id)
        ln = ax.scatter(sc[0], sc[1], marker="x" if ctr else "o", c="r" if xer else "b")
        ax.text(sc[0], sc[1], id)
        if ctr:
            if xer:
                lns[0] = ln
            else:
                lns[1] = ln
        else:
            if xer:
                lns[2] = ln
            else:
                lns[3] = ln
    ax.set_xlabel(f"Comp 1 ({exp_var[0]:.1f}%)")
    ax.set_ylabel(f"Comp 2 ({exp_var[1]:.1f}%)")

    xlims = np.array(ax.get_xlim())
    ylims = np.array(ax.get_ylim())
    ax.plot(xlims, [0, 0], color='0.4', linestyle='dashed', linewidth=1)
    ax.plot([0, 0], ylims, color='0.4', linestyle='dashed', linewidth=1)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.legend(lns[:4], ["Control xer", "Control not xer", "Irr xer", "Irr not xer"])
    # ax.grid(visible=True, which="major")
    ax.set_title(f"PCA scores plot (N={len(idvals)})")

    plt.show() if showbool else 0
    return 0


def plot_loadings(model, varnames, figax=[]):
    if not figax:
        fig, ax = plt.subplots()
        showbool = True
    else:
        fig, ax = figax
        showbool = False
    try:
        loading = model.X_loadings()
        exp_var = model.X_calExplVar()
        # print(loading)

    except Exception as e:
        print(*e.args)
        return 0

    xbuff, ybuff = np.max(loading[:, 0]), np.max(loading[:, 1])
    xbuff *= 0.05
    ybuff *= 0.04

    for i, name in enumerate(varnames):
        xpos, ypos = loading[i, 0], loading[i, 1]
        ax.scatter(xpos, ypos, s=10, c="w", marker="o", edgecolor="grey")
        ax.text(xpos + xbuff, ypos + ybuff, name, fontsize=12)

    ax.set_xlabel(f"Comp 1 ({exp_var[0]:.1f}%)")
    ax.set_ylabel(f"Comp 2 ({exp_var[1]:.1f}%)")

    xlims = np.array(ax.get_xlim())
    ylims = np.array(ax.get_ylim())
    ax.plot(xlims, [0, 0], color='0.4', linestyle='dashed', linewidth=1)
    ax.plot([0, 0], ylims, color='0.4', linestyle='dashed', linewidth=1)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_title(f"PCA scores plot (N={len(fts)})")
    plt.show() if showbool else 0
    return 0


if __name__ == "__main__":
    LRMODE = "aggregated"
    MODE = "NO P"
    # WGT = "T1"
    WGT = "T2"
    splitnum = 1
    # late = False
    late = "baseline"
    # late = "after irr"
    latemode = "acute" if late == False else late

    # trainset = False
    trainset = True     # use training data to find best k features
    k_fts = 5
    num_pcs = 2         # Number of PC vectors to calculate

    # title = f"PCA on top {k_fts} features selected from {latemode} {MODE} {WGT} LR {LRMODE} coreg T1 / T2 split {splitnum}"
    # title = f"PCA on ALL features {latemode} {MODE} {WGT} LR {LRMODE} coreg T1 / T2"
    # title = f"PCA on (time, dose) {latemode} {MODE} LR {LRMODE} coreg T1 / T2"
    # title = f"PCA on (time, dose) using validation data from {latemode} {MODE} LR {LRMODE} coreg T1 / T2 split {splitnum}"

    # SELECT BEST K FEATURES WITH TRAINING DATA

    # df1, df2, y = load_T1T2_coreg(MODE=MODE, LRMODE=LRMODE, predict_late=late, training=True, xer=True, SPLIT_NUMBER=splitnum, keep_names=True)
    # df = df2 if WGT == "T2" else df1
    # df, y = load_predict_late(MODE=MODE, WEIGHT=WGT, baseline=True, xer=True, training=True)

    # Y = y["xer"].values
    # top_fts = ["dose", "time"]
    # df = df.drop(["dose"], axis=1)  # dominating the PCA plots for acute T2
    # top_fts = mrmr_classif(df, Y, K=k_fts, return_scores=False)
    # top_fts = list(df.columns)
    # top_fts = ["dose", "time", top_fts[0]]
    # print(top_fts)

    # LOAD DATA FOR PCA
    # df1, df2, y = load_T1T2_coreg(MODE=MODE, LRMODE=LRMODE, predict_late=late, training="all", xer=True, SPLIT_NUMBER=splitnum, keep_names=True)
    # df1, df2, y = load_T1T2_coreg(MODE=MODE, LRMODE=LRMODE, predict_late=late, training=trainset, xer=True, SPLIT_NUMBER=splitnum, keep_names=True)
    # df = df2 if WGT == "T2" else df1

    # df, y = load_predict_late(MODE="DELTA P", WEIGHT="T2", baseline=True, xer=True, training=True, keep_names_times=True)
    # df, y = load_predict_late(MODE="NO P", WEIGHT="T2", baseline=False, xer=True, training=True, keep_names_times=True)
    # df, y = load_predict_late(MODE="NO P", WEIGHT="T2", baseline=False, xer=True, training=True, keep_names_times=True)

    lrmode = "aggregated"
    # lrmode = "average"
    mode = "NO P"
    # mode = "DELTA"
    # weight = "T1"
    weight = "T2"
    # latemode = "baseline"
    latemode = "after irr"
    include_td = True
    from classification import get_best_fts_loocv

    if mode == "DELTA":
        df, y = load_delta(WEIGHT=weight, LRMODE=lrmode, training="all", xer=True, keep_names=True, keep_time=False)
        df.loc[:, "id"] = [nm.split("_")[0] for nm in df["name"]]
    else:
        df, y = load_predict_late_not_delta(MODE=mode, WEIGHT=weight, LRMODE=lrmode, training="all", baseline=True, xer=True, keep_id=True)

    fts_best = get_best_fts_loocv(mode, weight, lrmode, NFTS=5, CLASSIFIER="LOGREG", predict_late=latemode, pick_thresh=.5, plot=False)
    # title = f"PCA on (time, dose) using validation data from {latemode} {MODE} LR {LRMODE} coreg T1 / T2 split {splitnum}"
    # title = f"PCA using data from {latemode} {mode} {weight} LR-{lrmode}"
    title = f"PCA best k={len(fts_best)} loocv-features from {latemode} {mode} {weight} LR-{lrmode} LOGREG"

    # df.loc[:, "ctr"] = [is_control(nm, t) for nm, t in zip(df["id"], df["time"])]
    df.loc[:, "ctr"] = [is_control(nm) for nm in df["id"]]
    df = df.drop(["name"], axis=1).rename(columns={"id":"name"})

    print(f"Have k={len(fts_best)} best loocv fts:")
    print(fts_best)
    if include_td:
        if mode == "DELTA":
            fts_best = np.append(fts_best, "dose")
            print("+ dose", fts_best)
        else:
            fts_best = np.append(fts_best, ["time", "dose"])
            print("+ time and dose", fts_best)

    Y = y
    print("LOADED:", df.shape, y.shape)
    # print(df["time"].values)
    id_vals = list(df["name"])
    df = df.drop("name", axis=1)
    fts = df.columns.values
    dose_vals = df["dose"].values   # get dose values BEFORE reduce df to top_fts

    # if mode == "DELTA":
    #     pass
    # else:
    #     df = df.drop(["time", "ctr", "dose"], axis=1)
    # df = df[["time", "dose", fts[2]]]
    df = df[fts_best]

    # Y = y["xer"].values
    # print(df.shape)
    X = df.values
    fts = list(df.columns)  # features
    data_objNames = list(df.index)
    print(X.shape)

    pcamodel = ho.nipalsPCA(arrX=X, Xstand=False, cvType=["loo"], numComp=num_pcs)
    print(num_pcs, "principal components calculated")

    # print(pcamodel.X_calExplVar())
    # scores = pcamodel.X_scores()
    # exp_var = pcamodel.X_calExplVar()

    fig, axes = plt.subplots(ncols=2)
    ax1, ax2 = axes
    # loadings = [pcamodel.X_loadings(), pcamodel.Y_loadings()]

    plot_scores(pcamodel, idvals=id_vals, xervals=Y, dosevals=dose_vals, figax=(fig, ax1))
    plot_loadings(pcamodel, fts, figax=(fig, ax2))

    fig.suptitle(title)
    plt.show()
    # hoplt.plot(pcamodel, comp=[1, 2], plots=[2], objNames=id_vals, XvarNames=fts) #1: X scores, 2: loadings plot, 6: expl variance in X
    # hoplt.loadings(pcamodel, line=True)
    # hoplt.conv_scoresPlot.scores(pcamodel, comp=[1, 2], objNames=data_objNames, which=["X"])#, objNames=[], newX=[], newY=[], newObjNames=[], figsize=None)
    # hoplt.plot(pcamodel, comp=[1, 2],
    #          plots=[1, 2, 3, 4, 6],
    #          objNames=data_objNames,
    #          XvarNames=data_varNames)