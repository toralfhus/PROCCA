from select_utils import *
from endpoints import binary_thresh_xerostomia
from preprocessing import norm_stscore

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from mrmr import mrmr_classif
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    # Nsamp = 60  # total number of (fake) samples
    # Nsamp = 120
    Nsamp = 512
    Ntest = int(Nsamp * 0.30)
    Nfts = 10    # Total number of features
    Ninfo = 3    # informative features
    Nred = 0    # linear comb of informative features
    sep_dist = 1.0  #
    # mu_ctrl = 127       # mean control days 26-75
    # mu_irr = 71         # mean irradiated days 26-75
    mu_ctrl = 110
    mu_irr = 50
    k_fts_select = 5    # number of features to select

    cv = 0.10   # sd / mu: defines sigma for gaussians from which fake saliva measurements are drawn from
    timepoint_late_saliva = 50      # day for which fake (late) saliva samples are "drawn from" --> affects xerostomic threshold value
    xer_thresh = (91.30 + 0.55 * timepoint_late_saliva) / 2   # threshold value for xerostomia

    # First: make fake feature space (X) with Y being separable control / irr clusters
    X_ftspace, Y_ctrlirr = make_classification(n_samples=Nsamp, n_informative=Ninfo, n_features=Nfts, n_redundant=Nred,
                                               class_sep=sep_dist, n_clusters_per_class=1, shuffle=False) # No shuffle: Ninfo informative features first

    print(X_ftspace.shape, Y_ctrlirr.shape)

    dfX = pd.DataFrame(columns=[f"ft {x}" for x in range(1, Nfts + 1)], data=X_ftspace)
    dfY = pd.DataFrame(data=Y_ctrlirr, columns=["ctr"])
    print(all(dfX.index == dfY.index))
    idx_ctrl = list(dfY[dfY["ctr"] == True].index)
    idx_irr = list(dfY[dfY["ctr"] == False].index)
    Nctrl, Nirr = len(idx_ctrl), len(idx_irr)

    # Make fake saliva data, drawn from two gaussian distributions (control / irr) with SAME cv = sd / mean
    X_ftspace_informative = X_ftspace[:, :Ninfo]        # Informative feature space
    X_ftspace_informative_ctrl = X_ftspace_informative[idx_ctrl]
    X_ftspace_informative_irr = X_ftspace_informative[idx_irr]

    ecld_ftdist_ctrl = np.sum(X_ftspace_informative_ctrl, axis=1)
    ecld_ftdist_irr = np.sum(X_ftspace_informative_irr, axis=1)
    # ecld_ftdist_ctrl = np.sqrt(np.sum(X_ftspace_informative_ctrl, axis=1))
    # ecld_ftdist_irr = np.sqrt(np.sum(X_ftspace_informative_irr, axis=1))

    dfY.loc[idx_ctrl, "ft eucl"] = ecld_ftdist_ctrl
    dfY.loc[idx_irr, "ft eucl"] = ecld_ftdist_irr
    # dfY.loc[:, "ft eucl"] = np.sqrt(list(dfY["ft eucl"]))
    dfY["idx"] = dfY.index

    # fig, ax = plt.subplots(ncols=2, nrows=2)
    # sns.scatterplot(data=dfY, x=dfY.index, y="ft eucl", hue="ctr", ax=ax[0, 0])
    # sns.histplot(data=dfY, x="ft eucl", kde=True, ax=ax[0, 1], hue="ctr")

    # dfY.loc[idx_ctrl, "saliva"] = norm_stscore(ecld_ftdist_ctrl, new_mean=-sep_dist, weight=1)
    # dfY.loc[idx_irr, "saliva"] = norm_stscore(ecld_ftdist_irr, new_mean=+sep_dist, weight=1)
    dfY.loc[idx_ctrl, "saliva"] = norm_stscore(ecld_ftdist_ctrl, new_mean=mu_ctrl, weight=mu_ctrl*cv)
    dfY.loc[idx_irr, "saliva"] = norm_stscore(ecld_ftdist_irr, new_mean=mu_irr, weight=mu_irr*cv)

    # sns.scatterplot(data=dfY, x=dfY.index, y="saliva", hue="ctr", ax=ax[1, 0])
    # sns.histplot(data=dfY, x="saliva", kde=True, ax=ax[1, 1], hue="ctr")
    # plt.show()

    # ecld_ftdist = dfY["ft eucl"].values
    # print(f"Euclidean distance between values in {Ninfo} informative features w/ stscore norm: shape={ecld_ftdist.shape}, mean={ecld_ftdist.mean():.3f}, sd={ecld_ftdist.std():.3f}")


    # saliv_ctrl = np.random.normal(loc=mu_ctrl, scale=cv * mu_ctrl, size=(len(idx_ctrl)))
    # saliv_ctrl = np.array([0 if x < 0 else x for x in saliv_ctrl])    # no values < 0
    # saliv_irr = np.random.normal(loc=mu_irr, scale=cv * mu_irr, size=len(idx_irr))
    # saliv_irr = np.array([0 if x < 0 else x for x in saliv_irr])
    saliv_ctrl = dfY.loc[idx_ctrl, "saliva"].values
    saliv_irr = dfY.loc[idx_irr, "saliva"].values

    # print(saliv_ctrl.shape, saliv_irr.shape)
    # dfY.loc[idx_ctrl, "val"] = saliv_ctrl
    # dfY.loc[idx_irr, "val"] = saliv_irr
    dfY = dfY.rename(columns={"saliva":"val"})
    dfY.loc[:, "time"] = [timepoint_late_saliva]*Nsamp
    print(dfY.columns.values, dfY.shape)
    Y = binary_thresh_xerostomia(dfY, plot=False)        # xer thresholding on fake saliva data

    # FEATURE SELECTION + RF MODELLING / EVALUATION
    X_train, X_test, y_train, y_test = train_test_split(dfX, Y, test_size=Ntest, stratify=dfY["ctr"])
    # print(y_test)
    # print(dfY.loc[y_test.index])
    print(f"Test: have {len(y_test[y_test == True])} of {len(y_test)} xer,", end=" ")
    print(f"{len(dfY.loc[y_test.index][dfY.loc[y_test.index]['''ctr'''] == True])} control")
    print("Train shape:", X_train.shape)

    top_fts = mrmr_classif(dfX, Y, K=k_fts_select)
    # top_fts = list(dfX.columns)[:2]
    print("TOP FTS:", top_fts)
    dfX = dfX[top_fts]
    X_train, X_test = X_train[top_fts], X_test[top_fts]

    params = {}
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    print(f"AUC = {auc:.2f}")


    dfX_ctrl, dfY_ctrl = dfX.loc[idx_ctrl], dfY.loc[idx_ctrl]
    dfX_irr, dfY_irr = dfX.loc[idx_irr], dfY.loc[idx_irr]
    xer_ctrl, xer_irr = Y.loc[idx_ctrl], Y.loc[idx_irr]

    idx_ctrl_xer = list(xer_ctrl[xer_ctrl == True].index)
    idx_ctrl_noxer = list(xer_ctrl[xer_ctrl == False].index)
    idx_irr_xer = list(xer_irr[xer_irr == True].index)
    idx_irr_noxer = list(xer_irr[xer_irr == False].index)

    fig, axes = plt.subplots(ncols=2, figsize=(14, 7))
    ax1, ax2 = axes.ravel()
    # ax1.scatter(dfX_ctrl.values[:, 0], dfX_ctrl.values[:, 1], marker="x", c="b", edgecolors="k", label="Control")
    # ax1.scatter(dfX_irr.values[:, 0], dfX_irr.values[:, 1], marker="o", c="r", edgecolors="k", label="Irradiated")
    ax1.scatter(dfX_ctrl.loc[idx_ctrl_xer].values[:, 0], dfX_ctrl.loc[idx_ctrl_xer].values[:, 1], marker="x", c="r", edgecolors="k", label=f"Control xer (N={len(idx_ctrl_xer)})")
    ax1.scatter(dfX_ctrl.loc[idx_ctrl_noxer].values[:, 0], dfX_ctrl.loc[idx_ctrl_noxer].values[:, 1], marker="x", c="b", edgecolors="k", label=f"Control not xer (N={len(idx_ctrl_noxer)})")
    ax1.scatter(dfX_irr.loc[idx_irr_xer].values[:, 0], dfX_irr.loc[idx_irr_xer].values[:, 1], marker="o", c="r", edgecolors="k", label=f"Irradiated xer (N={len(idx_irr_xer)})")
    ax1.scatter(dfX_irr.loc[idx_irr_noxer].values[:, 0], dfX_irr.loc[idx_irr_noxer].values[:, 1], marker="o", c="b", edgecolors="k", label=f"Irradiated not xer (N={len(idx_irr_noxer)})")
    ax1.set_xlabel(top_fts[0]); ax1.set_ylabel(top_fts[1])
    ax1.set_title("Manufactured feature space")
    ax1.legend()

    # PLOT contour plot of decision boundary for RF classifier https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html
    plotres = 0.1
    xmin, xmax = np.min(dfX.values[:, 0]), np.max(dfX.values[:, 0])
    ymin, ymax = np.min(dfX.values[:, 1]), np.max(dfX.values[:, 1])
    xx, yy = np.meshgrid(np.arange(xmin, xmax, plotres), np.arange(ymin, ymax, plotres))
    print("mesgrid = ", xx.shape, yy.shape)
    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = ax1.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)

    ax2.plot(list(range(Nctrl, Nctrl + Nirr)), saliv_irr, marker="o", linestyle="", label="Irradiated")
    ax2.plot(list(range(Nctrl)), saliv_ctrl, marker="x", linestyle="", label="Control")
    ax2.plot(list(range(Nsamp)), [xer_thresh]*Nsamp, "--", c="black", label=f"Xer thresh={xer_thresh:.0f}")   # Xer threshold line
    ax2.plot(list(range(Nsamp)), [mu_ctrl]*Nsamp, ":", c="orange")   # Mean control saliva line
    ax2.plot(list(range(Nsamp)), [mu_irr]*Nsamp, ":", c="blue")   # Mean irradiated saliva line
    ax2.set_xticks(list(range(Nctrl + Nirr)), [*idx_ctrl, *idx_irr], fontsize=6)
    ax2.set_xlabel("ID number")
    ax2.set_title("Manufactured saliva data")
    ax2.legend()

    fig.suptitle(f"Manufactured feature space: make {Ninfo} informative of {Nfts} features, with $N_{{samp}}$={Nsamp} samples, $N_{{ctrl}}$={Nctrl} control, $N_{{irr}}$={Nirr} irradiated."
                 f"\nDraw saliva data from two gaussian distributions with $\mu_{{ctrl}}$={mu_ctrl}, $\mu_{{irr}}$={mu_irr}, CV={cv}")
    plt.show()
