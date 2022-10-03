from select_utils import *
from data_loader import *
from model_utils import *
from endpoints import load_sg_area

import six
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, plot_roc_curve, auc, confusion_matrix
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from mrmr import mrmr_classif


def find_num_sign_different_features(df1, df2, difference_names="1, 2", p_thresh=0.05):
    # assume df1 and df2 have equal columns:
    if not all(df1.columns.values == df2.columns.values):
        print(">>>>> UNEQUAL COLUMNS")
        return 0
    count_equals = 0
    count_total = 0
    count_sign = 0
    for ft, vals1 in six.iteritems(df1):
        count_total += 1
        if not vals1.dtype == "object":
            vals1 = vals1.values
            vals2 = df2[ft].values
            if not all(vals1 == vals2):
                # st, pval = stats.ttest_ind(vals1, vals2, equal_var=True)    # 2 sided t-test
                # print(ft, f"{difference_names} two-sided t-test: stat={st:.3g}, pval={pval:.3g}")
                st, pval = stats.ttest_ind(vals1, vals2, equal_var=False)   #Welch's t-test
                print(ft, f"{difference_names} Welch's t-test: stat={st:.3g}, pval={pval:.3g}")
                is_sign = pval < p_thresh
                count_sign += 1 if is_sign else 0
                if is_sign:
                    sns.histplot(vals1, kde=True, color="blue")
                    sns.histplot(vals2, kde=True, color="red")
                    plt.title(ft + f"\nWelch's t-test: p={pval:.3e}")
                    plt.show()
            else:
                count_equals += 1
    print(f"Of {count_total} fts have:\n\t{count_equals} equals\n\t{count_sign} significantly different (p < {p_thresh})")
    return 1


def compare_roi_areas_to_measured():
    X = "L"

    df_area = load_sg_area("all")   # LEFT SMG + SL
    # print(df_area)
    id_list = df_area["name"].values
    df = load_fsps_data("T2", "NO P", TRAIN_SET="ALL")
    df_t1 = load_fsps_data("T1", "NO P", TRAIN_SET="ALL")
    df = df[["name", "time", "original_shape2D_PixelSurface"]].rename({"original_shape2D_PixelSurface":"area"}, axis=1)
    df_t1 = df_t1[["name", "time", "original_shape2D_PixelSurface"]].rename({"original_shape2D_PixelSurface":"area"}, axis=1)

    df = df[[x.split("_")[-1] == X for x in df["name"]]]    # only consider left or right ROI
    df_t1 = df_t1[[x.split("_")[-1] == X for x in df_t1["name"]]]    # only consider left or right ROI
    # df_red = df[[x.split("_")[0] in id_list and x.split("_")[-1] == X for x in df["name"]]]
    # print(f"Corr SMG / SLGL all {len(df_area)} area measurements: {df_area.corr().loc['''SMG''','''SLGL''']:.3f}")
    corr, p = stats.pearsonr(df_area.dropna()["SMG"], df_area.dropna()["SLGL"])
    print(f"Corr SMG / SLGL all {len(df_area)} area measurements: {corr:.3f} with p-value = {p:.3g}")

    # sns.heatmap(df_area.corr(), annot=True)
    # plt.show()

    smg_vals = []
    sg_vals = []
    summed_vals = []
    roi_vals = []
    roi_vals_t1 = []
    id_vals = []
    times_late = []
    df_reg = pd.DataFrame()
    for id, area_smg, area_slgl in df_area.values:
        # print(id, area_smg, area_slgl)
        df_id = df[[x.split("_")[0] == id for x in df["name"]]]
        df_id_t1 = df_t1[[x.split("_")[0] == id for x in df_t1["name"]]]
        times = [int(x[:-3]) for x in df_id["time"].values]
        times.sort()
        if any(times) and times[-1] > 5:
            times_late.append(times[-1])
            df_tmp = df_id[df_id["time"] == str(times[-1]) + "day"]
            area_roi = df_tmp["area"].values[0]
            print(id, area_smg, area_slgl, times[-1], area_roi)
            smg_vals.append(area_smg);  sg_vals.append(area_slgl);  summed_vals.append(area_smg + area_slgl)
            roi_vals.append(area_roi);  id_vals.append(id)

            df_tmp_t1 = df_id_t1[df_id_t1["time"] == str(times[-1]) + "day"]
            area_roi_t1 = df_tmp_t1["area"].values[0]
            roi_vals_t1.append(area_roi_t1)

            df_reg.loc[id, "ROI T2 no p"] = area_roi
            df_reg.loc[id, "ROI T1 no p"] = area_roi_t1
            df_reg.loc[id, "SMG"] = area_smg
            df_reg.loc[id, "SLG"] = area_slgl
            # df_reg.loc[id, "smg + slgl"] = area_smg + area_slgl

    print("Times ROI:", np.unique(times_late, return_counts=True))
    rho, p = stats.pearsonr(roi_vals, sg_vals)
    print(f"Corr T2 ROI to SLGL: rho={rho:.3f}, p={p:.4f}")

    vals_list = [roi_vals, roi_vals_t1, smg_vals, sg_vals]
    corrvals = [stats.pearsonr(y, x) for x in vals_list for y in vals_list]
    kws = [f"$\\rho$ = {rho:.2f}\np = {p:.2f}" for rho, p in corrvals]
    kws = np.reshape(kws, (4, 4))   # must be same shape as corr matrix
    fig2, ax2 = plt.subplots()
    # sns.heatmap(df_reg.corr(), annot=True, ax=ax2, fmt=".2f", cbar=False)
    sns.heatmap(df_reg.corr(), ax=ax2, fmt="", cbar=False, annot=kws)
    fig2.suptitle(f"Pearson correlation between N={len(df_reg)} area measurements from biopsies of the SLG + SMG.")

    rhos = {}
    for vals, gland in zip([smg_vals, sg_vals, summed_vals, roi_vals_t1], ["SMG", "SG", "SMG + SG", "ROI T1"]):
        rho, p = stats.pearsonr(vals, roi_vals)
        rhos[gland] = rho
        print(f"Pearson corr between ROI area and {gland}: {rho:.3g}, p={p:.3g}")

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    lns = ax.plot(id_vals, roi_vals, "o-", label="ROI (T2)", markersize=4, c="r")
    lns += ax.plot(id_vals, roi_vals_t1, "o-", label=f"ROI (T1): $\\rho={rhos['''ROI T1''']:.3f}$", markersize=4, c="g")
    lns += ax2.plot(id_vals, smg_vals, "x:", label=f"SMG:         $\\rho={rhos['''SMG''']:.3f}$")
    lns += ax2.plot(id_vals, sg_vals, "x:", label=f"SG:           $\\rho={rhos['''SG''']:.3f}$")
    lns += ax2.plot(id_vals, summed_vals, "x:", label=f"SMG + SG: $\\rho={rhos['''SMG + SG''']:.3f}$")
    ax.grid()
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    ax.set_ylabel("# pixels", fontsize=12)
    ax2.set_ylabel("mm2", fontsize=12)
    ax.set_xlabel("ID", fontsize=12)
    fig.tight_layout()
    plt.show()
    # plt.close()
    return 1


def compare_roi_areas(X="L", return_pvals=True):
    # COMPARE SEGMENTED ROI AREAS BETWEEN: T1, T2, No-P, after-p
    # df1 = pd.read_csv(os.path.join(ftsDir, "LR_split_FSPS_extracted_T1.csv")).drop("name.1", axis=1)
    df1 = pd.read_csv(os.path.join(ftsDir, "LR_split_FSPS_extracted_T1.csv"))
    df1 = df1[["name", "time", "original_shape2D_PixelSurface"]].rename({"original_shape2D_PixelSurface":"area"}, axis=1)
    df1["id"] = [x.split("_")[0] for x in df1["name"].values]
    df1 = df1[[x.split("_")[-1] == X for x in df1["name"].values]]
    print("Loaded T1:", df1.shape)
    df1_p = df1[["p" in x for x in df1["name"].values]]
    df1 = df1.drop(df1_p.index)

    df2 = pd.read_csv(os.path.join(ftsDir, "LR_split_FSPS_extracted_T2.csv"))
    df2 = df2[["name", "time", "original_shape2D_PixelSurface"]].rename({"original_shape2D_PixelSurface":"area"}, axis=1)
    df2["id"] = [x.split("_")[0] for x in df2["name"].values]
    df2 = df2[[x.split("_")[-1] == X for x in df2["name"].values]]
    print("Loaded T2:", df2.shape)
    df2_p = df2[["p" in x for x in df2["name"].values]]
    df2 = df2.drop(df2_p.index)

    id_vals = list(set(df2["id"].values))
    print(f"Having {len(id_vals)} id's in T2 no-p")

    df_main = pd.DataFrame()
    for id in id_vals:
        times = df2[df2["id"] == id]["time"]
        for t in times:
            idt = id + ":" + t
            df_main.loc[idt, "T2 no-p"] = df2[(df2["id"] == id) & (df2["time"] == t)]["area"].values[0]
            t1_nop = df1[(df1["id"] == id) & (df1["time"] == t)]["area"].values
            if any(t1_nop):
                df_main.loc[idt, "T1 no-p"] = t1_nop[0]
            t2_p = df2_p[(df2_p["id"] == id) & (df2_p["time"] == t)]["area"].values
            if any(t2_p):
                df_main.loc[idt, "T2 p"] = t2_p[0]
            t1_p = df1_p[(df1_p["id"] == id) & (df1_p["time"] == t)]["area"].values
            if any(t1_p):
                df_main.loc[idt, "T1 p"] = t1_p[0]
            if any([len(x) > 1 for x in [t1_nop, t1_p, t2_p]]):
                print(">>>>> Multiple instances found....")
                return 0

    sns.heatmap(df_main.corr(), annot=True, fmt=".2f", cbar=False)
    plt.title(f"Corr between number of pixels in ROI, {'''left''' if X=='''L''' else '''right'''} SG")

    corr_annot = corr_matrix_with_pvals(df_main)
    # fig2, ax2 = plt.subplots()
    # sns.heatmap(df_main.corr(), annot=corr_annot, fmt="", cbar=False)

    # plt.show()
    plt.close()

    # print(df_main)
    print(df_main.shape)
    print("T2 no p:", df_main["T2 no-p"].dropna().shape)
    print("T2 p:", df_main["T2 p"].dropna().shape)
    print("T1 no-p:", df_main["T1 no-p"].dropna().shape)
    print("T1 p:", df_main["T1 p"].dropna().shape)
    print("All:", df_main.melt().dropna().shape)
    # print(df_main.loc["8-3:105day"])
    # df_main = df_main.dropna()
    # print(df_main.shape)
    # sns.heatmap(df_main.corr())
    # plt.show()
    return df_main.corr(), corr_annot



def saliv_statistics(ids = []):
    # ids = ["11-", "12-", "13-", "14-"]  # Pilot 4
    # ids += ["5-", "6-", "7-", "8-", "9-"]   # Pilot 3
    # ids = ["8-", "9-"]  # 7.5 Gy / frac 2x per day data

    saliv = load_saliva(melt=True, include_baseline_as_ctrl=False)
    print("Loaded: ", saliv.shape)
    if ids:
        saliv = saliv[[x in ids for x in saliv["name"].values]]
        print("Reduced to ", len(ids), "ids:", saliv.shape)
    # print(saliv["name"].values)
    # saliv = saliv[[x[:4][:-1] in ids for x in saliv["name"]]]

    # print("Reduced: ", saliv.shape)
    # print(np.unique(saliv["name"].values))
    timegroups = ["baseline",  "day 3 - 12", "day 26 - 75"]
    # saliv["timegroup"] = pd.cut(saliv["time"], [-10, 0, 12, 35, 75], right=True, labels=["baseline",  "3 - 12", "26 - 35", "56 - 75"])
    # saliv["timegroup"] = pd.cut(saliv["time"], [-10, 0, 8, 26, 75], right=True, labels=["baseline",  "day 3 - 8", "day 12 - 26", "day 35 - 75"])
    saliv["timegroup"] = pd.cut(saliv["time"], [-10, 0, 12, 75], right=True, labels=timegroups)
    sal_ctr = saliv[saliv["ctr"] == True]
    sal_irr = saliv[saliv["ctr"] == False]
    print("Ctr:", sal_ctr.shape, "Irr:", sal_irr.shape)
    print(np.unique(saliv["timegroup"].values, return_counts=True))

    # pd.options.display.float_format = '{:,.0f}'.format
    print("CONTROL + IRR:\n", saliv.groupby(["timegroup"])["val"].describe())
    print("CONTROL:\n", sal_ctr.groupby(["timegroup"])["val"].describe())
    print("IRRADIATED:\n", sal_irr.groupby(["timegroup"])["val"].describe())

    import matplotlib.patheffects as PathEffects
    txt_outline_width = 1.5
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=saliv, x="timegroup", y="val", hue="ctr", ax=ax)#, labels=["Control", "Irradiated"])
    ax.set_ylabel("Saliva measured [$\mu L$]")
    ax.set_xlabel("")

    medians_ctr = sal_ctr.groupby(["timegroup"])["val"].median().values
    medians_irr = sal_irr.groupby(["timegroup"])["val"].median().values
    counts_ctr = sal_ctr["timegroup"].value_counts(sort=False).values
    counts_irr = sal_irr["timegroup"].value_counts(sort=False).values
    for x in ax.get_xticks():
        txt = ax.text(x + 0.1, medians_ctr[x] + 5, f"N={counts_ctr[x]}", size=10, color="w", weight="semibold")
        txt.set_path_effects([PathEffects.withStroke(linewidth=txt_outline_width, foreground='black')])
        txt = ax.text(x - 0.35, medians_irr[x] - 12, f"N={counts_irr[x]}", size=10, color="w", weight="semibold")
        txt.set_path_effects([PathEffects.withStroke(linewidth=txt_outline_width, foreground='black')])

    handles, _ = ax.get_legend_handles_labels()
    print(handles)
    ax.legend(handles, ["Irradiated", "Control"], loc="upper left")
    ax.set_title(f"Saliva measurements over time for irradiated (N={len(sal_irr)}) + control (N={len(sal_ctr)}) mice.")
    # plt.show()
    # plt.close()

    # fig4, ax4 = plt.subplots()
    # sns.boxplot(data=saliv, x="time", y="val", hue="ctr", ax=ax4)
    fig4, axes4 = plt.subplots(ncols=3)
    for ax4, timegroup in zip(axes4, timegroups):
        sns.histplot(data=saliv[saliv["timegroup"] == timegroup], x="val", hue="ctr", ax=ax4, kde=True, legend=False)
        ax4.set_title(timegroup)
        handles, _ = ax4.get_legend_handles_labels()
        print(ax4.get_legend_handles_labels())
        print(handles)
        # ax4.legend(handles, ["Irradiated", "Control"], loc="best")
        ax4.legend(["Control", "Irradiated"], loc="best")
    [axx.set_ylabel("") for axx in axes4[1:]]
    [axx.set_xlabel("") for axx in axes4]
    axes4[1].set_xlabel("Saliva production [$\mu$L / 15 min]")

    timestats = pd.DataFrame(index=np.unique(saliv["time"].values))
    st = sal_ctr["time"].value_counts(sort=False)
    timestats.loc[st.index, "num control"] = st.values
    st = sal_irr["time"].value_counts(sort=False)
    timestats.loc[st.index, "num irr"] = st.values
    st = sal_ctr.groupby(["time"])["val"].mean()
    timestats.loc[st.index, "ctr means"] = st.values
    st = sal_irr.groupby(["time"])["val"].mean()
    timestats.loc[st.index, "irr means"] = st.values
    st = sal_ctr.groupby(["time"])["val"].median()
    timestats.loc[st.index, "ctr medians"] = st.values
    st = sal_irr.groupby(["time"])["val"].median()
    timestats.loc[st.index, "irr medians"] = st.values
    print(timestats)

    fig3, ax3 = plt.subplots()
    print(saliv.columns.values)

    sns.heatmap(saliv.corr(), cbar=False, ax=ax3, annot=True)
    # plt.show()

    fig2, ax2 = plt.subplots(nrows=2, ncols=len(medians_ctr), figsize=(10, 6))
    for i, timegroup in enumerate(saliv["timegroup"].unique()):
        # df_tmp = saliv[saliv["timegroup"] == timegroup]

        vals_ctr = sal_ctr[sal_ctr["timegroup"] == timegroup]["val"].values
        vals_irr = sal_irr[sal_irr["timegroup"] == timegroup]["val"].values
        st, p = stats.ttest_ind(vals_ctr, vals_irr, equal_var=True)
        print("\n", timegroup, f"\tirr / ctr different means (t-test equal vars): {st:.3g}, p={p:.2g}")
        st, p = stats.ttest_ind(vals_ctr, vals_irr, equal_var=False)
        print(timegroup, f"\tirr / ctr different means (t-test unequal vars): {st:.3g}, p={p:.2g}")
        print("\tControl:", len(vals_ctr), "Irr:", len(vals_irr))
        # if i == 0:
        #     vals_ctr_baseline = vals_ctr
        #     vals_irr_baseline = vals_irr
        # else:
        #     st, p = stats.ttest_ind(vals_ctr, vals_ctr_baseline)
        #     # st, p = stats.ttest_rel(vals_ctr, vals_ctr_baseline)
        #     print(timegroup, f"ctr different to baseline: st={st:.3f}, p={p:.2g}")
        #     st, p = stats.ttest_ind(vals_irr, vals_irr_baseline)
        #     # st, p = stats.ttest_rel(vals_irr, vals_irr_baseline)
        #     print(timegroup, f"irr different to baseline: st={st:.3f}, p={p:.2g}")
        # # df_tmp = sal_ctr[sal_ctr["timegroup"] == timegroup]
        df_tmp = sal_irr[sal_irr["timegroup"] == timegroup]


        sns.histplot(df_tmp, x="val", kde=True, ax=ax2[0, i])
        ax2[0, i].set_title(timegroup)
        ax2[0, i].set_xlabel("Saliva [$\mu L$]")

        stats.probplot(df_tmp["val"], dist="norm", plot=ax2[1, i])
        ax2[1, i].set_ylabel("Saliva [$\mu L$]")
        ax2[1, i].set_title("")

    plt.show()
    return 1


def xerostomia_statistics():
    saliv = load_saliva(melt=True, include_baseline_as_ctrl=False)
    xer = binary_thresh_xerostomia(dfy=saliv, only_return_xer=False)
    xer["timegroup"] = pd.cut(xer["time"], [-10, 0, 12, 75], right=True, labels=["baseline",  "day 3 - 12", "day 26 - 75"])
    # print(xer)

    # xer_ctr = xer[xer["ctr"] == True]
    # xer_irr = xer[xer["ctr"] == False]

    xer_baseline = xer[xer["timegroup"] == "baseline"]
    xer_acute = xer[xer["timegroup"] == "day 3 - 12"]
    xer_late = xer[xer["timegroup"] == "day 26 - 75"]

    numtot, numxer = xer_baseline["xer"].describe()[["count", "freq"]].values
    numctr, numctrxer = xer_baseline[xer_baseline["ctr"] == True]["xer"].describe()[["count", "freq"]].values
    print("Baseline:\tTot=", numtot, f"({numxer} xer), ctr={numctr} ({numctrxer} xer)")
    numtot, numxer = xer_acute["xer"].describe()[["count", "freq"]].values
    numctr, numctrxer = xer_acute[xer_acute["ctr"] == True]["xer"].describe()[["count", "freq"]].values
    print("Acute:\t\tTot=", numtot, f"({numxer} xer), ctr={numctr} ({numctrxer} xer)")
    numtot, numxer = xer_late["xer"].describe()[["count", "freq"]].values
    numctr, numctrxer = xer_late[xer_late["ctr"] == True]["xer"].describe()[["count", "freq"]].values
    print("Late:\t\tTot=", numtot, f"({numxer} xer), ctr={numctr} ({numctrxer} xer)")

    # print(xer[["xer", "ctr"]].describe())
    # print(xer.groupby("xer")[["ctr", "timegroup"]].sum())

    # Find ID overlap between baseline and acute or late
    ids_baseline = xer_baseline["name"].values
    ids_acute = xer_acute["name"].values
    ids_late = xer_late["name"].values

    id_overlap_baseline_acute = set(ids_baseline).intersection(set(ids_acute))
    id_overlap_baseline_late = set(ids_baseline).intersection(set(ids_late))
    print("ID overlap baseline acute:", len(id_overlap_baseline_acute))
    print("ID overlap baseline late:", len(id_overlap_baseline_late))

    df_overlap_base_acute = find_overlap_values(xer_baseline, xer_acute, pick_latest_value=True, col_to_save="xer")
    df_overlap_base_late = find_overlap_values(xer_baseline, xer_late, pick_latest_value=True, col_to_save="xer")

    df_overlap_base_acute_ctr = df_overlap_base_acute[df_overlap_base_acute["ctr"] == True]
    df_overlap_base_acute_irr = df_overlap_base_acute[df_overlap_base_acute["ctr"] == False]
    df_overlap_base_late_ctr = df_overlap_base_late[df_overlap_base_late["ctr"] == True]
    df_overlap_base_late_irr = df_overlap_base_late[df_overlap_base_late["ctr"] == False]


    # McNemar's test: based on contingency table of xer baseline / later for SAME individuals!
    from statsmodels.stats.contingency_tables import mcnemar

    ctab = create_contingency_table(df=df_overlap_base_acute_ctr, c1="xer1", c2="xer2")
    print("Baseline-acute CONTROL:", ctab, f"N={np.sum(ctab)}")
    print(mcnemar(ctab, exact=False, correction=True))
    ctab = create_contingency_table(df=df_overlap_base_acute_irr, c1="xer1", c2="xer2")
    print("\nBaseline-acute IRR:", ctab,f"N={np.sum(ctab)}\n", mcnemar(ctab, exact=False, correction=True))

    ctab = create_contingency_table(df=df_overlap_base_late_ctr, c1="xer1", c2="xer2")
    print("\nBaseline-late CONTROL:", ctab, f"N={np.sum(ctab)}")
    print(mcnemar(ctab, exact=False, correction=True))
    ctab = create_contingency_table(df=df_overlap_base_late_irr, c1="xer1", c2="xer2")
    print("\nBaseline-late IRR:", ctab, f"N={np.sum(ctab)}\n", mcnemar(ctab, exact=False, correction=True))


    pass



def univariate_AUC(x_train_orig, y_train_orig, x_test_orig, y_test_orig, fts, n_repeats=1,
                   classifier="RF", tune_hp = False, title="", use_together=True, bootstrap=False, figax=[], showplot=True):
    # classifier needs to be a (binary?) classifier, as AUC is not defined for regression
    # y: binary outcome values of same length as df
    # ft: name of feature (column in df) for use as single predictor
    # If ft is list: evaluate all fts in list independently, plot together..
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    from sklearn.ensemble import RandomForestClassifier
    from classification import rf_hyperparamtuning
    print(f"----- CALCULATING UNIVARIATE AUC's FOR {len(fts)} features using {classifier} ------")
    print(x_train_orig.shape, x_test_orig.shape)
    if not figax:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig, ax = figax

    for j, ft in enumerate(fts):
        x_train, x_test = x_train_orig[ft].to_numpy().reshape(-1, 1), x_test_orig[ft].to_numpy().reshape(-1, 1)
        # print(x_train.shape, y_train.shape)
        # print(ft, x_test)
        # print(y_test_orig)
        if classifier == "RF":
            params = {}
            if tune_hp:
                params = rf_hyperparamtuning(x_train, y_train_orig, mode="classifier")
            classif = RandomForestClassifier(**params)
        else:
            print("Invalid clasif", classifier)
            return 0

        tpr_vals = []
        fpr_ax = np.linspace(0, 1, 100)   # interpolate tpr results to this shape
        auc_vals = []

        for i in range(n_repeats):
            if bootstrap:
                x_test, y_test = resample(x_test_orig[ft].to_numpy().reshape(-1, 1), y_test_orig)
            else:
                x_test, y_test = x_test, y_test_orig

            classif.fit(x_train, y_train_orig)
            y_pred = classif.predict(x_test)
            # print(y_pred.shape, y_test_orig.shape)
            acc_val = accuracy_score(y_test, y_pred)
            try:
                auc_val = roc_auc_score(y_test, y_pred)
                fpr, tpr, thresh = roc_curve(y_test, y_pred)
                # fpr_vals.append(fpr)
                tpr_interp = np.interp(fpr_ax, fpr, tpr)    # interpolate tpr to higher "resolution"
                tpr_interp[0] = 0.0
                # tpr_vals.append(tpr)
                tpr_vals.append(tpr_interp)
                auc_vals.append(auc_val)
            except Exception as e:
                pass

            # print(f"\nUnivariate prediction using {ft}: acc={acc_val:.3f}, auc={auc_val:.3f}")
            # print(fpr, tpr, thresh)
            # print("Predicted:\t", list(y_pred))
            # print("Test:\t\t", list(y_test_orig))
            # ax.plot(fpr, tpr, marker="o", linestyle="dashed", label=f"{ft}: AUC={auc_val:.2f}")

        tpr_mean = np.mean(tpr_vals, axis=0)
        tpr_mean[-1] = 1.0
        # print(f"Avg AUC = {auc(fpr_ax, tpr_mean):.3f}, {np.mean(auc_vals):.3f}")
        auc_mean = np.mean(auc_vals)
        auc_sd = np.std(auc_vals)
        lab = f"{j + 1}: {ft}: AUC={auc_mean:.3f} $\pm$ {auc_sd:.3f}" if not set(fts) == set(["time", "dose"]) else f"{ft}: AUC={auc_mean:.3f} $\pm$ {auc_sd:.3f}"
        # ax.plot(fpr_ax, tpr_mean, linestyle="dashed", label=f"{j}: {ft}: AUC={auc_mean:.3f} $\pm$ {auc_sd:.3f}")
        ax.plot(fpr_ax, tpr_mean, linestyle="dashed", label=lab)
        print(ft, f"auc={auc_mean:.3f} +- {auc_sd:3f}")

    if use_together:
        # Calculate / plot AUC for classifier where all features in fts are included
        x_train, x_test = x_train_orig[fts].to_numpy(), x_test_orig[fts].to_numpy()
        print(x_train.shape, x_test.shape)
        if classifier == "RF":
            params = {}
            if tune_hp:
                params = rf_hyperparamtuning(x_train, y_train_orig, mode="classifier")
            classif = RandomForestClassifier(**params)

        tpr_vals = []
        fpr_ax = np.linspace(0, 1, 100)   # interpolate tpr results to this shape
        auc_vals = []

        for i in range(n_repeats):
            if bootstrap:
                # x_train, y_train = resample(x_train_orig[fts].to_numpy(), y_train[fts].to_numpy())
                x_test, y_test = resample(x_test_orig[fts].to_numpy(), y_test_orig)
            else:   pass

            classif.fit(x_train, y_train)
            y_pred = classif.predict(x_test)
            # print(y_pred.shape, y_test.shape)
            # acc_val = accuracy_score(y_test, y_pred)
            try:
                auc_val = roc_auc_score(y_test, y_pred)
                fpr, tpr, thresh = roc_curve(y_test, y_pred)
                # ax.plot(fpr, tpr, "-o", label=f"All: AUC={auc_val:.2f}")

                auc_vals.append(auc_val)
                tpr_interp = np.interp(fpr_ax, fpr, tpr)
                tpr_interp[0] = 0.0
                tpr_vals.append(tpr_interp)
            except Exception as e:  pass
        tpr_mean = np.mean(tpr_vals, axis=0)
        tpr_mean[-1] = 1.0
        tpr_sd = np.std(tpr_vals, axis=0)
        auc_mean = np.mean(auc_vals)
        auc_sd = np.std(auc_vals)
        ax.plot(fpr_ax, tpr_mean, "-", label=f"All: AUC={auc_mean:.3f} $\pm$ {auc_sd:.3f}")
        ax.fill_between(fpr_ax, tpr_mean - tpr_sd, tpr_mean + tpr_sd, color="grey", alpha=0.2, label="")

    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), ":", c="black")#, label="AUC=0.50")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    # figtitle = title + "\n" if title else ""
    # figtitle += f"{'''with bootstrap''' if bootstrap else ''''''}{''' repeated ''' + str(n_repeats) + ''' times''' if n_repeats > 1 else ''''''}. Ntrain={len(x_train)}, Ntest={len(x_test)}"
    # fig.suptitle(figtitle)
    if not title:
        ax.set_title(f"{'''with bootstrap''' if bootstrap else ''''''}{''' repeated ''' + str(n_repeats) + ''' times''' if n_repeats > 1 else ''''''}. Ntrain={len(x_train)}, Ntest={len(x_test)}")
    else:
        ax.set_title(title)
    ax.grid()
    ax.legend()
    plt.show() if showplot else 0
    return 1


def show_confusion_matrix(cm, figax=[], showplot=True):
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()
    pred_var_name = "xer"
    # annot_matrix = [["TN", "FP"], ["FN", "TP"]]
    annot_matrix = np.reshape([f"{txt} = {c}" for c, txt in zip(cm.ravel(), ["TN", "FP", "FN", "TP"])], (2, 2))
    # print(annot_matrix)
    sns.heatmap(data=cm, annot=annot_matrix, fmt="", cbar=False, ax=ax)
    # sns.heatmap(data=cm, annot=True, fmt="", cbar=False, ax=ax)
    ax.set_xticks(ax.get_xticks(), ["Predicted False", "Predicted True"])
    ax.set_yticks(ax.get_yticks(), ["Actual False", "Actual True"])
    ax.xaxis.tick_top()

    plt.show() if showplot else 0
    pass


def longitudinal_paired_ttest(df):
    # assuming values of type for column: boolean ctr, float val, int time, str name
    TIMES_BASELINE = [-3, -7]
    TIMES_ACUTE = [3, 5, 8, 12]
    TIMES_LATE = [26, 35, 56, 75]
    df_ctr = df[df["ctr"] == True]
    df_irr = df[df["ctr"] == False]
    names_all = np.unique(df["name"].values)
    names_ctr = np.unique(df_ctr["name"].values)
    names_irr = np.unique(df_irr["name"].values)
    if any(np.intersect1d(names_irr, names_ctr)):
        print(">>>>>OVERLAPPING INDIVIDUALS CTR / IRR!!!")
        return 0

    baseline_reg_acute = []     # indices for baseline ID's having an acute measurement, and vice versa, of shape (n, 2)
    baseline_reg_late = []      # etc
    acute_reg_late = []
    # acute_reg_base = []
    # late_reg_base = []
    # late_reg_acute = []

    for nm in names_all:
        df_name = df[df["name"] == nm]
        times_for_name = df_name["time"].values
        times_for_name.sort()
        times_b = list(filter(lambda t: t in TIMES_BASELINE, times_for_name))
        times_ac = list(filter(lambda t: t in TIMES_ACUTE, times_for_name))
        times_l = list(filter(lambda t: t in TIMES_LATE, times_for_name))
        # print(nm, times_b, times_ac, times_l)
        if any(times_b) and any(times_ac):
            t_b = times_b[0]        # earliest time
            t_ac = times_ac[-1]     # latest acute time
            idx_b = int(df_name[df_name["time"] == t_b].index.values)   # get error if multiple rows are selected (should not happen)
            idx_ac = int(df_name[df_name["time"] == t_ac].index.values)   # get error if multiple rows are selected (should not happen)
            baseline_reg_acute.append([idx_b, idx_ac])
        if any(times_b) and any(times_l):
            t_b = times_b[0]
            t_l = times_l[-1]       # latest late time
            idx_b = int(df_name[df_name["time"] == t_b].index.values)
            idx_l = int(df_name[df_name["time"] == t_l].index.values)
            baseline_reg_late.append([idx_b, idx_l])
        if any(times_ac) and any(times_l):
            t_ac = times_ac[-1]
            t_l = times_l[-1]
            idx_ac = int(df_name[df_name["time"] == t_ac].index.values)
            idx_l = int(df_name[df_name["time"] == t_l].index.values)
            acute_reg_late.append([idx_ac, idx_l])

    baseline_reg_acute = np.array(baseline_reg_acute)
    baseline_reg_late = np.array(baseline_reg_late)
    acute_reg_late = np.array(acute_reg_late)

    print("Baseline reg acute:", baseline_reg_acute.shape, "Num control=", df.loc[baseline_reg_acute[:, 0]].value_counts("ctr")[True], "Num irr=", df.loc[baseline_reg_acute[:, 0]].value_counts("ctr")[False])
    print("Baseline reg late:", baseline_reg_late.shape, "Num control=", df.loc[baseline_reg_late[:, 0]].value_counts("ctr")[True], "Num irr=", df.loc[baseline_reg_late[:, 0]].value_counts("ctr")[False])
    print("Acute reg late:", acute_reg_late.shape, "Num control=", df.loc[acute_reg_late[:, 0]].value_counts("ctr")[True], "Num irr=", df.loc[acute_reg_late[:, 0]].value_counts("ctr")[False])

    # Loop over modes, indices, do paired t-test
    for REG_IDX, cs in zip([baseline_reg_acute, baseline_reg_late, acute_reg_late], ["baseline-acute", "basline-late", "acute-late"]):
        print("\n", cs.upper(), np.shape(REG_IDX))
        ctr1, ctr2 = [], []
        irr1, irr2 = [], []
        for [idx1, idx2] in REG_IDX:
            df1 = df.loc[idx1]
            df2 = df.loc[idx2]
            is_ctr = df1["ctr"]
            # print(idx1, idx2, df1["name"], df2["name"], df1["time"], df2["time"], is_ctr)
            if not df2["ctr"] == is_ctr:
                print("ID CONTROL ERROR")
                return 0
            val1, val2 = df1["val"], df2["val"]
            if is_ctr:
                ctr1.append(val1)
                ctr2.append(val2)
            else:
                irr1.append(val1)
                irr2.append(val2)
        stat, pval = stats.ttest_rel(ctr1, ctr2)
        print(f"Control: N={len(ctr1)}, t-stat={stat:.2f}, p={pval:.2e}")
        stat, pval = stats.ttest_rel(irr1, irr2)
        print(f"Irradiated: N={len(irr1)}, t-stat={stat:.2f}, p={pval:.2e}")
    pass


def compare_roi_to_saliva_over_time():
    saliv = load_saliva(melt=True)
    saliv["timegroup"] = pd.cut(saliv["time"], [-10, 0, 12, 75], right=True,
                                labels=["baseline", "day 3 - 12", "day 26 - 75"])
    # names_saliv = np.unique(saliv["name"].values)
    # print(len(names_saliv))
    df_img = load_all_roi_areas()
    area_modes = list(filter(lambda col: col not in ["name", "time", "ctr"], df_img.columns.values))

    for col, count in zip(area_modes, df_img[area_modes].count().values):
        print(f"{col}:{count}", end="\t")
    print()

    df_img = df_img.melt(id_vars=["name", "time", "ctr"], value_vars=area_modes, value_name="area")
    df_img = df_img.dropna()
    df_img.loc[:, "Lbool"] = [var[-1] == "L" for var in df_img["variable"].values]

    df_img["timegroup"] = pd.cut(df_img["time"], [-10, 0, 12, 75], right=True,
                                labels=["baseline", "day 3 - 12", "day 26 - 75"])

    print(df_img)
    df_img_t1 = df_img[["T1" in var for var in df_img["variable"]]]
    df_img_t2 = df_img[["T2" in var for var in df_img["variable"]]]
    df_img_L = df_img[["L" in var for var in df_img["variable"]]]
    df_img_R = df_img[["R" in var for var in df_img["variable"]]]
    df_img_p = df_img[["_p" in var for var in df_img["variable"]]]
    df_img_nop = df_img[["nop" in var for var in df_img["variable"]]]
    df_img_ctr = df_img[df_img["ctr"] == True]
    df_img_irr = df_img[df_img["ctr"] == False]

    # print(df_img.groupby(["timegroup"]).count())
    # print(df_img_ctr.groupby(["timegroup"]).count())
    # print(df_img_irr.groupby(["timegroup"]).count())
    df_img_late = df_img[df_img["timegroup"] == "day 26 - 75"]
    print(df_img_late)
    df_img_late_ctr = df_img_late[df_img_late["ctr"] == True]
    df_img_late_irr = df_img_late[df_img_late["ctr"] == False]
    print(df_img_late_ctr.shape, df_img_late_irr.shape)
    t, p = stats.ttest_ind(df_img_late_ctr["area"], df_img_late_irr["area"])
    print(f"Different mean late control (N={len(df_img_late_ctr)}) to irr (N={len(df_img_late_irr)}): p={p:.2e}")
    print(f"\tmean late control = {df_img_late_ctr['''area'''].mean():.0f}, irr = {df_img_late_irr['''area'''].mean():.0f}")

    fig, (ax, ax2) = plt.subplots(ncols=2)
    sns.boxplot(data=df_img, x="timegroup", y="area", hue="ctr", ax=ax, hue_order=[False, True])
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[f"Irradiated (N={len(df_img_irr)})", f"Control (N={len(df_img_ctr)})"])
    ax.set_ylabel("ROI area [# of pixels]")
    ax.set_xlabel("")
    sns.boxplot(data=df_img, x="timegroup", y="area", hue="Lbool", ax=ax2, hue_order=[True, False], palette="Set2")
    handles, labs = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles, labels=[f"Left unit (N={len(df_img[df_img['''Lbool''']==True])})", f"Right unit (N={len(df_img[df_img['''Lbool''']==False])})"])
    ax2.set_ylabel("")
    ax2.set_xlabel("")

    # ax.legend(title="", loc="best", labels=["Irradiated", "Control"])

    # fig, ax = plt.subplots(nrows=3, ncols=3)
    # sns.boxplot(data=df_img, x="timegroup", y="area", hue="ctr", ax=ax[0, 0])
    # sns.boxplot(data=df_img_t1, x="timegroup", y="area", hue="ctr", ax=ax[0, 1])
    # sns.boxplot(data=df_img_t2, x="timegroup", y="area", hue="ctr", ax=ax[0, 2])
    # sns.boxplot(data=df_img_L, x="timegroup", y="area", hue="ctr", ax=ax[1, 1])
    # sns.boxplot(data=df_img_R, x="timegroup", y="area", hue="ctr", ax=ax[1, 2])
    # sns.boxplot(data=df_img_p, x="timegroup", y="area", hue="ctr", ax=ax[2, 1])
    # sns.boxplot(data=df_img_nop, x="timegroup", y="area", hue="ctr", ax=ax[2, 2])
    # ax[0, 0].set_title("All ROI's")
    # ax[0, 1].set_title("T1 ROI's")
    # ax[0, 2].set_title("T2 ROI's")
    plt.show()
    pass


def calculate_roi_statistics_by_normalizations(make_kdeplots=False, savestats=True):
    # May also produce kdeplots of all images, split into whole image and mask for T1 and T2 separately
    from extract_utils import nrrdDir, PreprocessDir
    from feature_extractor import plot_nrrd
    from SimpleITK import GetArrayFromImage, ReadImage

    FONTSIZE = 20   # for plotting
    # norm_modes = ["no norm no n4", "no norm", "stscore", "nyul"]
    norm_modes = ["stscore", "nyul"]
    for norm in norm_modes:
        print("\n", norm.upper())
        ParentDir = os.path.join(nrrdDir, f"LR split {norm}")
        n_t1 = 0
        n_t2 = 0
        j = 0
        df = pd.DataFrame()

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
        for exp in os.listdir(ParentDir):
            for time in os.listdir(os.path.join(ParentDir, exp)):
                path_time = os.path.join(ParentDir, exp, time)
                files = os.listdir(path_time)
                for image in list(filter(lambda f: "image" in f, files)):
                    j += 1
                    name = "_".join(image.split("_")[:-1])  # drop _image.nrrd
                    t1bool = "T1" in name
                    pbool = "p" in name

                    print(j, time + name, end="\t")
                    mask = name + "_mask.nrrd"

                    path_image = os.path.join(path_time, image)
                    path_mask = os.path.join(path_time, mask)
                    # plot_nrrd(path_image, path_mask, name=time+name)

                    im = GetArrayFromImage(ReadImage(path_image))
                    msk = GetArrayFromImage(ReadImage(path_mask))
                    vals_image = im.ravel()

                    # im_msk = im * msk
                    # vals_roi = np.flatnonzero(im_msk)
                    vals_roi = im[msk != 0].ravel()
                    print("roi size =", len(vals_roi))

                    mean_image = np.mean(vals_image)
                    median_image = np.median(vals_image)
                    sd_image = np.std(vals_image)
                    cv_image = sd_image / mean_image
                    q3_image = np.quantile(vals_image, 0.75)    # third quartile (= 75th percentile)
                    q1_image = np.quantile(vals_image, 0.25)    # third quartile (= 75th percentile)
                    iqr_image = q3_image - q1_image                 # inter quartile range
                    qcd_image = (q3_image - q1_image) / (q3_image + q1_image)   # quartile coefficient of dispersion

                    mean_roi = np.mean(vals_roi)
                    median_roi = np.median(vals_roi)
                    sd_roi = np.std(vals_roi)
                    cv_roi = sd_roi / mean_roi
                    q3_roi = np.quantile(vals_roi, 0.75)    # third quartile (= 75th percentile)
                    q1_roi = np.quantile(vals_roi, 0.25)    # third quartile (= 75th percentile)
                    iqr_roi = q3_roi - q1_roi                 # inter quartile range
                    qcd_roi = (q3_roi - q1_roi) / (q3_roi + q1_roi)   # quartile coefficient of dispersion

                    df.loc[j, "name"] = name
                    df.loc[j, "mean image"] = mean_image
                    df.loc[j, "median image"] = median_image
                    df.loc[j, "sd image"] = sd_image
                    df.loc[j, "cv image"] = cv_image
                    df.loc[j, "iqr image"] = iqr_image
                    df.loc[j, "qcd image"] = qcd_image

                    df.loc[j, "roi size"] = len(vals_roi)
                    df.loc[j, "mean roi"] = mean_roi
                    df.loc[j, "median roi"] = median_roi
                    df.loc[j, "sd roi"] = sd_roi
                    df.loc[j, "cv roi"] = cv_roi
                    df.loc[j, "iqr roi"] = iqr_roi
                    df.loc[j, "qcd roi"] = qcd_roi

                    if savestats:
                        path_df = os.path.join(PreprocessDir, f"stats {norm}.csv")
                        df.to_csv(path_df)
                    
                    if make_kdeplots:
                        if t1bool:
                            n_t1 += 1
                            sns.kdeplot(vals_image, ax=ax[1, 0], lw=0.5, color="orange", alpha=0.8)
                            sns.kdeplot(vals_roi, ax=ax[1, 1], lw=0.5, color="orange", alpha=0.8)
                        else:
                            n_t2 += 1
                            sns.kdeplot(vals_image, ax=ax[0, 0], lw=0.5, color="orange", alpha=0.8)
                            sns.kdeplot(vals_roi, ax=ax[0, 1], lw=0.5, color="orange", alpha=0.8)

                    # break
                # break
            # break

        if make_kdeplots:
            ax[0, 0].set_title("Image T2", fontsize=FONTSIZE)
            ax[0, 1].set_title("ROI T2", fontsize=FONTSIZE)
            ax[1, 0].set_title("Image T1", fontsize=FONTSIZE)
            ax[1, 1].set_title("ROI T1", fontsize=FONTSIZE)

            ax[0, 0].set_ylabel("KDE images", fontsize=FONTSIZE)
            ax[1, 0].set_ylabel("KDE ROIs", fontsize=FONTSIZE)
            ax[0, 1].set_ylabel("")
            ax[1, 1].set_ylabel("")

            ax[0, 0].set_xlabel("")
            ax[0, 1].set_xlabel("")
            ax[1, 0].set_xlabel("pixel intensity [a.u.]", fontsize=FONTSIZE-2)
            ax[1, 1].set_xlabel("pixel intensity [a.u.]", fontsize=FONTSIZE-2)

            for axx in ax.ravel():
                axx.grid(True)
            fig.suptitle(f"{norm}")
            # plt.show()
            savepath = os.path.join(FigDir, "preprocessing\KDE hist", f"kdegrid {norm}.png")
            plt.savefig(savepath)
            plt.close()
        else:
            plt.close()
        print("Num T1 / T2 ROIs:", n_t1, n_t2)
    pass


def compare_roi_statistics_by_norm():
    # Load calculated values from calculate_roi_statistics_by_normalizations
    files = os.listdir(PreprocessDir)
    files = list(filter(lambda f: "stats" in f, files))
    print(files)
    df_summary = pd.DataFrame()
    for f in files:
        norm = " ".join(f.split(" ")[1:])[:-4]
        print(norm)
        df = pd.read_csv(os.path.join(PreprocessDir, f), index_col=0)
        df.loc[:, "pbool"] = ["p" in nm for nm in df["name"]]
        df.loc[:, "t1bool"] = ["T1" in nm for nm in df["name"]]
        print(df.columns)

        # add all metrics
        # metrics = df.drop(["name", "pbool", "t1bool"], axis=1).columns  # column names of statistical values
        # avgs = np.average(df.drop(["name", "pbool", "t1bool"], axis=1).values, axis=0)
        # df_summary.loc[norm, metrics] = avgs

        # split metric by pbool, t1bool
        metric = "cv means"

        if metric == "cv means":    # calculate cv as sd(means) / mean(means)
            means_all = df["mean roi"]
            means_t1 = df[df["t1bool"] == True]["mean roi"]
            means_t2 = df[df["t1bool"] == False]["mean roi"]
            means_nop = df[df["pbool"] == False]["mean roi"]
            means_p = df[df["pbool"] == True]["mean roi"]
            val_all = means_all.std() / means_all.mean()
            val_t1 = means_t1.std() / means_t1.mean()
            val_t2 = means_t2.std() / means_t2.mean()
            val_nop = means_nop.std() / means_nop.mean()
            val_p = means_p.std() / means_p.mean()
        else:
            val_all = df[f"{metric} roi"].mean()
            val_t1 = df[df["t1bool"] == True][f"{metric} roi"].mean()
            val_t2 = df[df["t1bool"] == False][f"{metric} roi"].mean()
            val_nop = df[df["pbool"] == False][f"{metric} roi"].mean()
            val_p = df[df["pbool"] == True][f"{metric} roi"].mean()

        vals = [val_all, val_t1, val_t2, val_nop, val_p]
        print(vals)
        df_summary.loc[norm, ["All", "T1", "T2", "No p", "p"]] = vals
        # print(avgs)
        # break
    print(df_summary.index.values)

    fig, ax = plt.subplots()
    rows = ["Raw", "No norm", "Nyul", "St. score"]
    df_summary_n = df_summary.div(df_summary.max(axis=1), axis=0)   # norm to max by values in row (for comparative colorscheme)
    sns.heatmap(data=df_summary_n, annot=df_summary, fmt=".2g", cmap="gray_r", cbar=False, vmin=-0.5, vmax=1.25, ax=ax)
    ax.set_title(f"avg {metric.upper()} values in ROI")
    ax.set_yticks(ax.get_yticks(), rows)
    plt.show()
    pass


def t_test_coregT12_classif(LRmode="aggregated", predmode="after-irr", Nfts=5, CVMODE="3fold", hptunebool=True):
    if CVMODE.lower() == "3fold":
        if predmode == "acute":
            predmodename = predmode
        else:
            predmodename = "predict_late_" + predmode + "_with"

        savename_root = f"pairwise_T1T2_LR{LRmode}_{predmodename}_NO-P_RF_classification_validate_Nfts={Nfts}"

        df = pd.DataFrame()
        for splitnum in [1, 2, 3]:
            savepath = os.path.join(ClassifDir, "pairwise T1T2 classif", savename_root + f"_split{splitnum}.csv")
            # print(savepath)
            df_split = pd.read_csv(savepath, index_col=0)
            print(df_split.shape)
            df = pd.concat([df, df_split], axis=0)
        df = df.reset_index().drop("index", axis=1)

        vals_t1 = list(df["auc_1"])
        vals_t2 = list(df["auc_2"])
        vals_comb = list(df["auc_comb"])
        vals_td = list(df["auc_td"])

    elif CVMODE.lower() == "loocv":
        hptuned = "_hptuned" if hptunebool else ""
        predmode = "simult" if predmode == "acute" else predmode
        predmode = " ".join(predmode.split("-"))

        folder = os.path.join(ClassifDir, "pairwise T1T2 classif", "loocv")
        loadname = f"pairwiseT12_loocv_LR{LRmode}_{predmode}_NO P_nfts={Nfts}{hptuned}.csv"
        # loadname_fts = f"pairwiseT12_loocv_LR{LRmode}_{predmode}_NO P_nfts={Nfts}_selectedfts{hptuned}.csv"
        df = pd.read_csv(os.path.join(folder, loadname), index_col=0)
        df = df.groupby("idx").mean()
        cols_rf = list(filter(lambda c: "rf" in c, df.columns.values))
        df = df[["y_gt", *cols_rf]] # reduce to RF models only

        vals_t1 = list(df["p_T1_rf"])
        vals_t2 = list(df["p_T2_rf"])
        vals_comb = list(df["p_T1 + T2_rf"])
        vals_td = list(df["p_td_rf"])
        y_true = list(df["y_gt"])

        aucvals = [roc_auc_score(y_true, probs) for probs in [vals_td, vals_t1, vals_t2, vals_comb]]
        vals_t1 = [(pi - yi)**2 for pi, yi in zip(vals_t1, y_true)]
        vals_t2 = [(pi - yi)**2 for pi, yi in zip(vals_t2, y_true)]
        vals_comb = [(pi - yi)**2 for pi, yi in zip(vals_comb, y_true)]
        vals_td = [(pi - yi)**2 for pi, yi in zip(vals_td, y_true)]
    else:
        print("CVmode", CVMODE, "indvalid. TRY: 3fold, loocv")
        return 0

    print(df)

    print(f"\nLR{LRmode} {predmode} {Nfts} fts {CVMODE}")
    # print("INDEPENDENT t-test:")
    # f, p = stats.ttest_ind(aucvals_t1, aucvals_t2, equal_var=False)
    # print(f"t-test T2 to T1:\t p={p:.2e}")
    # f, p = stats.ttest_ind(aucvals_comb, aucvals_t2, equal_var=False)
    # print(f"t-test T2 to T1+T2:\t p={p:.2e}")
    # f, p = stats.ttest_ind(aucvals_td, aucvals_t2, equal_var=False)
    # print(f"t-test T2 to td:\t p={p:.2e}")
    # f, p = stats.ttest_ind(aucvals_comb, aucvals_t1, equal_var=False)
    # print(f"t-test T1 to T1+T2:\t p={p:.2e}")
    # f, p = stats.ttest_ind(aucvals_td, aucvals_t1, equal_var=False)
    # print(f"t-test T1 to td:\t p={p:.2e}")
    # f, p = stats.ttest_ind(aucvals_comb, aucvals_td, equal_var=False)
    # print(f"t-test T1+T2 to td:\t p={p:.2e}")

    print("DEPENDENT (relative) t-test:")
    # f, p = stats.ttest_rel(vals_t1, vals_t2, alternative='less')
    f, p = stats.ttest_rel(vals_t1, vals_t2)
    print(f"t-test T2 to T1:\t p={p:.3f}")
    f, p = stats.ttest_rel(vals_comb, vals_t2)
    print(f"t-test T2 to T1+T2:\t p={p:.3f}")
    f, p = stats.ttest_rel(vals_td, vals_t2)
    print(f"t-test T2 to td:\t p={p:.3f}")
    f, p = stats.ttest_rel(vals_comb, vals_t1)
    print(f"t-test T1 to T1+T2:\t p={p:.3f}")
    f, p = stats.ttest_rel(vals_td, vals_t1)
    print(f"t-test T1 to td:\t p={p:.3f}")
    f, p = stats.ttest_rel(vals_comb, vals_td)
    print(f"t-test T1+T2 to td:\t p={p:.3f}")

    medians = [df[x].median() for x in df.columns.values]
    labels = ["T1", "T2", "T1 + T2", "time + dose"]
    # plt.rc('legend', fontsize=12)

    if CVMODE == "loocv":
        for model, vals, auc in zip(["td", "T1", "T2", "comb"], [vals_td, vals_t1, vals_t2, vals_comb], aucvals):
            bs = np.sum(vals) / len(vals)
            print(f"{model}: BS = {bs:.3f}, AUC = {auc:.3f}")

    g = sns.histplot(data=df.melt(), x="value", hue="variable", kde=True, multiple="dodge", legend=True)
    g.legend_.set_title("Feature type")
    for leg, lab, med in zip(g.legend_.texts, labels, medians):
        lab_new = f"{lab}\t\t($m={med:.2f}$)" if not len(lab)>7 else f"{lab}\t($m={med:.2f}$)"
        leg.set_text(lab_new)

    plt.title(f"LR{LRmode} {predmode} {Nfts} fts")
    plt.xlabel("ROC AUC")
    plt.xlim([0.4, 1.01])
    plt.ylim([0, 275]) if predmode == "acute" else 0
    plt.grid(True)
    if CVMODE == "loocv":
        plt.close()
    else:
        plt.show()
    pass


def feature_correlations(mode="all"):
    # x1, x2, y = load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", predict_late="after irr",
    #                                               training="all", xer=True, SPLIT_NUMBER=1)
    # # Combined T1 + T2 model: select best fts --> check corr + to time and dose
    # x = aggregate_T1_T2_on_features(x1, x2)
    # print(x.shape, x1.shape, x2.shape, y.shape)
    # td = x[["time", "dose"]]
    # x = x.drop(["time", "dose"], axis=1)
    # top_fts = mrmr_classif(x, y, K=5)
    # print(top_fts)
    # x = x[top_fts]
    # top_fts_new = ["_".join(ft.split("_")[:-2]) for ft in top_fts]
    # print(top_fts)
    # print(top_fts_new)
    #
    # ft_index = get_feature_index_global()
    # x = x.rename(columns=dict((k, f"Ft{v}") for k, v in zip(top_fts, [ft_index[ft] for ft in top_fts_new])))
    # df = pd.concat([x, td], axis=1)

    df = load_fsps_data(MODE="NO P", WEIGHT="T2", TRAIN_SET="all", LRMODE="average")
    df = df.filter(like="shape", axis=1)
    print(df)
    print(df.shape)
    # return 0
    vals, annot, ticks = corr_matrix_with_pvals(df, abs=False)
    print(ticks)
    fig, ax = plt.subplots()
    sns.heatmap(data=vals[-2:], annot=annot[-2:], fmt="", cbar=0, vmin=-1, vmax=1, ax=ax)
    ticks_x = [f"\n{t}" if i%2 else f"{t}" for i, t in enumerate(ticks)]
    print(ticks_x)
    ax.set_xticks(ax.get_xticks(), ticks_x)
    # ax.set_yticks(ax.get_yticks(), ticks)
    ax.set_yticks(ax.get_yticks(), ["time", "dose"])
    plt.title("T1T2 coreg combined model (T1 + T2) top 5 features corr with time, dose")

    fig, ax = plt.subplots()
    sns.heatmap(data=vals, annot=annot, fmt="", cbar=0, vmin=-1, vmax=1, ax=ax)
    ax.set_xticks(ax.get_xticks(), ticks_x)
    ax.set_yticks(ax.get_yticks(), ticks)

    plt.show()
    # stats.pearsonr()



    pass


def fsps_feature_stats():
    for mode in ["NO P", "DELTA P", "DELTA"]:
        df_master = pd.DataFrame()
        for lrmode in ["aggregated", "average", "split"]:
            for weight in ["T1", "T2"] if mode == "NO P" else ["T2"]:
                print("\n", lrmode, mode, weight)

                thresh = 0.05 if weight=="T2" else 0.15
                df_norms = get_fsps_best_norm(lrmode, weight, thresh)
                # print(df_norms.groupby("norm").size().values)
                df_norms.loc[:, ["lrmode", "weight"]] = [lrmode, weight]
                df_master = pd.concat([df_master, df_norms], axis=0)
                # COMPARE TO ACTUAL FSPS DATA (from loader)
                fts_dict = get_feature_index_fsps(lrmode, mode, weight, return_values=False)
                num_fts = len(fts_dict)    # number of features after FSPS
                x = load_fsps_data(weight, mode, lrmode, TRAIN_SET="all", verbose=False)
                num_shape = 18 if lrmode == "aggregated" else 9
                print(f"FSPS fts: {num_fts} ({len(x.T)}), after FSPS: {len(df_norms)} + {num_shape} shape = {len(df_norms) + num_shape}")
                # sns.histplot(data=df_norms, x="filter")
                # plt.show()

        if True:
            df_master = df_master.reset_index()
            print(df_master)
            df_t1 = df_master[df_master["weight"] == "T1"]
            df_t2 = df_master[df_master["weight"] == "T2"]

            for wgt in ["T1", "T2"]:
                df = df_t1 if wgt == "T1" else df_t2
                df_lrmodecounts = df_t1.groupby("lrmode").size()
                print(f"\n{wgt} FSPS STATS:")

                df_filterstats = pd.DataFrame(dtype=int)
                df_typestats = pd.DataFrame(dtype=int)
                df_normstats = pd.DataFrame()
                for lrmode in df_lrmodecounts.index.values:
                    df_filterstats.loc[lrmode, "fsps fts"] = df_lrmodecounts.loc[lrmode]
                    df_lrmode = df[df["lrmode"] == lrmode]
                    df_lrmode_filters = df_lrmode.groupby("filter").size()
                    for filter in df_lrmode_filters.index:
                        df_filterstats.loc[lrmode, filter] = df_lrmode_filters.loc[filter]
                    df_lrmode_types = df_lrmode.groupby("ft type").size()
                    for fttype in df_lrmode_types.index:
                        df_typestats.loc[lrmode, fttype] = df_lrmode_types.loc[fttype]
                    df_lrmode_norms = df_lrmode.groupby("norm").size()
                    for norm in df_lrmode_norms.index:
                        df_normstats.loc[lrmode, norm] = df_lrmode_norms.loc[norm]
                print(df_filterstats)
                print(df_typestats)
                print(df_normstats)

            # sns.histplot(df_t1, x="filter", hue="lrmode", multiple="dodge")
            fig, axes = plt.subplots(ncols=2)
            norms = ["no norm", "stscore", "nyul"]
            sns.histplot(df_t1, x="norm", hue="lrmode", multiple="dodge", ax=axes[0])
            sns.histplot(df_t2, x="norm", hue="lrmode", multiple="dodge", ax=axes[1])
            axes[0].set_title("T1")
            axes[1].set_title("T2")
            plt.show()
    pass


def segmentation_params():
    from extract_utils import SegSalivDir
    path = os.path.join(SegSalivDir, "segment params salivary.csv")
    df = pd.read_csv(path, index_col=0)
    print(df)
    print(df.groupby("params").size().sort_values(ascending=False))
    pass

if __name__ == "__main__":

    # segmentation_params()
    # fsps_feature_stats()
    # saliv = load_saliva(melt=True)
    saliv_statistics()
    # xerostomia_statistics()
    # longitudinal_paired_ttest(saliv)
    # compare_roi_to_saliva_over_time()
    # compare_roi_areas(X="L")
    # show_T12_coreg_classif_results()

    # calculate_roi_statistics_by_normalizations(make_kdeplots=True, savestats=True)
    # compare_roi_statistics_by_norm()

    # from classification import show_T12coreg_loocv_results
    # show_T12coreg_loocv_results(hptuned=True, CLASSIFIERS="RF")
    # t_test_coregT12_classif(LRmode="aggregated", predmode="acute", CVMODE="loocv")
    # t_test_coregT12_classif(LRmode="aggregated", predmode="baseline", CVMODE="loocv")
    # t_test_coregT12_classif(LRmode="aggregated", predmode="after-irr", CVMODE="loocv")

    # feature_correlations()

    # saliv = load_saliva(melt=True)
    # saliv.loc[:, "xer"] = binary_thresh_xerostomia(saliv)
    # saliv = saliv[(saliv["name"] == "C2") | (saliv["name"] == "H3")]
    # print(saliv)

    sys.exit()

    sn = 1                    # split number
    predmode = "baseline"
    df1_train, df2_train, y_train = load_T1T2_coreg("NO P", "aggregated", predict_late=predmode, training=True, xer=True, SPLIT_NUMBER=sn)
    df1_test, df2_test, y_test = load_T1T2_coreg("NO P", "aggregated", predict_late=predmode, training=False, xer=True, SPLIT_NUMBER=sn)
    fts = ["time", "dose"]
    x_train = df1_train[fts]
    x_test = df1_test[fts]

    # PLOT CONFUSION MATRIX, ROC CURVE, AND DESICION BOUNDARY FOR TOP TWO FEATURES IN MODEL
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    # title = f"{predmode.capitalize()} split {sn}"

    fig, axes = plt.subplots(ncols=3, figsize=(15, 6))
    ax1, ax2, ax3 = axes.ravel()
    show_decision_boundary(rf, x_train, y_train, x_test, y_test, xlabel="time [day]", ylabel="dose [Gy]",
                           title="Decision boundary", show_train=True, showplot=False, figax=(fig, ax3))
    univariate_AUC(x_train, y_train, x_test, y_test, fts=fts, classifier="RF", tune_hp=False,
                   title="ROC curve", n_repeats=1, use_together=True, bootstrap=False, showplot=False, figax=(fig, ax2))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    show_confusion_matrix(cm, figax=(fig, ax1), showplot=False)

    print("True:", y_test.values)
    print("Pred:", y_pred)
    print(roc_auc_score(y_test, y_pred))
    print(roc_curve(y_test, y_pred))

    fig.suptitle(f"{predmode.capitalize()} split {sn}, $N_{{train}}$={len(x_train)}, $N_{{test}}$={len(x_test)}, accuracy={acc:.2f}, auc={auc:.2f}")
    plt.show()
    sys.exit()

    # from RF import visualize_model_performance_pairwise
    # visualize_model_performance_pairwise(path=os.path.join(ModelDir, "RF\pairwise T1T2",
    #     "pairwise_T1T2_LRaggregated_predict_late_after-irr_with_NO-P_RF_classification_validate_Nfts=15_split1.csv"))
    # saliv_statistics()

    # show_T12_coreg_classif_results()


    # CALCULATE UNIVARIATE (+ combined?) ROC CURVES FOR VARIOUS MODELS
    # Nrep = 1000
    Nrep = 100
    # latemode = "acute"
    # latemode = "baseline"
    latemode = "after irr"
    LRMODE = "aggregated"
    # LRMODE = "average"
    # LRMODE = "split"
    WGT = "T2"
    # WGT = "T1"

    # WGT = "BOTH"
    split_num = 1
    title = f"ROC curve {latemode} using NO P {WGT} LR {LRMODE} (co-reg T1 / T2 split {split_num})"
    # if latemode == "acute":
        # df_train, y_train = load_nop(WEIGHT=WGT, LRMODE=LRMODE, training=True, xer=True)
        # df_test, y_test = load_nop(WEIGHT=WGT, LRMODE=LRMODE, training=False, xer=True)
    if latemode in ["after irr", "baseline", "acute"]:
        pred_late = False if latemode == "acute" else latemode
        t1_train, t2_train, y_train = load_T1T2_coreg(MODE="NO P", LRMODE=LRMODE, predict_late=pred_late,
                                                      training=True, xer=True, SPLIT_NUMBER=split_num)
        t1_test, t2_test, y_test = load_T1T2_coreg(MODE="NO P", LRMODE=LRMODE, predict_late=pred_late,
                                                      training=False, xer=True, SPLIT_NUMBER=split_num, keep_names=True)
        id_valid = list(y_test["name"])
        # print(y_test)
        y_test = y_test["xer"]
        # print(id_valid)
        # sys.exit()
        if WGT == "T1":
            df_train, df_test = t1_train, t1_test
        elif WGT == "T2":
            df_train, df_test = t2_train, t2_test
        elif WGT == "BOTH":
            df_train = aggregate_T1_T2_on_features(t1_train, t2_train)
            df_test = aggregate_T1_T2_on_features(t1_test, t2_test)
        else:
            sys.exit()
    else:
        sys.exit()


    saliv_statistics(id_valid)
    sys.exit()
    top_fts = mrmr_classif(df_train, y_train, K=5, return_scores=False)
    print("TOP 5 features:", top_fts)

    ft_list = top_fts
    # ft_list = list(set(["time", "dose", *top_fts[:3]]))     # remove duplicates
    # ft_list = ["time", "dose"]
    # ft_list = [top_fts[2], top_fts[4]]
    # ft_list = ["time", "dose", top_fts[0]]
    # print(ft_list)

    univariate_AUC(df_train, y_train, df_test, y_test, fts=ft_list, classifier="RF", tune_hp=False,
                   title=title, n_repeats=Nrep, use_together=True, bootstrap=True)
    # univariate_AUC(df_train, y_train, df_test, y_test, fts=["time", "dose"], classifier="RF", tune_hp=False,
    #                title=title, n_repeats=Nrep, use_together=True, bootstrap=True)



    # EVALUATE INTER-VARIABILITY between ROIs created for same mouse (ID) at same time (day)
    # compare_roi_areas_to_measured()
    # c1, ann1 = compare_roi_areas("L")
    # c2, ann2 = compare_roi_areas("R")
    #
    # fig, (ax1, ax2) = plt.subplots(ncols=2)
    # # sns.heatmap(c1, annot=True, fmt=".2f", cbar=False, ax=ax1)
    # # sns.heatmap(c2, annot=True, fmt=".2f", cbar=False, ax=ax2)
    # sns.heatmap(c1, annot=ann1, fmt="", cbar=False, ax=ax1)
    # sns.heatmap(c2, annot=ann2, fmt="", cbar=False, ax=ax2)
    # ax1.set_title("Left")
    # ax2.set_title("Right")
    # plt.show()


    # Count number of significantly different features from L + R SMG
    # df = load_fsps_data("T2", "ALL")
    # df_l = df[[x.split("_")[-1] == "L" for x in df["name"]]]
    # print(df_l.shape)
    # df_r = df.drop(df_l.index, axis=0)
    # print(df_r.shape)
    # find_num_sign_different_features(df_l, df_r, difference_names="L, R", p_thresh=1e-10)


