import pandas as pd
import os
import six
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import sys

import numpy as np
from select_utils import *
from model_utils import *
from name_dose_relation import is_control, dose_to_name


# def load_saliva(melt=False, impute=False, verbose=True):
#     # saliva1 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=2)[:16].drop("Unnamed: 5", axis=1)
#     saliva1 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=2)[:16].drop("Gender", axis=1)
#     # print(saliva1)
#     saliva2 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=23)[:16].drop("Unnamed: 5", axis=1)
#     saliva3 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=42)#[:16].drop("Unnamed: 5", axis=1)
#     saliva3 = saliva3.drop(saliva3.columns.values[saliva3.columns.str.contains("Unnamed")], axis=1)[:60]
#     saliva4 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=105).drop("Unnamed: 5", axis=1)    # 42 + 60 = 102
#
#     saliva1.columns = [int(c[4:-20]) for c in saliva1.columns]
#     saliva2.columns = [int(c[4:-19]) for c in saliva2.columns]
#     saliva3.columns = [int(c[4:]) for c in saliva3.columns]
#     saliva4.columns = [int(c[4:]) for c in saliva4.columns]
#     # print(saliva3)
#     # print(saliva4)
#     # saliva = saliva1.append(saliva2).append(saliva3)
#     saliva = saliva1.append(saliva2).append(saliva3).append(saliva4)
#     # saliva = saliva1.concat(saliva2).concat(saliva3)
#     # print(saliva.index)
#     saliva["name"] = saliva.index.values
#     # print(saliva)
#     if impute:
#         # SORT TO HAVE TIMES AS COLUMNS
#         # https://towardsdatascience.com/advanced-missing-data-imputation-methods-with-sklearn-d9875cbcc6eb
#         saliva_melt = pd.melt(saliva, id_vars="name", var_name="time", value_name="val")
#         Ndata_old = len(saliva_melt.dropna())
#         Nmissing = len(saliva_melt[saliva_melt["val"].isna()])
#         print("SALIV ORIGINAL data N=", Ndata_old, "MISSING N=", Nmissing)
#         # print(saliva_melt[saliva_melt["val"] == "-"])
#         saliva_melt["ctr"] = [is_control(time=t, name=nm) for t, nm in
#                             zip(saliva_melt["time"].values, saliva_melt["name"].values)]
#         # print(saliva_melt)
#         names = saliva_melt["name"].values
#         saliva_melt = saliva_melt.drop("name", axis=1)
#         saliva_melt.index = names
#         # print(saliva_melt)
#         # print(saliva_melt.isna().mean().sort_values())
#
#         from sklearn.impute import KNNImputer
#         saliva_imputed = saliva_melt.copy(deep=True)
#         knnimp = KNNImputer(n_neighbors=5)
#         saliva_imputed.loc[:, :] = knnimp.fit_transform(saliva_imputed)
#         # print(saliva_imputed.isnull().mean().sort_values())
#         print("\tIMPUTED", Nmissing, "SALIVA VALUES")
#         saliva_melt = saliva_imputed
#         saliva_melt["name"] = names
#         saliva_melt["ctr"] = [bool(x) for x in saliva_melt["ctr"].values]
#         # print(saliva_melt)
#     else:
#         saliva_melt = pd.melt(saliva, id_vars="name", var_name="time", value_name="val").dropna()
#         saliva_melt = saliva_melt[saliva_melt["val"] != "-"]
#         saliva_melt["ctr"] = [is_control(time=t, name=nm) for t, nm in
#                             zip(saliva_melt["time"].values, saliva_melt["name"].values)]
#     if melt:
#         # saliva_melt["ctr"] = [is_control(time=t, name=nm) for t, nm in
#         #                     zip(saliva_melt["time"].values, saliva_melt["name"].values)]
#         saliva_melt["time"] = [int(x) for x in saliva_melt["time"].values]
#         saliva_melt["val"] = [float(x) for x in saliva_melt["val"].values]
#         # return saliva
#         # print(saliva_melt)
#         print(f"----- SALIVA DATA LOADED FOR {len(saliva_melt)} INSTANCES (name, time) -----")
#         return saliva_melt
#     print(f"----- SALIVA DATA LOADED FOR {len(saliva_melt)} INSTANCES (name, time) -----") if verbose else 0
#     return saliva


def load_saliva(melt=False, impute=False, verbose=True, include_baseline_as_ctrl=False, include_TGFbeta_groups=False):
    tgf_beta_groups = ["7", "10"]

    # saliva1 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=2)[:16].drop("Unnamed: 5", axis=1)
    saliva1 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=2)[:16].drop("Gender", axis=1)
    # print(saliva1)
    saliva2 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=23)[:16].drop("Unnamed: 5", axis=1)
    saliva3 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=42)#[:16].drop("Unnamed: 5", axis=1)
    saliva3 = saliva3.drop(saliva3.columns.values[saliva3.columns.str.contains("Unnamed")], axis=1)[:60]
    saliva4 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=105).drop("Unnamed: 5", axis=1)    # 42 + 60 = 102

    saliva1.columns = [int(c[4:-20]) for c in saliva1.columns]
    saliva2.columns = [int(c[4:-19]) for c in saliva2.columns]
    saliva3.columns = [int(c[4:]) for c in saliva3.columns]
    saliva4.columns = [int(c[4:]) for c in saliva4.columns]
    # print(saliva4.shape)
    # print(saliva4)
    # print(saliva4.melt().dropna().drop_duplicates())
    # print(saliva3)
    # print(saliva4)
    # saliva = saliva1.append(saliva2).append(saliva3)
    saliva = saliva1.append(saliva2).append(saliva3).append(saliva4)
    saliva["name"] = saliva.index.values

    if not include_TGFbeta_groups:
        # id_vals = [x[:4] for x in saliva["name"].values]    #
        id_vals = saliva["name"]
        # print(id_vals)
        saliva = saliva.loc[list(filter(lambda x: x.split("-")[0] not in tgf_beta_groups, id_vals))]
        # saliva = saliva[[x[:-1] not in tgf_beta_groups for x in id_vals]]
        # saliva = saliva[[x[:2] not in tgf_beta_groups for x in saliva["name"].values]]

    if impute:
        # SORT TO HAVE TIMES AS COLUMNS
        # https://towardsdatascience.com/advanced-missing-data-imputation-methods-with-sklearn-d9875cbcc6eb
        saliva_melt = pd.melt(saliva, id_vars="name", var_name="time", value_name="val")
        Ndata_old = len(saliva_melt.dropna())
        Nmissing = len(saliva_melt[saliva_melt["val"].isna()])
        print("SALIV ORIGINAL data N=", Ndata_old, "MISSING N=", Nmissing)
        # print(saliva_melt[saliva_melt["val"] == "-"])
        saliva_melt["ctr"] = [is_control(time=t, name=nm, include_baseline=include_baseline_as_ctrl) for t, nm in
                            zip(saliva_melt["time"].values, saliva_melt["name"].values)]
        # print(saliva_melt)
        names = saliva_melt["name"].values
        saliva_melt = saliva_melt.drop("name", axis=1)
        saliva_melt.index = names
        # print(saliva_melt)
        # print(saliva_melt.isna().mean().sort_values())

        from sklearn.impute import KNNImputer
        saliva_imputed = saliva_melt.copy(deep=True)
        knnimp = KNNImputer(n_neighbors=5)
        saliva_imputed.loc[:, :] = knnimp.fit_transform(saliva_imputed)
        # print(saliva_imputed.isnull().mean().sort_values())
        print("\tIMPUTED", Nmissing, "SALIVA VALUES")
        saliva_melt = saliva_imputed
        saliva_melt["name"] = names
        saliva_melt["ctr"] = [bool(x) for x in saliva_melt["ctr"].values]
        # print(saliva_melt)
    else:
        saliva_melt = pd.melt(saliva, id_vars="name", var_name="time", value_name="val").dropna()
        saliva_melt = saliva_melt[saliva_melt["val"] != "-"]
        saliva_melt["ctr"] = [is_control(time=t, name=nm, include_baseline=include_baseline_as_ctrl) for t, nm in
                            zip(saliva_melt["time"].values, saliva_melt["name"].values)]
    if melt:
        # saliva_melt["ctr"] = [is_control(time=t, name=nm) for t, nm in
        #                     zip(saliva_melt["time"].values, saliva_melt["name"].values)]
        saliva_melt["time"] = [int(x) for x in saliva_melt["time"].values]
        saliva_melt["val"] = [float(x) for x in saliva_melt["val"].values]
        # print(saliva_melt.shape)
        # print("DUPLICATES:\n", saliva_melt[saliva_melt.duplicated(keep=False)])    # show duplicates
        saliva_melt = saliva_melt.drop_duplicates()
        # print(saliva_melt.shape)
        names = set(saliva_melt["name"].values)
        # names_ctr = set(saliva_melt[saliva_melt["ctr"] == True]["name"].values)
        names_ctr = set(filter(lambda x: is_control(x, include_baseline=False), saliva_melt["name"].values))
        names_irr = set(filter(lambda x: not(is_control(x, include_baseline=False)), saliva_melt["name"].values))
        # print(saliva_melt[saliva_melt["ctr"] == True])
        ctr_name = "0Gy" if include_baseline_as_ctrl else "control"
        print(f"----- SALIVA DATA LOADED FOR {len(saliva_melt)} INSTANCES ({len(saliva_melt[saliva_melt['''ctr'''] == True])} {ctr_name}, {len(saliva_melt[saliva_melt['''ctr'''] == False])} irr), {len(names)} individuals ({len(names_ctr)} {ctr_name}, {len(names_irr)} irr) -----")
        return saliva_melt
    names = set(saliva["name"].values)
    print(f"----- SALIVA DATA LOADED FOR {len(saliva_melt)} INSTANCES (name, time), {len(names)} individuals -----") if verbose else 0
    return saliva


def binary_thresh_xerostomia(dfy=np.array([0]), thresh_factor=0.5, plot=False, impute=False, only_return_xer=True):
    # print(dfy)
    # RETURN DFY WITH XER BOOL: TRUE IF BELOW LIN REG PREDICTION (on time) * THRESH_FACTOR
    print("----- FINDING XEROSTOMIC THRESHOLD FROM CONTROL + BASELINE SALIVA DATA -----")
    df_saliva = load_saliva(melt=True, impute=impute, include_baseline_as_ctrl=True)
    # print("SALIV: ", df_saliva.shape)
    if not(any(dfy)):
        # df_saliva = load_saliva(melt=True)
        dfy = df_saliva.copy()
    elif plot and not(any([x in dfy.columns.values for x in ["ctr", "ctrl"]])):
        dfy["ctr"] = [d == 0 for d in dfy["dose"].values]  # CONTROL (no dose) + BASELINE (-7day)d

    # ESTABLISH LINEAR RELATIONSHIP BETWEEN 0Gy SALIVA DATA AND TIME
    ctr = df_saliva[df_saliva["ctr"]][["val", "time"]]
    irr = df_saliva[df_saliva["ctr"] == False][["val", "time"]]
    # print("CTR: ", ctr.shape)
    # print("IRR: ", irr.shape)
    # print("NO IRR TIME DATA:", np.unique(ctr["time"].values, return_counts=True))
    times = ctr["time"].values.reshape(-1, 1)
    vals = ctr["val"].values.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(times, vals)
    # print(reg.coef_[0, 0], reg.intercept_[0])
    a = reg.intercept_[0]   #y = a + bx
    b = reg.coef_[0, 0]
    r2 = reg.score(times, vals)

    # XEROSTOMIA THRESHOLDING
    xer_thresh = reg.predict(dfy["time"].values.reshape(-1, 1)).reshape(-1, ) * thresh_factor
    # dfy["xer"] = dfy["val"].values < xer_thresh
    dfy.loc[:, "xer"] = dfy["val"].values < xer_thresh
    # print(dfy)


    if plot:
        import statsmodels.api as sm
        from scipy import stats
        X = sm.add_constant(times)
        ols = sm.OLS(vals, X)
        fit = ols.fit()

        fig, ax = plt.subplots(nrows=1)

        plt.plot(times, vals, "x", label=f"0Gy control + baseline (N={len(ctr)})", c="b", ms=6)
        plt.plot(times, reg.predict(times), "o:", label=f"Predicted 0Gy", c="b")
        plt.plot(irr["time"], irr["val"], "x", c="r", ms=4, label=f"Irradiated (N={len(irr)})")
        plt.xticks(np.unique(times), np.unique(times))
        plt.title(f"Linear fit on 0Gy data (N={len(ctr)}): y = {a:.2f} + {b:.2f}x\nR2 = {r2:.3f}        p-value={fit.f_pvalue:.1e}")
        plt.xlabel("Time [day]")
        plt.ylabel(f"Saliva amount [$\mu L$]")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

        times = dfy["time"].values.reshape(-1, 1)
        dfy_control = dfy[dfy["ctr"] == True][["val", "time", "xer"]]
        dfy_irr = dfy[dfy["ctr"] == False][["val", "time", "xer"]]
        dfy_control_xer = dfy_control[dfy_control["xer"] == True]
        dfy_control_noxer = dfy_control[dfy_control["xer"] == False]
        dfy_irr_xer = dfy_irr[dfy_irr["xer"] == True]
        dfy_irr_noxer = dfy_irr[dfy_irr["xer"] == False]
        N_xer = len(dfy_control_xer) + len(dfy_irr_xer)
        plt.plot(times, reg.predict(times), "-", label=f"Predicted 0Gy saliva $\hat{{y}}$", c="black")
        plt.plot(times, xer_thresh, ":", c="black",
                 label=f"Xer thresh = {thresh_factor:.1f}$\hat{{y}}$")
        plt.plot(dfy_control_xer["time"].values, dfy_control_xer["val"], "x", c="r", ms=6, label=f"0Gy xer=True (N={len(dfy_control_xer)})")
        plt.plot(dfy_control_noxer["time"].values, dfy_control_noxer["val"], "x", c="b", ms=6, label=f"0Gy xer=False (N={len(dfy_control_noxer)})")
        plt.plot(dfy_irr_xer["time"].values, dfy_irr_xer["val"], "o", c="r", ms=3, label=f"Irr xer=True (N={len(dfy_irr_xer)})")
        plt.plot(dfy_irr_noxer["time"].values, dfy_irr_noxer["val"], "o", c="b", ms=3, label=f"Irr xer=False (N={len(dfy_irr_noxer)})")
        plt.title(f"Xerostomic thresholding on all saliva data: True if observed value is below expected times threshold factor:"
                  f"$y(t) < \hat{{y}}(t)$ * {thresh_factor}"
        # plt.title(f"Xerostomic thresholding on saliva data (co-registered to no-p MRI data) - True if observed value is below expected times threshold factor: $y(t) < \hat{{y}}(t)$ * {thresh_factor}"
        # plt.title(f"Xerostomic thresholding on saliva data (co-registered to $\Delta$-p MRI data) - True if observed value is below expected times threshold factor: $y(t) < \hat{{y}}(t)$ * {thresh_factor}"
                  f"\n$N_{{xer}}$={N_xer} of $N_{{tot}}$={len(dfy)} ({N_xer / len(dfy) * 100:.1f}%)")
        plt.xlabel("Time [day]")
        plt.ylabel(f"Saliva amount [$\mu L$]")
        plt.xticks(np.unique(times), np.unique(times))
        plt.grid()
        plt.legend()
        plt.show()
        # plt.plot(times2, vals2, "x", label="f")
    num_xer = len(dfy[dfy['''xer'''] == True])
    if "ctr" in dfy.columns:
        ctr_bool = True
        num_xer_ctr = len(dfy[(dfy["ctr"] == True) & (dfy["xer"] == True)])
        num_xer_irr = len(dfy[(dfy["ctr"] == False) & (dfy["xer"] == True)])
        num_ctr = len(dfy[dfy["ctr"] == True])
        num_irr = len(dfy[dfy["ctr"] == False])
    else:
        ctr_bool = False

    print(f"----- Having {num_xer} of {len(dfy)} ({num_xer / len(dfy) * 100:.1f}%) xerostomic measurements with thresh_factor={thresh_factor}------")
    if ctr_bool:
        print(f"\t Number of xer measurements control: {num_xer_ctr} of {num_ctr} ({num_xer_ctr / num_ctr * 100:.1f}%)"
              f"\tirradiated: {num_xer_irr} of {num_irr} ({num_xer_irr / num_irr * 100:.1f}%)")
    return dfy["xer"] if only_return_xer else dfy


def load_cytok_table(path):
    ds = pd.read_csv(path)
    for i in ds.index:
        if pd.isna(ds.loc[i].values).all():
            # print(ds.loc[i])
            ds = ds.drop(index=i)
    ds.index = ds["Unnamed: 0"]
    ds.index.name = "Mouse ID"
    ds = ds.drop(columns=["Unnamed: 0"])
    return ds


def what_times(ds):
    times = []
    print(ds.index.name)
    if "Pilot" in ds.index.name:    #saliva 1 & 2 - here time is on columns
        for c in ds.columns:
            time = c[:6]
            # print(time[-1])
            if time[-1] in [":", " "]: time = time[:-1]
            times.append(time) if time not in times else 0
        return times

    for ind in ds.index:
        # print(ind)
        if "d" in ind:
            # print("d")
            loc = ind.index("d")
            time = ind[loc - 2:loc + 1]
        elif "W" in ind:
            # print("W")
            loc = ind.index("W") + 2
            time = ind[loc - 2:loc]
        times.append(time) if time not in times else 0
        # print(time)
    return times


def load_sg_area(gland="all"):
    # Measurements of LEFT SMG / SG, biopsies taken at day ?100 ish??
    Dir = os.path.join(RawDir, "Area salivary glands overview.csv")
    df = pd.read_csv(Dir)
    # print(df.columns)
    df = df.rename(columns={"Mouse ID":"name", "Area (mm^2) SMGL":"SMG", "Area (mm^2) SLGL":"SLGL"})
    # print(df)
    if gland in ["", "all"]:
        print("----- Loaded", len(df), "values for SMG & SG area (mm2) ----")
        # df = df.dropna()
        return df

    elif gland == "sum":
        df["value"] = df["SMG"].values + df["SLGL"].values
        df = df.drop(["SLGL", "SMG"], axis=1)
        # print(df)
        print("----- Loaded", len(df), "values for SUMMED SMG + SG area (mm2) ----")
        return df

    elif gland.lower() in ["smg", "smgl"]:
        df = df.drop("SLGL", axis=1)
        # print(df)
        df = df.rename(columns={"SMG":"value"})
        df = df.dropna()
        return df
    # for row, val in six.iteritems(df.T):
    #     print(row, val)
    # print(df.dropna())
    return df


def what_names(ds):
    names = []
    for ind in ds.index:
        # print(ind[-2:])
        if ind[-2] == "-":  #PILOT 2 A-B
            name = ind[-3:]
        else:               #PILOT 1 C2, H4, L2 etc
            name = ind[-2:]
        names.append(name) if name not in names else 0
    return names


def plot_saliva_distributions(saliv=[]):
    from scipy import stats
    if not any(saliv):
        saliv = load_saliva(melt=True)
        # print(saliv["name"])
        saliv["dose"] = [dose_to_name(nm, t, ignore_exp=True) for nm, t in zip(saliv["name"].values, saliv["time"].values)]
    else:
        pass
    print(saliv.shape)
    print(saliv)
    saliv = saliv[["val", "time", "dose", "name", "ctr"]]
    # saliv = saliv.rename(columns={"val":"saliva"})

    times_in_saliv = np.unique(saliv["time"])
    print(times_in_saliv)
    saliv_ctr = saliv[saliv["ctr"] == True]
    saliv_irr = saliv[saliv["ctr"] == False]
    print(saliv_ctr.shape, saliv_irr.shape)
    vals_irr = saliv.drop(saliv_ctr.index.values)["val"].values
    vals_ctr = saliv_ctr["val"].values

    stat, pval = stats.ttest_ind(vals_ctr, vals_irr, equal_var=False)
    print(stat, pval)
    print(f"Difference 0Gy vs irr:\t\t pval={pval:.4g}")

    sns.boxplot(data=saliv, x="ctr", y="val")
    plt.title("All saliva measurements for mice having recieved no irr (ctrl + baseline days) vs irr\n"
              f"No irr: N={len(vals_ctr)}, mean={np.mean(vals_ctr):.2f}. Irr: N={len(vals_irr)}, mean={np.mean(vals_irr):.2f}"
              "\n$H_0: \mu_{irr}$ = $\mu_{no irr}$ rejected with p-value of " + f"{pval:.3g} (Welch's t-test)")
    # plt.show()
    plt.close()

    # saliv_irr.loc[:, "baseline"] = [t in [-7, -3] for t in saliv_irr["time"].values]
    saliv.loc[:, "baseline"] = [t in [-7, -3] for t in saliv["time"].values]
    saliv_ctr.loc[:, "baseline"] = [t in [-7, -3] for t in saliv_ctr["time"].values]
    vals_ctr_baseline = saliv_ctr[saliv_ctr["baseline"] == True]["val"].values
    vals_ctr_later = saliv_ctr[saliv_ctr["baseline"] == False]["val"].values
    print("Ctr / baseline\n", saliv.value_counts(["ctr", "baseline"]))

    stat, pval = stats.ttest_ind(vals_ctr_baseline, vals_ctr_later, equal_var=False)
    print("Different mean baseline vs later (0Gy): p=", pval)
    # print(saliv_ctr)
    # sns.boxplot(data=saliv_ctr, y="val", x="baseline")
    sns.boxplot(data=saliv_ctr, y="val", x="baseline")
    # plt.show()
    plt.close()
    
    # PLOT ALL SALIVA DATA OVER TIME, SEP CONTROL / IRR
    fig2, ax = plt.subplots()
    # fig2, (ax, ax2) = plt.subplots(ncols=2)
    # fig2 = plt.figure()
    # ax = fig2.add_subplot(1, 2, 1)
    sns.scatterplot(data=saliv, x="time", y="val", hue="ctr", ax=ax, zorder=5)
    ax.legend(title='', loc='best', labels=[f"Control (N={len(saliv_ctr)})", f"Irradiated (N={len(saliv_irr)})"])
    ax.set_xlabel("Time [Day]")
    ax.set_ylabel(saliva_label)
    ax.set_xticks(times_in_saliv)
    ax.grid(zorder=0)
    # ax2 = fig2.add_subplot(1, 2, 2, projection="3d")
    # ax2.scatter(saliv_ctr["time"].values, saliv_ctr["dose"].values, saliv_ctr["val"].values, color="orange")
    # ax2.scatter(saliv_irr["time"].values, saliv_irr["dose"].values, saliv_irr["val"].values, color="blue")
    # ax2.set_xlabel("Time [Day]")
    # ax2.set_ylabel("Dose [Gy]")
    # ax2.set_zlabel(saliva_label)
    # sns.boxplot(data=saliv, x="ctr", y="val", ax=ax2)
    # corr, annot, ticks = corr_matrix_with_pvals(df_main=saliv.drop(["baseline", "ctr", "name"], axis=1), text=True, num_digits=2)
    # sns.heatmap(data=corr, annot=annot, fmt="", ax=ax2, cbar=False)
    # ax2.set_xticks(ax2.get_xticks(), ticks)
    # ax2.set_yticks(ax2.get_yticks(), ticks)
    # plt.close()
    # CORRELATION BETWEEN val, time, dose FOR IRR AND CONTROL
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3)
    corr, annot, ticks = corr_matrix_with_pvals(df_main=saliv.drop(["ctr", "name", "baseline"], axis=1), text=True, num_digits=2)
    # triangle = np.triu(corr)    # triangle mask
    sns.heatmap(data=corr, annot=annot, fmt="", ax=ax0, cbar=False)
    ticks = ["saliva", "time", "dose"]
    ax0.set_xticks(ax0.get_xticks(), ticks);    ax0.set_yticks(ax0.get_yticks(), ticks)

    corr, annot, ticks = corr_matrix_with_pvals(df_main=saliv_ctr.drop(["ctr", "name", "baseline", "dose"], axis=1), text=True, num_digits=2)
    # triangle = np.triu(corr)    # triangle mask
    sns.heatmap(data=corr, annot=annot, fmt="", ax=ax1, cbar=False)
    ticks = ["saliva", "time"]
    ax1.set_xticks(ax1.get_xticks(), ticks);    ax1.set_yticks(ax1.get_yticks(), ticks)

    corr, annot, ticks = corr_matrix_with_pvals(df_main=saliv_irr.drop(["ctr", "name"], axis=1), text=True, num_digits=2)
    # triangle = np.triu(corr)
    sns.heatmap(data=corr, annot=annot, fmt="", ax=ax2, cbar=False, annot_kws={"fontsize":10})
    ticks = ["saliva", "time", "dose"]
    ax2.set_xticks(ax2.get_xticks(), ticks);    ax2.set_yticks(ax2.get_yticks(), ticks)
    ax1.set_title("control");   ax2.set_title("Irradiated")
    plt.show()

    fig, (ax, ax2, ax3) = plt.subplots(ncols=3)
    hist_stat = "count"
    sns.histplot(data=saliv, x="val", hue="ctr", ax=ax, kde=True, stat=hist_stat, hue_order=[False, True])#, legend=False)
    # ax.set_ylabel()
    ax.set_title("All times")
    ax.set_xlabel("")
    ax.legend(title="", loc="best", labels=["Control", "Irradiated"])

    sns.histplot(data=saliv[saliv["baseline"] == True], x="val", hue="ctr", ax=ax2, kde=True, stat=hist_stat)
    ax2.set_xlabel(saliva_label)
    ax2.set_title("Baseline")
    ax2.legend()
    # ax2.legend(title="", loc="best", labels=["Control", "Irradiated"])
    ax2.set_ylabel("")
    sns.histplot(data=saliv[saliv["baseline"] == False], x="val", hue="ctr", ax=ax3, kde=True, stat=hist_stat)
    ax3.set_title("After irr start (incl day 3)")
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.legend()

    plt.show()
    return 0


if __name__ == "__main__":
    print()
    # cytok_saliva = load_cytok_table(os.path.normpath(os.path.join(os.getcwd(), "..", "RAW DATA", "cytokines_saliva.csv")))
    # cytok_blood = load_cytok_table(os.path.normpath(os.path.join(os.getcwd(), "..", "RAW DATA", "cytokines_blood.csv")))
    # load_sg_area()
    plot_saliva_distributions()
    sys.exit()

    # saliv = load_saliva()
    saliv = load_saliva(melt=True, include_TGFbeta_groups=False, include_baseline_as_ctrl=False)
    counts = pd.DataFrame()
    counts["all"] = saliv.value_counts("time", sort=False)
    counts["num ctr"] = saliv[saliv["ctr"] == True].value_counts("time", sort=False)
    counts["num irr"] = saliv[saliv["ctr"] == False].value_counts("time", sort=False)
    print(counts.T)
    saliv["xer"] = binary_thresh_xerostomia(dfy=saliv, plot=False)
    # num_xer = saliv.value_counts("xer")[True]
    # num_noxer = saliv.value_counts("xer")[False]
    # print(num_xer, num_noxer)
    # print(f"Having ")
    # print(saliv.value_counts("xer"))

    sys.exit()
    saliv["dose"] = [dose_to_name(name=nm, time=t, ignore_exp=True) for nm, t in zip(saliv["name"].values, saliv["time"].values)]
    neq = np.not_equal([saliv["dose"] == 0], saliv["ctr"].values)[0]
    if np.count_nonzero(neq):
        if any([x not in [-7, -3] for x in np.unique(saliv.loc[neq]["time"])]):
            print("MISMATCH CONTROL / DOSE.....")
    print(saliv.value_counts("ctr"))
    print(saliv.value_counts("time", sort=False))
    print(saliv.value_counts(["time", "ctr"], sort=False).T)
    plot_saliva_distributions(saliv)
    sys.exit()
    binary_thresh_xerostomia(dfy=saliv, plot=True, impute=False)
    # binary_thresh_xerostomia(plot=True, impute=False)
    # saliv_ctr = saliv[saliv["ctr"] == True]
    # fig, ax = show_interpolated_surface(x1=saliv["time"].values, x2=saliv["dose"].values, y=saliv["val"].values,
    #                                     y_sep=saliv["ctr"].values, sep_names=["Control", "Irr"], xlab="Time [Day]",
    #                                     ylab="Dose [Gy]", cbarlab="Saliva amount [$\mu$L]", Nsteps=50)

    sys.exit()

    print(saliv.shape)
    id_set = set(saliv["name"].values)
    time_set = set(saliv["time"].values)
    print("TOTAL # IDS", len(id_set))
    # print(time_set)
    time_list = list(time_set)
    time_list.sort()
    # print(time_list)
    measures_list = []
    for t in time_list:
        # print(saliv[saliv["time"] == t])
        saliv_in_time = saliv[saliv["time"] == t]
        ids_in_time = set(saliv_in_time["name"].values)
        # print(ids_in_time)
        # print("Time", t, "have #IDs =", len(ids_in_time))
        measures_list.append(len(ids_in_time))
        # break
    print(time_list)
    print(measures_list)
    # print(sum(measures_list))
    print()
    # print(saliv["ctr"])
    num_ctrl = 0
    for id in id_set:
        saliv_for_id = saliv[saliv["name"] == id]
        times_to_id = saliv_for_id["time"].values
        times_to_id.sort()
        ctrl = int(saliv_for_id[saliv_for_id["time"] == times_to_id[-1]]["ctr"].values)
        # print(id, times_to_id, ctrl)
        num_ctrl += ctrl
    print("Of", len(id_set), "mice with saliva measured have", num_ctrl, "ctrl")


    # print(saliv)
    # names = saliv["name"].values
    # saliv_long = pd.melt(saliv, id_vars=["name"], var_name="time", value_name="val").dropna()
    # saliv_long = saliv_long[saliv_long["val"] != "-"]
    # saliv_long["val"] = [int(x) for x in saliv_long["val"].values]
    # print(saliv_long)
    # saliv_long["ctr"] = [is_control(time=t, name=nm) for t, nm in zip(saliv_long["time"].values, saliv_long["name"].values)]
    # saliv_ctr = saliv_long[saliv_long["ctr"] == 1]
    # print(saliv_ctr)
    #
    # #   TODO : REGRESSION EXPRESSION FOR SALIVA AS FUNCTION OF TIME
    # # sns.catplot(x="time", y="val", kind="swarm", data=saliv_ctr)
    # # plt.show()
    # # sns.violinplot(x="time", y="val", data=saliv_ctr)
    # # sns.violinplot(x="time", y="val", data=saliv_long, hue="ctr", split=True)
    # # sns.catplot(x="time", y="val", kind="swarm", data=saliv_long, hue="ctr")
    # xvals = saliv_ctr["time"].values.reshape(-1, 1)
    # print(np.unique(xvals))
    # yvals = saliv_ctr["val"].values.reshape(-1, 1)
    # reg = LinearRegression().fit(xvals, yvals)
    # print(reg.score(xvals, yvals))  # R2
    # y_pred = reg.predict(xvals)
    # fig, ax = plt.subplots()
    # fig.suptitle(f"{len(yvals)} 0Gy DATA POINTS")
    # ax.plot(xvals, yvals, "x")
    # ax.plot(xvals, y_pred, "--")
    # plt.show()


    # saliva1 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=2)[:16].drop("Unnamed: 5", axis=1)
    # saliva2 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=23)[:16].drop("Unnamed: 5", axis=1)
    # saliva3 = pd.read_csv(os.path.normpath(os.path.join(RawDir, "saliva.csv")), index_col=0, header=42)#[:16].drop("Unnamed: 5", axis=1)
    # saliva3 = saliva3.drop(saliva3.columns.values[saliva3.columns.str.contains("Unnamed")], axis=1)
    # saliva1.columns = [int(c[4:-20]) for c in saliva1.columns]
    # saliva2.columns = [int(c[4:-19]) for c in saliva2.columns]
    # saliva3.columns = [int(c[4:]) for c in saliva3.columns]
    # saliva = saliva1.append(saliva2).append(saliva3)
    # print(saliva)

    # print(saliva1.columns)
    # print(saliva2.columns)
    # print(cytok_saliva)
    # print(cytok_blood)
    # print(what_names(saliva1))
    # print(what_names(saliva2))
    # print(what_names(cytok_blood))
    # print(what_names(cytok_saliva))

    # print(what_times(cytok_saliva))
    # print(what_times(cytok_blood))
    # print(what_times(saliva1))
    # print(what_times(saliva2))

    # ds = cytok_saliva
    # time = "-7d"
    # for time in what_times(ds):
    #     print(time)
    #     ds_t = ds[ds.index.str.contains(time)]
    #     print(ds_t)

