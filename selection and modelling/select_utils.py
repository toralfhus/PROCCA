import os
import pandas as pd
import numpy as np
from scipy import stats
import six
from name_dose_relation import is_control, dose_to_name
from visualizations import plot_ids_over_time


global ftsDir, RawDir
ftsDir = os.path.join(os.getcwd(), "..", "..", "Radiomic features")
selectDir = os.path.join(os.getcwd(), "..", "..", "Radiomic feature selection and modelling")
RawDir = os.path.join(os.getcwd(), "..", "..", "RAW DATA")
FigDir = os.path.join(os.getcwd(), "..", "..", "master_plots")
ModelDir = os.path.join(os.getcwd(), "..", "..", "Radiomic Modelling")
PreprocessDir = os.path.join(os.getcwd(), "..", "..", "Preprocessing")
ALL_TIMES = [-7, -3, 5, 8, 12, 26, 35, 56, 70, 105] # ALL TIMES WHERE MRI DATA EXISTS
BASELINE_TIMES = [-7, -3]
AFTER_IRR_TIMES = [5, 8, 12, 26, 35]
FEATURE_TYPES = ["shape", "firstorder", "glcm", "glrlm", "gldm", "ngtdm", "glszm"]
FILTER_TYPES = ["original", "logarithm", "squareroot", "wavelet", "exponential", "gradient", "lbp", "square"]

# saliva_label = "Saliva amount per measurement time [$\mu$L / 15 min]"
saliva_label = "Saliva production [$\mu$L / 15 min]"


ID_DELTA_T1_TRAIN = ['11-3', '13-10', '14-3', '14-5', '11-5', '13-9', '14-4', '14-2', '11-10', '9-4', '13-6', '11-4', '11-6', '9-5', '11-2', '9-3']
ID_DELTA_T1_VALID = ['12-1', '11-1', '13-8', '11-8', '11-9']

# ONLY T2
ID_DELTA_TRAIN = ['11-2', '11-3', '11-6', '11-8', 'L1', '14-2', '9-3', '9-2', '14-4', '1-1', '13-8', '8-3', '11-1', 'H1',
                  '9-4', 'L2', '13-9', 'C2', '11-9', '4-2', '14-3', '11-4', '1-2', '2-1', '11-10', '4-1', '14-5', '13-6', '2-2', '8-4', 'H3']
ID_DELTA_VALID = ['3-2', '9-5', '3-1', '8-5', '12-1', '13-10', '9-1', '11-5']   # Note: 9-1 are missing after irr (day 5) images..

# ONLY T2
ID_DELTA_P_TRAIN = ['C2', 'H1', 'H3', 'L1', '1-1', '1-2', '2-1', '2-2', '4-1', '4-2', '8-3', '8-4', '9-2', '9-3', '9-4',
                    '5-1', '5-2', '5-3', '5-5', '6-1', '6-2', '6-4', '6-5', '8-1', '8-7', 'C3', 'H4']
ID_DELTA_P_VALID = ['3-1', '3-2', '8-5', '9-1', '9-5', '5-4', '6-3', '8-2']

# T1 + T2
ID_NOP_T2_TRAIN = ['C2', 'H1', 'H3', 'L1', 'L2', '1-1', '1-2', '2-1', '2-2', '4-1', '4-2','8-3', '8-4', '9-2', '9-3', '9-4',
                   '11-10', '11-1', '11-2', '11-3', '11-4', '11-6', '11-8', '11-9', '13-6', '13-8', '13-9', '14-2', '14-3', '14-4',
                   '14-5', 'C1', 'H4', '8-1', '8-7', '5-3', '5-5', '5-1', '14-1', '6-1', 'C6', '11-7', '6-4', '5-2', 'C3', '6-5', '6-2']
ID_NOP_T2_VALID = ['3-1', '3-2', '8-5', '9-1', '9-5', '11-5', '12-1', '13-10', '13-7', '6-3', '5-4', '8-2']


ID_NOP_T1_TRAIN = ['9-3', '9-4', '8-3', '8-4', '9-2', '11-10', '11-1', '11-2', '11-3', '11-4', '11-6', '11-8', '11-9', '13-6',
                   '13-8', '13-9', '14-2', '14-3', '14-4', '14-5', '11-7', '14-1', '8-7']
ID_NOP_T1_VALID = ['9-5', '8-5', '9-1', '11-5', '12-1', '13-10', '13-7']



def LRsplit_to_aggregated(df, include_dose=True, verbose=True):
    df_l = df[[x.split("_")[-1] == "L" for x in df["name"]]]
    df_r = df[[x.split("_")[-1] == "R" for x in df["name"]]]
    print(df.shape, df_l.shape, df_r.shape)
    df_agg = pd.DataFrame()
    j = 0
    print("----- LR split --> LR aggregated -----")
    print("-" * (len(df_l) // 3))
    for idx, name, time in zip(df_l.index.values, df_l["name"].values, df_l["time"].values):
        name_orig = name[:-2]  # remove _L
        dose = df_l.loc[idx, "dose"] if include_dose else 0

        # print(j, name_orig, time, dose)
        print("-", end="") if not j%3 else 0

        df_r_instance = df_r[np.logical_and(df_r["name"] == name_orig + "_R", df_r["time"] == time)]
        df_l_instance = df_l.loc[idx]

        if include_dose:
            if not len(df_r_instance == 1) or df_r_instance["dose"].values[0] != dose:
                print(">>>>>ERR: ", df_r_instance.shape, df_r_instance["dose"].values, dose, name_orig, time)
                return 0
            else:
                df_r_instance = df_r_instance.iloc[0]
        else:
            if not len(df_r_instance == 1):
                print(">>>>>ERR: ", df_r_instance.shape)
                return 0
            else:
                df_r_instance = df_r_instance.iloc[0]

        df_temp = pd.DataFrame()
        if include_dose:
            df_temp.loc[j, ["name", "time", "dose"]] = [name_orig, time, dose]  # should be same values for L + R
            df_r_instance = df_r_instance.drop(["name", "time", "dose"], axis=0)
            df_l_instance = df_l_instance.drop(["name", "time", "dose"], axis=0)

        else:
            df_temp.loc[j, ["name", "time"]] = [name_orig, time]  # should be same values for L + R
            df_r_instance = df_r_instance.drop(["name", "time"], axis=0)
            df_l_instance = df_l_instance.drop(["name", "time"], axis=0)

        df_r_instance.index = [ind + "_R" for ind in df_r_instance.index.values]    # mark feature by L / R origins
        df_l_instance.index = [ind + "_L" for ind in df_l_instance.index.values]
        df_temp.loc[j, df_r_instance.index.values] = df_r_instance.values   # append both L + R features to same row j
        df_temp.loc[j, df_l_instance.index.values] = df_l_instance.values

        df_agg = pd.concat([df_agg, df_temp])
        j += 1

    print("LR split:", df.shape, "--> LR aggregated:", df_agg.shape)
    return df_agg


def LRsplit_to_average(df, include_dose=True, verbose=True):
    df_l = df[[x.split("_")[-1] == "L" for x in df["name"]]]
    df_r = df[[x.split("_")[-1] == "R" for x in df["name"]]]
    # print(df.shape, df_l.shape, df_r.shape)
    df_avg = pd.DataFrame()
    j = 0
    print("----- LR split --> LR average -----")
    print("-" * (len(df_l) // 3))
    for idx, name, time in zip(df_l.index.values, df_l["name"].values, df_l["time"].values):
        name_orig = name[:-2]  # remove _L
        dose = df_l.loc[idx, "dose"] if include_dose else 0

        # print(j, name_orig, time, dose)
        print("-", end="") if not j%3 else 0

        df_r_instance = df_r[np.logical_and(df_r["name"] == name_orig + "_R", df_r["time"] == time)]
        df_l_instance = df_l.loc[idx]

        if include_dose:
            if not len(df_r_instance == 1) or df_r_instance["dose"].values[0] != dose:
                print(">>>>>ERR: ", df_r_instance.shape, df_r_instance["dose"].values, dose, name_orig, time)
                return 0
            else:
                df_r_instance = df_r_instance.iloc[0]
        else:
            if not len(df_r_instance == 1):
                print(">>>>>ERR: ", df_r_instance.shape)
                return 0
            else:
                df_r_instance = df_r_instance.iloc[0]

        df_temp = pd.DataFrame()
        if include_dose:
            df_temp.loc[j, ["name", "time", "dose"]] = [name_orig, time, dose]  # should be same values for L + R
            df_r_instance = df_r_instance.drop(["name", "time", "dose"], axis=0)
            df_l_instance = df_l_instance.drop(["name", "time", "dose"], axis=0)

        else:
            df_temp.loc[j, ["name", "time"]] = [name_orig, time]  # should be same values for L + R
            df_r_instance = df_r_instance.drop(["name", "time"], axis=0)
            df_l_instance = df_l_instance.drop(["name", "time"], axis=0)

        if "out_val" in df_l_instance.index.values:
            if not df_l_instance.loc["out_val"] == df_r_instance.loc["out_val"]:
                print(">>>>DIFFERENT OUTCOME VALUES L / R:", df_l_instance, df_r_instance)
                return 0
            else:
                df_temp.loc[j, "out_val"] = df_l_instance.pop("out_val")
                df_r_instance.pop("out_val")
        if not all(df_l_instance.index.values == df_r_instance.index.values):
            print("DIFFERENT FEATURE NAMES L / R:", df_l_instance.index.values, df_r_instance.index.values)
            return 0

        ft_names = df_l_instance.index.values
        feature_values_averaged = np.mean([df_l_instance.values, df_r_instance.values], axis=0)
        df_temp.loc[j, ft_names] = feature_values_averaged

        df_avg = pd.concat([df_avg, df_temp])
        j += 1

    print("\nLR split:", df.shape, "--> LR average:", df_avg.shape)
    return df_avg


def register_name_to_outcome(df, out, melt=True, make_70_exception=True):    # RETURN ORDERED LIST OF ENDPOINT (e.g. saliva) CORRESPONDING TO name, time IN df
    # OUT: y = out registerred to df
    # IF MELT = FALSE: out NEEDS TO HAVE COLUMNS name, time, value
    # ctr?
    y = pd.DataFrame({"idx":np.array([], dtype=int), "name":[], "time":np.array([], dtype=int), "val":[]})
    dose_bool = "dose" in df.columns.values

    if melt:
        out = out.melt(id_vars="name", var_name="time", value_name="value")
    elif "val" in out.columns.values:
        out = out.rename(columns={"val":"value"})
    else:
        pass
    # print(out)
    # print(df["time"])
    out = out.dropna()

    for j, name in enumerate(list(set(df["name"].values))):
        times, indexes = df[df["name"] == name]["time"].values, df[df["name"] == name]["time"].index.values   # TIMES FOR NAME IN INPUT (ft) DF
        name_id = name.split("_")[0]
        out_for_name = out[[n == name_id for n in out["name"].values]]
        for t, idx in zip(times, indexes):
            t = int(t[:-3])
            if t==70 and make_70_exception:
                t = 75  # Co-register day 75 saliv data to day 70 MRI
            val = out_for_name[out_for_name["time"] == t]["value"].values
            ctr = out_for_name[out_for_name["time"] == t]["ctr"].values
            if any(val):
            # if any(val) and val != "-":
                if len(val) > 1:
                    if not list(val).count(val[0]) == len(val):
                        print("DIFFERENT VALUES FOUND FOR SAME TIME, ID:", val)
                        return 0
                    else:
                        val = val[0]
                val = float(val)
                if len(ctr) > 1:
                    print(ctr)
                    print(out_for_name)
                ctr = bool(ctr)
                if dose_bool:
                    dose = df.loc[idx]["dose"]
                    y = y.append({"idx":int(idx), "name":name, "time":t, "dose":dose, "val":val, "ctr":ctr}, ignore_index=True)
                else:
                    y = y.append({"idx": int(idx), "name": name, "time": t, "val": val},
                                 ignore_index=True)
    y = y.sort_values("idx")
    # y.index = y["idx"]
    # print(y)
    print(f"\tREGISTERED {len(y)} OUTCOME VALS TO DF")
    return y.set_index(y["idx"]).drop(["idx"], axis=1)


def aggregate_T1_T2_on_features(df1, df2):
    # Join set of features from df1, df2 to new df with features (labelled T1 or T2)
    # Assuming df1, df2 to be co-registerred for each row
    # Shape features: calculates average between df1 / 2
    # print(df1.shape, df2.shape)
    time, dose = df1["time"].values, df1["dose"].values
    if not all(time == df2["time"].values):
        print(">>>ERR: different time values")
        return 0
    elif not all(dose == df2["dose"].values):
        print(">>>ERR: different dose values")
        print(dose, df2["dose"].values)
        return 0
    else:
        df1 = df1.drop(["time", "dose"], axis=1)
        df2 = df2.drop(["time", "dose"], axis=1)
    df1.columns = [c + "_T1" for c in df1.columns]
    df2.columns = [c + "_T2" for c in df2.columns]
    df = pd.DataFrame()
    df.loc[:, "time"] = time
    df.loc[:, "dose"] = dose
    df.loc[:, df1.columns.values] = df1.values
    # print(df.shape)
    df.loc[:, df2.columns.values] = df2.values
    # print(df.shape)

    # df = df.T.drop_duplicates().T   # drop duplicate columns (in values)
    df.index = df1.index
    print("Joined df1, df2 at columns:", df.shape)
    return df


def corr_matrix_with_pvals(df_main, text=True, num_digits=2, abs=True):
    shape_corr = df_main.corr().shape
    # print(df_main.shape, shape_corr)
    corrvals = []
    annot_matrix = []
    ticks = []
    for col1 in df_main.columns.values:
        ticks.append(col1)
        for col2 in df_main.columns.values:
            df_coreg = df_main[col1].dropna()
            overlap = np.intersect1d(df_main[col2].dropna().index.values, df_coreg.index.values)
            df_coreg = df_main.loc[overlap]
            # print(col1, col2, df_coreg.shape)
            vals1, vals2 = df_coreg[col1].values, df_coreg[col2].values
            # print(df_coreg)
            corr, pval = stats.pearsonr(vals1, vals2)
            # annot_matrix.append([corr, pval])
            if abs:
                corr = np.abs(corr)
            corrvals.append(corr)
            annot_matrix.append(f"{corr:.{num_digits}f}\np = {pval:.{num_digits-1}e}") if text else annot_matrix.append([corr, pval])

    annot_matrix = np.reshape(annot_matrix, shape_corr) if text else np.reshape(annot_matrix, (*shape_corr, 2))
    corrvals = np.reshape(corrvals, shape_corr)
    # print(corrvals)
    # print(annot_matrix)
    return (corrvals, annot_matrix, ticks) if text else (corrvals, ticks)


def get_feature_index_fsps(LRMODE="aggregated", MODE="NO P", WEIGHT="T2", return_values=False):
    # if return_values: returns dataframe with all radiomic feature columns and values
    # else:             returns dictionary containing ft_i:ft_number_i
    if MODE == "NO P":
        filename = f"LR_{LRMODE}_FSPS_extracted_{WEIGHT}.csv"   # no-p
    elif MODE == "DELTA P":
        filename = f"LR_{LRMODE}_FSPS_DELTA-P.csv"  # delta-p
    elif MODE == "DELTA":
        filename = f"LR_{LRMODE}_FSPS_DELTA-time.csv" if WEIGHT == "T2" else f"LR_{LRMODE}_FSPS_DELTA-time_T1.csv"  # delta
    else:
        print("TRY MODE: NO P, DELTA P, DELTA")
        return 0

    df = pd.read_csv(os.path.join(ftsDir, filename), index_col=0)
    if MODE in ["NO P", "DELTA P"]:
        df = df.drop(["name", "time", "dose"], axis=1)
    else:
        df = df.drop(["name", "dose", "delta saliv", "saliv late", "time saliv late"], axis=1)
    if return_values:
        return df
    else:
        num_fts = len(df.T)
        fts = list(df.columns)
        ft_num_dict = dict([(ft, idx) for ft, idx in zip(fts, list(range(1, num_fts + 1)))])
        del df
        return ft_num_dict
    return 0


def get_feature_index_global(drop_3d_wavelet=True, invert=False):
    path = os.path.join(ftsDir, "FSPS_LRaverage_T2_THRESH=0.05.csv")
    df = pd.read_csv(path, index_col=0)

    if drop_3d_wavelet:
        df = df.drop(df.filter(like="LH", axis=0).index)
        df = df.drop(df.filter(like="LL", axis=0).index)
        df = df.drop(df.filter(like="HL", axis=0).index)
        df = df.drop(df.filter(like="HH", axis=0).index)

    fts_shape = ["original_shape2D_Elongation", "original_shape2D_MajorAxisLength", "original_shape2D_MaximumDiameter",
                 "original_shape2D_MeshSurface", "original_shape2D_MinorAxisLength", "original_shape2D_Perimeter",
                 "original_shape2D_PerimeterSurfaceRatio", "original_shape2D_PixelSurface", "original_shape2D_Sphericity"]
    fts = [*fts_shape, *list(df.index)]

    index = dict([(ft, idx) for ft, idx in zip(fts, range(1, 1 + len(fts)))])
    if invert:
        index = dict((idx, ft) for ft, idx in index.items())
    return index


def get_fsps_best_norm(LRMODE="aggregated", WEIGHT="T2", THRESH=0.15, drop_wavelet_3d=True):
    filepath = os.path.join(ftsDir, f"FSPS_LR{LRMODE}_{WEIGHT}_THRESH={THRESH}.csv")
    df = pd.read_csv(filepath, index_col=0)["spearman"]

    if drop_wavelet_3d:
        for wl in ["LL", "LH", "HL", "HH"]:
            rows = df.filter(like=wl, axis=0).index.values
            df = df.drop(rows)

    df_drop = df[df == "drop"]
    df_survived = df[df != "drop"]
    df_nonorm = df[df == "T2:no norm:FBW"]
    df_nyul = df[df == "T2:nyul:FBW"]
    df_stscore = df[df == "T2:stscore:FBW"]
    print(f"Having {len(df_survived)} of {len(df)} fts after FSPS LR{LRMODE} {WEIGHT} thresh={THRESH}")
    # print(f"All fts={len(df)}, Dropped={len(df_drop)}, not dropped={len(df_survived)}")
    # print("no norm:", len(df_nonorm), "nyul:", len(df_nyul), "stscore:", len(df_stscore))

    df_survived = df_survived.to_frame(name="norm")
    df_survived.loc[:, "norm"] = [x.split(":")[1] for x in df_survived["norm"].values]
    df_survived.loc[:, "filter"] = [ft.split("_")[0] for ft in df_survived.index.values]
    df_survived.loc[:, "ft type"] = [ft.split("_")[1] for ft in df_survived.index.values]
    return df_survived


if __name__ == "__main__":
    pass
    ft_index = get_feature_index_global()
    print(ft_index.keys())
    ft_index_flipped = dict((val, key) for key, val in ft_index.items())
    # fts = ["lbp-2D_firstorder_RobustMeanAbsoluteDeviation", "square_glrlm_RunVariance"]
    fts = ["gradient_glszm_LargeAreaEmphasis", "original_firstorder_Energy",
           "wavelet-L_firstorder_RobustMeanAbsoluteDeviation",
           "exponential_glszm_GrayLevelNonUniformity"]
    for ft in fts:
        idx = ft_index[ft]
        print(ft, ft_index[ft])
        print(ft_index_flipped[idx])
        print()

    for idx in [340, 635]:
        print(idx, ft_index_flipped[idx])
    # print(ft_index["wavelet-H_gldm_LargeDependenceEmphasis"])
    import sys
    sys.exit()
    # for ft in ["original_shape2D_MajorAxisLength", "logarithm_glcm_JointAverage", "wavelet-H_gldm_LargeDependenceEmphasis"]
    # for ft in ["original_shape2D_MajorAxisLength", "logarithm_glcm_JointAverage", "wavelet-H_gldm_LargeDependenceEmphasis"]
    # print()

    # df_allfts = pd.DataFrame()
    # for ft, idx in ft_index.items():
    #     print(idx, ft)
    #     df_allfts.loc[idx, "ft"] = ft
    # print(df_allfts)
    # df_allfts.to_csv(os.path.join(ftsDir, "feature_index.csv"))

    df = pd.read_csv(os.path.join(ftsDir, "feature_index.csv")).rename(columns={"Unnamed: 0":"idx"})
    df.loc[:, "filter_type"] = ["_".join(ft.split("_")[:2]) for ft in df["ft"].values]

    # print(df)
    # print(df.groupby("filter_type").size())
    # print(df.groupby("filter_type").agg({"idx":min, "idx":max}))
    dff = pd.DataFrame(dtype="int64")
    df_idxmin = df.groupby("filter_type").agg({"idx":min})
    df_idxmax = df.groupby("filter_type").agg({"idx":max})
    for filter_type in df_idxmin.index:
        dff.loc[filter_type, "idx min"] = int(df_idxmin.loc[filter_type].values[0])
        dff.loc[filter_type, "idx max"] = int(df_idxmax.loc[filter_type].values[0])
    dff = dff.astype("int64").sort_values("idx min")
    print(dff)
    print(dff.shape)
#    df.reset_index().groupby('id').agg({'value': min, 'time (index)': min})
    # ft_dict = get_feature_index_fsps(MODE="NO P", WEIGHT="T2")
    # get_feature_index_fsps(MODE="NO P", WEIGHT="T1")
    # get_feature_index_fsps(MODE="DELTA")
    # ft_dict = get_feature_index_fsps(MODE="DELTA P", LRMODE="average")
    # ft_dict = get_feature_index_fsps(MODE="DELTA P", LRMODE="aggregated")
    # fts = ['wavelet-H_firstorder_Median_R', 'logarithm_glszm_GrayLevelNonUniformityNormalized_R', 'gradient_glszm_LargeAreaEmphasis_R', 'logarithm_firstorder_Kurtosis_L']
    # print(len(ft_dict))
    # for ft in fts:
    #     print(ft_dict[ft])