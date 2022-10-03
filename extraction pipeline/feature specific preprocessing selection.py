import yaml
import SimpleITK as sitk
import pandas as pd
import numpy as np
from radiomics import featureextractor
import os
import re
import six
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

from extract_utils import *
from select_utils import LRsplit_to_aggregated, LRsplit_to_average
from DICOM_reader import find_folders, dcm_files_indexed, dcm_folder_to_pixel_matrix
from nyul_histogram_matching import nyul_normalizer, nyul_initializer, nyul_scales_T2, nyul_scales_T1
from preprocessing import n4correction, mean_centering
from selection.endpoints import load_saliva

FSPS_Dir = os.path.join(os.getcwd(),"..", "..", r"Radiomic features\feature specific preprocessing selection and modelling")

def baseline_features_LR_split(MRI_weight="T2", norm="nyul", disc="fbw"):
    '''
    Method inspired by Delta-radiomic features for the prediction of patient outcomes in NSCLC Fave et al., 2022
    LOAD pre-treatment data (day -7 or day -3) MRI + SALIVA, CO-REGISTER INSTANCES
    '''

    import yaml
    # from yaml.loader import SafeLoader
    import SimpleITK as sitk
    from selection.endpoints import load_saliva
    # smoothname = "smooth" if smooth else "no smooth"
    SaveName = "FSPS_" + "_".join([MRI_weight, norm, disc])
    # print(SaveName)
    SavePath = os.path.join(FSPS_Dir, SaveName) + ".csv"
    print(SavePath)

    if not MRI_weight.upper() in ["T1", "T2"]:
        print("MRI WEIGHT", MRI_weight, "NOT VALID.....")
        return 0

    outcome_data = load_saliva(melt=True)
    # print(outcome_data)
    # print(np.unique(outcome_data["time"].values, return_counts=True))
    INCLUDE_TIMES = ["-3day", "-7day"]

    instances = []  # append all exp:time:name to be considered
    out_values = [] # append outcome values corresponding to instances above
    folder_names = [] # append folder names for dcm acquisition matrix extraction
    for experiment in find_folders(RawDir, condition="Pilot"):
        for time in find_folders(os.path.join(RawDir, experiment)):
            # print("\n", experiment, time)
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            for folder in find_folders(datadir, condition="sagittal"):
                name = get_name(experiment, folder, condition="sagittal")
                t1bool = "T1" in name
                pbool = "p" in name
                # print(name, t1bool)
                includebool = False
                if pbool:   # skip all after_p images
                    pass
                elif MRI_weight.upper() == "T2" and not(t1bool):
                    includebool = True
                    # instances.append(":".join([experiment, time, name]))    #pilot1:-7day:C2_sagittal
                elif MRI_weight.upper() == "T1" and t1bool:
                    includebool = True
                    # instances.append(":".join([experiment, time, name]))
                else:
                    pass
                if not(time in INCLUDE_TIMES):
                    includebool = False
                if includebool:
                    t = int(time[:-3])
                    temp = outcome_data[outcome_data["time"] == t]
                    # print(t, name)
                    temp = temp[[nm in name for nm in temp["name"].values]]
                    # print(len(temp))
                    instances.append(":".join([experiment, time, name]))
                    folder_names.append(folder)
                    if len(temp) != 1:  # BUG: selected both 11-1 AND 11-10 is name = 11-10_...
                        # print(temp["name"] == name[:5])
                        temp = temp[temp["name"] == name[:5]]
                        # print(len(temp))
                    if len(temp) == 1:
                        out_values.append(temp["val"].values[0])
    N_inst = len(instances)
    print("--- Having", len(instances), "instances with MRI weighting", MRI_weight, "at time", INCLUDE_TIMES,
          "\n\tregistered to", len(out_values), "outcome values ---")
    if len(out_values) != len(instances):
        print("MISMATCH BETWEEN instances AND out_values....")
        return 0

    # EXTRACT ALL FTS FOR ALL PREPROCESSING METHODS
    path_settings = os.path.join(ftsDir, "settings", "Params_nonorm_2D.yaml")
    with open(path_settings) as file:
        settings_orig = yaml.load(file, Loader=yaml.loader.SafeLoader)
    BW = FBW_dict_T2[norm] if MRI_weight.upper() == "T2" else FBW_dict_T1[norm]
    settings_orig["setting"]["binWidth"] = BW
    print(f"FBW discretization for {MRI_weight} images with norm = {norm}: BW={BW}")
    print(settings_orig)

    FORCE2D_FEATURECLASSES = ["glcm", "glrlm"]    # CALCULATION OF glcm AND glrlm TEXTURE FEATURES REQUIRE force2D=True -- pyradiomics bug maybe??
    # REST_FEATURECLASSES = ["shape2D", "firstorder", "glszm", "gldm", "ngtdm"]

    nrmny = nyul_initializer("nyul otsu decile")
    if MRI_weight.upper() == "T1":
        nrmny.SCALE_STD = nyul_scales_T1["otsu decile nop"]
    else:
        nrmny.SCALE_STD = nyul_scales["otsu decile nop"]

    # EXTRACT ALL FEATURES FOR ALL INCLUDED IMAGES
    df_tot = pd.DataFrame([])
    j = 1
    for instance, folder, out_val in zip(instances, folder_names, out_values):
        exp, time, name = instance.split(":")
        print("\n", j, "/", N_inst, exp, time, name, folder, out_val)
        j += 1
        # path_raw = os.path.join(RawDir, exp, time, name)
        path_raw = os.path.join(RawDir, exp, time, folder)
        path_roi = os.path.join(SegSalivDir, exp, time, name + ".npy")

        files = dcm_files_indexed(path_raw)
        MTR = dcm_folder_to_pixel_matrix(files, folder_path=path_raw, printbool=False)
        ROI = np.load(path_roi)

        for X in ["L", "R"]:
            idx = left_idx_dict[time + name] if X == "L" else right_idx_dict[time + name]
            img = MTR[idx]
            img = n4correction(img)[0]
            roi = ROI[idx]
            if norm == "nyul":
                img = nrmny.transform_image(img)
            elif norm == "stscore":
                img = mean_centering(img, mode="image", n=3)   # norm to stscore mu = 0 + 3*sd, sd=1
            elif norm in ["none", "raw", "", "no norm"]:
                pass
            else:
                print("NORMALIZATION", norm, "NOT IMPLEMENTED....")
                return 0
            # DISCRETIZING


            im = sitk.GetImageFromArray(img)
            msk = sitk.GetImageFromArray(roi)

            settings = settings_orig.copy()

            settings["setting"]["force2D"] = False
            extractor = featureextractor.RadiomicsFeatureExtractor(settings)
            results1 = extractor.execute(im, msk)
            df_temp1 = pd.DataFrame.from_dict(results1, orient="index")
            # print(df_temp1.shape)
            for fclass in FORCE2D_FEATURECLASSES:
                # print(df_temp1.filter(like=fclass, axis=0))
                filtered = df_temp1.filter(like=fclass, axis=0)
                df_temp1 = df_temp1.drop(index=filtered.index)
            # print(df_temp1.shape)

            settings["setting"]["force2D"] = True
            extractor = featureextractor.RadiomicsFeatureExtractor(settings)
            results2 = extractor.execute(im, msk)
            df_temp2 = pd.DataFrame.from_dict(results2, orient="index")
            overlap = df_temp1.index.intersection(df_temp2.index)
            # print(overlap.shape)
            # print(df_temp2.shape)
            df_temp2 = df_temp2.drop(index=overlap)
            # print(df_temp2.shape)
            df_temp = pd.concat([df_temp1, df_temp2], verify_integrity=True)
            # print(df_temp.shape)
            df_temp = df_temp.drop(index=df_temp.filter(like="diagnostics", axis=0).index)
            df_temp = df_temp.drop(index=df_temp.filter(like="shape", axis=0).index)
            # print(df_temp)
            df_temp = df_temp.T
            df_temp["name"] = name + "_" + X
            df_temp["time"] = time
            df_temp["out_val"] = out_val
            cols = df_temp.drop(["name", "time", "out_val"], axis=1).columns
            df_temp = df_temp[["name", "time", "out_val", *cols]]

            if df_tot.empty:
                df_tot = df_temp
            else:
                df_tot = df_tot.append(df_temp, ignore_index=True)
            # print(df_tot)
            df_tot.to_csv(SavePath)

    # if os.path.exists(SavePath):
    #     print("Path exists at ", SavePath)
    # df_tot.to_csv(SavePath)
    return 1


def compare_preprocessing_LRsplit(weight="T2", plot_features=False, P_THRESH=0.10):
    # P_THRESH = 0.10
    SAVEPATH = os.path.join(ftsDir, "FSPS_LRsplit_" + weight + f"_THRESH={P_THRESH}.csv")
    weight = weight.upper()
    if not weight in ["T1", "T2"]:
        print("Weight", weight, "NOT VALID - TRY AGAIN...")
        return 0
    files = os.listdir(FSPS_Dir)
    r = re.compile(".*" + weight)
    files = list(filter(r.match, files))
    # print(files)
    # DF_LIST = []
    DF_DICT = {}
    Y_LIST = []
    AREA_LIST = []
    for f in files:
        _, _, norm, disc = f.split("_")
        disc = disc[:-4]
        mode = ":".join([weight, norm, disc.upper()])
        print(mode, f)
        df = pd.read_csv(os.path.join(FSPS_Dir, f), index_col=0).drop(["name", "time"], axis=1)
        y = df.pop("out_val")
        # print(y)
        # print(df.name)
        # DF_LIST.append(df)
        DF_DICT[mode] = df
        Y_LIST.append(y)
        AREA_LIST.append(df.pop("out_area"))
    # FT_LIST = DF_LIST[0].columns.values
    MODES = list(DF_DICT)    # modes = keys
    FT_LIST = DF_DICT[MODES[0]].columns.values
    # if not all([all(FT_LIST == DF_LIST[i].columns.values) for i in range(len(DF_LIST))]):
    if not all([all(FT_LIST == DF_DICT[key].columns.values) for key in MODES]):
        print("UNEQUAL FEATURES IN FILES, CANNOT COMPARE...")
        return 0
    if not all([all(Y_LIST[0] == Y_LIST[i]) for i in range(len(Y_LIST))]):
        print("UNEQUAL OUTCOME VALUES, CANNOT COMPARE...")
        return 0
    else:
        y_vals = Y_LIST[0]
    if not all([all(AREA_LIST[0] == AREA_LIST[i]) for i in range(len(AREA_LIST))]):
        print("UNEQUAL AREA OUTCOME VALUES, CANNOT COMPARE...")
        return 0
    else:
        y_area = AREA_LIST[0]
    print("FOUND", np.shape(FT_LIST), "FEATURES WITH", np.shape(y_vals), "OUTCOME VALUES")

    df_sign_linreg = pd.DataFrame(columns=MODES, index=FT_LIST)
    df_sign_spearman = pd.DataFrame(columns=MODES, index=FT_LIST)
    df_best = pd.DataFrame(index=FT_LIST, columns=["linreg", "spearman"], data=[["none"]*2]*len(FT_LIST))
    # print(df_best)
    # FTS_DROPPED = []    # drop if not sign for any
    # MODE_BEST = {}      # append best fts for each mode
    # for mode in MODES:
    #     MODE_BEST[mode] = []
    # print(MODE_BEST)
    # print(df_sign)
    # plot_features = True
    print("-" * (len(FT_LIST) // 10))
    for i, ft in enumerate(FT_LIST):
        # print("\n", ft)
        if not(i%10):
            print("-", end="")
        # r2_vals = []
        p_list_linreg = []
        p_list_spearman = []
        # spearman = []
        # df_area_spearman = pd.Series(index=MODES, dtype=np.float64)
        df_area_spearman = pd.DataFrame(index=MODES, columns=["linreg", "spearman"], dtype=np.float64)
        if plot_features:   fig, ax = plt.subplots(figsize=(16, 9))
        for mode in MODES:
            df = DF_DICT[mode][ft]
            # print(mode, df.shape)
            # TODO: p-val < 0.05, 0.10? corr to volume????
            x = df.values
            X = sm.add_constant(x)
            ols = sm.OLS(y_vals, X)
            fit = ols.fit()
            r2_linreg, p_linreg_out = fit.rsquared, fit.f_pvalue
            coef_spear_out, p_spear_out = stats.spearmanr(x, y_vals)

            p_list_linreg.append(p_linreg_out < P_THRESH)
            p_list_spearman.append(p_spear_out < P_THRESH)

            # LIN REG
            if p_linreg_out < P_THRESH:
                coef_spear_area, _ = stats.spearmanr(x, y_area)
                coef_spear_area = abs(coef_spear_area)    # monotonically increasing OR decreasing!
                # spearman[mode] = coef_spear_area, p_spear
                # spearman.append([mode, coef_spear_area])
                df_area_spearman.loc[mode, "linreg"] = coef_spear_area
            else:
                df_area_spearman.loc[mode, "linreg"] = 1.0

            # SPEARMAN (to out)
            if p_spear_out < P_THRESH:
                coef_spear_area, _ = stats.spearmanr(x, y_area)
                coef_spear_area = abs(coef_spear_area)
                df_area_spearman.loc[mode, "spearman"] = coef_spear_area
            else:
                df_area_spearman.loc[mode, "spearman"] = 1.0

            # print(df_area_spearman.loc[mode, "linreg"])
            ax.plot(x, y, "x", label=mode +
                    f"\nLinreg:          r2={r2_linreg:.3f}, p={p_linreg_out:.3g}, area corr={df_area_spearman.loc[mode, '''linreg''']:.3g}" +
                    f"\nSpearman: corr={coef_spear_out:.3f}, p={p_spear_out:.3g}, area corr={df_area_spearman.loc[mode, '''spearman''']:.3g}") if plot_features else 0
        # print(df_sign.T[ft])\¨åæ\æ

        # LINEAR REGRESSION P-VALUE
        if not any(p_list_linreg):
            df_best.loc[ft, "linreg"] = "drop"
        else:
            df_best.loc[ft, "linreg"] = df_area_spearman["linreg"].idxmin()

        # SPEEARMAN P-VALUE
        if not any(p_list_spearman):
            df_best.loc[ft, "spearman"] = "drop"
        else:
            df_best.loc[ft, "spearman"] = df_area_spearman["spearman"].idxmin()
        # print(df_best)
        # print(df_best)
        # if plot_features:
        #     plot_features = True if ft == "original_glszm_LowGrayLevelZoneEmphasis" else False

        if plot_features:
            print(df_best.loc[ft, "linreg"])
            print(df_best.loc[ft, "spearman"])
            fig.suptitle(weight + " " + ft + "\nbest linreg:" + df_best.loc[ft, "linreg"] + "\nbest spearman:" + df_best.loc[ft, "spearman"])
            fig.legend()
            plt.show()
    print()
    # print(len(FTS_DROPPED), FTS_DROPPED)
    # print(len(df_best[df_best["mode"] == "drop"]))
    # print(df_best["linreg"])
    print("Dropped linreg:", len(df_best[df_best["linreg"] == "drop"]))
    print("Dropped sprman:", len(df_best[df_best["spearman"] == "drop"]))
    # for mode in MODES:
    #     print("\nMode:", mode)
    #     print("\tHaving", df_sign_linreg[df_sign_linreg[mode] != 0][mode].shape, "sign fts")
    #     print("\tBeing best preprocessing for", len(df_best[df_best["mode"] == mode]), "features")
    # print("FEATURES DROPPED:", len(df_best[df_best["mode"] == "drop"]))
    df_best.to_csv(SAVEPATH)
    print("---- DF saved ----")
    return 1


def compare_preprocessing(LRMODE="aggregated", weight="T2", plot_features=False, P_THRESH=0.10):
    print(f"----- Finding best preprocessing for all LR {LRMODE} features -----")
    SAVEPATH = os.path.join(ftsDir, f"FSPS_LR{LRMODE}_" + weight + f"_THRESH={P_THRESH}.csv")
    weight = weight.upper()
    if not weight in ["T1", "T2"]:
        print("Weight", weight, "NOT VALID - TRY AGAIN...")
        return 0
    if not LRMODE in ["aggregated", "average"]:
        print("LR mode", LRMODE, "invalid")
        return 0

    files = os.listdir(FSPS_Dir)
    r = re.compile(".*" + weight)
    files = list(filter(r.match, files))

    DF_DICT = {}
    Y_LIST = []
    AREA_LIST = []
    for f in files:
        _, _, norm, disc = f.split("_")
        disc = disc[:-4]
        mode = ":".join([weight, norm, disc.upper()])
        print(mode, f)
        # df = pd.read_csv(os.path.join(FSPS_Dir, f), index_col=0).drop(["name", "time"], axis=1)
        df = pd.read_csv(os.path.join(FSPS_Dir, f), index_col=0)

        if LRMODE == "aggregated":
            df = LRsplit_to_aggregated(df, include_dose=False)
            y = df["out_val_L"]
            if not all(y.values == df["out_val_R"].values):
                print(">>> ERR: different out vals L / R")
                return 0
            df.pop("out_val_R")
            out_area = np.mean([df.pop("out_area_R").values, df.pop("out_area_L").values], axis=0)


        elif LRMODE == "average":
            df = LRsplit_to_average(df, include_dose=False)
            y = df.pop("out_val")
            out_area = df.pop("out_area")

        # print(df.filter(like="out", axis=1))
        # print(out_area)
        DF_DICT[mode] = df.drop(["name", "time"], axis=1)
        Y_LIST.append(y)
        AREA_LIST.append(out_area)

    MODES = list(DF_DICT) # keys
    FT_LIST = DF_DICT[MODES[0]].columns.values

    if not all([all(FT_LIST == DF_DICT[key].columns.values) for key in MODES]):
        print("UNEQUAL FEATURES IN FILES, CANNOT COMPARE...")
        return 0
    if not all([all(Y_LIST[0] == Y_LIST[i]) for i in range(len(Y_LIST))]):
        print("UNEQUAL OUTCOME VALUES, CANNOT COMPARE...")
        return 0
    else:
        y_vals = Y_LIST[0]
    if not all([all(AREA_LIST[0] == AREA_LIST[i]) for i in range(len(AREA_LIST))]):
        print("UNEQUAL AREA OUTCOME VALUES, CANNOT COMPARE...")
        return 0
    else:
        y_area = AREA_LIST[0]
    print("FOUND", np.shape(FT_LIST), "FEATURES WITH", np.shape(y_vals), "OUTCOME VALUES")

    df_best = pd.DataFrame(index=FT_LIST, columns=["linreg", "spearman"], data=[["none"]*2]*len(FT_LIST))
    print("-" * (len(FT_LIST) // 10))
    for i, ft in enumerate(FT_LIST):
        print(i, ft)
        # if not (i%10):
        #     print("-", end="")
        p_list_linreg = []
        p_list_spearman = []
        df_area_spearman = pd.DataFrame(index=MODES, columns=["linreg", "spearman"], dtype=np.float64)
        if plot_features: fig, ax = plt.subplots(figsize=(16, 9))
        for mode in MODES:
            df = DF_DICT[mode][ft]
            # print(mode, df.shape)
            # TODO: p-val < 0.05, 0.10? corr to volume????
            x = df.values
            X = sm.add_constant(x)
            ols = sm.OLS(y_vals, X)
            fit = ols.fit()
            r2_linreg, p_linreg_out = fit.rsquared, fit.f_pvalue
            coef_spear_out, p_spear_out = stats.spearmanr(x, y_vals)

            p_list_linreg.append(p_linreg_out < P_THRESH)
            p_list_spearman.append(p_spear_out < P_THRESH)

            # LIN REG
            if p_linreg_out < P_THRESH:
                coef_spear_area, _ = stats.spearmanr(x, y_area)
                coef_spear_area = abs(coef_spear_area)  # monotonically increasing OR decreasing!
                # spearman[mode] = coef_spear_area, p_spear
                # spearman.append([mode, coef_spear_area])
                df_area_spearman.loc[mode, "linreg"] = coef_spear_area
            else:
                df_area_spearman.loc[mode, "linreg"] = 1.0

            # SPEARMAN (to out)
            if p_spear_out < P_THRESH:
                coef_spear_area, _ = stats.spearmanr(x, y_area)
                coef_spear_area = abs(coef_spear_area)
                df_area_spearman.loc[mode, "spearman"] = coef_spear_area
            else:
                df_area_spearman.loc[mode, "spearman"] = 1.0

            # print(df_area_spearman.loc[mode, "linreg"])
            ax.plot(x, y, "x", label=mode +
                                     f"\nLinreg:          r2={r2_linreg:.3f}, p={p_linreg_out:.3g}, area corr={df_area_spearman.loc[mode, '''linreg''']:.3g}" +
                                     f"\nSpearman: corr={coef_spear_out:.3f}, p={p_spear_out:.3g}, area corr={df_area_spearman.loc[mode, '''spearman''']:.3g}") if plot_features else 0
            # print(df_sign.T[ft])\¨åæ\æ

            # LINEAR REGRESSION P-VALUE
        if not any(p_list_linreg):
            df_best.loc[ft, "linreg"] = "drop"
        else:
            df_best.loc[ft, "linreg"] = df_area_spearman["linreg"].idxmin()

            # SPEEARMAN P-VALUE
        if not any(p_list_spearman):
            df_best.loc[ft, "spearman"] = "drop"
        else:
            df_best.loc[ft, "spearman"] = df_area_spearman["spearman"].idxmin()

        if plot_features:
            print(df_best.loc[ft, "linreg"])
            print(df_best.loc[ft, "spearman"])
            fig.suptitle(weight + " " + ft + "\nbest linreg:" + df_best.loc[ft, "linreg"] + "\nbest spearman:" + df_best.loc[ft, "spearman"])
            fig.legend()
            plt.show()
    print()
    print("Dropped linreg:", len(df_best[df_best["linreg"] == "drop"]))
    print("Dropped sprman:", len(df_best[df_best["spearman"] == "drop"]))
    df_best.to_csv(SAVEPATH)
    print("---- DF saved ----")
    return 1


def find_baseline_instances(MRI_weight="T2"):
    if not MRI_weight.upper() in ["T1", "T2"]:
        print("MRI WEIGHT", MRI_weight, "NOT VALID.....")
        return 0

    outcome_data = load_saliva(melt=True)
    # print(outcome_data)
    # print(np.unique(outcome_data["time"].values, return_counts=True))
    INCLUDE_TIMES = ["-3day", "-7day"]

    instances = []  # append all exp:time:name to be considered
    out_values = [] # append outcome values corresponding to instances above
    folder_names = [] # append folder names for dcm acquisition matrix extraction
    for experiment in find_folders(RawDir, condition="Pilot"):
        for time in find_folders(os.path.join(RawDir, experiment)):
            # print("\n", experiment, time)
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            for folder in find_folders(datadir, condition="sagittal"):
                name = get_name(experiment, folder, condition="sagittal")
                t1bool = "T1" in name
                pbool = "p" in name
                # print(name, t1bool)
                includebool = False
                if pbool:   # skip all after_p images
                    pass
                elif MRI_weight.upper() == "T2" and not(t1bool):
                    includebool = True
                    # instances.append(":".join([experiment, time, name]))    #pilot1:-7day:C2_sagittal
                elif MRI_weight.upper() == "T1" and t1bool:
                    includebool = True
                    # instances.append(":".join([experiment, time, name]))
                else:
                    pass
                if not(time in INCLUDE_TIMES):
                    includebool = False
                if includebool:
                    t = int(time[:-3])
                    temp = outcome_data[outcome_data["time"] == t]
                    # print(t, name)
                    temp = temp[[nm in name for nm in temp["name"].values]]
                    # print(len(temp))
                    instances.append(":".join([experiment, time, name]))
                    folder_names.append(folder)
                    if len(temp) != 1:  # BUG: selected both 11-1 AND 11-10 is name = 11-10_...
                        # print(temp["name"] == name[:5])
                        temp = temp[temp["name"] == name[:5]]
                        # print(len(temp))
                    if len(temp) == 1:
                        out_values.append(temp["val"].values[0])
    N_inst = len(instances)
    print("--- Having", N_inst, "instances with MRI weighting", MRI_weight, "at time", INCLUDE_TIMES,
          "\n\tregistered to", len(out_values), "outcome values ---")
    if len(out_values) != len(instances):
        print("MISMATCH BETWEEN instances AND out_values....")
        return 0
    return instances, folder_names


def plot_norm_distribution(weight="T2", thresh=0.05, LRmode="split", drop_3d_wavelet=True):
    import seaborn as sns
    df = get_best_feature_normalization(weight=weight, THRESH=thresh, LRmode=LRmode)
    print(df.shape)
    if drop_3d_wavelet:
        df = df.drop(df.filter(like="wavelet-HH", axis=0).index, axis=0)
        df = df.drop(df.filter(like="wavelet-HL", axis=0).index, axis=0)
        df = df.drop(df.filter(like="wavelet-LH", axis=0).index, axis=0)
        df = df.drop(df.filter(like="wavelet-LL", axis=0).index, axis=0)
    print(df.shape)
    Nfts = len(df)
    Nfts_remaining = len(df[df["norm"] != "drop"])
    print(df.value_counts())
    df = df[df["norm"] != "drop"]
    counts = df.value_counts()
    print(counts.index.values)
    norms = [str(x)[2:-3] for x in counts.index.values]
    print(norms)
    print(counts.array)
    countvals = counts.to_numpy()

    fig, ax = plt.subplots(figsize=(4.5, 5))
    ax.bar(norms, countvals)
    ax.set_yticks(countvals, countvals)
    ax.grid(axis="y", which="major")
    ax.set_title(f"{weight} LR{LRmode} thresh={thresh}\n{Nfts_remaining} of {Nfts} features remaining after FSPS.")
    plt.show()
    return 1


if __name__ == "__main__":
    # baseline_features_LR_split(MRI_weight="T2", norm="nyul", disc="fbw")  # todo: add roi area to df's as out_area
    # baseline_features_LR_split(MRI_weight="T1", norm="nyul", disc="fbw")
    # baseline_features_LR_split(MRI_weight="T2", norm="no norm", disc="fbw")
    # baseline_features_LR_split(MRI_weight="T1", norm="no norm", disc="fbw")
    # baseline_features_LR_split(MRI_weight="T2", norm="stscore", disc="fbw")
    # baseline_features_LR_split(MRI_weight="T1", norm="stscore", disc="fbw")

    # compare_preprocessing_LRsplit(weight="T2", plot_features=False, P_THRESH=0.05)
    # compare_preprocessing_LRsplit(weight="T1", plot_features=False, P_THRESH=0.20)
    # find_baseline_instances("T2")
    # find_baseline_instances("T1")

    # compare_preprocessing(LRMODE="average", weight="T1", plot_features=False, P_THRESH=0.15)
    # compare_preprocessing(LRMODE="average", weight="T2", plot_features=False, P_THRESH=0.05)
    # compare_preprocessing_LRaggregated(weight="T2", plot_features=False, P_THRESH=0.05)

    # df = get_best_feature_normalization(weight="T2", THRESH=0.05, LRmode="aggregated")
    # df = get_best_feature_normalization(weight="T2", THRESH=0.05, LRmode="average")


    plot_norm_distribution("T2", thresh=0.05, LRmode="average", drop_3d_wavelet=True)
    plot_norm_distribution("T1", thresh=0.15, LRmode="average", drop_3d_wavelet=True)

    # plot_norm_distribution("T2", thresh=0.05, LRmode="split", drop_3d_wavelet=True)
    # plot_norm_distribution("T1", thresh=0.15, LRmode="aggregated", drop_3d_wavelet=True)
    # plot_norm_distribution("T1", thresh=0.15, LRmode="split", drop_3d_wavelet=True)
    # df = get_best_feature_normalization(weight="T1", THRESH=0.15, LRmode="aggregated")
    # print(df)
    # print(df.shape)
    # df = df.drop(df.filter(like="wavelet-HH", axis=0).index, axis=0)
    # df = df.drop(df.filter(like="wavelet-HL", axis=0).index, axis=0)
    # df = df.drop(df.filter(like="wavelet-LH", axis=0).index, axis=0)
    # df = df.drop(df.filter(like="wavelet-LL", axis=0).index, axis=0)
    # print(df.shape)
    # print(np.unique(df["norm"].values, return_counts=True))
    # df = df[df["norm"] != "drop"]
    # print(df.shape)



    # d = get_best_feature_normalization(weight="T1")
    # print(d)

    # for THRESH in [0.1, 0.15, 0.2]:
    #     df_fsps_T1 = pd.read_csv(os.path.join(ftsDir, "FSPS_LRsplit_" + "T1_" + f"THRESH={THRESH}" + ".csv"), index_col=0)
    #     print("THRESH=", THRESH, df_fsps_T1.shape)
    #     # print("T1 LINREG:", np.unique(df_fsps_T1["linreg"].values, return_counts=True))
    #     print("T1 SPEARMAN:", np.unique(df_fsps_T1["spearman"].values, return_counts=True))
    #     print()
    # df_fsps_T2 = pd.read_csv(os.path.join(ftsDir, "FSPS_LRsplit_" + "T2_" + f"THRESH=0.1" + ".csv"), index_col=0)
    # # print(df_fsps_T2.shape)
    # # print("T2 LINREG:", np.unique(df_fsps_T2["linreg"].values, return_counts=True))
    # print("T2 SPEARMAN:", np.unique(df_fsps_T2["spearman"].values, return_counts=True))


    # # WEIGHT = "T2"
    # WEIGHT = "T1"
    # MODES = ["no norm", "nyul", "stscore"]
    # DF_DICT = {}
    # DF_PATHS = {}
    # for mode in MODES:
    #     DF_PATHS[mode] = os.path.join(FSPS_Dir, "_".join(["FSPS", WEIGHT, mode, "fbw.csv"]))
    #     df = pd.read_csv(DF_PATHS[mode], index_col=0)
    #     print(df.shape)
    #     # print(df["name"].values)
    #     df["out_area"] = None
    #     cols = df.drop(["name", "time", "out_val", "out_area"], axis=1).columns
    #     df = df[["name", "time", "out_val", "out_area", *cols]]
    #     print(df.shape)
    #     DF_DICT[mode] = df
    #
    # instances, folders = find_baseline_instances(MRI_weight=WEIGHT)
    # # find_baseline_instances(MRI_weight="T1")
    # equals = True
    #
    # for inst, folder in zip(instances, folders):
    #     # break
    #     exp, time, name = inst.split(":")
    #     # print(exp, time, name, end="\t")
    #     # break
    #     path_raw = os.path.join(RawDir, exp, time, folder)
    #     path_roi = os.path.join(SegSalivDir, exp, time, name + ".npy")
    #     files = dcm_files_indexed(path_raw, printbool=0)
    #     M = dcm_folder_to_pixel_matrix(files, path_raw, printbool=0)
    #     ROI = np.load(path_roi)
    #     for X in ["L", "R"]:
    #         idx = left_idx_dict[time+name] if X=="L" else right_idx_dict[time+name]
    #         img = M[idx]
    #         roi = ROI[idx]
    #         # plt.imshow(img, cmap="gray")
    #         # plt.imshow(img * roi, alpha=0.5)
    #         # plt.show()
    #         extractor = featureextractor.RadiomicsFeatureExtractor()
    #         extractor.disableAllFeatures()
    #         # extractor.enableFeaturesByName()
    #         # extractor.enableAllFeatures()
    #         extractor.enableFeatureClassByName("shape2D")
    #         result = extractor.execute(sitk.GetImageFromArray(img), sitk.GetImageFromArray(roi))
    #         # dfres = pd.DataFrame.from_dict(result, orient="index")
    #         dfres = pd.DataFrame.from_dict(result, orient="index").T
    #         # print(dfres.filter(like=))
    #         area = dfres["original_shape2D_PixelSurface"].values[0]
    #         area_roi = len(roi[roi!= 0].ravel())
    #         print(exp, time, name, X,  end="\t")
    #         print(f"Area - Radiomics={area}, Numpy roi={area_roi}")
    #         equals = round(area, 0) == round(area_roi, 0)
    #         for mode in MODES:
    #             df = DF_DICT[mode]
    #             # path = DF_PATHS[mode]
    #             df.loc[df["name"] == name + "_" + X, "out_area"] = area
    #             DF_DICT[mode] = df
    #
    # for mode in MODES:
    #     df = DF_DICT[mode]
    #     path = DF_PATHS[mode]
    #     # print(df["out_area"].values)
    #     # print(df)
    #     df.to_csv(path)
    # print("ALL EQUAL??", equals)