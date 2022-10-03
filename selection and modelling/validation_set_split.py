import os
import numpy as np
import six
from select_utils import *
from endpoints import load_saliva
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from data_loader import load_T1T2_coreg, load_fsps_data, load_predict_late_not_delta, load_delta
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles


def VALIDATION_SPLIT_ANALYSIS_SALIVA(MODE="DELTA", WEIGHT="both", STATE=42, verbose=False, plot=False, return_dataframes=False):
    # DONT CHANGE STATE - ENSURES THE SAME SPLIT ALL TIMES CODE IS RUN
    # MAIN TRAIN / VALIDATION SPLIT: DONE BEFORE ANY SELECTION (except FSPS) OR MODELLING
    # REDUCE ID SETS BASED ON MODE
    # SPLIT REMAINING ID'S WITH STRATIFICATION ON CTR BOOL
    WEIGHT = WEIGHT.upper()
    MODE = MODE.upper()
    if not WEIGHT in ["T1", "T2", "BOTH", "ALL"]:
        print("INVALID WEIGHT", WEIGHT)
        return 0
    if not MODE in ["DELTA", "DELTA P", "NO P"]:
        print("INVALID MODE", MODE, "TRY: DELTA, DELTA P, NO P")
        return 0
    print("\n------ MAKING TRAIN / VALIDATION SPLIT FOR WEIGHT=", WEIGHT, ", MODE=", MODE, "------")
    # FIRST: FIND WHAT EXISTS FOR EACH ID: CTR / IRR, P / NO P, TIME, T1 / T2

    if WEIGHT == "BOTH" or WEIGHT == "ALL":
        df_t1 = pd.read_csv(os.path.join(ftsDir, "LR_split_FSPS_extracted_T1.csv"))
        df_t2 = pd.read_csv(os.path.join(ftsDir, "LR_split_FSPS_extracted_T2.csv"))
        df = pd.concat([df_t1, df_t2], ignore_index=True)
        print("T1:", df_t1.shape, "T2:", df_t2.shape) if verbose else 0
    else:
        path = os.path.join(ftsDir, "LR_split_FSPS_extracted_")
        path += WEIGHT + ".csv"
        # print(path)
        df = pd.read_csv(path)

    print("DATA:", df.shape) if verbose else 0
    df.loc[:, "ctr"] = [d == 0.0 for d in df["dose"].values]
    fts = df.drop(["name", "time", "ctr"], axis=1).columns # drop all radiomic features, not interesting here
    df = df.drop(fts, axis=1)

    df_main = pd.DataFrame(columns=["times"], data=list([]))    # IDs AS COLUMNS: WHAT EXISTS OF T1, P DATA? TIMES? IS CONTROL?

    # REGISTER INDIVIDUALS TO TIMES, WHETHER T1 / P EXISTS, CONTROL / IRR
    for idx, vals in six.iteritems(df.T):
        name, time, ctr = vals
        # print(name, time, ctr)
        id = name.split("_")[0]
        pbool = "p" in name
        t1bool = "T1" in name
        # id_set.add(id)
        if not id in df_main.index.values:
            s = set([time])
            obj = [s, t1bool, pbool, ctr]
            df_main.loc[id, ["times", "T1", "P", "CTR"]] = obj # have to do this to understand that s is a set, not string
        else:
            df_main.loc[id, "times"].add(time)
        try:
            if df_main.loc[id, "T1"] == True:
                pass
            else:
                df_main.loc[id, "T1"] = t1bool
        except Exception as e:
            df_main.loc[id, "T1"] = t1bool
            # print(*e.args)
            pass
        try:
            if df_main.loc[id, "P"] == True:
                pass
            else:
                df_main.loc[id, "P"] = pbool
        except Exception as e:
            df_main.loc[id, "P"] = pbool
            pass
        if not "-" in time:
            df_main.loc[id, "CTR"] = ctr
    print("HAVE ", len(df_main), f"IMAGING INDIVIDUALS (weight={WEIGHT})")

    # LOAD / SORT SALIVA DATA TO INDIVIDUALS WITH TIMES
    df_saliv_orig = load_saliva(melt=True, verbose=verbose)
    df_saliv_orig.index = df_saliv_orig["name"]
    # REGISTER SALIVA DATA OVER TIME TO INDIVIDUALS
    df_saliv = pd.DataFrame(columns=["times"], data=list([]))
    times_saliv = set()
    for idx, vals in six.iteritems(df_saliv_orig.T):
        id, time, _, ctr = vals
        # print(idx, id, time, ctr)
        times_saliv.add(time)
        if not id in df_saliv.index.values:
            l = [set([time]), ctr]
            df_saliv.loc[id, ["times", "ctr"]] = l
        else:
            df_saliv.loc[id, "times"].add(time)
            df_saliv.loc[id, "ctr"] = ctr if (not df_saliv.loc[id, "ctr"] == True) and time not in ["-3day", "-7day"] else False

    print("HAVE", len(df_saliv), "SALIV DATA INDIVIDUALS") if verbose else 0
    print("BOTH IN IMAGES AND SALIV:", len(df_saliv.index.intersection(df_main.index))) if verbose else 0
    times_saliv = list(times_saliv);    times_saliv.sort()
    print("SALIV TIMES = ", times_saliv) if verbose else 0
    time_sets_mri = df_main["times"].values

    times_baseline = ["-3day", "-7day"]
    df_have_baseline = df_main[[any([t in times_baseline for t in ts]) for ts in time_sets_mri]]    # have baseline times
    print("HAVE BASELINE:", len(df_have_baseline)) if verbose else 0

    if MODE == "DELTA":
        # DELTA OPTION A: MAXIMIZE SAMPLE SIZE
        times_after_irr = ["5day", "8day", "12day", "35day"]
        times_predict_saliv = [26, 35, 56, 75]

        # DELTA OPTION B: MINIMIZE SALIVA (LATE) TIMEPOINT VARIATION
        # times_after_irr = ["5day", "8day", "12day"]
        # times_predict_saliv = [26, 35, 56, 75]

        df_delta = df_have_baseline[[any([t in times_after_irr for t in ts]) for ts in df_have_baseline["times"].values]]   # have baseline + after irr
        print("BASELINE + AFTER IRR:", len(df_delta)) if verbose else 0
        df_delta_saliv = df_saliv.loc[df_delta.index.intersection(df_saliv.index).values]       # select IDs in saliv from df_delta
        df_delta_saliv = df_delta_saliv[[any([t in times_predict_saliv for t in ts]) for ts in df_delta_saliv["times"].values]] # have time in times_predict_saliv
        df_delta = df_delta.loc[df_delta.index.intersection(df_delta_saliv.index).values]       # co-register remaining MRI and saliv IDs
        print("HAVE SALIVA LATE --> SAMPLE SIZE DELTA_A:", len(df_delta)) if verbose else 0

        # REDUCE TIMES TO ONLY RELEVANT DATAPOINTS (FOR PLOTTING)
        # MRI:      baseline, after irr
        # SALIV:    baseline, after irr, late
        for id in df_delta.index.values:
            # print(id, df_delta.loc[id, "times"])
            times = [int(t[:-3]) for t in list(df_delta.loc[id, "times"])]
            times.sort()
            times_red = times[:2]
            times_sal = df_delta_saliv.loc[id, "times"]
            # print(id, times_sal, times_red)
            times_sal_last = list(times_sal);   times_sal_last.sort();  times_sal_last = times_sal_last[-1]
            times_sal_red = set(times_red).intersection(times_sal)
            times_sal_red.add(times_sal_last)
            times_red = [str(t) + "day" for t in times_red]
            df_delta.loc[id, "times"] = times_red
            df_delta_saliv.loc[id, "times"] = times_sal_red

        df_main = df_delta
        df_saliv = df_delta_saliv


    if MODE == "DELTA P":
        df_main = df_main[df_main["P"] == True]
        # print(df_main.shape)
        # print(df_main)
        df_main = df_main.drop("C6")
        print("HAVE P-DATA:", len(df_main)) if verbose else 0
        # return 0

    if MODE == "NO P" or MODE == "DELTA P":
        # NEED TO HAVE SALIVA MEASUREMENTS AT SAME TIME AS MRI DATA
        overlap = df_saliv.index.intersection(df_main.index).values
        df_saliv = df_saliv.loc[overlap]
        # print(df_saliv)
        for id in df_main.index.values:
            times_mri = [int(t[:-3]) for t in list(df_main.loc[id, "times"])]
            times_sal = list(df_saliv.loc[id, "times"])
            times_sal_new = []
            times_mri_new = []
            for t in times_sal:
                if any([abs(t - tmri) <= 5 for tmri in times_mri]) and not t==3: # a bit of fuzzy logic: include day 75 saliv for day 70 mri
                    times_sal_new.append(t)
            for tmri in times_mri:
                if any([abs(t - tmri) <= 5 for t in times_sal_new]): # a bit of fuzzy logic: include day 75 saliv for day 70 mri
                    times_mri_new.append(tmri)
            times_mri_new = [str(t) + "day" for t in times_mri_new]
            # print(times_mri, times_sal_new, times_mri_new)
            df_main.loc[id, "times"] = set(times_mri_new)
            df_saliv.loc[id, "times"] = set(times_sal_new)
            if not any(times_mri_new):
                df_main = df_main.drop(id, axis=0)  # drop empty
                df_saliv = df_saliv.drop(id, axis=0)
        print("HAVING", len(df_main), "MICE FOR MODE", MODE, "WEIGHT", WEIGHT) if verbose else 0


    if plot:  # PLOT THE RELEVANT DATA FOR THE MODEL (MODE, WEIGHT)
    # if False:
        fig, ax = plt.subplots()
        j = 0
        yvals = list(range(len(df_main)))
        ylabels = []

        plt.plot(ALL_TIMES[:-1], [0]*len(ALL_TIMES[:-1]), linewidth=0)
        for id, y in zip(df_main.index.values, yvals):
            times = list(df_main.loc[id, "times"])
            times = [int(t[:-3]) for t in times]
            times.sort()
            # print(id, times)
            times_saliv = list(df_saliv.loc[id, "times"])
            # print(id, times_saliv, times)

            # c = "r" if df_main.loc[id, "CTR"] else "b"
            c = "r"
            cc = "b"
            plt.plot(times, [y]*len(times), "x-", color=c) # IMAGES
            plt.plot(times_saliv, [y]*len(times_saliv), "o:", color=cc, markersize=3)
            ylabels.append(id)
        plt.yticks(yvals, ylabels)
        # plt.xticks(ALL_TIMES, ALL_TIMES, fontsize=16) if MODE=="DELTA" else plt.xticks(ALL_TIMES[:-1], ALL_TIMES[:-1], fontsize=16)
        plt.xticks(ALL_TIMES[:-1], ALL_TIMES[:-1], fontsize=16)
        plt.xlabel("Day (relative to irradiation start at day 0)", fontsize=16)
        plt.ylabel("Mouse ID", fontsize=16)
        plt.grid()
        # plt.title(f"Times when the {WEIGHT}-w MR-images were aquired for N={len(df_main)} mice")
        # plt.title(f"{MODE} data\nTimes when the {WEIGHT}-w MR-images (red crosses), and saliva measurements (blue dots), were aquired for N={len(df_main)} mice")
        # plt.title(f"Times when the {WEIGHT}-w MR-images (red crosses), and saliva measurements (blue dots),\n"
        #           f"were aquired for the N={len(df_main)} mice avaliable for {MODE} analysis", fontsize=16)
        fig.tight_layout()
        plt.show()

    # TRAIN / VALIDATION SET SPLIT:
    if MODE == "DELTA":
        x_train, x_val, y_train, y_val = train_test_split(df_main, df_saliv, random_state=STATE, test_size=0.20, stratify=df_main["CTR"])
        ID_TRAIN = x_train.index.values
        ID_VAL = x_val.index.values
        print("SPLIT INTO", len(ID_TRAIN), " TRAINING ID's AND", len(ID_VAL), "VALIDATION ID's") if verbose else 0
        num_ctr_train = len(x_train[x_train["CTR"]])
        num_irr_train = len(x_train[x_train["CTR"] == False])
        num_ctr_val = len(x_val[x_val["CTR"]])
        num_irr_val = len(x_val[x_val["CTR"] == False])

        # print("\tCTR / IRR TRAIN:", num_ctr_train, "/", num_irr_train, f"Ratio ctrl = {num_ctr_train / len(x_train):.3f}") if verbose else 0
        # print("\tCTR / IRR VAL:  ", num_ctr_val, "/", num_irr_val, f"Ratio ctrl {num_ctr_val / len(x_val):.3f}") if verbose else 0
        #
        # print("TRAIN:", ID_TRAIN) if verbose else 0
        # print("VAL:", ID_VAL) if verbose else 0
    else:
        # MAKE OVERLAP IN TEST / VALIDATION ID'S WITH DELTA MODE
        print("--- MAKING OVERLAP WITH DELTA VALIDATION SPLIT ----") if verbose else 0
        ID_VAL = df_main.index.intersection(ID_DELTA_VALID).values
        ID_TRAIN = df_main.index.intersection(ID_DELTA_TRAIN).values
        # print(df_main.index.values)
        df_val = df_main.loc[ID_VAL]
        df_train = df_main.loc[ID_TRAIN]
        # print(len(df_val) + len(df_train))
        df_saliv_val = df_saliv.loc[ID_VAL]
        df_saliv_train = df_saliv.loc[ID_TRAIN]

        df_main = df_main.drop([*ID_TRAIN, *ID_VAL], axis=0)
        df_saliv = df_saliv.drop([*ID_TRAIN, *ID_VAL], axis=0)
        # print(df_main.shape, df_saliv.shape)

        if not (WEIGHT == "T2" and MODE == "NO P"):
            print("--- MAKING OVERLAP WITH NO P T2 VALIDATION SPLIT ---") if verbose else 0
            ID_VAL_tmp = list(np.intersect1d(ID_NOP_T2_VALID, df_main.index.values))
            print(ID_VAL_tmp)
            ID_TRAIN_tmp = list(np.intersect1d(ID_NOP_T2_TRAIN, df_main.index.values))
            df_saliv_val = pd.concat([df_saliv_val, df_saliv.loc[ID_VAL_tmp]])
            df_saliv_train = pd.concat([df_saliv_train, df_saliv.loc[ID_TRAIN_tmp]])
            df_val = pd.concat([df_val, df_main.loc[ID_VAL_tmp]])
            df_train = pd.concat([df_train, df_main.loc[ID_TRAIN_tmp]])

            df_main = df_main.drop([*ID_TRAIN_tmp, *ID_VAL_tmp])
            df_saliv = df_saliv.drop([*ID_TRAIN_tmp, *ID_VAL_tmp])
            # print(ID_VAL_tmp)
            # print(ID_TRAIN_tmp)

        if len(df_main) > 5:
            print("--- MAKING NEW SPLIT ---") if verbose else 0
            x_train, x_val, y_train, y_val = train_test_split(df_main, df_saliv, random_state=STATE, test_size=0.20,
                                                              stratify=df_main["CTR"])
        elif len(df_main) > 1:
            # print(df_main)
            x_train, x_val, y_train, y_val = train_test_split(df_main, df_saliv, random_state=STATE, test_size=0.20)
        elif len(df_main) == 1:
            x_train = df_main.iloc[0]
            y_train = df_saliv.iloc[0]
            x_val, y_val = [], []
        else:
            x_train, x_val, y_train, y_val = [], [], [], []
        x_train = pd.concat([df_train, x_train]) if any(x_train) else df_train
        x_val = pd.concat([df_val, x_val]) if any(x_val) else df_val
        y_train = pd.concat([df_saliv_train, y_train]) if any(y_train) else df_saliv_train
        y_val = pd.concat([df_saliv_val, y_val]) if any(y_val) else df_saliv_val
        print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
        ID_VAL = x_val.index.values
        ID_TRAIN = x_train.index.values
        # print(x_train)
    num_ctr_train = len(x_train[x_train["CTR"]])
    num_irr_train = len(x_train[x_train["CTR"] == False])
    num_ctr_val = len(x_val[x_val["CTR"]])
    num_irr_val = len(x_val[x_val["CTR"] == False])
    num_ctr_all = num_ctr_train + num_ctr_val
    num_irr_all = num_irr_train + num_irr_val
    print(f"---- LOADED VALIDATION SET FOR {num_ctr_all + num_irr_all} INDIVIDUALS -----")
    print("TOTAL ID's:", num_ctr_all + num_irr_all, "\tCTRL / IRR = ", num_ctr_all, " / ", num_irr_all, f"PRC CTRL = {(num_ctr_all / (num_irr_all + num_ctr_all))*100:.2f}%") if verbose else 0
    print("\tCTR / IRR TRAIN:", num_ctr_train, "/", num_irr_train, f"Perc ctrl = {(num_ctr_train / len(x_train))*100:.2f}%") if verbose else 0
    print("\tCTR / IRR VAL:  ", num_ctr_val, "/", num_irr_val, f"Perc ctrl = {(num_ctr_val / len(x_val))*100:.2f}%") if verbose else 0
    print("TRAIN:", len(ID_TRAIN)) if verbose else 0
    print(ID_TRAIN) if verbose else 0
    print("VAL:", len(ID_VAL)) if verbose else 0
    print(ID_VAL) if verbose else 0

    if plot:
        fig, ax = plt.subplots()
        IDVALS = [*ID_TRAIN, *ID_VAL]
        yvals = list(range(len(IDVALS)))
        ylabels = []
        df_main = pd.concat([x_train, x_val])
        df_saliv = pd.concat([y_train, y_val])
        plt.plot(ALL_TIMES, [0]*len(ALL_TIMES), linewidth=0)
        # for id, y in zip(df_main.index.values, yvals):
        for id, y in zip(IDVALS, yvals):
            times = list(df_main.loc[id, "times"])
            # print(id, times)
            try:
                times = [int(t[:-3]) for t in list(times)]
                times.sort()
            except Exception as e:
                t = str(times[0])
                # print(t[2:4])
                print(t)
                times = [int(t[2:4])]   # dont ask....
            times_saliv = list(df_saliv.loc[id, "times"])
            if type(times_saliv[0]) == set:
                times_saliv = list(times_saliv[0])
            c = "r" if id in ID_VAL else "b"
            plt.plot(times, [y] * len(times), "x-", color=c)  # IMAGES

            plt.plot(times_saliv, [y] * len(times_saliv), "o:", color=c, markersize=3)
            ylabels.append(id)
        plt.yticks(yvals, ylabels)
        plt.xticks(ALL_TIMES, ALL_TIMES, fontsize=16)
        plt.xlabel("Day (relative to irradiation start at day 0)", fontsize=16)
        plt.grid()
        # plt.title(f"Times when the {WEIGHT}-w MR-images were aquired for N={len(df_main)} mice")
        plt.title(
            f"Model: {MODE}. Times when the {WEIGHT}-w MR-images (crosses), and saliva measurements (dots) were aquired for N={len(df_main)} mice\n"
            f"Split into training (blue, N={len(ID_TRAIN)}) and validation (red, N={len(ID_VAL)})")
        plt.show()

    # SECOND: IDENTIFY WHAT DATA EXISTS FOR DELTA-P, DELTA, NO P ANALYSIS
    #       --> THEN STATIFY SPLIT BY
    #       existing saliva data?
    #       control / no control?
    if return_dataframes:
        return ID_TRAIN, ID_VAL, x_train, x_val, y_train, y_val
    else:
        return ID_TRAIN, ID_VAL


def plot_valid_train_sets_overlap():
    # Plot overlap NO P T1 with NO P T2
    # Plot overlap T2 NO P, DELTA, DELTA-P
    from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
    from matplotlib import pyplot as plt
    # s1 = set(ID_NOP_T2_TRAIN)
    s1 = set(ID_NOP_T2_TRAIN)
    # s2 = set(ID_NOP_T2_VALID)
    s2 = set(ID_DELTA_P_TRAIN)
    # s2 = set(ID_NOP_T1_VALID)
    # print(s1.intersection(s2))
    # print(s1.difference(s2))
    print(s2.difference(s1))
    label1 = "No P T2 Training"
    label2 = "Delta-P T2 Training"
    fig, ax = plt.subplots()
    venn2([s1, s2], set_labels=[label1, label2], ax=ax)
    fig.tight_layout()
    plt.show()
    return 0


def make_grand_test_train_split_late():
    df_nopt2_baseline, y = load_predict_late_not_delta(MODE="NO P", WEIGHT="T2", LRMODE="aggregated", training="all", baseline=True, xer=True)
    df_nopt2_afterirr, y = load_predict_late_not_delta(MODE="NO P", WEIGHT="T2", LRMODE="aggregated", training="all", baseline=False, xer=True)
    df_nopt1_baseline, y = load_predict_late_not_delta(MODE="NO P", WEIGHT="T1", LRMODE="aggregated", training="all", baseline=True, xer=True)
    df_nopt1_afterirr, y = load_predict_late_not_delta(MODE="NO P", WEIGHT="T1", LRMODE="aggregated", training="all", baseline=False, xer=True)

    DF_LIST = [df_nopt2_baseline, df_nopt2_afterirr, df_nopt1_baseline, df_nopt1_afterirr]
    DF_NAMES = ["NO P T2 baseline", "NO P T2 after irr", "NO P T1 baseline", "NO P T1 after irr"]
    for df, mode in zip(DF_LIST, DF_NAMES):
        n = len(df)
        n_id = len(df["name"].drop_duplicates())
        print(f"{mode}\tn={n}, nid={n_id}")
    pass


def summary_T12_coreg_splits(latemode=False):
    # latemode: False, baseline, after irr

    summary = pd.DataFrame()
    for split_n in [1, 2, 3]:
        for trainmode in ["all", True, False]:

            df1, df2, y = load_T1T2_coreg(MODE="NO P", predict_late=latemode, SPLIT_NUMBER=split_n, training=trainmode, keep_names=True)
            y["ctr"] = [is_control(nm) for nm in y["name"].values]
            print(y)
            y_ctr = y[y["ctr"] == True]
            y_irr = y[y["ctr"] == False]

            if trainmode != "all":
                trainmode = "train" if trainmode else "test"
            row = f"split {split_n} {trainmode}"
            summary.loc[row, "Tot"] = len(y)
            summary.loc[row, "Tot xer"] = len(y[y["xer"] == True])
            summary.loc[row, "Ctr"] = len(y_ctr)
            summary.loc[row, "Ctr xer"] = len(y_ctr[y_ctr["xer"] == True])
            summary.loc[row, "Irr"] = len(y_irr)
            summary.loc[row, "Irr xer"] = len(y_irr[y_irr["xer"] == True])
    print(summary)

    import seaborn as sns
    sns.heatmap(summary, cbar=False, annot=True)
    plt.title(f"late = {latemode}")
    plt.show()
    return 1


if __name__ == "__main__":
    # pass
    # make_grand_test_train_split_late()
    summary_T12_coreg_splits(latemode=False)
    # df1_b, df2_b, y_b = load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", training="ALL", xer=True, predict_late="baseline")
    # df1_a, df2_a, y_a = load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", training="ALL", xer=True, predict_late="after irr")

    # x_train, x_val, y_train, y_val = train_test_split(df1_a, y_b)

    # for MODE in ["DELTA P"]:
    # for MODE in ["DELTA", "NO P", "DELTA P"]:
    #     for WEIGHT in ["T1", "T2"]:
    #         try:
    #             ID_TRAIN, ID_VALID = VALIDATION_SPLIT_ANALYSIS_SALIVA(MODE=MODE, WEIGHT=WEIGHT, verbose=True, plot=False, return_dataframes=False)
    #         except Exception as e:
    #             ID_TRAIN, ID_VALID = [], []
    #         print(MODE, WEIGHT, "TRAIN:", "', '".join(ID_TRAIN))
    #         print(MODE, WEIGHT, "VALID:", "', '".join(ID_VALID))
    # VALIDATION_SPLIT_ANALYSIS_SALIVA(MODE="NO P", WEIGHT="T1", plot=False, verbose=True)
    # VALIDATION_SPLIT_ANALYSIS_SALIVA(MODE="NO P", WEIGHT="T2", plot=True, verbose=True)
    # VALIDATION_SPLIT_ANALYSIS_SALIVA(MODE="DELTA", WEIGHT="T1", plot=True, verbose=True)
    # plot_valid_train_sets_overlap()
