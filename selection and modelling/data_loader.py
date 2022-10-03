from select_utils import *
from endpoints import binary_thresh_xerostomia, load_saliva
import sys
from model_utils import *

t12_coreg_split_val_dict = {
    False: {
        1: ['11-8', '8-3', '11-1', '9-5', '11-10', '13-7', '14-2', '13-8', '12-1'],
        2: ['11-5', '9-4', '14-5', '11-6', '14-4', '11-2', '13-10', '11-3'],
        3: ['9-3', '8-4', '8-5', '8-7', '9-2', '9-3', '11-4', '11-7', '11-9', '13-6', '13-9', '14-1', '14-3', '13-6',
            '13-9', '14-1', '14-3', '11-4', '11-9', '13-6', '13-9', '11-4', '11-9']
    },
    "baseline": {
        1: ['11-2', '11-8', '14-5', '11-5', '11-10', '14-1', '11-9', '11-7'],
        2: ['13-9', '13-6', '13-10', '9-4', '11-3', '12-1', '9-5', '13-8'],
        3: ['11-1', '14-2', '11-6', '11-4', '14-4', '9-3', '13-7', '14-3']
    },
    "after irr": {
        1: ['8-7', '11-8', '9-4', '12-1', '8-3', '11-4', '11-3', '9-2'],
        2: ['14-4', '11-9', '13-10', '11-6', '14-2', '9-3', '11-2', '8-4'],
        3: ['11-10', '11-1', '14-5', '11-5', '9-5', '13-9', '14-3', '13-6']
    }
}


def load_titanic_data():
    dataDir = os.path.join(os.getcwd(), "..", "..", r"Radiomic Modelling\other data model verification\titanic")
    df_train = pd.read_csv(os.path.join(dataDir, "train.csv"))
    df_test = pd.read_csv(os.path.join(dataDir, "test.csv"))
    rm = []
    for c in df_train.columns:
        num_nan = len(df_train[df_train[c].isna()])
        # print(c, "num nan = ", len(df_train[df_train[c].isna()])) # count number of NaN values in column
        if num_nan != 0 or (not df_train[c].dtypes in ["int64", "float64"]):
            rm.append(c)
    print(rm)
    df_train = df_train.drop(rm, axis=1)
    df_test = df_test.drop(rm, axis=1)
    X = df_train.drop(["Survived"], axis=1)
    Y = df_train["Survived"]
    return X, Y


def load_fsps_data(WEIGHT="T2", MODE="NO P", LRMODE="aggregated",  TRAIN_SET=True, verbose=True, drop_wavelet_3d=True):
    # If TRAIN_SET=False: Load Validation data
    # MODE == "ALL" loads all training / validation images based on TRAIN_SET
    # TRAIN_SET == "ALL" loads all images (both training / validation) for MODE
    # Thus: MODE == TRAIN_SET == "ALL" loads ALL images for WEIGHT
    # LRMODE: split, average, agg - how to use the L + R segmented SMG data ?
    if not MODE in ["NO P", "DELTA", "DELTA P", "ALL"]:
        print("INVALID MODE", MODE, "TRY: NO P, DELTA, DELTA-P")
        return 0
    if not LRMODE in ["split", "average", "agg", "aggregated"]:
        print("INVALID LR MODE", MODE, "TRY: split, average")
        return 0
    if not WEIGHT in ["T1", "T2"]:
        print("INVALID WEIGHT", WEIGHT)
        return 0
    # if MODE in ["DELTA", "DELTA P"] and WEIGHT=="T1":
    if MODE in ["DELTA P"] and WEIGHT=="T1":
        print("CANNOT USE MODEL", MODE, "FOR WEGITH", WEIGHT, "- not enough samples...")
        return 0

    idx_include = []
    id_set_included = set()
    LRMODE = "aggregated" if LRMODE == "agg" else LRMODE

    if MODE == "NO P" or MODE == "ALL":
        # filename = f"LR_{LRMODE}_FSPS{'''_extracted''' if LRMODE=='''split''' else ''''''}_{WEIGHT}.csv"
        # filename = f"LR_{LRMODE}_FSPS{'''''' if LRMODE =='''average''' else '''_extracted'''}_{WEIGHT}.csv"
        filename = f"LR_{LRMODE}_FSPS_extracted_{WEIGHT}.csv"
        df = pd.read_csv(os.path.join(ftsDir, filename), index_col=0)
        try:
            df = df.drop(["name.1"], axis=1)
        except Exception as e:
            pass
        shape_loaded = df.shape

        if str(TRAIN_SET).upper() == "ALL":
            # idx_include = df.index.values
            id_set_included = set([x.split("_")[0] for x in df["name"].values])
            ID_SET = list(id_set_included)
        elif TRAIN_SET == True or TRAIN_SET == False:
            # print("LOADED", df.shape) if verbose else 0
            if TRAIN_SET:
                ID_SET = ID_NOP_T2_TRAIN if WEIGHT == "T2" else ID_NOP_T1_TRAIN
            else:
                ID_SET = ID_NOP_T2_VALID if WEIGHT == "T2" else ID_NOP_T1_VALID
        else:
            print(">>>TRAIN SET", TRAIN_SET, "INVALID")
            return 0

        if MODE == "ALL":
            idx_include = df.index.values
        else:
            for idx, name in zip(df.index.values, df["name"].values):
                id = name.split("_")[0]
                # time = df.loc[idx, "time"]
                pbool = "p" in name
                t1bool = "T1" in name

                if (t1bool and WEIGHT == "T1") or (not(t1bool) and WEIGHT == "T2"):
                    include = True
                else:
                    include = False

                if not(pbool) and include:
                    include = id in ID_SET
                    # if TRAIN_SET:
                    #     include = id in ID_NOP_T2_TRAIN if WEIGHT == "T2" else id in ID_NOP_T1_TRAIN
                    # else:
                    #     include = id in ID_NOP_T2_VALID if WEIGHT == "T2" else id in ID_NOP_T1_VALID
                else:
                    include = False     # Omit P in NO P

                if include:
                    idx_include.append(idx)
                    id_set_included.add(id)

    elif MODE == "DELTA P": # ONLY T2 DATA
        filename = f"LR_{LRMODE}_FSPS_DELTA-P.csv"
        df = pd.read_csv(os.path.join(ftsDir, filename), index_col=0)
        shape_loaded = df.shape
        if str(TRAIN_SET).upper() == "ALL":
            id_set_included = set([x.split("_")[0] for x in df["name"].values])
            ID_SET = list(id_set_included)
        elif TRAIN_SET in [True, False]:
            ID_SET = ID_DELTA_P_TRAIN if TRAIN_SET else ID_DELTA_P_VALID
        else:
            print(">>>>>>TRAIN_SET", TRAIN_SET, "not valdi")
            return 0

        for idx, name in zip(df.index.values, df["name"].values):
            id = name.split("_")[0]
            # pbool = "p" in name
            # t1bool = "T1" in name
            # if (TRAIN_SET and id in ID_DELTA_P_TRAIN) or (not(TRAIN_SET) and id in ID_DELTA_P_VALID):
            if id in ID_SET:
                idx_include.append(idx)
                id_set_included.add(id)

    elif MODE == "DELTA":
        filename = f"LR_{LRMODE}_FSPS_DELTA-time.csv" if WEIGHT == "T2" else f"LR_{LRMODE}_FSPS_DELTA-time_T1.csv"
        df = pd.read_csv(os.path.join(ftsDir, filename), index_col=0)
        # print(df)
        shape_loaded = df.shape

        if str(TRAIN_SET).upper() == "ALL":
            id_set_included = set([x.split("_")[0] for x in df["name"].values])
            ID_SET = list(id_set_included)
        elif TRAIN_SET in [True, False]:
            if WEIGHT == "T2":
                ID_SET = ID_DELTA_TRAIN if TRAIN_SET else ID_DELTA_VALID
            else:
                ID_SET = ID_DELTA_T1_TRAIN if TRAIN_SET else ID_DELTA_T1_VALID
        else:
            print(">>>> TRAIN SET", TRAIN_SET, "invalid")
            return 0

        for idx, name in zip(df.index.values, df["name"].values):
            id = name.split("_")[0]
            if id in ID_SET:
                idx_include.append(idx)
                id_set_included.add(id)

    else:
        print("MODE", MODE, "NOT IMPLEMENTED YET!")
        return 0

    length_expected = len(ID_SET)
    if len(id_set_included) != length_expected:
        print("\tEXPECTED", length_expected, "MOUSE ID's, GOT", len(id_set_included))
        id_diff = set(ID_SET).difference(id_set_included)
        print("\tEXPECTED IDs NOT INCLUDED = ", id_diff)

    df_red = df.loc[idx_include]

    # cols = df_red.filter(like="LL", axis=1).columns.values    # maybe drop these: LH / LL wavelets assume 3D images??
    # df_red = df_red.drop(cols, axis=1)
    # cols = df_red.filter(like="LH", axis=1).columns.values
    # df_red = df_red.drop(cols, axis=1)
    # cols = df_red.filter(like="HL", axis=1).columns.values
    # df_red = df_red.drop(cols, axis=1)
    # cols = df_red.filter(like="HH", axis=1).columns.values
    # df_red = df_red.drop(cols, axis=1)
    # print(cols)
    # print(df_red)
    # id_vals = [nm.split("_")[0] for nm in df_red["name"]]
    # print(id_vals)

    num_control_ids = np.count_nonzero([is_control(nm, include_baseline=False) for nm in id_set_included])

    print(f"\n----- Loaded LR {LRMODE} FSPS data successively for {MODE} {WEIGHT} ------"
          "\nFound", len(id_set_included), f"{'''TRAINING''' if TRAIN_SET == True else '''VALIDATION''' if TRAIN_SET==False else ''''''} individuals, "
                                           f"{num_control_ids} control ({num_control_ids / len(id_set_included) * 100 :.0f}%)"
            f" with", len(idx_include), "images for mode", MODE, "weight", WEIGHT, f"(expected {len(ID_SET)})") if verbose else 0

    print("Loaded =", shape_loaded, "Reduced =", df_red.shape) if verbose else 0
    # print() if verbose else 0
    return df_red


def load_nop(WEIGHT="T2", LRMODE="aggregated", training=True, xer=False, impute=True, keep_names=False):
    if WEIGHT == "T1":
        df = load_fsps_data("T1", "NO P", LRMODE=LRMODE, TRAIN_SET=training)
    elif WEIGHT == "T2":
        df = load_fsps_data("T2", "NO P", LRMODE=LRMODE, TRAIN_SET=training)
    else:
        print("Weight", WEIGHT, "not valid. Try: T1, T2")
        return 0, 0
    # df_y = load_saliva(melt=False)
    # df_y = register_name_to_outcome(df, df_y, melt=True, make_70_exception=True)
    df_y = load_saliva(melt=True)
    df_y = register_name_to_outcome(df, df_y, melt=False, make_70_exception=True)

    df_red = df.loc[df_y.index.values]

    if not keep_names:
        df_red = df_red.drop(["name"], axis=1)
    df_red["time"] = [int(x[:-3]) for x in df_red["time"].values]
    if xer:
        df_y = binary_thresh_xerostomia(dfy=df_y)
        # print(df_y.shape)
    else:
        df_y = df_y["val"]
    return df_red, df_y


def load_delta(WEIGHT="T2", LRMODE="split", training=True, xer=False, keep_time=False, keep_names=False):
    # df = load_fsps_data("T2", "DELTA", LRMODE=LRMODE, TRAIN_SET=training)
    df = load_fsps_data(WEIGHT, "DELTA", LRMODE=LRMODE, TRAIN_SET=training)

    if not xer:
        df_y = df["saliv late"]
        if not keep_time:
            df = df.drop("time saliv late", axis=1)
        else:
            df = df.rename(columns={"time saliv late":"time"})
    else:
        # print(df)
        df_y = df[["saliv late", "time saliv late"]]
        df_y = df_y.rename(columns={"time saliv late":"time", "saliv late":"val"})
        df_y = binary_thresh_xerostomia(df_y)
        df = df.drop("time saliv late", axis=1) if not keep_time else df.rename(columns={"time saliv late":"time"})
    if keep_names:
        df_red = df.drop(["saliv late", "delta saliv"], axis=1)
    else:
        df_red = df.drop(["name", "saliv late", "delta saliv"], axis=1)
    print("Loaded delta:", df_red.shape, df_y.shape)
    return df_red, df_y


def load_delta_P(WEIGHT="T2", LRMODE="split", training=True, xer=False, keep_names=False):
    if WEIGHT == "T1":
        print("T1 is not a valid weight for delta-P, too little data..")
        # df = load_fsps_data("T1", "DELTA P", TRAIN_SET=training)
        return 0, 0
    elif WEIGHT == "T2":
        df = load_fsps_data("T2", "DELTA P", LRMODE=LRMODE, TRAIN_SET=training)
    else:
        print("Weight", WEIGHT, "not valid. Try: T1, T2")
        return 0, 0
    df_y = load_saliva(melt=True)
    df_y = register_name_to_outcome(df, df_y, melt=False, make_70_exception=True)
    df_red = df.loc[df_y.index.values]
    if not keep_names:
        df_red = df_red.drop(["name"], axis=1)

    df_red["time"] = [int(x[:-3]) for x in df_red["time"].values]
    if not xer:
        df_y = df_y["val"]
    else:
        df_y = binary_thresh_xerostomia(df_y)
    return df_red, df_y


def load_predict_late(MODE="NO P", WEIGHT="T2", LRMODE="aggregated", training=True, xer=False, baseline=True, keep_names_times=False, keep_times=True):
    # MODE is NO P or DELTA P
    if not MODE in ["NO P", "DELTA P"]:
        print("Mode", MODE, "NOT VALID")
        return 0
    # Load NoP data from either baseline or after irr
    # Only load id's in train / test set for delta-features
    # Load saliva from late measurements (same as delta-features)
    df_delta = load_fsps_data(MODE="DELTA", TRAIN_SET=training, LRMODE=LRMODE)
    id_delta = [n.split("_")[0] for n in df_delta["name"].values]

    df = load_fsps_data(MODE=MODE, WEIGHT=WEIGHT, TRAIN_SET=training, LRMODE=LRMODE)
    df = pd.concat([df, load_fsps_data(MODE=MODE, LRMODE=LRMODE, WEIGHT=WEIGHT, TRAIN_SET=not(training))]) # load everything, follow delta train / valid IDs
    id_list = [n.split("_")[0] for n in df["name"].values]

    df_late = pd.DataFrame()
    for id in list(set(id_delta)):
        df_temp = df[[id == idd for idd in id_list]]
        df_delta_temp = df_delta[[id == idd for idd in id_delta]]

        id_times = list(set(df_temp["time"].values))
        id_times = [int(t[:-3]) for t in id_times]
        if id_times:
            id_times.sort()
            saliv_late = df_delta_temp["saliv late"].values[0]
            time_late = df_delta_temp["time saliv late"].values[0]
            if not all([s == saliv_late for s in df_delta_temp["saliv late"].values]):
                print("VARYING SALIVA LATE MEASUREMENTS: CANNOT LOAD LATE PREDICT FOR", MODE, WEIGHT)
                return 0
            # print(id_times)
            if len(id_times) == 1:
                time_extract = id_times[0]
                keep_bool = time_extract < 0 if baseline else time_extract > 0
            else:
                time_extract = id_times[0] if baseline else id_times[1]
                keep_bool = True
            if keep_bool:
                df_curr = df_temp[df_temp["time"] == str(time_extract) + "day"]
                df_curr["saliv late"] = [saliv_late] * len(df_curr)  # y-value to predict
                df_curr["time saliv late"] = time_late  # needed for xer thresholding
                df_late = pd.concat([df_late, df_curr])

    # print(df_late)
    # print(list(set(df_late["time"].values)))
    # print(list(set(df_late["name"].values)))

    if not xer:
        if not keep_names_times:
            df_y = df_late["saliv late"]
            df_late = df_late.drop("time saliv late", axis=1)
        else:
            df_y = df_late[["name", "saliv late", "time saliv late"]]
    else:
        df_y = df_late[["saliv late", "time saliv late"]]
        df_y = df_y.rename(columns={"time saliv late":"time", "saliv late":"val"})
        # print(df_y)
        df_y = binary_thresh_xerostomia(df_y)

    ids_in_dfred = [n.split("_")[0] for n in df_late["name"].values]
    num_ids = len(list(set(ids_in_dfred)))

    # print(df_late["name"])
    df_late["time"] = [int(t[:-3]) for t in df_late["time"]]
    if keep_names_times:
        df_red = df_late.drop(["saliv late"], axis=1)
    elif keep_times:
        df_red = df_late.drop(["saliv late", "name"], axis=1)
    else:
        df_red = df_late.drop(["name", "saliv late", "time"], axis=1)

    print("----- LOADED", MODE, WEIGHT, f"for late prediction using {'''BASELINE''' if baseline else '''AFTER IRR'''} data: shape=", df_red.shape, " N_ids=", num_ids)
    print()
    # print(df_y)
    return df_red, df_y


def load_predict_late_not_delta(MODE="NO P", WEIGHT="T2", LRMODE="aggregated", training=True, baseline=True, xer=False, keep_id=False, reset_y_index=False):
    TIMES_ALLOWED = BASELINE_TIMES if baseline else AFTER_IRR_TIMES

    if not MODE in ["NO P", "DELTA P"]:
        print("INVALDI MODE ", MODE)
        return 0

    df = load_fsps_data(WEIGHT=WEIGHT, MODE=MODE, LRMODE=LRMODE, TRAIN_SET=training)
    print(f"LOADED training={training}:", MODE, WEIGHT, "LR", LRMODE, f"TRAIN={training}", f"baseline={baseline}", df.shape)
    df["time_val"] = [int(x[:-3]) for x in df["time"].values]
    name_set = df["name"].drop_duplicates().values
    # print(df[["name", "time_val", "dose"]])

    idx_list = []
    for name in name_set:
        df_name = df[df["name"] == name]
        df_name = df_name.sort_values(by="time_val")
        df_name = df_name[[x in TIMES_ALLOWED for x in df_name["time_val"].values]]
        if not df_name.empty:
            # print(set(df_name["name"].values), df_name["time"].values, df_name["time"].values[0])
            # print(df_name.iloc[0].name)
            idx_list.append(df_name.iloc[0].name)

    df = df.loc[idx_list]
    print(f"Have {'''baseline''' if baseline else '''after irr'''}:", df.shape)
    df["id"] = [x.split("_")[0] for x in df["name"].values]

    sal = load_saliva(melt=True)
    id_set_saliv = sal["name"].drop_duplicates().values
    # print(name_set)

    sal_coreg = pd.DataFrame()
    for idx, id, time_img in zip(df.index.values, df["id"].values, df["time_val"].values):
        df_sal_name = sal[sal["name"] == id].sort_values(by="time")
        # print(idx, id, time_img)
        # print(df_sal_name)
        # print()
        if not(df_sal_name["time"].values[-1] <= time_img) and not(df_sal_name.empty):
            sal_coreg = pd.concat([sal_coreg, df_sal_name.iloc[[-1]]])
        else:
            pass

    df = df[[id in sal_coreg["name"].values for id in df["id"].values]]
    if reset_y_index:
        sal_coreg.index = df.index
    print("REG TO FUTURE SALIV:", df.shape, sal_coreg.shape)

    if not xer:
        y = sal_coreg
    else:
        # print(sal_coreg)
        y = binary_thresh_xerostomia(dfy=sal_coreg)
    if not keep_id:
        df = df.drop(["id", "name", "time"], axis=1)
    else:
        df = df.drop(["time"], axis=1)
    df = df.rename(columns={"time_val":"time"})
    # print(df)
    return df, y


def load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", predict_late=False, training=True, xer=True, SPLIT_NUMBER=1, keep_names=False):
    # CO-REGISTER name, time between loaded T1 and T2 df's
    # USES DIFFERENT SPLIT THAN T1 / T2 individual analysis: three train / validate set splits, with no overlap
        # splits vary for acute, baseline, after irr analysis

    if not SPLIT_NUMBER in [1, 2, 3]:
        print(">>>>SPLIT_NUMBER must be 1-3")
        return 0

    if not predict_late:
        # df_t1 = load_fsps_data("T1", MODE=MODE, LRMODE=LRMODE, TRAIN_SET=training)
        # df_t2 = load_fsps_data("T2", MODE=MODE, LRMODE=LRMODE, TRAIN_SET=training)
        df_t1 = load_fsps_data("T1", MODE=MODE, LRMODE=LRMODE, TRAIN_SET="ALL")
        df_t2 = load_fsps_data("T2", MODE=MODE, LRMODE=LRMODE, TRAIN_SET="ALL")

    elif predict_late in ["baseline", "after irr"]:
        baseline_bool = predict_late == "baseline"
        # df_t1, dfy_t1 = load_predict_late(MODE=MODE, WEIGHT="T1", LRMODE=LRMODE, training="ALL", xer=False, baseline=baseline_bool, keep_names_times=True)
        # df_t2, dfy_t2 = load_predict_late(MODE=MODE, WEIGHT="T2", LRMODE=LRMODE, training="ALL", xer=False, baseline=baseline_bool, keep_names_times=True)
        df_t1, dfy_t1 = load_predict_late_not_delta(MODE=MODE, WEIGHT="T1", LRMODE=LRMODE, training="ALL", baseline=baseline_bool, keep_id=True)
        df_t2, dfy_t2 = load_predict_late_not_delta(MODE=MODE, WEIGHT="T2", LRMODE=LRMODE, training="ALL", baseline=baseline_bool, keep_id=True)
        # print(df_t1)
    else:
        print(">>>>PREDICT LATE", predict_late, "invalid")
        return 0

    df_t1["id"] = [x.split("_")[0] for x in df_t1["name"].values]
    df_t2["id"] = [x.split("_")[0] for x in df_t2["name"].values]
    # df_t1["id"] = [x[:5] for x in df_t1["name"]]
    # df_t2["id"] = [x[:5] for x in df_t2["name"]]


    print("T1 / T2 before co-reg:", df_t1.shape, df_t2.shape)
    idtimes_t1 = [tuple(x) for x in df_t1[["id", "time"]].to_numpy()]
    idtimes_t2 = [tuple(x) for x in df_t2[["id", "time"]].to_numpy()]
    overlap = set(idtimes_t1).intersection(set(idtimes_t2))
    t1_not_overlap = set(idtimes_t1).difference(overlap)
    print("MISSING T1 rel T2:", t1_not_overlap) if any(t1_not_overlap) else 0
    id_set = set([item[0] for item in overlap])

    df_t1 = df_t1[[tuple(x) in overlap for x in df_t1[["id", "time"]].to_numpy()]]
    df_t2 = df_t2[[tuple(x) in overlap for x in df_t2[["id", "time"]].to_numpy()]]
    df_t1 = df_t1.drop_duplicates()
    df_t2 = df_t2.drop_duplicates()
    print("T1 / T2 after co-reg:", df_t1.shape, df_t2.shape, "with", len(id_set), "IDs")

    if not predict_late:
        dfy = load_saliva(melt=True)
        dfy_t1 = register_name_to_outcome(df_t1, dfy, melt=False, make_70_exception=True)
        dfy_t2 = register_name_to_outcome(df_t2, dfy, melt=False, make_70_exception=True)
        print("Loaded co-reg saliv:", dfy_t1.shape, dfy_t2.shape)

        if not np.all(dfy_t1[["time", "dose", "val"]].to_numpy() == dfy_t2[["time", "dose", "val"]].to_numpy()):
            # print(dfy_t1[[any(x) for x in dfy_t1.isna().values]])
            # print(dfy_t2[[any(x) for x in dfy_t2.isna().values]])
            print(">>>>>ERR: UNEQUAL SALIVA VALUES...")
            return 0
        else:
            dfy = dfy_t1
            dfy.loc[:, "name"] = [x.split("_")[0] for x in dfy["name"].values]

        df_t1_red = df_t1.loc[dfy_t1.index.values].drop(["name", "id"], axis=1)
        df_t2_red = df_t2.loc[dfy_t2.index.values].drop(["name", "id"], axis=1)
        print("Reduced:", df_t1_red.shape, df_t2_red.shape)
        df_t1_red["time"] = [int(x[:-3]) for x in df_t1_red["time"].values]
        df_t2_red["time"] = [int(x[:-3]) for x in df_t2_red["time"].values]

    else:
        # dfy_t1["id"] = [x[:5] for x in dfy_t1["name"]]
        # dfy_t2["id"] = [x[:5] for x in dfy_t2["name"]]

        dfy_t1 = dfy_t1.drop_duplicates()
        dfy_t2 = dfy_t2.drop_duplicates()
        # print(dfy_t1)
        # print(dfy_t2)

        sal_idx_overlap = dfy_t1.index.intersection(dfy_t2.index)
        dfy = dfy_t1.loc[sal_idx_overlap]
        # print(dfy)

        # index stuff
        df_t2 = df_t2.reset_index()
        df_t1 = df_t1.reset_index()
        t1_idx_new = []
        for idx, id, t in zip(df_t2.index.values, df_t2["id"].values, df_t2["time"].values):
            df_t1_curr = df_t1.loc[(df_t1["id"] == id) & (df_t1["time"] == t)]
            if len(df_t1_curr) > 1:
                print("ERR: multiple instances found...")
                return 0, 0, 0
            # print("\n", idx, id, t)
            # print(df_t1_curr)
            df_t1_idx = df_t1_curr.index.values[0]
            t1_idx_new.append(df_t1_idx)
        df_t1 = df_t1.loc[t1_idx_new]
        if not all(df_t1["id"].values == df_t2["id"].values):
            print(">>>ERR: different order of IDs in df's")
            return 0, 0, 0

        # print(df_t1["id"])
        # print(df_t2["id"])
        if "time_val" in df_t1.columns:
            try:
                df_t1_red = df_t1.drop(["name", "time", "id", "index"], axis=1).rename(columns={"time_val":"time"})
                df_t2_red = df_t2.drop(["name", "time", "id", "index"], axis=1).rename(columns={"time_val":"time"})
            except Exception as e:
                df_t1_red = df_t1.drop(["name", "time", "id"], axis=1).rename(columns={"time_val":"time"})
                df_t2_red = df_t2.drop(["name", "time", "id"], axis=1).rename(columns={"time_val":"time"})
        else:
            try:
                df_t1_red = df_t1.drop(["name", "id", "index"], axis=1)
                df_t2_red = df_t2.drop(["name", "id", "index"], axis=1)
            except Exception as e:
                df_t1_red = df_t1.drop(["name", "id"], axis=1)
                df_t2_red = df_t2.drop(["name", "id"], axis=1)

        # print(dfy)
        # print(df_t1_red)

    if xer:
        dfy_old = dfy.copy()
        # print(dfy)
        dfy = binary_thresh_xerostomia(dfy=dfy)
        # print(dfy)
        dfy_old = dfy_old.loc[dfy.index]
        # print(dfy_old)
    else:
        dfy_old = dfy.copy()
        dfy = dfy["val"]

    print(df_t1_red.shape, df_t2_red.shape, dfy.shape)
    # print("Control:", df_t1_red[df_t1_red["dose"] == 0])
    # INDEX_VALS = df_t2_red.index if not keep_names else dfy_old["name"].values


    df_t1_red.index = df_t2_red.index   # need to have same index for joined bootstrapping etc
    dfy.index = df_t2_red.index
    dfy_old.index = df_t2_red.index

    if keep_names:
        # id_names = dfy_old["name"].values
        print("Times of saliva sampling:", np.unique(list(dfy_old["time"]), return_counts=True))
        dfy = pd.concat([dfy, dfy_old["name"]], axis=1)
        # print(dfy)
        # print(id_names)
    # print(df_t1_red)
    if not(all(df_t1_red["time"].values == df_t2_red["time"].values)):
        print(">>>ERR: different time values....")
        return 0, 0, 0

    if not(all(df_t1_red["dose"].values == df_t2_red["dose"].values)):
        print(">>>ERR: different dose values....")
        print(df_t1_red["dose"].values == df_t2_red["dose"].values)
        return 0, 0, 0

    # if predict_late and not training == "ALL": # NEW TRAIN TEST SPLIT
    if type(training) == str:
        training = training.upper()

    if not training == "ALL":   # NEW TRAIN TEST SPLIT CO-REG T1 T2
        SPLIT_VAL_ID = t12_coreg_split_val_dict[predict_late][SPLIT_NUMBER]
        print(f"VALID ID SET SPLIT {SPLIT_NUMBER} =", SPLIT_VAL_ID)
        idx_valid = dfy_old[[nm in SPLIT_VAL_ID for nm in dfy_old["name"].values]].index.values
        idx_train = dfy_old.drop(idx_valid, axis=0).index.values

        x1_train, x1_val, y_train, y_val = df_t1_red.loc[idx_train], df_t1_red.loc[idx_valid], dfy.loc[idx_train], dfy.loc[idx_valid]
        x2_train, x2_val = df_t2_red.loc[idx_train], df_t2_red.loc[idx_valid]

        # if keep_names:
        #     y_train["id"] =

        # from sklearn.model_selection import train_test_split
        # strat = dfy if xer else False
        # splitoverlap = [1, 1, 1, 1, 1, 1, 1, 1]
        # SPLIT_VAL_ID = [1, 1, 1, 1, 1, 1, 1, 1]
        # SPLIT_1_VAL_ID = list(set(split_val_dict[predict_late][1]))
        # SPLIT_2_VAL_ID = list(set(split_val_dict[predict_late][2]))
        # print("SPLIT 1 # ids: ", len(SPLIT_1_VAL_ID), SPLIT_1_VAL_ID)
        # print("SPLIT 2 # ids: ", len(SPLIT_2_VAL_ID), SPLIT_2_VAL_ID)
        # STATE = 0
        # VALID_SIZE = 21     # baseline: 8, after irr: 8, False (acute): 21
        # # VALID_SIZE = 0.30   # fraction
        # idx_valid_existing_1 = dfy_old[[nm in SPLIT_1_VAL_ID for nm in dfy_old["name"].values]].index.values
        # idx_valid_existing_2 = dfy_old[[nm in SPLIT_2_VAL_ID for nm in dfy_old["name"].values]].index.values
        # print("EXISTING IMGS Spl 1 =", len(idx_valid_existing_1))
        # print("EXISTING IMGS Spl 2 =", len(idx_valid_existing_2))
        # # print(np.intersect1d(idx_valid_existing_1, idx_valid_existing_2))
        # # print(dfy_old.loc[np.intersect1d(idx_valid_existing_1, idx_valid_existing_2)])
        #
        # df_t1_red = df_t1_red.drop(idx_valid_existing_1, axis=0)
        # df_t1_red = df_t1_red.drop(idx_valid_existing_2, axis=0)
        # dfy = dfy.drop(idx_valid_existing_1, axis=0)
        # dfy = dfy.drop(idx_valid_existing_2, axis=0)
        # strat = dfy
        # print(df_t1_red.shape)
        #
        # # while len(splitoverlap) != 0:
        # # while any(set(SPLIT_1_VAL_ID).difference(set(SPLIT_VAL_ID))):
        # for i in range(1):
        # #     STATE += 1
        #     STATE = 42
        #     # x1_train, x1_val, y_train, y_val = train_test_split(df_t1_red, dfy, random_state=STATE, test_size=VALID_SIZE,
        #     #                                                   stratify=strat)
        #     # x2_train, x2_val = df_t2_red.loc[x1_train.index], df_t2_red.loc[x1_val.index]
        #     #
        #     # SPLIT_VAL_ID = dfy_old.loc[y_val.index]["name"].values
        #
        #     SPLIT_VAL_ID = dfy_old.loc[df_t1_red.index]["name"].values
        #     # splitoverlap = []
        #     splitoverlap = np.intersect1d(SPLIT_VAL_ID, SPLIT_1_VAL_ID)
        #     splitoverlap = list(set(splitoverlap).union(set(np.intersect1d(SPLIT_VAL_ID, SPLIT_2_VAL_ID))))
        #     print(STATE, "split_overlap=", splitoverlap)
        # print("Split state =", STATE, "--> valid ID set = ", SPLIT_VAL_ID)


        if training:
            num_xer_train = len(y_train[y_train.values == True])
            print("Training NUM XER:", num_xer_train, "of", len(y_train), f"({num_xer_train / len(y_train) *100 :.2f}%)")
            print(
                f"----- LOADED T1T2 registerred instances mode {MODE} {LRMODE} predict late {predict_late} split num {SPLIT_NUMBER} ------\n")
            return x1_train, x2_train, y_train
        else:
            num_xer_val = len(y_val[y_val.values == True])
            print("Valid NUM XER:", num_xer_val, "of", len(y_val), f"({num_xer_val / len(y_val) *100 :.2f}%)")
            print(
                f"----- LOADED T1T2 registerred instances mode {MODE} {LRMODE} predict late {predict_late} split num {SPLIT_NUMBER} ------\n")
            return x1_val, x2_val, y_val
    print(f"----- LOADED T1T2 registerred instances mode {MODE} {LRMODE} predict late {predict_late} split num {SPLIT_NUMBER} ------\n")
    return df_t1_red, df_t2_red, dfy


def load_all_roi_areas():
    # LOAD ALL ROI AREAS FOR L / R
    df1 = pd.read_csv(os.path.join(ftsDir, "LR_aggregated_FSPS_extracted_T1.csv"), index_col=0)
    df2 = pd.read_csv(os.path.join(ftsDir, "LR_aggregated_FSPS_extracted_T2.csv"), index_col=0)
    # print("Loaded T1, T2:", df1.shape, df2.shape)
    df1 = df1[["name", "time", "original_shape2D_PixelSurface_L", "original_shape2D_PixelSurface_R"]].rename({
        "original_shape2D_PixelSurface_L":"area_L", "original_shape2D_PixelSurface_R":"area_R"}, axis=1)
    df2 = df2[["name", "time", "original_shape2D_PixelSurface_L", "original_shape2D_PixelSurface_R"]].rename({
        "original_shape2D_PixelSurface_L":"area_L", "original_shape2D_PixelSurface_R":"area_R"}, axis=1)
    df1["id"] = [x.split("_")[0] for x in df1["name"].values]
    df2["id"] = [x.split("_")[0] for x in df2["name"].values]
    df1["pbool"] = ["p" in nm for nm in df1["name"].values]
    df2["pbool"] = ["p" in nm for nm in df2["name"].values]
    df1["ctr"] = [is_control(nm) for nm in df1["id"].values]
    df2["ctr"] = [is_control(nm, include_baseline=False) for nm in df2["id"].values]
    # print("Red T1, T2:", df1.shape, df2.shape)

    ids_t1 = np.unique(df1["id"].values)
    ids_t2 = np.unique(df2["id"].values)
    all_ids_images = np.unique([*ids_t1, *ids_t2])
    ids_t1_ctr = np.unique(df1[df1["ctr"] == True]["id"].values)
    ids_t2_ctr = np.unique(df2[df2["ctr"] == True]["id"].values)
    all_ctr_ids_images = np.unique([*ids_t1_ctr, *ids_t2_ctr])
    print("Number of image IDs:", len(all_ids_images), f"({len(ids_t1)} T1, {len(ids_t2)} T2), {len(all_ctr_ids_images)} control ({len(ids_t1_ctr)} T1, {len(ids_t2_ctr)} T2)")

    df_areas_img = pd.DataFrame()
    i = 0
    for nm in all_ids_images:
        df1_nm = df1[df1["id"] == nm]
        df2_nm = df2[df2["id"] == nm]
        times = np.unique([*df1_nm["time"].values, *df2_nm["time"].values])
        ctr_bool = all([*df1_nm["ctr"].values, *df2_nm["ctr"].values])
        # print(nm, times, "ctr =", ctr_bool)
        for t in times:
            df1_nm_t = df1_nm[df1_nm["time"] == t]
            df1_nm_t_nop = df1_nm_t[df1_nm_t["pbool"] == False]
            df1_nm_t_p = df1_nm_t[df1_nm_t["pbool"] == True]
            df2_nm_t = df2_nm[df2_nm["time"] == t]
            df2_nm_t_nop = df2_nm_t[df2_nm_t["pbool"] == False]
            df2_nm_t_p = df2_nm_t[df2_nm_t["pbool"] == True]
            t = int(t[:-3])
            # print(nm, t, df1_nm_t.values)
            df_areas_img.loc[i, ["name", "time", "ctr"]] = [nm, t, ctr_bool]
            if any(df1_nm_t_nop.values.ravel()):
                df_areas_img.loc[i, ["T1_nop_L", "T1_nop_R"]] = df1_nm_t_nop[["area_L", "area_R"]].values[0]
            if any(df1_nm_t_p.values.ravel()):
                df_areas_img.loc[i, ["T1_p_L", "T1_p_R"]] = df1_nm_t_p[["area_L", "area_R"]].values[0]
            if any(df2_nm_t_nop.values.ravel()):
                df_areas_img.loc[i, ["T2_nop_L", "T2_nop_R"]] = df2_nm_t_nop[["area_L", "area_R"]].values[0]
            if any(df2_nm_t_p.values.ravel()):
                df_areas_img.loc[i, ["T2_p_L", "T2_p_R"]] = df2_nm_t_p[["area_L", "area_R"]].values[0]
            i += 1
    return df_areas_img


def load_classif_results(PREDMODE="SIMULT", MODE="NO P", WEIGHT="T1", LRMODE="aggregated", CLASSIFIER="RF", NFTS=5, loocv=True, boot=False):
    PREDMODE = PREDMODE.lower()
    LOADNAMES = []
    if PREDMODE in ["late", "after irr", "baseline", "delta"]:
        if loocv:
            load_folder = os.path.join(ClassifDir, "no coreg", "loocv late")
            if boot:
                load_folder = os.path.join(load_folder, "boot")
        else:
            load_folder = os.path.join(ClassifDir, LRMODE)
    elif PREDMODE in ["acute", "simult", "contemp"]:
        if loocv:
            load_folder = os.path.join(ClassifDir, "loocv simult")
            if boot:
                load_folder = os.path.join(load_folder, "boot")
                loadname = "_".join([LRMODE, CLASSIFIER, MODE, WEIGHT, PREDMODE, f"nfts={NFTS}", "bootmatrix.csv"])
                LOADNAMES.append(loadname)
                loadname = "_".join([LRMODE, CLASSIFIER, MODE, WEIGHT, PREDMODE, f"nfts={NFTS}", "bootmatrix_td.csv"])
                LOADNAMES.append(loadname)
                loadname = "_".join([LRMODE, CLASSIFIER, MODE, WEIGHT, PREDMODE, f"nfts={NFTS}", "featurematrix.csv"])
                LOADNAMES.append(loadname)
                loadname = "_".join([LRMODE, CLASSIFIER, MODE, WEIGHT, PREDMODE, f"nfts={NFTS}", "truthvalues.csv"])
                LOADNAMES.append(loadname)
            else:
                loadname = "_".join([LRMODE, CLASSIFIER, MODE, WEIGHT, PREDMODE, f"nfts={NFTS}.csv"])
                LOADNAMES.append(loadname)
                loadname = "_".join([LRMODE, CLASSIFIER, MODE, WEIGHT, PREDMODE, f"nfts={NFTS}", "selected features.csv"])
                LOADNAMES.append(loadname)
    else:
        return 0
    DFL = []
    for load in LOADNAMES:
        df = pd.read_csv(os.path.join(load_folder, load), index_col=0)
        DFL.append(df)
    print(f"LOADED {len(DFL)} dfs for {PREDMODE} {MODE} {WEIGHT} {LRMODE} {CLASSIFIER} {NFTS}fts loocv={loocv}, boot={boot}")
    return DFL


if __name__ == "__main__":


    # x, y = load_delta("T2", LRMODE="aggregated", training=True, xer=False, keep_time=True)
    # x, y = load_nop("T2", LRMODE="aggregated", training=False, xer=True)
    # x, y = load_delta_P(WEIGHT="T2", LRMODE="aggregated", xer=True, training=False)
    # print(x["time"])
    # x_train, y_train = load_predict_late_not_delta(MODE="NO P", WEIGHT="T2", LRMODE="aggregated", baseline="baseline",
    #                                                training=True, xer=True)

    # x, y = load_nop(WEIGHT="T1", LRMODE="aggregated", training="all", xer=True, keep_names=True)
    # x, y = load_delta(WEIGHT="T2", LRMODE="aggregated", training="all", xer=False, keep_time=True, keep_names=True)
    # x, y = load_delta_P(WEIGHT="T2", LRMODE="aggregated", training="all", xer=True, keep_names=True)
    # y = y.to_frame()
    # y["name"] = [nm.split("_")[0] for nm in list(x1["name"])]
    # y["ctr"] = [is_control(nm) for nm in y["name"].values]
    # ids = list(set(y["name"].values))
    # ids_ctr = list(set(y[y["ctr"] == True]["name"].values))
    # print(f"Having {len(ids)} individuals, where {len(ids_ctr)} is control ({len(ids_ctr) / len(ids) * 100 :.0f}%)")
    # print(f"\tNumber of data points: {len(y)}, {len(y[y['''ctr''']==True])} control.")

    d = {"all":{}, "test":{}, "train":{}}
    # d = {"all":{}, "test1":{}, "test2":{}, "test3":{}}
    # for i in range(1, 4):
    for i in range(1):
        for train in ["all", "test"]:
            trainbool = train if train == "all" else False
            # x1, x2, y = load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", predict_late="after irr", training=trainbool, xer=True, SPLIT_NUMBER=i, keep_names=True)
            # train = str(train) + str(i) if not train == "all" else "all"
            x, y = load_nop("T1", LRMODE="aggregated", training=trainbool, xer=True, keep_names=True)
            y = y.to_frame()
            y.loc[:, "name"] = x["name"]

            y["ctr"] = [is_control(nm) for nm in y["name"].values]
            y_ctr = y[y["ctr"] == True]
            y_irr = y[y["ctr"] == False]

            d[train]["tot"] = len(y)
            d[train]["tot xer"] = len(y[y["xer"] == True])
            d[train]["ctr"] = len(y_ctr)
            d[train]["ctr xer"] = len(y_ctr[y_ctr["xer"] == True])
            d[train]["irr"] = len(y_irr)
            d[train]["irr xer"] = len(y_irr[y_irr["xer"] == True])

    for key, dict in d.items():
        print(key, dict)
        all = dict["tot"]
        allxer = dict["tot xer"]
        ctr = dict["ctr"]
        ctrxer = dict["ctr xer"]
        print(f"tot: {all} ({allxer}), control: {ctr} ({ctrxer})")
    sys.exit()

    # print(x)
    # print(y)



    # MD, WG = "NO P", "T2"
    # basel = True
    # train = "all"
    # df, y = load_predict_late(MODE=MD, WEIGHT=WG, baseline=basel, xer=True, training=train, keep_names_times=True)
    # df, y = load_predict_late_not_delta(MODE=MD, WEIGHT=WG, baseline=basel, training=train);    df["time"] = [int(t[:-3]) for t in df["time"].values];    y = binary_thresh_xerostomia(dfy=y)
    # df, df2, y = load_T1T2_coreg(MODE="NO P", predict_late="baseline", training="all")

    # df["xer"] = y
    # df = df.drop_duplicates()
    # y = df["xer"]

    # print(df[["time", "dose"]])
    # print(df[df["time"] > 0][["time", "dose"]])
    # print(y)
    # print(df.shape, y.shape)
    # print(f"ALL: {len(y)}, XER: {len(y[y == True])}")
    # sys.exit()

    # load_T1T2_coreg(MODE="NO P",LRMODE="aggregated", predict_late=False, training=True, xer=False, SPLIT_NUMBER=1, keep_names=False)
    # MODE, WEIGHT = "NO P", "T1"
    MODE, WEIGHT = "DELTA", "T2"
    # baseline = True
    baseline = False
    NOT_DELTA = True

    if MODE == "DELTA": # training
        x, y = load_delta(WEIGHT=WEIGHT, LRMODE="aggregated", training=True, xer=True, keep_time=True, keep_names=True)
        x["id"] = [nm.split("_")[0] for nm in x["name"].values]
    else:
        if NOT_DELTA:
            x, y = load_predict_late_not_delta(MODE=MODE, WEIGHT=WEIGHT, LRMODE="aggregated", training=True, xer=True, baseline=baseline, keep_id=True, reset_y_index=True)
        else:
            x, y = load_predict_late(MODE=MODE, WEIGHT=WEIGHT, LRMODE="aggregated", training=True,
                                     baseline=baseline, xer=True, keep_names_times=True)
            x["id"] = [nm.split("_")[0] for nm in x["name"].values]

    print(x["time"])

    sys.exit()
    # x["id"] = [nm.split("_")[0] for nm in x["name"].values]
    x["ctr"] = [is_control(nm) for nm in x["id"].values]
    # print(x[["id", "ctr"]])
    y_ctr = y.loc[x[x["ctr"] == True].index]
    y_irr = y.loc[x[x["ctr"] == False].index]
    train_tot, train_tot_xer = len(x), len(y[y==True])
    train_control, train_control_xer = len(y_ctr), len(y_ctr[y_ctr == True])
    train_irr, train_irr_xer = len(y_irr), len(y_irr[y_irr == True])

    if MODE == "DELTA": # test
        x, y = load_delta(WEIGHT=WEIGHT, LRMODE="aggregated", training=False, xer=True, keep_time=True, keep_names=True)
        x["id"] = [nm.split("_")[0] for nm in x["name"].values]
    else:
        if NOT_DELTA:
            x, y = load_predict_late_not_delta(MODE=MODE, WEIGHT=WEIGHT, LRMODE="aggregated", training=False, xer=True, baseline=baseline, keep_id=True, reset_y_index=True)
        else:
            x, y = load_predict_late(MODE=MODE, WEIGHT=WEIGHT, LRMODE="aggregated", training=False,
                                     baseline=baseline, xer=True, keep_names_times=True)
            x["id"] = [nm.split("_")[0] for nm in x["name"].values]

    x["ctr"] = [is_control(nm) for nm in x["id"].values]
    # print(x[["id", "ctr"]])
    y_ctr = y.loc[x[x["ctr"] == True].index]
    y_irr = y.loc[x[x["ctr"] == False].index]
    test_tot, test_tot_xer = len(x), len(y[y==True])
    test_control, test_control_xer = len(y_ctr), len(y_ctr[y_ctr == True])
    test_irr, test_irr_xer = len(y_irr), len(y_irr[y_irr == True])

    if MODE == "DELTA": # ALL
        x, y = load_delta(WEIGHT=WEIGHT, LRMODE="aggregated", training="all", xer=True, keep_time=True, keep_names=True)
        x["id"] = [nm.split("_")[0] for nm in x["name"].values]
    else:
        if NOT_DELTA:
            x, y = load_predict_late_not_delta(MODE=MODE, WEIGHT=WEIGHT, LRMODE="aggregated", training="all", xer=True, baseline=baseline, keep_id=True, reset_y_index=True)
        else:
            x, y = load_predict_late(MODE=MODE, WEIGHT=WEIGHT, LRMODE="aggregated", training="all",
                                     baseline=baseline, xer=True, keep_names_times=True)
            x["id"] = [nm.split("_")[0] for nm in x["name"].values]

    x["ctr"] = [is_control(nm) for nm in x["id"].values]
    # print(x[["id", "ctr"]])
    y_ctr = y.loc[x[x["ctr"] == True].index]
    y_irr = y.loc[x[x["ctr"] == False].index]
    all_tot, all_tot_xer = len(x), len(y[y==True])
    all_control, all_control_xer = len(y_ctr), len(y_ctr[y_ctr == True])
    all_irr, all_irr_xer = len(y_irr), len(y_irr[y_irr == True])

    print(f"{MODE} {WEIGHT} {'''baseline''' if baseline else '''after irr'''}:")
    print(f"ALL count (xer):\t\t{all_tot} ({all_tot_xer}), control: {all_control} ({all_control_xer}), irradiated: {all_irr} ({all_irr_xer})")
    print(f"TRAIN + TEST tot (xer):\t{train_tot + test_tot} ({train_tot_xer + test_tot_xer})")
    print(f"Train: count (xer) tot:\t{train_tot} ({train_tot_xer}), control: {train_control} ({train_control_xer})"
          f", irradiated: {train_irr} ({train_irr_xer})")

    print(f"test: count (xer) tot:\t{test_tot} ({test_tot_xer}), control: {test_control} ({test_control_xer})"
          f", irradiated: {test_irr} ({test_irr_xer})")



    sys.exit()

    # late = "after irr"
    late = "baseline"
    # late = False
    latemode = "acute" if late == False else late

    SPLNM = 2
    # train = "all"
    train = False
    df1, df2, y = load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", predict_late=late, training=train, xer=True,
                                  SPLIT_NUMBER=SPLNM, keep_names=True)

    if not (all(df1.index.values == y.index.values) or all(df2.index.values == y.index.values)):
        print("DIFFERENT INDEXES: FIX!!!!!!!")
        sys.exit()
    y["time"] = df1["time"]
    y["dose"] = df1["dose"]
    print(y)

    # binary_thresh_xerostomia(dfy=y)

    sys.exit()


    y["ctrl"] = [is_control(nm) for nm in y["name"].values]
    y["time"] = df1["time"]
    print("All times:", np.unique(list(y["time"]), return_counts=True))
    print(f"All control:", len(y[y["ctrl"] == True]), "of", len(y))
    print("All xer:\t", len(y[y["xer"] == True]), "of", len(y))

    # plot_ids_over_time(y, title=f"T12 co-reg {latemode} SPLIT {SPLNM}")
    # sys.exit()
    # print(y)

    SPLIT_1_VALID = t12_coreg_split_val_dict[latemode][1]
    SPLIT_2_VALID = t12_coreg_split_val_dict[latemode][2]
    SPLIT_3_VALID = t12_coreg_split_val_dict[latemode][3]

    y = y[[nm not in SPLIT_1_VALID for nm in list(y["name"])]]
    y = y[[nm not in SPLIT_2_VALID for nm in list(y["name"])]]
    y = y[[nm not in SPLIT_3_VALID for nm in list(y["name"])]]
    plot_ids_over_time(y)
    # df1 = df1.loc[y.index]
    # df2 = df2.loc[y.index]
    # print("Remove valid 1:", y.shape)

    sys.exit()

    from mrmr import mrmr_classif
    mrmr_classif(t1_train, y_train, K=Nfts, return_scores=False)
    print(df1.shape, df2.shape, y.shape)

    # print(df1)
    # print(df2)
    sys.exit()

    y_indices = set()
    LRMODE = "average"
    MODE = "NO P"
    # LRMODE = "aggregated"
    for SPLN in [1, 2, 3]:
        # for SPLN in [2]:
        df1, df2, y = load_T1T2_coreg(MODE=MODE, LRMODE=LRMODE, training=False, xer=True, predict_late="after irr", SPLIT_NUMBER=SPLN)
        # df1, df2, y = load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", training=False, xer=True, predict_late="baseline", SPLIT_NUMBER=SPLN)
        # df1, df2, y = load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", training=False, xer=True, predict_late=False, SPLIT_NUMBER=SPLN)
        print(y.index.values)
        y_indices = y_indices.union(set(y.index.values))
    print(len(y_indices), y_indices)

    # df1, df2, y = load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", training=False, xer=True, predict_late="baseline")
    # df = load_fsps_data(WEIGHT="T2", MODE="NO P", TRAIN_SET=True)
    # mode = "DELTA"
    # df = load_fsps_data(WEIGHT="T1", MODE="DELTA", TRAIN_SET=True, LRMODE="agg")
    # df = load_fsps_data(WEIGHT="T2", MODE=mode, TRAIN_SET=False, LRMODE="average")

    # df = load_fsps_data(WEIGHT="T2", MODE="NO P", LRMODE="agg", TRAIN_SET="ALL")
    # load_nop("T2", LRMODE="agg", training=False, xer=True)
    # df, y = load_delta_P("T2", LRMODE="agg", training=False, xer=True)
    # df, y = load_nop("T2", LRMODE="agg", training=False, xer=True)
    # df, y = load_predict_late(MODE="NO P", WEIGHT="T1", LRMODE="agg", training=True, xer=True, baseline=True)
    # df, y = load_predict_late(MODE="DELTA P", WEIGHT="T2", LRMODE="agg", training=True, xer=True, baseline=True)
    # df, y = load_delta(WEIGHT="T1", LRMODE="agg", training="ALL", xer=True)
    # print(df.T.head(5))
    # load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", training=True, xer=True)
    # load_T1T2_coreg(MODE="NO P", LRMODE="average", training=True, xer=True)

    # load_predict_late_not_delta(MODE="NO P", WEIGHT="T1", LRMODE="aggregated", training="ALL", baseline=False)
    # load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", training=True, xer=True, predict_late="baseline")
    # load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", training="ALL", xer=True, predict_late="after irr")


    # df1, df2, y = load_T1T2_coreg(MODE="NO P", LRMODE="average", training=False, xer=True, predict_late="after irr")
    # df1, df2, y = load_T1T2_coreg(MODE="NO P", LRMODE="aggregated", training=False, xer=True, predict_late=False, SPLIT_NUMBER=3)


    # load_T1T2_coreg(MODE="NO P", LRMODE="average", training=True, xer=True, predict_late="baseline")
    # print(df)


    # print(y)

    # load_predict_late(MODE="DELTA P", WEIGHT="T2", LRMODE="split", training=True, xer=True, baseline=True)
    # df, y = load_predict_late(MODE="DELTA P", WEIGHT="T2", LRMODE="average", training=True, xer=False, baseline=True)
    # df, y = load_predict_late(MODE="DELTA P", WEIGHT="T2", LRMODE="average", training=False, xer=False, baseline=True)

    # load_fsps_data("T2", MODE="NO P", LRMODE="average", TRAIN_SET="ALL")
    # load_fsps_data("T1", MODE="NO P", LRMODE="average", TRAIN_SET="ALL")

    # load_nop("T1", LRMODE="average", training="ALL")
    # load_nop("T2", LRMODE="average", training="ALL")
    # load_delta_P(WEIGHT="T2", LRMODE="average", training="ALL")
    # load_delta(LRMODE="average", training="ALL")
    # load_fsps_data("T2", MODE="DELTA", LRMODE="average", TRAIN_SET=False)


    # x, y = load_nop("T2", LRMODE="split", training=True, xer=True)
    # x, y = load_nop("T2", LRMODE="average", training=True, xer=True)
    # x, y = load_delta(LRMODE="split", training=True, xer=False)
    # x, y = load_delta(LRMODE="average", training=True, xer=False)
    # x, y = load_delta_P(LRMODE="split", WEIGHT="T2", training=True, xer=False)
    # x, y = load_delta_P(LRMODE="average", WEIGHT="T2", training=True, xer=False)
    # print(x.shape)
    # x, y = load_delta_P("T2", training=False, xer=False)
    # print(x.shape)

    # df = pd.read_csv(os.path.join(ftsDir, "LR_average_FSPS_DELTA-P.csv"), index_col=0)
    # print(df.shape)
    # print(len(set([x.split("_")[0] for x in df["name"]])))
    # df = pd.read_csv(os.path.join(ftsDir, "LR_split_FSPS_DELTA-P.csv"), index_col=0)
    # print(df.shape)
    # print(len(set([x.split("_")[0] for x in df["name"]])))
    #
    # df = pd.read_csv(os.path.join(ftsDir, "LR_split_FSPS_DELTA-time.csv"), index_col=0)
    # print(df.shape)
    # print(len(set([x.split("_")[0] for x in df["name"]])))
    # df = pd.read_csv(os.path.join(ftsDir, "LR_average_FSPS_DELTA-time.csv"), index_col=0)
    # print(df.shape)
    # print(len(set([x.split("_")[0] for x in df["name"]])))


    # x, y = load_predict_late(MODE="NO P", WEIGHT="T2", training=True, baseline=False, xer=True)
    # print(x)
    # print(y)

    # df = load_fsps_data(WEIGHT="T2", MODE="DELTA P", TRAIN_SET=True)
    # df = load_fsps_data(WEIGHT="T2", MODE="DELTA", TRAIN_SET=True)
    # df = load_fsps_data(WEIGHT="T1", MODE="NO P", TRAIN_SET=True)