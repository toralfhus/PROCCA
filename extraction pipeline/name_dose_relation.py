import pandas as pd
import os
# import six
# import numpy as np


def load_naming_data():
    namingpath = os.path.normpath(os.path.join(os.getcwd(), "..", r"RAW DATA\naming.csv"))
    if not os.path.exists(namingpath):
        print("CANNOT FIND FILE @ ", namingpath, " NOTHING LOADED")
        return 0
    else:
        df1 = pd.read_csv(namingpath, skiprows=1, nrows=16, index_col=0)    #experiment 1 (pilot)
        df2 = pd.read_csv(namingpath, skiprows=range(23), nrows=16, index_col=0)    #Experiment 2 (pilot)
        df1 = df1.rename(columns={'Dose/fraction*':'doserate'})
        df2 = df2.rename(columns={'Dose/fraction*':'doserate'})
    df = pd.concat([df1, df2])
    df["doserate"] = df["doserate"].map(lambda x: float(x[:-3]))   # REMOVE space + Gy
    # print(df["doserate"])
    # for name in list(df1.index.values) + list(df2.index.values):
    #     print(name)
    return df

def dose_to_name(name, time, exp="", ignore_exp=False):
    # FIGURE OUT IF 7- AND 10- ARE CONTROL OR IRR 
    if time == "-7day" or time == "-3day" or time in [-3, -7]:
        # print("BASELINE", name)
        return 0    # NO DOSE AT BASELINE, ELSE dose/frac * 5 * 2

    if exp.lower() == "pilot1" or ignore_exp:
        if name[0] == "C":
            return 0
        elif name[0] == "L":
            return 3 * 10
        elif name[0] == "H":
            return 4.4 * 10
        elif not ignore_exp:
            print(">>>> NO DOSE DATA FOR", exp, time, name)
            return 1 / 0    #break

    if exp.lower() == "pilot4" or ignore_exp:
        CTR = ["13-6", "13-8", "13-10", "14-2", "14-3",
               "11-2", "11-4", "11-5", "11-8", "11-10"]
        IRR1 = ["13-7", "13-9", "14-1", "14-4", "14-5"]  # 7.4 Gy per fraction (once per day, 10 days)
        IRR2 = ["11-1", "11-3", "11-6", "11-7", "11-9", "12-1"] # 7.5 Gy per fraction --||--
        for id in CTR:
            if id in name:
                return 0
        for id in IRR1:
            if id in name:
                return 7.4 * 10
        for id in IRR2:
            if id in name:
                return 7.5 * 10
        if not ignore_exp:
            print(">>>> NOT ANY DOSE DATA????")
            return 1 / 0
    if ignore_exp or exp.lower() in ["pilot2", "pilot3"]:
        if "1-" in name or "5-" in name or "8-" in name:
            # print("CONTROL")
            return 0
        elif "2-" in name:
            return 5 * 10
        elif "3-" in name:
            return 5.75 * 10
        elif "4-" in name:
            return 6.5 * 10
        elif "6-" in name:
            return 8.5 * 10
        elif "9-" in name:
            return 7.5 * 10
        else:
            print(f"\n\n\n\n\n>>>>>>NO DATA FOR {name}, {time}....\nn\n\n\n\n\n")
            return 1 / 0
    print("\n\n\n\n\n>>>>>NO DATA\nn\n\n\n\n\n")
    return 1 / 0

def is_control(name, time=0, include_baseline=False):
    # NEED TO HAVE NAME ON FORM X-Y or CX / LX / HX (X, Y are whole numbers)
    CTR_exp4 = ["13-6", "13-8", "13-10", "14-2", "14-3", "11-2", "11-4", "11-5", "11-8", "11-10"]
    if (time in [-7, -3, "-7day", "-3day"]) and include_baseline:
        return True
    id_group = name.split("-")[0]
    # print(name, id_group)
    # if any([nm in name for nm in ["1-", "5-", "8-", "C"]]):
    if any([id_group in ["1", "5", "8"]]):
        return True
    # elif any([nm in name for nm in CTR_exp4]):
    elif name[0] == "C":
        return True
    elif name in CTR_exp4:
        return True
    else:
        # print(time, name)
        return False
# load_naming_data()


if __name__ == "__main__":
    print(is_control("C5"))