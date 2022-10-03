import os
import pandas as pd
import numpy as np
from scipy import stats
import six
import sys

global RawDir, SegBrainDir, SegSalivDir, nrrdDir, ftsDir
RawDir = os.path.join(os.getcwd(), "..", "..", "RAW DATA")
SegBrainDir = os.path.join(os.getcwd(), "..", "..", "Segmentations", "brain")
SegSalivDir = os.path.join(os.getcwd(), "..", "..", "Segmentations", "salivary")
ftsDir = os.path.join(os.getcwd(), "..", "..", "Radiomic features")
nrrdDir = os.path.join(os.getcwd(), "..", "..", "Radiomic features", "nrrd files")
PlotDir = os.path.join(os.getcwd(), "..", "..", "master_plots")
PreprocessDir = os.path.join(os.getcwd(), "..", "..", "Preprocessing")

print(os.getcwd())
global central_idx_dict, all_timenames
all_timenames = []
central_idx_dict = {}
right_idx_dict = {}
left_idx_dict = {}
df = pd.read_csv(os.path.join(os.getcwd(), "..", "..", r"Segmentations\salivary\segment params salivary.csv"),
                 index_col=0)
df_lr = pd.read_csv(os.path.join(SegSalivDir, "salivary LR split indexes.csv"), index_col=0)
# print(df_lr)
for timename, idx in zip(df["time_name"].values, df["idx_central"].values):
    central_idx_dict.update({timename: idx})
    all_timenames.append(timename)

for timename, idx_l, idx_r in zip(df_lr["time_name"].values, df_lr["idx_l"].values, df_lr["idx_r"].values):
    left_idx_dict.update({timename:idx_l})
    right_idx_dict.update({timename:idx_r})

FBW_dict_T2 = {"no norm":100, "stscore":0.075, "nyul":950}  # stscore: n=3, nyul: otsu decile
FBW_dict_T1 = {"no norm":100, "stscore":0.050, "nyul":800}


def get_best_feature_normalization(weight="T2", THRESH=0, return_df=True, LRmode="split"):
    weight = weight.upper()
    if not weight in ["T1", "T2"]:
        print("INVALID WEIGHT", weight)
        return 0
    if not LRmode in ["split", "aggregated", "average"]:
        print(">>>>>Invalid LR mode", LRmode)
        return 0
    if THRESH == 0:
        if weight == "T2":
            THRESH = 0.05
        else:
            THRESH = 0.15
    PATH = os.path.join(ftsDir, f"FSPS_LR{LRmode}_" + weight + f"_THRESH={THRESH}" + ".csv")
    try:
        df = pd.read_csv(PATH, index_col=0)["spearman"]
    except Exception as e:
        print(*e.args)
        return 0
    norm_dict = {}
    # print(df)
    norm_df = pd.DataFrame()
    for key, val in six.iteritems(df):
        # print(key, val)
        l = val.split(":")
        if len(l) != 1:
            norm = l[1]
        else:
            norm = l[0]
        norm_dict[key] = norm
        norm_df.loc[key, "norm"] = norm

    print(f"--- Best normalization for {weight} images with {len(norm_dict)} features loaded successfully ----")
    return norm_df if return_df else norm_dict


def get_name(experiment, folder, condition="sagittal"):
    experiment = experiment.lower()
    if experiment == "pilot2":
        name = folder[:3] if not "p" in folder else folder[:12]     # differentiates small and big p!!
        name = name + "_" + condition if bool(condition) else name
        # print(name)
    elif experiment == "pilot1":
        name = folder.replace(" ", "_")
    elif experiment == "pilot3":
        if not "p" in folder:
            name = folder[:3]
        else:
            name = folder[:11]
        name += "_" + "T2_" if "t2" in folder.lower() else "_" + "T1_"
        name = name + condition if bool(condition) else name
        # name = name + condition if bool(condition) else ""
        # print(name)
    elif experiment == "pilot4":    # there is no p data here
        id, rest = folder.split("__")
        name = id + rest[2:6]
        name = name + condition if bool(condition) else name
    else:
        print("NAMING PROCEDURE FOR EXPERIMENT", experiment, "NOT RECOGNIZED.")
        return 0
    return name


def minmax_norm(x, min, max, lower, upper, rounded):
    a = lower + (x - min) * (upper - lower) / (max - min)
    return round(a) if rounded else a


def qcd(vals):
    # Quartile coefficient of dispersion
    # (Q3 - Q1) / (Q3 + Q1)
    Q1, Q3 = np.quantile(vals, [0.25, 0.75])
    return (Q3 - Q1) / (Q3 + Q1)


def norm_minmax_featurescaled(m, lower=0, upper=255, rounded=False, printbool=False):
    min, max = np.min(m), np.max(m)
    func = np.vectorize(minmax_norm)
    print("Matrix of shape", m.shape, "with range [{0:.2f},{1:.2f}] normalized by feature-scaled min-max norm to [{2},{3}].".format(min, max, lower, upper))    if printbool else 0
    try:
        return func(m, min, max, lower=lower, upper=upper, rounded=rounded)
    except Exception as e:
        print("ERROR MINMAX NORM:", e.args)
        return m


def percentile_truncation(m, lower=0, upper=99, settozero=False, verbose=False):
    m = m.copy()
    min, max = np.percentile(m, [lower, upper])
    if verbose:
        print("PERCENTILE TRUNCATION:")
        print(f"From {len(m.ravel())} voxels, {len(m[m < min])} lower than {lower} percentile = {min} set to {0 if settozero else min}, {len(m[m > max])} above {upper} percentile = {max:.3f}.")
        print(len(m[m < min]) + len(m[m > max]), "voxels adjusted.")
    m[m < min] = 0 if settozero else min
    m[m > max] = max# if not settozero else 0
    return m

def crop_to_mask(img, mask):
    imin, imax = min(np.argwhere(mask).T[0]), max(np.argwhere(mask).T[0])
    jmin, jmax = min(np.argwhere(mask).T[1]), max(np.argwhere(mask).T[1])
    img_crop = img[imin:imax, jmin:jmax]
    # print(imax-imin, jmax-jmin, (jmax-jmin)*(imax-imin))
    return img_crop


def resegment(im, roi, n=3):
    print(f"    --- RESEGMENTING IMAGE: excluding outliers outside mu +- {n} * sd ---")
    '''adjust ROI such that all px values in im are within mu +- 3 * sd (calc from roi)'''
    mu, sd = np.mean(im[roi != 0]), np.std(im[roi != 0])
    f = lambda x: abs(x - mu) < n * sd
    vf = np.vectorize(f)
    return np.multiply(roi, vf(im))

def discretize_FBW(im, roi, bw):
    # print(f"    --- DISCRETIZING IMAGE by FBW w/ bw = {bw:.1f} ---")
    # min_val = np.min(np.ceil(im[roi].ravel() / bw))
    min_val = np.min(np.ceil(im[roi != 0].ravel() / bw))
    # f = lambda x: np.ceil(x / bw) - min_val + 1
    print("MIN VAL = ", min_val)
    f = lambda x: int(np.ceil(x / bw) - min_val + 1)
    vf = np.vectorize(f)
    dim = vf(im)
    # print("     minimum value in roi: ", min_val)
    # print("     minimum value in whole image / ROI AFTER DISC:", np.min(dim), np.min(dim[roi]))
    print(f"     --- IMAGE DISCRETIZED into {len(np.unique(dim))} (whole image), {len(np.unique(dim[roi != 0]))} bins (ROI), of equal width {bw}.")
    return dim

def discretize_FBW_ISBI(im, roi, bw):
    min_val = np.min(im[roi != 0].ravel())
    # print("MIN VAL = ", min_val)
    f = lambda x: np.floor((x - min_val) / bw) + 1
    vf = np.vectorize(f)
    dim = vf(im)
    # f2 = lambda x: np.ceil((x - min_val) / bw) + 1
    # vf2 = np.vectorize(f2)
    # print(len(np.unique(vf(im))), len(np.unique(vf2(im))))
    print(f"     --- IMAGE DISCRETIZED into {len(np.unique(dim))} (whole image), {len(np.unique(dim[roi != 0]))} bins (ROI), of equal width {bw}.")
    return dim


def freedman_diaconis_rule(vals, return_nbins=False):
    '''
    Based on N voxels in VOI / ROI, and the IQR of image voxel values in VOI, calculate bin width for FBW discretization.
    '''
    # N = np.count_nonzero(roi)
    # iqr = stats.iqr(im[roi].ravel())
    vals = vals.ravel()
    N = len(vals)
    iqr = stats.iqr(vals)
    bw = 2 * iqr / N**(1 / 3)
    print(f"\tFreedman Diaconis BW (N={N}, IQR={iqr:.1f}) = {bw:.2f}")
    if return_nbins:
        return int((max(vals) - min(vals)) / bw)
    return bw
