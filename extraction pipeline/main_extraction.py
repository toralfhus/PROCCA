import os
import pandas as pd
import numpy as np
import six
from extract_utils import *
from DICOM_reader import find_folders, dcm_files_indexed, dcm_folder_to_pixel_matrix, find_raw_folders_from_name
from preprocessing import get_name, norm_minmax_featurescaled, percentile_truncation, mean_centering, norm_stscore
from MRI_denoising import n4correction
from nyul_histogram_matching import nyul_normalizer, nyul_initializer, nyul_scales_stsc, nyul_scales_T2, nyul_scales_T1
from feature_extractor import ndarray_to_nrrd, plot_nrrd
from name_dose_relation import dose_to_name


from visualizations import plot_masked
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.integrate import simps
from radiomics import featureextractor
import yaml
import SimpleITK as sitk


def plot_values_in_roi(n4=True, offsetIDX=0):
    if not n4:
        df = pd.DataFrame(
            {"name": [], "exp": [], "time": [], "p": [], "mu_img": [], "mu_brain": [], "mu_saliv": [], "sd_img": [],
             "sd_brain": [], "sd_saliv": []})
    else:
        df = pd.DataFrame(
            {"name": [], "exp": [], "time": [], "p": [], "n4":[], "mu_img": [], "mu_brain": [], "mu_saliv": [], "sd_img": [],
             "sd_brain": [], "sd_saliv": []})

    for exp in ["Pilot1", "Pilot2"]:
        for time in os.listdir(os.path.join(RawDir, exp)):
            print(exp, time)
            datadir = os.path.normpath(os.path.join(RawDir, exp, time))
            for i, folder in enumerate(find_folders(datadir, "sagittal")):
                name = get_name(exp, folder, "sagittal")
                idx_center = central_idx_dict[time+name]
                # print(name, folder)
                print(name)
                indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
                MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder), printbool=False)
                ROI_saliv = np.load(os.path.join(SegSalivDir, exp.lower(), time, name + ".npy"))
                ROI_brain = np.load(os.path.join(SegBrainDir, exp.lower(), time, name + ".npy"))
                im_raw = MATR_RAW[idx_center + offsetIDX]
                roi_saliv = ROI_saliv[idx_center + offsetIDX]
                roi_brain = ROI_brain[idx_center + offsetIDX]
                # plot_masked(im_raw, roi_saliv)

                if n4:
                    im_raw_corr, im_field, msk = n4correction(im_raw)
                    # fig, ax = plt.subplots(ncols=2)
                    # ax[0].imshow(im_field)#, alpha=0.5)
                    # ax[0].imshow(im_raw, cmap="gray", alpha=0.5)
                    # ax[1].imshow(im_raw_corr - im_raw, cmap="gray")
                    # plt.show()
                    mu_img_corr = np.mean(im_raw_corr)
                    sd_img_corr = np.std(im_raw_corr)
                    mu_brain_corr = np.mean(im_raw_corr[roi_brain != 0])
                    sd_brain_corr = np.std(im_raw_corr[roi_brain != 0])
                    mu_saliv_corr = np.mean(im_raw_corr[roi_saliv != 0])
                    sd_saliv_corr = np.std(im_raw_corr[roi_saliv != 0])

                mu_img = np.mean(im_raw)
                sd_img = np.std(im_raw)
                mu_brain = np.mean(im_raw[roi_brain != 0])
                sd_brain = np.std(im_raw[roi_brain != 0])
                mu_saliv = np.mean(im_raw[roi_saliv != 0])
                sd_saliv = np.std(im_raw[roi_saliv != 0])
                if not n4:
                    df = df.append({"name":name, "exp":exp.lower(), "time":time, "p":"p" in name, "mu_img":mu_img, "mu_brain":mu_brain, "mu_saliv":mu_saliv,
                                    "sd_img":sd_img, "sd_brain":sd_brain, "sd_saliv":sd_saliv}, ignore_index=True)
                if n4:
                    df = df.append({"name": name, "exp": exp.lower(), "time": time, "p": "p" in name, "n4":False,
                                    "mu_img": mu_img, "mu_brain": mu_brain, "mu_saliv": mu_saliv, "sd_img": sd_img, "sd_brain": sd_brain, "sd_saliv": sd_saliv}, ignore_index=True)
                    df = df.append({"name":name, "exp":exp.lower(), "time":time, "p":"p" in name, "n4":True, "mu_img":mu_img_corr,
                                    "mu_brain":mu_brain_corr, "mu_saliv":mu_saliv_corr, "sd_img":sd_img_corr, "sd_brain":sd_brain_corr, "sd_saliv":sd_saliv_corr}, ignore_index=True)
    print(df)
    # g = sns.PairGrid(df, vars=["mu_img", "mu_brain", "sd_img", "sd_brain"])#, hue="time")
    # g.map_upper(sns.scatterplot)
    # g.map_diag(sns.kdeplot)
    # g.map_lower(sns.kdeplot)
    # g.add_legend()

    # g = sns.PairGrid(data=df, y_vars=["mu_img", "mu_brain", "sd_img", "sd_brain"], x_vars=["time", "p", "exp"])
    # g.map(sns.boxplot)
    # g.add_legend()

    # sns.catplot(data=df, kind="box")#, hue="p")
    # sns.catplot(data=df, x="time", y="mu_img", hue="p", kind="box")
    # sns.relplot(x="sd_img", y="sd_brain", hue="p", data=df)

    if not n4:
        df_melt = df.melt(id_vars=["name", "exp", "time", "p"], value_vars=["mu_img", "mu_brain", "mu_saliv"])
        # df_melt = df.melt(id_vars=["name", "exp", "time", "p"], value_vars=["sd_img", "sd_brain", "sd_saliv"])
        fig, axes = plt.subplots(ncols=3)
        ax1, ax2, ax3 = axes.ravel()
        [ax.grid(axis="y") for ax in axes]
        sns.boxplot(data=df_melt, x="variable", y="value", hue="p", ax=ax1)
        sns.boxplot(data=df_melt, x="variable", y="value", hue="time", ax=ax2)
        sns.boxplot(data=df_melt, x="variable", y="value", hue="exp", ax=ax3)
        ax1.set_ylabel("SD of pixel intensity / a.u.")
        [ax.set_ylabel("") for ax in [ax2, ax3]]
        # plt.title("Pixel intensity average / sd for each image / ROI in image")
        # plt.show()
        plt.close()
        sns.violinplot(data=df_melt, x="variable", y="value")
        plt.ylabel("SD of pixel intensities / a.u.")
        plt.xlabel("")
        plt.grid(axis="y")
        plt.show()
    if n4:
        df_melt = df.melt(id_vars=["name", "exp", "time", "p", "n4"], value_vars=["mu_img", "mu_brain", "mu_saliv"])
        # df_melt = df.melt(id_vars=["name", "exp", "time", "p", "n4"], value_vars=["sd_img", "sd_brain", "sd_saliv"])
        fig, axes = plt.subplots(ncols=3)
        ax1, ax2, ax3 = axes.ravel()
        [ax.grid(axis="y") for ax in axes]
        sns.boxplot(data=df_melt, x="variable", y="value", hue="p", ax=ax1)
        sns.boxplot(data=df_melt, x="variable", y="value", hue="time", ax=ax2)
        sns.boxplot(data=df_melt, x="variable", y="value", hue="exp", ax=ax3)
        ax1.set_ylabel("SD of pixel intensity / a.u.")
        [ax.set_ylabel("") for ax in [ax2, ax3]]
        # plt.title("Pixel intensity average / sd for each image / ROI in image")
        # plt.show()
        plt.close()
        sns.violinplot(data=df_melt, x="variable", y="value", hue="n4")
        # plt.ylabel("SD of pixel intensities / a.u.")
        plt.ylabel("Mean of pixel intensities / a.u.")
        plt.xlabel("")
        plt.grid(axis="y")
        plt.show()
    return 0


def plot_hist_from_all(nbins=250, norm="none", n4=True, offsetIDX=0, pbool="all", t1bool=False, include_brain=False, savebool=False):
    fig, axes = plt.subplots(nrows=3) if include_brain else plt.subplots(nrows=2, figsize=(12, 8))
    FONTSIZE = 20
    if include_brain:
        ax1, ax2, ax3 = axes.ravel()    # change if more subplots
    else:
        ax1, ax3 = axes.ravel()
    pc1, pc2, s1, s2 = 2, 98, 1, 100

    if norm == "nyul mean decile":
        nrmny = nyul_normalizer(pbool=True, MASK_MODE="mean", n4bool=n4, L=np.arange(10, 100, 10), pc1=2, pc2=98, s1=1, s2=5e4)
        # nrmny = nyul_normalizer(pbool=True, MASK_MODE="mean", n4bool=n4, L=np.arange(10, 100, 10), pc1=1, pc2=99, s1=1, s2=256)
        # scale_p = nrmny.train_standard_histogram()
        scale_p = [1.00000000e+00, 1.39334273e+03, 3.15713912e+03, 4.79883128e+03, 6.38858544e+03, 8.25087320e+03,
                        1.07163858e+04, 1.37280698e+04, 1.79087290e+04, 2.53118737e+04, 5.00000000e+04]

        # scale_p = [1, 7.09238083, 13.97876528, 20.37835568, 26.55424457, 33.77192383, 43.348061, 55.03168975, 71.21403826, 99.78669316, 256]
        # nrmny = nyul_normalizer(pbool=False, MASK_MODE="mean", n4bool=True, L=np.arange(10, 100, 10), pc1=1, pc2=99, s1=1, s2=256)
        # scale_nop = nrmny.train_standard_histogram()
        # scale_nop = [1., 10.99693371, 22.66121363, 32.99404327, 41.51079168, 50.47239637, 62.91600329, 79.10789209, 99.73595561, 133.20188819, 256.]
        scale_nop = [1.00000000e+00, 2.09484671e+03, 4.82741201e+03, 7.24832438e+03, 9.24464907e+03, 1.13434589e+04, 1.42562717e+04, 1.80503356e+04, 2.28803051e+04, 3.07337163e+04, 5.00000000e+04]
    elif norm == "nyul brain decile":
        nrmny = nyul_normalizer(pbool=True, MASK_MODE="brain", n4bool=n4, L=[10, 20, 30, 40, 50, 60, 70, 80, 90], pc1=2, pc2=98, s1=1, s2=5e4)
        # nrmny = nyul_normalizer(pbool=True, threshmode="brain", n4bool=n4, L=[10, 20, 30, 40, 50, 60, 70, 80, 90], pc1=1, pc2=99, s1=1, s2=256)
        # scale_p = nrmny.train_standard_histogram()
        # scale_p = [1., 34.23755408, 47.37877586, 57.10493923, 65.77756881, 75.01711199, 86.0064134, 101.08750368, 120.76504172, 156.59937444, 256.]
        scale_p = [1.00000000e+00, 5.51478431e+03, 8.49028859e+03, 1.06895568e+04, 1.26494816e+04, 1.47360021e+04,
                         1.72168060e+04, 2.06242499e+04, 2.50680921e+04, 3.31180883e+04, 5.00000000e+04]
        # nrmny = nyul_normalizer(pbool=False, threshmode="brain", n4bool=True, L=[10, 20, 30, 40, 50, 60, 70, 80, 90], pc1=1, pc2=99, s1=1, s2=256)
        # scale_nop = nrmny.train_standard_histogram()
        # scale_nop = [1., 34.95178368, 46.19512203, 54.35308416, 61.94424755, 70.2827526, 80.98071465, 98.03055332, 121.14998526, 152.60979797, 256.]
        scale_nop = [1.00000000e+00, 5.48630905e+03, 8.11602435e+03, 1.00209666e+04, 1.17940187e+04, 1.37388780e+04, 1.62289771e+04, 2.01914164e+04, 2.55586951e+04, 3.28261892e+04, 5.00000000e+04]
    elif norm == "nyul brain mode":
        nrmny = nyul_normalizer(pbool=True, MASK_MODE="brain", n4bool=True, L=[], pc1=2, pc2=98, s1=1, s2=5e4)
        # scale_p = nrmny.train_standard_histogram(plot=True)
        scale_p = [1.0, 1.22860372e+04, 5.0e+04]

        # nrmny = nyul_normalizer(pbool=False, MASK_MODE="brain", n4bool=True, L=[], pc1=2, pc2=98, s1=1, s2=5e4)
        # scale_nop = nrmny.train_standard_histogram(verbose=True, plot=True)
        scale_nop = [1.0, 1.10861289e+04, 5.0e+04]

    elif norm == "nyul otsu decile":
        nrmny = nyul_normalizer(pbool=True, MASK_MODE="otsu", n4bool=n4, L=np.arange(10, 100, 10), pc1=2,
                                            pc2=98, s1=1, s2=5e4)
        # scale_otsu_decile_p = nrmny_otsu_decile.train_standard_histogram()
        # scale_otsu_decile_p = [1.00000000e+00, 2.84130128e+03, 5.04186555e+03, 7.24607624e+03, 9.48030195e+03, 1.16017962e+04, 1.38711264e+04, 1.70069385e+04, 2.12181681e+04, 2.83314151e+04, 5.00000000e+04]
        # nrmny_otsu_decile = nyul_normalizer(pbool=False, MASK_MODE="otsu", n4bool=n4, L=np.arange(10, 100, 10), pc1=2, pc2=98, s1=1, s2=5e4)
        # scale_otsu_decile_nop = nrmny_otsu_decile.train_standard_histogram()
        scale_p_T2 = nyul_scales["otsu decile p"]
        scale_nop_T2 = nyul_scales["otsu decile nop"]
        scale_p_T1 = nyul_scales_T1["otsu decile p"]
        scale_nop_T1 = nyul_scales_T1["otsu decile nop"]

    first_im = True
    brainbool = include_brain
    # for exp in ["Pilot1", "Pilot2"]:
    j = 1
    for exp in find_folders(RawDir, condition="Pilot"):
        for time in os.listdir(os.path.join(RawDir, exp)):
            print(exp, time)
            datadir = os.path.normpath(os.path.join(RawDir, exp, time))
            for i, folder in enumerate(find_folders(datadir, "sagittal")):
                name = get_name(exp, folder, "sagittal")

                includebool = True      # change this to exclude e.g. p

                if pbool == False:
                    includebool = not("p" in name)
                elif pbool == True:
                    includebool = "p" in name
                elif pbool == "all":
                    includebool = True
                else:
                    print("pbool", pbool, "not valid...")
                    return 0

                if includebool:
                    if t1bool == "all":
                        pass
                    else:
                        includebool = "T1" in name if t1bool else not("T1") in name

                if includebool:
                    print(j, exp, time, name)
                    j += 1
                    t1bool_curr = "T1" in name
                    pbool_curr = "p" in name
                    # idx_center = central_idx_dict[time + name]
                    indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
                    MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder), printbool=False)
                    ROI_saliv = np.load(os.path.join(SegSalivDir, exp.lower(), time, name + ".npy"))
                    for X in ["L", "R"]:
                        # print(j, exp, time, name + "_" + X)
                        # j += 1
                        idx = left_idx_dict[time+name] if X=="L" else right_idx_dict[time+name]
                        # im_raw = MATR_RAW[idx_center + offsetIDX]
                        # roi_saliv = ROI_saliv[idx_center + offsetIDX]
                        im_raw = MATR_RAW[idx]
                        roi_saliv = ROI_saliv[idx]

                        if include_brain:
                            try:
                                ROI_brain = np.load(os.path.join(SegBrainDir, exp.lower(), time, name + ".npy"))
                                roi_brain = ROI_brain[idx_center + offsetIDX]   #todo: fix LR if brain included
                                brainbool = True
                            except Exception as e:
                                # print(*e.args)
                                brainbool = False
                        # N4 BEFORE NORM: SEE Palumbo et al. (2011)
                        if n4:
                            im, _, _ = n4correction(im_raw, verbose=True)
                            # im_n4corr, _, _ = n4correction(im_raw)
                            # im = im_raw
                        else:
                            im = im_raw

                        if norm == "stscore image":
                            im = mean_centering(im, mode="image", n=3)
                        elif norm == "stscore ROI brain":
                            im = mean_centering(im, ROI=roi_brain, mode="roi", n=3)
                        elif norm == "none" or norm == "raw":
                            pass
                        elif norm == "stscore ROI saliv":
                            im = mean_centering(im, ROI=roi_saliv, mode="roi", n=3)
                        # elif norm == "nyul brain":
                        elif "nyul brain" in norm:
                            nrmny.SCALE_STD = scale_p if "p" in name else scale_nop
                            im = nrmny.transform_image(image=im, experiment=exp, time=time, name=name, verbose=True)
                        elif norm == "nyul mean decile":
                            # nrmny = nyul_normalizer(pbool=False, threshmode="mean", n4bool=True, L=np.arange(10, 100, 10), pc1=1, pc2=99, s1=1, s2=256)
                            nrmny.SCALE_STD = scale_p if "p" in name else scale_nop
                            im = nrmny.transform_image(image=im, experiment=exp, time=time, name=name, verbose=True)
                        elif norm == "nyul otsu decile":
                            if t1bool_curr:
                                nrmny.SCALE_STD = scale_p_T1 if pbool_curr else scale_nop_T1
                            else:
                                nrmny.SCALE_STD = scale_p_T2 if pbool_curr else scale_nop_T2
                            im = nrmny.transform_image(image=im, experiment=exp, time=time, name=name, verbose=True)

                        else:
                            print("NORM", norm, "NOT IMPLEMENTED..")
                            return 0

                        pxvals = im.flatten()
                        kde = stats.gaussian_kde(pxvals)
                        # xx = np.linspace(min(pxvals), max(pxvals), 500)
                        xmax = np.percentile(pxvals, 98)
                        xx = np.linspace(min(pxvals), xmax, 500)
                        ax1.plot(xx, kde(xx), color="orange", lw=.5, label="Raw" if first_im else "")
                        # pxvals_corr = im_n4corr.flatten()
                        # kde = stats.gaussian_kde(pxvals_corr)
                        # xx = np.linspace(min(pxvals_corr), max(pxvals_corr), 500)
                        # ax1.plot(xx, kde(xx), color="green", lw=.5, label="N4 corr" if first_im else "")
                        ax1.set_ylabel("KDE IMG", fontsize=FONTSIZE)

                        # ROI BRAIN - ONLY RUN IF ROI BRIAN EXISTS
                        if brainbool and include_brain:
                            pxvals = im[roi_brain != 0].flatten()
                            kde = stats.gaussian_kde(pxvals)
                            # xx = np.linspace(min(pxvals), max(pxvals), 500)
                            xmax = np.percentile(pxvals, 98)
                            xx = np.linspace(min(pxvals), xmax, 500)
                            ax2.plot(xx, kde(xx), color="orange", lw=.5, label="Raw" if first_im else "")
                            # pxvals_corr = im_n4corr[roi_brain != 0].flatten()
                            # kde = stats.gaussian_kde(pxvals_corr)
                            # xx = np.linspace(min(pxvals_corr), max(pxvals_corr), 500)
                            # ax2.plot(xx, kde(xx), color="green", lw=.5, label="N4 corr" if first_im else "")
                            ax2.set_ylabel("kde ROI brain")
                        else:
                            pass
                        # ROI SALIV
                        pxvals = im[roi_saliv != 0].flatten()
                        kde = stats.gaussian_kde(pxvals)
                        # xx = np.linspace(min(pxvals), max(pxvals), 500)
                        xmax = np.percentile(pxvals, 98)
                        xx = np.linspace(min(pxvals), xmax, 500)
                        ax3.plot(xx, kde(xx), color="orange", lw=.5, label="Raw" if first_im else "")
                        # pxvals_corr = im_n4corr[roi_saliv != 0].flatten()
                        # kde = stats.gaussian_kde(pxvals_corr)
                        # xx = np.linspace(min(pxvals_corr), max(pxvals_corr), 500)
                        # ax3.plot(xx, kde(xx), color="green", lw=.5, label="N4 corr" if first_im else "")
                        ax3.set_ylabel("KDE ROI", fontsize=FONTSIZE)
                        # ax3.set_title("KDE ROI SMG L + R")
                        first_im = False
                    # if j > 1:
                    #     break
                # if j > 1:
                #    break
    # ax2.legend()
    for ax in axes:
        ax.set_xlabel("px intensity / a.u.", fontsize=FONTSIZE)
        ax.grid(1)
        # ax.set_ylabel("kde")
    norm_description = {"raw":"no", "no norm":"no", "none":"no", "nyul otsu decile":"nyul (decile)", "nyul otsu mode":"nyul (mode)", "stscore image":"z-score"}
    fig.suptitle(f"Kernel density estimate for {'''T1''' if t1bool else '''T2'''} images (N={j-1} instances, L+R slice per instance)\n"
                 f"{''',''' if not n4 else ''' after N4 correction and'''} {norm_description[norm]} normalization.", fontsize=FONTSIZE)

    # if norm == "none" or "raw":
    #     fig.suptitle(f"Kernel Density Estimator for {'''raw''' if not n4 else '''N4 corrected raw'''} values from whole image, ROI brain, ROI salivary gland.")
    # if norm == "stscore image":
    #     fig.suptitle(f"Kernel Density Estimator for st. score normalized values ($\mu_j$, $\sigma_j$ from whole image $I_j$) {'''''' if not n4 else '''with N4 correction'''} from: whole image, ROI brain, ROI salivary gland")
    # if norm == "stscore ROI brain":
    #     fig.suptitle(
    #         f"Kernel Density Estimator for st. score normalized values ($\mu_j$, $\sigma_j$ from image values within ROI_brain $I_j$[ROI != 0]) {'''''' if not n4 else '''with N4 correction'''} from: whole image, ROI brain, ROI salivary gland")
    # if norm == "stscore ROI saliv":
    #     fig.suptitle(
    #         f"Kernel Density Estimator for st. score normalized values ($\mu_j$, $\sigma_j$ from image values within ROI_saliv $I_j$[ROI != 0]) {'''''' if not n4 else '''with N4 correction'''} from: whole image, ROI brain, ROI salivary gland")
    # if "nyul brain" in norm:
    #     fig.suptitle(
    #         f"Kernel Density Estimator for {norm.upper()} normalized values ($\mu_j$, $\sigma_j$ from image values within ROI BRAIN) {'''''' if not n4 else '''with N4 correction'''} from: whole image, ROI brain, ROI salivary gland")
    # if norm == "nyul mean decile":
    #     fig.suptitle(
    #         f"Kernel Density Estimator for NYUL_MEAN DECILE normalized values ($\mu_j$, $\sigma_j$ from image values within MEAN THRESH) {'''''' if not n4 else '''with N4 correction'''} from: whole image, ROI brain, ROI salivary gland")
    # if norm == "nyul brain mode":
    if savebool:
        savepath = os.path.join(PlotDir, f"preprocessing\KDE hist\kde_{'''T1''' if t1bool else '''T2'''}_{'''n4''' if n4 else '''NOn4'''}_{norm}.png")
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()
    return 0


def compare_norm_methods(normmodes=["raw"], n4=True, stscore_first=False, offsetIDX=0, show_sample=False, save=False):
    pc1, pc2, s1, s2 = 2, 98, 1, 100
    if stscore_first:
        print("PLEASE DONT USE THIS, NOT UPDATED WITH FUNCTION")
        return 0
    if "nyul brain decile" in normmodes:
        if not stscore_first:
            # TRAIN NYUL NORM
            # nrmny_brain_decile = nyul_normalizer(pbool=True, MASK_MODE="brain", n4bool=n4, L=np.arange(10, 100, 10), pc1=1, pc2=99, s1=1, s2=256)
            # scale_brain_p = [1., 34.23755408, 47.37877586, 57.10493923, 65.77756881, 75.01711199, 86.0064134, 101.08750368, 120.76504172, 156.59937444, 256.]
            # nrmny_brain_decile = nyul_normalizer(pbool=False, MASK_MODE="brain", n4bool=n4, L=np.arange(10, 100, 10), pc1=1, pc2=99, s1=1, s2=256)
            # scale_brain_nop = [1., 34.95178368, 46.19512203, 54.35308416, 61.94424755, 70.2827526, 80.98071465, 98.03055332, 121.14998526, 152.60979797, 256.]
            nrmny_brain_decile = nyul_normalizer(pbool=True, MASK_MODE="brain", n4bool=n4, L=np.arange(10, 100, 10), pc1=2, pc2=98, s1=1, s2=5e4)
            # scale_brain_p = nrmny_brain_decile.train_standard_histogram()
            # print("NYUL BRAIN DECILE SCLAE P")
            # print(scale_brain_p)
            scale_brain_p = [1.00000000e+00, 5.51478431e+03, 8.49028859e+03, 1.06895568e+04, 1.26494816e+04, 1.47360021e+04, 1.72168060e+04, 2.06242499e+04, 2.50680921e+04, 3.31180883e+04, 5.00000000e+04]

            nrmny_brain_decile = nyul_normalizer(pbool=False, MASK_MODE="brain", n4bool=n4, L=np.arange(10, 100, 10), pc1=2, pc2=98, s1=1, s2=5e4)
            # scale_brain_nop = nrmny_brain_decile.train_standard_histogram()
            # print("NYUL BRAIN DECILE SCALE NO P:")
            # print(scale_brain_nop)
            scale_brain_nop = [1.00000000e+00, 5.48630905e+03, 8.11602435e+03, 1.00209666e+04, 1.17940187e+04, 1.37388780e+04, 1.62289771e+04, 2.01914164e+04, 2.55586951e+04, 3.28261892e+04, 5.00000000e+04]
        else:
            nrmny_brain_decile = nyul_normalizer(pbool=True, MASK_MODE="brain", n4bool=n4, L=np.arange(10, 100, 10), pc1=pc1,
                                          pc2=pc2, s1=s1, s2=s2)
            # scale_brain_p = nrmny_brain_decile.train_standard_histogram(stscore_first=True)
            scale_brain_p = nyul_scales_stsc["brain_decile_p"]
            nrmny_brain_decile = nyul_normalizer(pbool=False, MASK_MODE="brain", n4bool=n4, L=np.arange(10, 100, 10), pc1=pc1,
                                          pc2=pc2, s1=s1, s2=s2)
            # scale_brain_nop = nrmny_brain_decile.train_standard_histogram(stscore_first=True)
            scale_brain_nop = nyul_scales_stsc["brain_decile_nop"]

    if "nyul brain mode" in normmodes:
        if not stscore_first:
            nrmny_brain_mode = nyul_normalizer(pbool=True, MASK_MODE="brain", n4bool=True, L=[], pc1=2, pc2=98, s1=1, s2=5e4)
            # scale_brain_mode_p = nrmny_brain_mode.train_standard_histogram(plot=True)
            scale_brain_mode_p = [1.0, 1.22860372e+04, 5.0e+04]

            # nrmny_brain_mode = nyul_normalizer(pbool=False, MASK_MODE="brain", n4bool=True, L=[], pc1=2, pc2=98, s1=1, s2=5e4)
            # scale_brain_mode_nop = nrmny_brain_mode.train_standard_histogram(verbose=True, plot=True)
            scale_brain_mode_nop = [1.0, 1.10861289e+04, 5.0e+04]
        else:
            nrmny_brain_mode = nyul_normalizer(pbool=True, MASK_MODE="brain", n4bool=n4, L=[], pc1=pc1,
                                          pc2=pc2, s1=s1, s2=s2)
            # scale_brain_mode_p = nrmny_brain_mode.train_standard_histogram(stscore_first=True)
            scale_brain_mode_p = nyul_scales_stsc["brain_mode_p"]
            nrmny_brain_mode = nyul_normalizer(pbool=False, MASK_MODE="brain", n4bool=n4, L=[], pc1=pc1,
                                          pc2=pc2, s1=s1, s2=s2)
            # scale_brain_mode_nop = nrmny_brain_mode.train_standard_histogram(stscore_first=True)
            scale_brain_mode_nop = nyul_scales_stsc["brain_mode_nop"]

    if "nyul mean decile" in normmodes:
        if not stscore_first:
            nrmny_mean = nyul_normalizer(pbool=True, MASK_MODE="mean", n4bool=n4, L=np.arange(10, 100, 10), pc1=2, pc2=98, s1=1, s2=5e4)
            # scale_mean_p = nrmny_mean.train_standard_histogram()
            # print("NYUL MEAN DECILE SCALE MEAN P:")
            # print(scale_mean_p)
            scale_mean_p = [1.00000000e+00, 1.39334273e+03, 3.15713912e+03, 4.79883128e+03, 6.38858544e+03, 8.25087320e+03, 1.07163858e+04, 1.37280698e+04, 1.79087290e+04, 2.53118737e+04, 5.00000000e+04]
            # nrmny_mean = nyul_normalizer(pbool=True, MASK_MODE="mean", n4bool=n4, L=np.arange(10, 100, 10), pc1=1, pc2=99, s1=1, s2=256)
            # scale_mean_p = [1, 7.09238083, 13.97876528, 20.37835568, 26.55424457, 33.77192383, 43.348061, 55.03168975, 71.21403826, 99.78669316, 256]
            # nrmny_mean = nyul_normalizer(pbool=False, MASK_MODE="mean", n4bool=n4, L=np.arange(10, 100, 10), pc1=1, pc2=99, s1=1, s2=256)
            # scale_mean_nop = [1., 10.99693371, 22.66121363, 32.99404327, 41.51079168, 50.47239637, 62.91600329, 79.10789209, 99.73595561, 133.20188819, 256.]
            nrmny_mean = nyul_normalizer(pbool=False, MASK_MODE="mean", n4bool=n4, L=np.arange(10, 100, 10), pc1=2, pc2=98, s1=1, s2=5e4)
            # scale_mean_nop = nrmny_mean.train_standard_histogram()
            # print("NYUL MEAN DECILE SCALE MEAN NO P:")
            # print(scale_mean_nop)
            scale_mean_nop = [1.00000000e+00, 2.09484671e+03, 4.82741201e+03, 7.24832438e+03, 9.24464907e+03, 1.13434589e+04, 1.42562717e+04, 1.80503356e+04, 2.28803051e+04, 3.07337163e+04, 5.00000000e+04]
        else:
            nrmny_mean = nyul_normalizer(pbool=True, MASK_MODE="mean", n4bool=n4, L=np.arange(10, 100, 10), pc1=pc1, pc2=pc2, s1=s1, s2=s2)
            # scale_mean_p = nrmny_mean.train_standard_histogram(stscore_first=True)
            scale_mean_p = nyul_scales_stsc["mean_decile_p"]
            nrmny_mean = nyul_normalizer(pbool=False, MASK_MODE="mean", n4bool=n4, L=np.arange(10, 100, 10), pc1=pc1, pc2=pc2, s1=s1, s2=s2)
            # scale_mean_nop = nrmny_mean.train_standard_histogram(stscore_first=True)
            scale_mean_nop = nyul_scales_stsc["mean_decile_nop"]

    if "nyul otsu decile" in normmodes:
        if not stscore_first:
            nrmny_otsu_decile = nyul_initializer(norm="nyul otsu decile", n4=n4)
            # nrmny_otsu_decile = nyul_normalizer(pbool=True, MASK_MODE="otsu", n4bool=n4, L=np.arange(10, 100, 10), pc1=2, pc2=98, s1=1, s2=5e4)
            # scale_otsu_decile_p = nrmny_otsu_decile.train_standard_histogram()
            # scale_otsu_decile_p = [1.00000000e+00, 2.84130128e+03, 5.04186555e+03, 7.24607624e+03, 9.48030195e+03, 1.16017962e+04, 1.38711264e+04, 1.70069385e+04, 2.12181681e+04, 2.83314151e+04, 5.00000000e+04]
            # nrmny_otsu_decile = nyul_normalizer(pbool=False, MASK_MODE="otsu", n4bool=n4, L=np.arange(10, 100, 10), pc1=2, pc2=98, s1=1, s2=5e4)
            # scale_otsu_decile_nop = nrmny_otsu_decile.train_standard_histogram()
            scale_otsu_decile_p_T2 = nyul_scales["otsu decile p"]
            scale_otsu_decile_nop_T2 = nyul_scales["otsu decile nop"]
            scale_otsu_decile_p_T1 = nyul_scales_T1["otsu decile p"]
            scale_otsu_decile_nop_T1 = nyul_scales_T1["otsu decile nop"]

        else:
            nrmny_otsu_decile = nyul_normalizer(pbool=True, MASK_MODE="otsu", n4bool=n4, L=np.arange(10, 100, 10), pc1=pc1,
                                          pc2=pc2, s1=s1, s2=s2)
            # scale_otsu_decile_p = nrmny_otsu_decile.train_standard_histogram()
            scale_otsu_decile_p = nyul_scales_stsc["otsu_decile_p"]
            nrmny_otsu_decile = nyul_normalizer(pbool=False, MASK_MODE="otsu", n4bool=n4, L=np.arange(10, 100, 10), pc1=pc1,
                                          pc2=pc2, s1=s1, s2=s2)
            # scale_otsu_decile_nop = nrmny_otsu_decile.train_standard_histogram()
            scale_otsu_decile_nop = nyul_scales_stsc["otsu_decile_nop"]

    if "nyul otsu mode" in normmodes:
        if stscore_first:
            nrmny_otsu_mode = nyul_normalizer(pbool=True, MASK_MODE="otsu", n4bool=n4, L=[], pc1=pc1,
                                          pc2=pc2, s1=s1, s2=s2)
            # scale_otsu_mode_p = nrmny_otsu_mode.train_standard_histogram()
            scale_otsu_mode_p = nyul_scales_stsc["otsu_mode_p"]
            nrmny_otsu_mode = nyul_normalizer(pbool=False, MASK_MODE="otsu", n4bool=n4, L=[], pc1=pc1,
                                          pc2=pc2, s1=s1, s2=s2)
            # scale_otsu_mode_nop = nrmny_otsu_mode.train_standard_histogram()
            scale_otsu_mode_nop = nyul_scales_stsc["otsu_mode_nop"]
        else:

            nrmny_otsu_mode = nyul_normalizer(pbool=True, MASK_MODE="otsu", n4bool=n4, L=[], pc1=2, pc2=98, s1=1, s2=5e4)
            # scale_otsu_mode_p = nrmny_otsu_decile.train_standard_histogram()
            scale_otsu_mode_p = [1.00000000e+00, 2.84130128e+03, 5.04186555e+03, 7.24607624e+03, 9.48030195e+03, 1.16017962e+04, 1.38711264e+04, 1.70069385e+04, 2.12181681e+04, 2.83314151e+04, 5.00000000e+04]
            nrmny_otsu_mode = nyul_normalizer(pbool=False, MASK_MODE="otsu", n4bool=n4, L=[], pc1=2, pc2=98, s1=1, s2=5e4)
            scale_otsu_mode_nop = nrmny_otsu_mode.train_standard_histogram()


    # df = pd.DataFrame({"name":[], "exp":[], "time":[], "p":[],  "weight":[], "n4":[], "norm":[], "avg im":[], "median im":[], "sd im":[], "avg brain":[], "median brain":[], "sd brain":[],
    #                     "avg saliv":[], "median saliv":[], "sd saliv":[]})   # DF for NORMALIZED DATA
    df = pd.DataFrame({"name":[], "exp":[], "time":[], "p":[],  "weight":[], "n4":[], "norm":[], "avg im":[], "median im":[],
                       "sd im":[], "avg saliv":[], "median saliv":[], "sd saliv":[]})   # DF for NORMALIZED DATA


    df2 = pd.DataFrame({"name":[], "exp":[], "time":[], "p":[], "weight":[], "n4":[], "norm":[], "roi":[], "perc":[], "value":[]}) # CV PROFILE - calculate PERCENTILE values for img, brain, saliv then COMPARE NORM METHODS ACROSS PERCENTILES --> AUC??


    # for experiment in find_folders(RawDir, condition="Pilot"):    # TODO: CHANGE TO THIS WHEN PILOT3 IS SEGMENTED
    j = 1
    # for experiment in ["Pilot1", "Pilot2", "Pilot3"]:
    # for experiment in ["Pilot3", "Pilot1", "Pilot2"]:
    for experiment in find_folders(RawDir, "Pilot"):
        for time in find_folders(os.path.join(RawDir, experiment)):
            print("\n", experiment, time)
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            for folder in find_folders(os.path.join(RawDir, experiment, time), condition="sagittal"):
                name = get_name(experiment, folder, condition="sagittal")
                print("j=", j, experiment, time, name)
                j += 1
                pbool = "p" in name
                t1bool = "T1" in name

                idx_center = central_idx_dict[time + name]
                indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
                MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder), printbool=False)
                ROI_saliv = np.load(os.path.join(SegSalivDir, experiment.lower(), time, name + ".npy"))
                # ROI_brain = np.load(os.path.join(SegBrainDir, experiment.lower(), time, name + ".npy"))

                for X in ["L", "R"]:
                    idx = left_idx_dict[time + name] if X=="L" else right_idx_dict[time + name]
                    im_raw = MATR_RAW[idx]
                    roi_saliv = ROI_saliv[idx]

                    if n4:
                        im = n4correction(im_raw)[0]
                    else:
                        im = im_raw

                    if stscore_first:
                        im = mean_centering(im)
                        print("---- standardizing image -----")
                    else:
                        pass

                    if show_sample:
                        # fig, ax = plt.subplots(ncols=2, nrows=len(normmodes))
                        # fig, ax = plt.subplots(nrows=2, ncols=len(normmodes));  ax=ax.T
                        fig, ax = plt.subplots(nrows=2, ncols=4);  ax=ax.ravel()
                        fig.tight_layout()


                    for n, norm in enumerate(normmodes):
                        print(n, norm)
                        if norm == "raw":
                            im_norm = im
                        elif norm == "stscore image":
                            im_norm = mean_centering(im, n=3)
                        # elif norm == "stscore brain":
                        #     im_norm = mean_centering(im, n=0, mode="ROI", ROI=roi_brain)
                        # elif norm == "nyul mean" or norm == "nyul mean decile":
                        #     nrmny_mean.SCALE_STD = scale_mean_p if pbool else scale_mean_nop
                        #     im_norm = nrmny_mean.transform_image(im)
                        # elif norm == "nyul brain decile":# or norm == "nyul brain":
                        #     nrmny_brain_decile.SCALE_STD = scale_brain_p if pbool else scale_brain_nop
                        #     im_norm = nrmny_brain_decile.transform_image(im, ROI=roi_brain)
                        # elif norm == "nyul brain mode":
                        #     nrmny_brain_mode.SCALE_STD = scale_brain_mode_p if pbool else scale_brain_mode_nop
                        #     im_norm = nrmny_brain_mode.transform_image(im, ROI=roi_brain)
                        elif norm == "nyul otsu decile":
                            if not t1bool:
                                nrmny_otsu_decile.SCALE_STD = scale_otsu_decile_p_T2 if pbool else scale_otsu_decile_nop_T2
                            else:
                                nrmny_otsu_decile.SCALE_STD = scale_otsu_decile_p_T1 if pbool else scale_otsu_decile_nop_T1
                            im_norm = nrmny_otsu_decile.transform_image(im)
                        # elif norm == "nyul otsu mode":
                        #     nrmny_otsu_mode.SCALE_STD = scale_otsu_mode_p if pbool else scale_otsu_mode_nop
                        #     im_norm = nrmny_otsu_mode.transform_image(im)
                        else:
                            print("NORM MODE", norm, "NOT IMPLEMENTED....")
                            return 0

                        vals_im = im_norm.ravel()
                        # vals_brain = im_norm[roi_brain != 0].ravel()
                        vals_saliv = im_norm[roi_saliv != 0].ravel()

                        # df = df.append({"name":name, "exp":experiment, "time":time, "p":pbool, "n4":n4, "norm":norm,
                        #     "avg im":np.average(vals_im), "median im":np.median(vals_im), "sd im":np.std(vals_im), "avg brain":np.average(vals_brain),
                        #     "median brain":np.median(vals_brain), "sd brain":np.std(vals_brain), "avg saliv":np.average(vals_saliv),
                        #         "median saliv":np.median(vals_saliv), "sd saliv":np.std(vals_saliv)}, ignore_index=True)
                        weight = "T1" if t1bool else "T2"
                        namevar = name + "_" + X
                        df = df.append({"name":namevar, "exp":experiment, "time":time, "weight":weight, "p":pbool, "n4":n4, "norm":norm,
                            "avg im":np.average(vals_im), "median im":np.median(vals_im), "sd im":np.std(vals_im), "avg saliv":np.average(vals_saliv),
                                "median saliv":np.median(vals_saliv), "sd saliv":np.std(vals_saliv)}, ignore_index=True)

                        perc = np.arange(10, 100, 10)
                        pcvals_img = np.percentile(vals_im, perc)
                        # pcvals_brain = np.percentile(vals_brain, perc)
                        pcvals_saliv = np.percentile(vals_saliv, perc)
                        # print(pcvals_img)
                        df2 = df2.append({"name":namevar, "exp":experiment, "time":time, "weight":weight, "p":pbool, "n4":n4, "norm":norm, "roi":"img", "perc":perc, "value":pcvals_img}, ignore_index=True)
                        # df2 = df2.append({"name":name, "exp":experiment, "time":time, "p":pbool, "n4":n4, "norm":norm, "roi":"brain", "perc":perc, "value":pcvals_brain}, ignore_index=True)
                        df2 = df2.append({"name":namevar, "exp":experiment, "time":time, "weight":weight, "p":pbool, "n4":n4, "norm":norm, "roi":"saliv", "perc":perc, "value":pcvals_saliv}, ignore_index=True)
                        # print(df2)
                        if show_sample:
                            ax[n].imshow(im_norm, cmap="hot")
                            ax[n].axis("off")
                            ax[n].set_title(norm)
                            # ax[n, 0].imshow(im_norm, cmap="hot")
                            # ax[n, 0].axis("off")
                            # ax[n, 0].set_title(norm)
                            # ax[n, 1].hist(im_norm.ravel(), bins=256)
                # print("bruh")
                plt.show() if show_sample else 0
    if save:
        pth1 = os.path.join(os.getcwd(), "..", "..", r"Preprocessing\norm_data")
        pth1 += "_stscore_first.csv"  if stscore_first else ".csv"
        df.to_csv(path_or_buf=pth1)
        pth2 = os.path.join(os.getcwd(), "..", "..", "Preprocessing\cv_profile_data")
        pth2 += "_stscore_first.csv" if stscore_first else ".csv"
        df2.to_csv(path_or_buf=pth2)    # PERCENTILE VALUE DATA FOR VARIOUS ROI, NORM -- USED BY plot_cv_profile_norm_comparison

    # df_melt = df.melt(id_vars=["name", "exp", "time", "p", "n4", "norm"], value_vars=["avg im", "median im", "sd im", "avg brain", "median brain", "sd brain", "avg saliv", "median saliv", "sd saliv"])
    df_melt = df.melt(id_vars=["name", "exp", "time", "weight", "p", "n4", "norm"], value_vars=["avg im", "median im", "sd im", "avg saliv", "median saliv", "sd saliv"])
    g = sns.FacetGrid(df_melt, row="norm", sharey=False)#, hue="p")
    print(df_melt.columns)
    g.map(sns.boxplot, "variable", "value")#, hue="p")
    # g.add_legend()
    # g = sns.catplot(data=df_melt, x="variable", y="value", row="norm")
    for ax, norm in zip(g.axes.ravel(), normmodes):
        min, max = np.min(df_melt.loc[df_melt["norm"] == norm]["value"].values), np.max(df_melt.loc[df_melt["norm"] == norm]["value"].values)
        ax.set_ylim(bottom=min, top=max)
        ax.set_ylabel("a.u.")
        ax.set_xlabel("")
    # plt.show()
    plt.close()
    return 1


def plot_cv_profile_norm_comparison(stscore_first=False, qcd_bool=False):
    # qcd_bool = True: use quartile coeff of dispersion instead of CV
    pth = os.path.join(os.path.join(os.getcwd(), "..", "..", "Preprocessing\cv_profile_data"))
    pth += "_stscore_first.csv" if stscore_first else ".csv"
    # df = pd.read_csv(os.path.join(os.getcwd(), "..", "..", "Preprocessing\cv_profile_data.csv"), index_col=0)
    df = pd.read_csv(pth, index_col=0)
    print(df.columns)
    percentiles = [int(x) for x in df["perc"].values[0][1:-1].split(" ")]
    # print((max(percentiles) - min(percentiles)) / (len(percentiles) - 1))

    print(set(df["roi"].values))
    print(set(df["norm"].values))
    num_norm = len(set(df["norm"].values))
    df_im = pd.DataFrame({"norm":list(set(df["norm"].values)), "prcvals":np.array([{i:[] for i in percentiles} for i in range(num_norm)]), "count":[0]*len(set(df["norm"]))})#, dtype="float64")
    # df_brain = pd.DataFrame({"norm":list(set(df["norm"].values)), "prcvals":np.array([{i:[] for i in percentiles} for i in range(num_norm)]), "count":[0]*len(set(df["norm"]))})#, dtype="float64")
    df_saliv = pd.DataFrame({"norm":list(set(df["norm"].values)), "prcvals":np.array([{i:[] for i in percentiles} for i in range(num_norm)]), "count":[0]*len(set(df["norm"]))})#, dtype="float64")
    print(df_im)
    # TODO: PLOT CV_prc = st(pval given mode, roi) / mean(pval given mode, roi)     FOR EACH percentile value (then compare norm modes in plot, plot for each ROI)
    for idx, row in six.iteritems(df.T):
        pcvals = np.array([float(x) for x in row["value"][1:-1].replace("\n", "").split(" ") if x])
        # print(len(pcvals), pcvals)
        norm = row["norm"]
        roi = row["roi"]
        # print(row["roi"], row["norm"])
        print(idx, norm) if not idx % 100 else 0
        if roi == "img":
            df_im.loc[df_im["norm"] == norm, "count"] += 1
            for pc, p in zip(percentiles, pcvals):
                df_im.loc[df_im["norm"] == norm, "prcvals"].values[0][pc].append(p)
        # elif roi == "brain":
        #     df_brain.loc[df_brain["norm"] == norm, "count"] += 1
        #     for pc, p in zip(percentiles, pcvals):
        #         df_brain.loc[df_brain["norm"] == norm, "prcvals"].values[0][pc].append(p)
        elif roi == "saliv":
            df_saliv.loc[df_saliv["norm"] == norm, "count"] += 1
            for pc, p in zip(percentiles, pcvals):
                df_saliv.loc[df_saliv["norm"] == norm, "prcvals"].values[0][pc].append(p)
        else:
            print("ROI", roi, "NOT VALID, BRO")
            return 0
        # if idx > 100:
        #     break
    print()
    print(df_im)
    # print(df_brain)
    print(df_saliv)
    # print(df_im.loc[df_im["norm"] == "stscore image"])
    # fig, ax = plt.subplots(1, 3)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"raw":"blue", "stscore image":"green", "nyul otsu decile":"orange"}
    for norm in list(set(df["norm"].values)):
        cv_img = []
        cv_brain = []
        cv_saliv = []
        for pc in percentiles:
            # print(norm, pc)
            vals_im = df_im.loc[df_im["norm"] == norm, "prcvals"].values[0][pc]  # ROI, NORM, all vals_im_im for percentile pc
            # vals_brain = df_brain.loc[df_brain["norm"] == norm, "prcvals"].values[0][pc]  # ROI, NORM, all vals_im_im for percentile pc
            vals_saliv = df_saliv.loc[df_saliv["norm"] == norm, "prcvals"].values[0][pc]  # ROI, NORM, all vals_im_im for percentile pc
            if not qcd_bool:
                cv_img.append(abs(np.std(vals_im) / np.mean(vals_im)))
                # cv_brain.append(abs(np.std(vals_brain) / np.mean(vals_brain)))
                cv_saliv.append(abs(np.std(vals_saliv) / np.mean(vals_saliv)))
            else:
                cv_img.append(abs(qcd(vals_im)))
                # cv_brain.append(abs(qcd(vals_brain)))
                cv_saliv.append(abs(qcd(vals_saliv)))
            # cv_img.append(abs(np.std(vals_im) / np.mean(vals_im)))
            # cv_brain.append(abs(np.std(vals_brain) / np.mean(vals_brain)))
            # cv_saliv.append(abs(np.std(vals_saliv) / np.mean(vals_saliv)))
        auc_img = simps(np.abs(cv_img), dx=(max(percentiles) - min(percentiles)) / (len(percentiles) - 1))
        # auc_brain = simps(np.abs(cv_brain), dx=(max(percentiles) - min(percentiles)) / (len(percentiles) - 1))
        auc_saliv = simps(np.abs(cv_saliv), dx=(max(percentiles) - min(percentiles)) / (len(percentiles) - 1))
        ax[0].plot(percentiles, cv_img, "--", label=norm + f", auc={auc_img:.2g}", color=colors[norm]);  ax[0].set_title("Whole image")
        # ax[1].plot(percentiles, cv_brain, "--", label=norm + f", auc={auc_brain:.2g}");    ax[1].set_title("ROI brain")
        ax[1].plot(percentiles, cv_saliv, "--", label=norm + f", auc={auc_saliv:.2g}", color=colors[norm]);    ax[1].set_title("ROI saliv")
    for axx in ax:
        axx.legend()
        axx.grid()
        axx.set_xlabel("Percentile")
    ax[0].set_ylabel("$CV_{perc}$")
    fig.suptitle(f"Comparing normalization modes for {'''CV_perc''' if not qcd_bool else '''QCD_perc'''} at various image ROI's")
    fig.tight_layout()
    plt.show()
    return 1


def cv_compare_norm(stscore_first=False):
    pth = os.path.join(os.getcwd(), "..", "..", r"Preprocessing\norm_data")
    pth += "_stscore_first.csv" if stscore_first else ".csv"
    # df = pd.read_csv(os.path.join(os.getcwd(), "..", "..", r"Preprocessing\norm_data.csv"), index_col=0)
    df = pd.read_csv(pth, index_col=0)
    norms = list(set(df["norm"].values))
    # norms = ["raw", "stscore image", "stscore brain", "nyul brain decile", "nyul brain mode", "nyul mean decile", "nyul otsu decile"]
    print(df.describe)
    print(df.head)
    print(df.columns)
    print(norms)
    # cvdata = np.zeros(shape=(len(norms), 3))
    cvdata = np.zeros(shape=(len(norms), 2))
    qcddata = np.zeros(shape=(len(norms), 2))
    for i, norm in enumerate(norms):
        print("\n", norm)
        # print(df.loc[df["norm"] == norm])
        df_norm = df.loc[df["norm"] == norm]
        sdvals_img = df_norm["sd im"]
        muvals_img = df_norm["avg im"]
        # sdvals_brain = df_norm["sd brain"]
        # muvals_brain = df_norm["avg brain"]
        sdvals_saliv = df_norm["sd saliv"]
        muvals_saliv = df_norm["avg saliv"]
        cv_mu_img = np.std(muvals_img) / np.mean(muvals_img)
        cv_sd_img = np.std(sdvals_img) / np.mean(sdvals_img)
        # cv_mu_br = np.std(muvals_brain) / np.mean(muvals_brain)
        # cv_sd_br = np.std(sdvals_brain) / np.mean(sdvals_brain)
        cv_mu_sal = np.std(muvals_saliv) / np.mean(muvals_saliv)
        cv_sd_sal = np.std(sdvals_saliv) / np.mean(sdvals_saliv)
        mse_img = np.sqrt(cv_mu_img**2 + cv_sd_img**2) # mean square error ish
        # mse_br = np.sqrt(cv_mu_br**2 + cv_sd_br**2) # mean square error ish
        mse_sal = np.sqrt(cv_mu_sal**2 + cv_sd_sal**2) # mean square error ish
        print(f"IMG: cv_mu = {cv_mu_img:.3g}, cv_sd = {cv_sd_img:.3g}, mse_img = {mse_img:.4g}")
        # print(f"BRAIN: cv_mu = {cv_mu_br:.3g}, cv_sd = {cv_sd_br:.3g}, mse_img = {mse_br:.4g}")
        print(f"SALIV: cv_mu = {cv_mu_sal:.3g}, cv_sd = {cv_sd_sal:.3g}, mse_img = {mse_sal:.4g}")
        cvdata[i][0] = mse_img
        # cvdata[i][1] = mse_br
        cvdata[i][1] = mse_sal

    # cvdata[1, 0] = 0
    # cvdata[2, 1] = 0
    sns.heatmap(cvdata, linewidths=1, annot=True, cmap="winter", cbar=False)
    locs, _ = plt.xticks()
    ylocs, _ = plt.yticks()
    # plt.xticks(locs, ["Img", "Brain", "Saliv"])
    plt.xticks(locs, ["Img", "ROI SMG"])
    plt.yticks(ylocs, norms, rotation=0)
    # plt.tight_layout()
    plt.title("Normalization influence on the coefficient of variation (cv) for various ROI's"
              "\n$\sqrt{cv_{\mu}^2 + cv_{\sigma}^2} = \sqrt{(s(\sigma_i) / \\bar{x}(\sigma_i))^2 + (s(\mu_i) / \\bar{x}(\mu_i))^2}$")
    plt.show()
    pass


def disc_FBW_count_bins(norm, offsetIDX=0, bw=0, t1_images=False, show_sample=True, make_nrrd=False, n4=True):    # USING FIXED BIN WIDTH DISCRETIZATION
    # bw: bin width for discretizing, if 0 use Freedman-Diaconis rule
    # t1bool: T1 and T2 should have different BW !!!

    if "nyul" in norm:
        nrmny = nyul_initializer(norm)
        # print(norm[5:])
        try:
            mode = norm[5:]
            scale_p_t2 = nyul_scales[mode + " p"]
            scale_nop_t2 = nyul_scales[mode + " nop"]
            scale_p_t1 = nyul_scales_T1[mode + " p"]
            scale_nop_t1 = nyul_scales_T1[mode + " nop"]
        except Exception as e:
            print(*e.args)
            print("NORM", norm, "NOT IMPLEMENTED - TRY AGAIN")
            return 0

    fbw_bins_roi = []
    fbw_bins_im = []
    fd_binwidths = []
    j = 1
    for experiment in find_folders(RawDir, condition="Pilot"):
    # for experiment in ["Pilot3"]:
        for time in find_folders(os.path.join(RawDir, experiment)):
            print("\n", experiment, time)
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            for folder in find_folders(os.path.join(RawDir, experiment, time), condition="sagittal"):
                name = get_name(experiment, folder, condition="sagittal")

                pbool = "p" in name
                t1bool = "T1" in name
                includebool = bool(t1_images == t1bool)
                if includebool:
                    print(j, experiment, time, name)
                    j += 1
                    idx_center = central_idx_dict[time + name]
                    indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
                    MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder), printbool=False)
                    im_raw = MATR_RAW[idx_center + offsetIDX]
                    ROI_saliv = np.load(os.path.join(SegSalivDir, experiment.lower(), time, name + ".npy"))
                    roi_saliv = ROI_saliv[idx_center + offsetIDX]
                    try:
                        ROI_brain = np.load(os.path.join(SegBrainDir, experiment.lower(), time, name + ".npy"))
                        roi_brain = ROI_brain[idx_center + offsetIDX]
                        brainbool = True
                    except Exception as e:
                        # print(*e.args)
                        brainbool = False

                    if n4:
                        im = n4correction(im_raw)[0]     # N4 bias field correction (ONLY TURN OFF TO SPEED UP TESTING)
                    else:
                        im = im_raw

                    if norm == "none":
                        pass
                    elif norm == "stscore image":
                        im = mean_centering(im, mode="image", n=0)
                    elif norm == "stscore brain":
                        im = mean_centering(im, mode="roi", ROI=roi_brain, n=0)
                    elif "nyul" in norm:
                        if t1bool:
                            nrmny.SCALE_STD = scale_p_t1 if pbool else scale_nop_t1
                        else:
                            nrmny.SCALE_STD = scale_p_t2 if pbool else scale_nop_t2
                        im = nrmny.transform_image(im)

                    roi_shape = roi_saliv.copy()                # SPLIT ROI MASK INTO SHAPE AND INTENSITY
                    roi_reseg = resegment(im, roi_saliv, n=3)

                    if bw == 0:
                        # bw_fd = freedman_diaconis_rule(im , roi_reseg) # old
                        bw_fd = freedman_diaconis_rule(im[roi_reseg].ravel())
                        fd_binwidths.append(bw_fd)
                        # bw = bw_fd
                        im_disc = discretize_FBW_ISBI(im, roi_reseg, bw=bw_fd)
                    else:
                        # im_disc = discretize_FBW(im, roi_reseg, bw=bw)
                        im_disc = discretize_FBW_ISBI(im, roi_reseg, bw=bw)
                        # im_disc = discretize_FBW_ISBI(crop_to_mask(im, roi_reseg), crop_to_mask(roi_reseg, roi_reseg), bw=bw)
                        # im_disc = discretize_FBW(crop_to_mask(im, roi_reseg), crop_to_mask(roi_reseg, roi_reseg), bw=bw)

                    # im_disc = discretize_FBW_ISBI(im, roi_reseg, bw)

                    fbw_bins_roi.append(len(np.unique(im_disc[roi_reseg != 0])))
                    fbw_bins_im.append(len(np.unique(im_disc)))
                    # show_sample = True if len(np.unisque(im_disc)) in [28, 118] else False   # look at outliers in whoel img
                    #                 # show_sample = True if len(np.unique(im_dic[roi_reseg])) in [5, 17] else False   # look at outliers in ROI
                    # show_sample = False
                    if show_sample:
                        fig, ax = plt.subplots(2, 2)
                        ax = ax.ravel()
                        fig.tight_layout()
                        nbins = 50
                        ax[0].axis("off")
                        # [axx.axis("off") for axx in ax]
                        roi_cropped = np.ma.masked_where(crop_to_mask(roi_saliv, roi_saliv) == 0, crop_to_mask(roi_saliv, roi_saliv))
                        im1 = np.ma.masked_where(crop_to_mask(roi_reseg, roi_saliv) == 0, crop_to_mask(im_raw, roi_saliv))
                        im2 = np.ma.masked_where(crop_to_mask(roi_reseg, roi_saliv) == 0, crop_to_mask(im, roi_saliv))
                        # ax[0].imshow(crop_to_mask(im_raw, roi_saliv), cmap="gray")
                        # ax[0].imshow(np.ma.masked_where(crop_to_mask(roi_saliv, roi_saliv) == 0, crop_to_mask(im_raw, roi_saliv)), cmap="Blues")
                        # ax[0].imshow(im1, cmap="hot")
                        # ax[0].set_title("avg={}, sd={}".format(*[round(x, 1) for x in [np.average(im_raw[roi_saliv != 0]), np.std(im_raw[roi_saliv != 0])]]))
                        ax[0].imshow(crop_to_mask(im, roi_saliv), cmap="gray")
                        ax[0].imshow(np.ma.masked_where(crop_to_mask(roi_saliv, roi_saliv) == 0, crop_to_mask(im, roi_saliv)), cmap="Blues")
                        ax[0].imshow(im2, cmap="hot")
                        ax[0].set_title(norm+" norm w/ avg={}, sd={}".format(*[round(x, 1) for x in [np.average(im[roi_saliv != 0]), np.std(im[roi_saliv != 0])]])
                                        +" - blue pixels excluded by re-segmentation (outlier filter)")

                        ax[1].hist(im[roi_reseg != 0].ravel(), bins=nbins)
                        ax[2].imshow(crop_to_mask(im_disc, roi_saliv), cmap="gray")
                        ax[2].imshow(np.ma.masked_where(crop_to_mask(roi_reseg, roi_saliv) == 0, crop_to_mask(im_disc, roi_saliv)), cmap="hot")
                        ax[2].set_title(f"Discretized by FBW (bw={bw if bw else '''FD-bw = ''' + str(round(bw_fd, 2))}) into {len(np.unique(im_disc[roi_reseg != 0]))} (ROI),"
                                        f"{len(np.unique(im_disc))} (whole img) bins.")
                        ax[3].hist(im_disc[roi_reseg != 0].ravel(), bins=nbins)
                        plt.show()

    print("FBW number of bins in ROI: median, var [range]", np.median(fbw_bins_roi), round(np.var(fbw_bins_roi), 1), f"[{np.min(fbw_bins_roi)}, {np.max(fbw_bins_roi)}]")
    print("FBW number of bins in whole image: median, var [range]", np.median(fbw_bins_im), round(np.var(fbw_bins_im), 1), f"[{np.min(fbw_bins_im)}, {np.max(fbw_bins_im)}]")
    print("Freedman Diaconis BW: median, var [range]", np.median(fd_binwidths), round(np.var(fd_binwidths), 1), f"[{np.min(fd_binwidths)}, {np.max(fd_binwidths)}]") if bw == 0 else 0
    return [np.median(fbw_bins_roi), np.std(fbw_bins_roi), np.min(fbw_bins_roi), np.max(fbw_bins_roi)]


def disc_FBW_count_bins_LRsplit(norm, bw=0, t1_images=False, show_sample=True, make_nrrd=False, n4=True):    # USING FIXED BIN WIDTH DISCRETIZATION
    # bw: bin width for discretizing, if 0 use Freedman-Diaconis rule
    # t1bool: T1 and T2 should have different BW !!!
    # mode = "LR split"
    # import SimpleITK as sitk


    if "nyul" in norm:
        nrmny = nyul_initializer(norm)
        # print(norm[5:])
        try:
            nyul_mode = norm[5:]
            scale_p_t2 = nyul_scales[nyul_mode + " p"]
            scale_nop_t2 = nyul_scales[nyul_mode + " nop"]
            scale_p_t1 = nyul_scales_T1[nyul_mode + " p"]
            scale_nop_t1 = nyul_scales_T1[nyul_mode + " nop"]
        except Exception as e:
            print(*e.args)
            print("NORM", norm, "NOT IMPLEMENTED - TRY AGAIN")
            return 0

    fbw_bins_roi = []
    fbw_bins_im = []
    fd_binwidths = []
    j = 1
    for experiment in find_folders(RawDir, condition="Pilot"):
        for time in find_folders(os.path.join(RawDir, experiment)):
            print("\n", experiment, time)
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            for folder in find_folders(os.path.join(RawDir, experiment, time), condition="sagittal"):
                name = get_name(experiment, folder, condition="sagittal")
                pbool = "p" in name
                t1bool = "T1" in name
                includebool = bool(t1_images == t1bool)
                if includebool:
                    path_raw = os.path.join(RawDir, experiment, time, folder)
                    path_roi = os.path.join(SegSalivDir, experiment, time, name + ".npy")

                    files = dcm_files_indexed(path_raw)
                    MTR = dcm_folder_to_pixel_matrix(files, folder_path=path_raw, printbool=False)
                    ROI = np.load(path_roi)
                    for X in ["L", "R"]:
                        # name = name_orig + "_" + X
                        print("\n", j, experiment, time, name)
                        j += 1

                        # if j > 3:
                        #     return [np.mean(fbw_bins_roi), np.median(fbw_bins_roi), np.std(fbw_bins_roi),
                        #             np.min(fbw_bins_roi), np.max(fbw_bins_roi),
                        #             np.mean(fbw_bins_im), np.median(fbw_bins_im), np.std(fbw_bins_im),
                        #             np.min(fbw_bins_im), np.max(fbw_bins_im)]

                        idx = left_idx_dict[time + name] if X == "L" else right_idx_dict[time + name]
                        im = MTR[idx]
                        roi = ROI[idx]
                        # im = n4correction(img)[0]

                        if n4:
                            im = n4correction(im)[0]     # N4 bias field correction (ONLY TURN OFF TO SPEED UP TESTING)
                        else:
                            pass

                        if norm in ["none", "raw", "no norm"]:
                            pass
                        elif norm in ["stscore image", "stscore"]:
                            print(f"Stscore norm: mean={np.mean(im):.3f}", end="")
                            im = mean_centering(im, mode="image", n=3)
                            print(f"--> mean={np.mean(im):.3f}")
                        # elif norm == "stscore brain":
                        #     im = mean_centering(im, mode="roi", ROI=roi_brain, n=0)
                        elif "nyul" in norm:
                            if t1bool:
                                nrmny.SCALE_STD = scale_p_t1 if pbool else scale_nop_t1
                            else:
                                nrmny.SCALE_STD = scale_p_t2 if pbool else scale_nop_t2
                            im = nrmny.transform_image(im)
                        else:
                            print("INVALID NORM", norm)
                            return 0

                        roi_shape = roi.copy()                # SPLIT ROI MASK INTO SHAPE AND INTENSITY
                        roi_reseg = resegment(im, roi, n=3)

                        if bw == 0:
                            # bw_fd = freedman_diaconis_rule(im , roi_reseg) # old
                            bw_fd = freedman_diaconis_rule(im[roi_reseg].ravel())
                            fd_binwidths.append(bw_fd)
                            # bw = bw_fd
                            im_disc = discretize_FBW_ISBI(im, roi_reseg, bw=bw_fd)
                        else:
                            # im_disc = discretize_FBW(im, roi_reseg, bw=bw)
                            im_disc = discretize_FBW_ISBI(im, roi_reseg, bw=bw)
                            # im_disc = discretize_FBW_ISBI(crop_to_mask(im, roi_reseg), crop_to_mask(roi_reseg, roi_reseg), bw=bw)
                            # im_disc = discretize_FBW(crop_to_mask(im, roi_reseg), crop_to_mask(roi_reseg, roi_reseg), bw=bw)

                        # im_disc = discretize_FBW_ISBI(im, roi_reseg, bw)

                        fbw_bins_roi.append(len(np.unique(im_disc[roi_reseg != 0])))
                        fbw_bins_im.append(len(np.unique(im_disc)))
                        # show_sample = True if len(np.unisque(im_disc)) in [28, 118] else False   # look at outliers in whoel img
                        #                 # show_sample = True if len(np.unique(im_dic[roi_reseg])) in [5, 17] else False   # look at outliers in ROI
                        # show_sample = False
                        if show_sample:
                            fig, ax = plt.subplots(2, 2)
                            ax = ax.ravel()
                            fig.tight_layout()
                            nbins = 50
                            ax[0].axis("off")
                            # [axx.axis("off") for axx in ax]
                            roi_cropped = np.ma.masked_where(crop_to_mask(roi, roi) == 0, crop_to_mask(roi, roi))
                            im1 = np.ma.masked_where(crop_to_mask(roi_reseg, roi) == 0, crop_to_mask(im_raw, roi))
                            im2 = np.ma.masked_where(crop_to_mask(roi_reseg, roi) == 0, crop_to_mask(im, roi))
                            # ax[0].imshow(crop_to_mask(im_raw, roi), cmap="gray")
                            # ax[0].imshow(np.ma.masked_where(crop_to_mask(roi_saliv, roi_saliv) == 0, crop_to_mask(im_raw, roi_saliv)), cmap="Blues")
                            # ax[0].imshow(im1, cmap="hot")
                            # ax[0].set_title("avg={}, sd={}".format(*[round(x, 1) for x in [np.average(im_raw[roi_saliv != 0]), np.std(im_raw[roi_saliv != 0])]]))
                            ax[0].imshow(crop_to_mask(im, roi), cmap="gray")
                            ax[0].imshow(np.ma.masked_where(crop_to_mask(roi, roi) == 0, crop_to_mask(im, roi)), cmap="Blues")
                            ax[0].imshow(im2, cmap="hot")
                            ax[0].set_title(norm+" norm w/ avg={}, sd={}".format(*[round(x, 1) for x in [np.average(im[roi != 0]), np.std(im[roi != 0])]])
                                            +" - blue pixels excluded by re-segmentation (outlier filter)")

                            ax[1].hist(im[roi_reseg != 0].ravel(), bins=nbins)
                            ax[2].imshow(crop_to_mask(im_disc, roi), cmap="gray")
                            ax[2].imshow(np.ma.masked_where(crop_to_mask(roi_reseg, roi) == 0, crop_to_mask(im_disc, roi)), cmap="hot")
                            ax[2].set_title(f"Discretized by FBW (bw={bw if bw else '''FD-bw = ''' + str(round(bw_fd, 2))}) into {len(np.unique(im_disc[roi_reseg != 0]))} (ROI),"
                                            f"{len(np.unique(im_disc))} (whole img) bins.")
                            ax[3].hist(im_disc[roi_reseg != 0].ravel(), bins=nbins)
                            plt.show()



    print("FBW number of bins in ROI: median, var [range]", np.median(fbw_bins_roi), round(np.var(fbw_bins_roi), 1), f"[{np.min(fbw_bins_roi)}, {np.max(fbw_bins_roi)}]")
    print("FBW number of bins in whole image: median, var [range]", np.median(fbw_bins_im), round(np.var(fbw_bins_im), 1), f"[{np.min(fbw_bins_im)}, {np.max(fbw_bins_im)}]")
    print("Freedman Diaconis BW: median, var [range]", np.median(fd_binwidths), round(np.var(fd_binwidths), 1), f"[{np.min(fd_binwidths)}, {np.max(fd_binwidths)}]") if bw == 0 else 0
    return [np.mean(fbw_bins_roi), np.median(fbw_bins_roi), np.std(fbw_bins_roi), np.min(fbw_bins_roi), np.max(fbw_bins_roi),
            np.mean(fbw_bins_im), np.median(fbw_bins_im), np.std(fbw_bins_im), np.min(fbw_bins_im), np.max(fbw_bins_im)]
    # return [np.median(fbw_bins_roi), np.std(fbw_bins_roi), np.min(fbw_bins_roi), np.max(fbw_bins_roi)]



def make_nrrd(mode, norm, disc="none", n4=True):
    if mode == "center":
        offsetIDX = 0
        pass
    else:
        print("Mode", mode, "NOT VALID --> CANNOT MAKE NRRD FILES")
        return 0

    if "nyul" in norm:
        nrmny = nyul_initializer(norm)
        # print(norm[5:])
        try:
            scale_p_t2 = nyul_scales[norm[5:] + " p"]
            scale_nop_t2 = nyul_scales[norm[5:] + " nop"]
            scale_p_t1 = nyul_scales_T1[norm[5:] + " p"]
            scale_nop_t1 = nyul_scales_T1[norm[5:] + " nop"]
        except Exception as e:
            print(*e.args)
            print("NORM", norm, "NOT IMPLEMENTED - TRY AGAIN")
            return 0

    j = 1
    for experiment in find_folders(RawDir, condition="Pilot"):
        for time in find_folders(os.path.join(RawDir, experiment)):
            print("\n", experiment, time)
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            for folder in find_folders(os.path.join(RawDir, experiment, time), condition="sagittal"):
                name = get_name(experiment, folder, condition="sagittal")
                pbool = "p" in name
                t1bool = "T1" in name
                print(j, experiment, time, name)
                j += 1

                idx_center = central_idx_dict[time + name]
                indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
                MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder), printbool=False)
                im_raw = MATR_RAW[idx_center + offsetIDX]
                ROI_saliv = np.load(os.path.join(SegSalivDir, experiment.lower(), time, name + ".npy"))
                roi_saliv = ROI_saliv[idx_center + offsetIDX]
                try:
                    ROI_brain = np.load(os.path.join(SegBrainDir, experiment.lower(), time, name + ".npy"))
                    roi_brain = ROI_brain[idx_center + offsetIDX]
                    brainbool = True
                except Exception as e:
                    # print(*e.args)
                    brainbool = False

                if n4:
                    im = n4correction(im_raw)[0]  # N4 bias field correction (ONLY TURN OFF TO SPEED UP TESTING)
                else:
                    im = im_raw

                if norm in ["none", "raw", ""]:
                    pass
                elif norm == "stscore image":
                    im = mean_centering(im, mode="image", n=0)
                elif norm == "stscore brain":
                    im = mean_centering(im, mode="roi", ROI=roi_brain, n=0)
                elif "nyul" in norm:
                    if t1bool:
                        nrmny.SCALE_STD = scale_p_t1 if pbool else scale_nop_t1
                    else:
                        nrmny.SCALE_STD = scale_p_t2 if pbool else scale_nop_t2
                    im = nrmny.transform_image(im)  # CHANGE TO INCLUDE BRAIN FOR NYUL BRAIN
                else:
                    print("\tNorm", norm, "NOT IMPLEMENTED, EXITING.")
                    return 0
                path_nrrd = os.path.join(nrrdDir, mode + " " + norm, experiment, time)

                ndarray_to_nrrd(im, path_nrrd, name, mode="image")  # UNCOMMENT TO SAVE AS NRRD
                ndarray_to_nrrd(roi_saliv, path_nrrd, name, mode="mask")
    return 1


def make_nrrd_LRsplit(mode, norm, disc="none", n4=True):
    if mode == "center":
        offsetIDX = 0
        print("DEPRECATED")
        return 0
    elif mode == "LR split":
        # df = pd.read_csv(os.path.join(SegSalivDir, "salivary LR split indexes.csv"), index_col=0)
        pass
    else:
        print("Mode", mode, "NOT VALID --> CANNOT MAKE NRRD FILES")
        return 0
    folder_parent_name = mode + " " + norm if n4 else mode + " " + norm + " no n4"
    print(folder_parent_name)

    nrmny = nyul_initializer(norm="nyul otsu decile")
    nyul_scale_T1_p = nyul_scales_T1["otsu decile p"]
    nyul_scale_T1_nop = nyul_scales_T1["otsu decile p"]
    nyul_scale_T2_p = nyul_scales_T2["otsu decile p"]
    nyul_scale_T2_nop = nyul_scales_T2["otsu decile p"]

    j = 1
    for experiment in find_folders(RawDir, condition="Pilot"):
        for time in find_folders(os.path.join(RawDir, experiment)):
            print("\n", experiment, time)
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            # folder_nrrd = os.path.join(nrrdDir, mode + " " + norm, experiment, time)
            folder_nrrd = os.path.join(nrrdDir, folder_parent_name, experiment, time)
            for folder in find_folders(os.path.join(RawDir, experiment, time), condition="sagittal"):
                name = get_name(experiment, folder, condition="sagittal")
                pbool = "p" in name
                t1bool = "T1" in name
                print("\n", j, experiment, time, name)
                j += 1

                if t1bool:
                    nyul_scale = nyul_scale_T1_p if pbool else nyul_scale_T1_nop
                else:
                    nyul_scale = nyul_scale_T2_p if pbool else nyul_scale_T2_nop
                nrmny.SCALE_STD = nyul_scale

                path_raw = os.path.join(RawDir, experiment, time, folder)
                path_roi = os.path.join(SegSalivDir, experiment, time, name + ".npy")
                files = dcm_files_indexed(path_raw)
                MTR = dcm_folder_to_pixel_matrix(files, folder_path=path_raw, printbool=False)
                ROI = np.load(path_roi)

                for X in ["L", "R"]:
                    idx = left_idx_dict[time + name] if X=="L" else right_idx_dict[time+name]
                    img = MTR[idx]
                    roi = ROI[idx]

                    # PREPROCESSING
                    if n4:
                        img = n4correction(img)[0]

                    if norm == "no norm":
                        pass
                    elif norm == "stscore":
                        img = mean_centering(img, mode="image", n=3)
                    elif norm == "nyul":
                        img = nrmny.transform_image(img)
                    else:
                        print("\tNorm", norm, "NOT IMPLEMENTED, EXITING.")
                        return 0

                    ndarray_to_nrrd(img, savepath=folder_nrrd, mousename=name + "_" + X, mode="image")  # UNCOMMENT TO SAVE AS NRRD
                    ndarray_to_nrrd(roi, savepath=folder_nrrd, mousename=name + "_" + X, mode="mask")
    return 1


def entropy(imvals):
    # MANUAL CHECK OF FEATURE CALCULATION: ENTROPY
    # print("Calculating entropy")
    imvals = np.ravel(imvals)
    imvals -= np.min(imvals) - 1
    count, vals = np.histogram(imvals, bins=np.max(imvals) - np.min(imvals))
    # print(count)
    # print(vals)
    # print(np.sum(count), len(imvals))
    Np = np.sum(count)
    eps = np.spacing(1)
    sum = 0
    for i, p in zip(vals, count):
        pi = p / Np
        sum += -pi * np.log2(pi + eps)
    return sum


def main_extract(mode, norm, compare_to_manual=False):
    j = 0
    offsetIDX = 0
    savepath = os.path.join(ftsDir, "_".join([mode, *norm.split(" ")]) + "_extracted.csv")
    # print(savepath)
    df_tot = pd.DataFrame()
    if compare_to_manual:
        from radiomics.ngtdm import RadiomicsNGTDM
        from radiomics.glcm import RadiomicsGLCM
        from radiomics.firstorder import RadiomicsFirstOrder
        from radiomics.shape2D import RadiomicsShape2D
        from radiomics.imageoperations import getLBP2DImage, getWaveletImage, getGradientImage, binImage, getBinEdges
        import SimpleITK as sitk
        import nrrd
    for experiment in find_folders(RawDir, condition="Pilot"):
    # for experiment in ["Pilot2", "Pilot3"]:
        for time in find_folders(os.path.join(RawDir, experiment)):
            print("\n", experiment, time)
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            for folder in find_folders(datadir, condition="sagittal"):
                name = get_name(experiment, folder, condition="sagittal")
                pbool = "p" in name
                t1bool = "T1" in name
                j += 1
                print("\n",j, experiment, time, name)
                path_nrrd = os.path.join(nrrdDir, mode+" "+norm, experiment, time)
                path_nrrd_image = os.path.join(path_nrrd, name + "_image.nrrd")
                path_nrrd_mask = os.path.join(path_nrrd, name + "_mask.nrrd")

                params = os.path.join(ftsDir, "settings", "Params_nonorm_2D.yaml")  # path to extractor settings
                extractor = featureextractor.RadiomicsFeatureExtractor(params)
                results = extractor.execute(path_nrrd_image, path_nrrd_mask)  # TODO: actual feature extracting - UNDERSTAND THIS
                # print(results)
                # df = pd.DataFrame.from_dict(results, orient="index", columns=[name]).T
                df = pd.DataFrame.from_dict(results, orient="index", columns=[j]).T#, columns=[name]).T
                # df = pd.DataFrame.from_dict(results, orient="columns")#, columns=[name]).T
                # print(df)
                df = df.drop(list(df.filter(regex="diagnostics")), axis=1)  # drop diagnostics "features" - not really features!
                df["name"] = name
                df["time"] = time[:-3]
                df["dose"] = dose_to_name(experiment, time, name)
                cols = df.drop(["name", "time", "dose"], axis=1).columns
                df = df[["name", "time", "dose", *cols]]        # make name, time, dose first "features" in dataframe
                if j == 1:
                    df_tot = df
                else:
                    df_tot = df_tot.append(df)
                # print(df_tot.T.head(5))
                # print(df_tot)
                # print(len(df_tot), len(df_tot.T))

                # print(df_tot.T.head(10))
                # print(df.T.tail(5))
                # print(len(np.nonzero(ft.feats.columns.str.contains(word))[0]), f"features having {word} in name.\n")
                # print(np.nonzero(df.columns.str.contains("gradient_n")))
                # print(df.T[df.columns.str.contains("lbp-2D_n")])
                # print(df.T[df.columns.str.contains("gradient_n")])
                # print(df.T[df.columns.str.contains("original_ngtdm")].head(5))
                # print(df.T[df.columns.str.contains("original_glcm")].head(5))
                # print(df["original_shape2D_PixelSurface"])
                # print(df.T[df.columns.str.contains("original_first")].head(8))
                # print(df.T[df.columns.str.contains("lbp-2D")].head(8))
                if compare_to_manual:
                    # MANUAL COMPARISON OF CALCULATED FEATURES -- IGNORE WHEN RUNNING EXTRACTOR
                    im, _ = nrrd.read(path_nrrd_image)
                    roi_, _ = nrrd.read(path_nrrd_mask)
                    # roi_ = np.ones(shape=im.shape)
                    img = sitk.GetImageFromArray(im)
                    roi = sitk.GetImageFromArray(roi_)

                    shape = RadiomicsShape2D(img, roi)
                    shape._initCalculation()
                    print("Pixel count ROI:", len(im[roi_ != 0]), shape.getPixelSurfaceFeatureValue(), df["original_shape2D_PixelSurface"].values)

                    roi_ = resegment(im, roi_, n=3)
                    roi = sitk.GetImageFromArray(roi_)
                    firstord = RadiomicsFirstOrder(img, roi, binWidth=950)
                    firstord._initCalculation()
                    print("10 perc:", firstord.get10PercentileFeatureValue(), df["original_firstorder_10Percentile"].values)
                    print("90 perc:", firstord.get90PercentileFeatureValue(), df["original_firstorder_90Percentile"].values)
                    print("Energy:", firstord.getEnergyFeatureValue(), df["original_firstorder_Energy"].values)
                    print("Entropy:", firstord.getEntropyFeatureValue(), df["original_firstorder_Entropy"].values, entropy(binImage(im, binWidth=950)[0][roi_ != 0]))
                    print("IQR:", firstord.getInterquartileRangeFeatureValue(), df["original_firstorder_InterquartileRange"].values)
                    print("Mean:", firstord.getMeanFeatureValue(), df["original_firstorder_Mean"].values)
                    print("RobMeanabsdev:", firstord.getRobustMeanAbsoluteDeviationFeatureValue(), df["original_firstorder_RobustMeanAbsoluteDeviation"].values)
                    print("Totalenergy:", firstord.getTotalEnergyFeatureValue(), df["original_firstorder_TotalEnergy"].values)
                    print()
                    # im = discretize_FBW(im, roi_, bw=950)
                    # im = discretize_FBW_ISBI(im, roi_, bw=950)
                    # im, edges = binImage(im, binWidth=950)
                    # img = sitk.GetImageFromArray(im)
                    # print("---  IMAGE DISCRETIZED ---")

                    img_orig = img

                    print(df.T[df.columns.str.contains("original_glcm")].head(5))
                    glcm = RadiomicsGLCM(img_orig, roi, binWidth=950)
                    glcm._initCalculation()
                    print("COMPARING GLCM calculated values:")
                    print("Autocorr:", glcm.getAutocorrelationFeatureValue(), df["original_glcm_Autocorrelation"].values)
                    print("Jointavg:", glcm.getJointAverageFeatureValue(), df["original_glcm_JointAverage"].values)


                    # img_filt = [list(xx) for xx in getGradientImage(img, roi)][0][0]
                    img = sitk.GetImageFromArray(norm_minmax_featurescaled(im, 0, 255))
                    img_filt = [list(xx) for xx in getLBP2DImage(img, roi, lbp2DRadius=1, lbp2DSamples=9)][0][0]
                    # img_filt = [list(xx) for xx in getWaveletImage(img, roi, wavelet="haar", level=5)][0][0]
                    print(df.T[df.columns.str.contains("lbp-2D")].head(5))
                    # print(df.T[df.columns.str.contains("gradient_glcm")].head(5))
                    # glcm = RadiomicsGLCM(img_filt, roi, binWidth=950)
                    glcm = RadiomicsGLCM(img_filt, roi, binWidth=950)
                    glcm._initCalculation()
                    print("COMPARING GLCM calculated values:")
                    # print("Autocorr:", glcm.getAutocorrelationFeatureValue(), df["gradient_glcm_Autocorrelation"].values)
                    # print("Jointavg:", glcm.getJointAverageFeatureValue(), df["gradient_glcm_JointAverage"].values)
                    print("Autocorr:", glcm.getAutocorrelationFeatureValue(), df["lbp-2D_glcm_Autocorrelation"].values)
                    print("Jointavg:", glcm.getJointAverageFeatureValue(), df["lbp-2D_glcm_JointAverage"].values)

                    # ngtdm = RadiomicsNGTDM(img_orig, roi)
                    # ngtdm._initCalculation()
                    # f1 = ngtdm.getCoarsenessFeatureValue()[0]
                    # f2 = ngtdm.getComplexityFeatureValue()[0]
                    # f3 = ngtdm.getContrastFeatureValue()[0]
                    # f4 = ngtdm.getStrengthFeatureValue()[0]
                    # # TODO: WHY DO MANUALLY CALC FTS DIFFER FROM EXTRACTION PIPELINE???
                    # #   maybe disc algorithm different???? se documentation..
                    # print(f"Filter NGTDM coarseness / complexity / contrast / strength: {f1:.5g}, {f2:.5g}, {f3:.5g}, {f4:.5g}")
                    # print("Filter GLCM autocorr / JointAvg / ClusterProm / clustershade / clustertend: {}".format([round(x, 5) for x in [glcm.getAutocorrelationFeatureValue()[0], glcm.getJointAverageFeatureValue()[0], glcm.getClusterProminenceFeatureValue()[0], glcm.getClusterShadeFeatureValue()[0], glcm.getClusterTendencyFeatureValue()[0]]]))
                    fig, ax = plt.subplots(1, 2)
                    ax[0].imshow(im, cmap="gray")
                    ax[0].imshow(im*roi_, cmap="hot", alpha=0.3)
                    ax[1].imshow(sitk.GetArrayFromImage(img_filt), cmap="hot")
                    plt.show()
            # break
        # break
    print(len(df_tot.T), "FEATURES EXTRACTED FOR ", len(df_tot), "IMAGES.")
    if os.path.exists(savepath):
        print("PATH EXISTS - DO YOU WANT TO OVERWRITE?  y / n")
        answ = input()
        if answ == "y":
            df_tot.to_csv(savepath)
            print("\nEXTRACTED FEATURES saved at", savepath)
            return 1
        else:
            print("\nEXTRACTED FEATURES NOT SAVED.")
            return 0
    else:
        df_tot.to_csv(savepath)
        print("\nEXTRACTED FEATURES saved at", savepath)
        return 1
    return 0


def main_extract_LRsplit_old(norm):
    j = 0
    offsetIDX = 0
    mode = "LR_split"
    # savepath = os.path.join(ftsDir, "_".join([*"LR split".split(" "), *norm.split(" ")]) + "_extracted.csv")
    savepath = os.path.join(ftsDir, "_".join([mode, *norm.split(" ")]) + "_extracted.csv")
    print(savepath)
    df_tot = pd.DataFrame()
    for experiment in find_folders(RawDir, condition="Pilot"):
    # for experiment in ["Pilot2", "Pilot3"]:
        for time in find_folders(os.path.join(RawDir, experiment)):
            print("\n", experiment, time)
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            for folder in find_folders(datadir, condition="sagittal"):
                name_orig = get_name(experiment, folder, condition="sagittal")
                pbool = "p" in name_orig
                t1bool = "T1" in name_orig
                for X in ["L", "R"]:
                    name = name_orig + "_" + X
                    j += 1
                    print("\n",j, experiment, time, name)
                    path_nrrd = os.path.join(nrrdDir, mode+" "+norm, experiment, time)
                    path_nrrd_image = os.path.join(path_nrrd, name + "_image.nrrd")
                    path_nrrd_mask = os.path.join(path_nrrd, name + "_mask.nrrd")
                    # plot_nrrd(path_nrrd_image, path_nrrd_mask)
                    params = os.path.join(ftsDir, "settings", "Params_nonorm_2D.yaml")  # path to extractor settings
                    extractor = featureextractor.RadiomicsFeatureExtractor(params)
                    results = extractor.execute(path_nrrd_image, path_nrrd_mask)  # TODO: actual feature extracting - UNDERSTAND THIS
                    #
                    df = pd.DataFrame.from_dict(results, orient="index", columns=[j]).T#, columns=[name]).T
                    #
                    df = df.drop(list(df.filter(regex="diagnostics")), axis=1)  # drop diagnostics "features" - not really features!
                    df["name"] = name
                    df["time"] = time[:-3]
                    df["dose"] = dose_to_name(experiment, time, name)
                    cols = df.drop(["name", "time", "dose"], axis=1).columns
                    df = df[["name", "time", "dose", *cols]]        # make name, time, dose first "features" in dataframe
                    if j == 1:
                        df_tot = df
                    else:
                        df_tot = df_tot.append(df)

                    # if compare_to_manual:
                    if False:
                        # MANUAL COMPARISON OF CALCULATED FEATURES -- IGNORE WHEN RUNNING EXTRACTOR
                        im, _ = nrrd.read(path_nrrd_image)
                        roi_, _ = nrrd.read(path_nrrd_mask)
                        # roi_ = np.ones(shape=im.shape)
                        img = sitk.GetImageFromArray(im)
                        roi = sitk.GetImageFromArray(roi_)

                        shape = RadiomicsShape2D(img, roi)
                        shape._initCalculation()
                        print("Pixel count ROI:", len(im[roi_ != 0]), shape.getPixelSurfaceFeatureValue(), df["original_shape2D_PixelSurface"].values)

                        roi_ = resegment(im, roi_, n=3)
                        roi = sitk.GetImageFromArray(roi_)
                        firstord = RadiomicsFirstOrder(img, roi, binWidth=950)
                        firstord._initCalculation()
                        print("10 perc:", firstord.get10PercentileFeatureValue(), df["original_firstorder_10Percentile"].values)
                        print("90 perc:", firstord.get90PercentileFeatureValue(), df["original_firstorder_90Percentile"].values)
                        print("Energy:", firstord.getEnergyFeatureValue(), df["original_firstorder_Energy"].values)
                        print("Entropy:", firstord.getEntropyFeatureValue(), df["original_firstorder_Entropy"].values, entropy(binImage(im, binWidth=950)[0][roi_ != 0]))
                        print("IQR:", firstord.getInterquartileRangeFeatureValue(), df["original_firstorder_InterquartileRange"].values)
                        print("Mean:", firstord.getMeanFeatureValue(), df["original_firstorder_Mean"].values)
                        print("RobMeanabsdev:", firstord.getRobustMeanAbsoluteDeviationFeatureValue(), df["original_firstorder_RobustMeanAbsoluteDeviation"].values)
                        print("Totalenergy:", firstord.getTotalEnergyFeatureValue(), df["original_firstorder_TotalEnergy"].values)
                        print()
                        # im = discretize_FBW(im, roi_, bw=950)
                        # im = discretize_FBW_ISBI(im, roi_, bw=950)
                        # im, edges = binImage(im, binWidth=950)
                        # img = sitk.GetImageFromArray(im)
                        # print("---  IMAGE DISCRETIZED ---")

                        img_orig = img

                        print(df.T[df.columns.str.contains("original_glcm")].head(5))
                        glcm = RadiomicsGLCM(img_orig, roi, binWidth=950)
                        glcm._initCalculation()
                        print("COMPARING GLCM calculated values:")
                        print("Autocorr:", glcm.getAutocorrelationFeatureValue(), df["original_glcm_Autocorrelation"].values)
                        print("Jointavg:", glcm.getJointAverageFeatureValue(), df["original_glcm_JointAverage"].values)


                        # img_filt = [list(xx) for xx in getGradientImage(img, roi)][0][0]
                        img = sitk.GetImageFromArray(norm_minmax_featurescaled(im, 0, 255))
                        img_filt = [list(xx) for xx in getLBP2DImage(img, roi, lbp2DRadius=1, lbp2DSamples=9)][0][0]
                        # img_filt = [list(xx) for xx in getWaveletImage(img, roi, wavelet="haar", level=5)][0][0]
                        print(df.T[df.columns.str.contains("lbp-2D")].head(5))
                        # print(df.T[df.columns.str.contains("gradient_glcm")].head(5))
                        # glcm = RadiomicsGLCM(img_filt, roi, binWidth=950)
                        glcm = RadiomicsGLCM(img_filt, roi, binWidth=950)
                        glcm._initCalculation()
                        print("COMPARING GLCM calculated values:")
                        # print("Autocorr:", glcm.getAutocorrelationFeatureValue(), df["gradient_glcm_Autocorrelation"].values)
                        # print("Jointavg:", glcm.getJointAverageFeatureValue(), df["gradient_glcm_JointAverage"].values)
                        print("Autocorr:", glcm.getAutocorrelationFeatureValue(), df["lbp-2D_glcm_Autocorrelation"].values)
                        print("Jointavg:", glcm.getJointAverageFeatureValue(), df["lbp-2D_glcm_JointAverage"].values)

                        # ngtdm = RadiomicsNGTDM(img_orig, roi)
                        # ngtdm._initCalculation()
                        # f1 = ngtdm.getCoarsenessFeatureValue()[0]
                        # f2 = ngtdm.getComplexityFeatureValue()[0]
                        # f3 = ngtdm.getContrastFeatureValue()[0]
                        # f4 = ngtdm.getStrengthFeatureValue()[0]
                        # # TODO: WHY DO MANUALLY CALC FTS DIFFER FROM EXTRACTION PIPELINE???
                        # #   maybe disc algorithm different???? se documentation..
                        # print(f"Filter NGTDM coarseness / complexity / contrast / strength: {f1:.5g}, {f2:.5g}, {f3:.5g}, {f4:.5g}")
                        # print("Filter GLCM autocorr / JointAvg / ClusterProm / clustershade / clustertend: {}".format([round(x, 5) for x in [glcm.getAutocorrelationFeatureValue()[0], glcm.getJointAverageFeatureValue()[0], glcm.getClusterProminenceFeatureValue()[0], glcm.getClusterShadeFeatureValue()[0], glcm.getClusterTendencyFeatureValue()[0]]]))
                        fig, ax = plt.subplots(1, 2)
                        ax[0].imshow(im, cmap="gray")
                        ax[0].imshow(im*roi_, cmap="hot", alpha=0.3)
                        ax[1].imshow(sitk.GetArrayFromImage(img_filt), cmap="hot")
                        plt.show()
            # break
        # break
    print(len(df_tot.T), "FEATURES EXTRACTED FOR ", len(df_tot), "IMAGES.")
    if os.path.exists(savepath):
        print("PATH EXISTS - DO YOU WANT TO OVERWRITE?  y / n")
        answ = input()
        if answ == "y":
            df_tot.to_csv(savepath)
            print("\nEXTRACTED FEATURES saved at", savepath)
            return 1
        else:
            print("\nEXTRACTED FEATURES NOT SAVED.")
            return 0
    else:
        df_tot.to_csv(savepath)
        print("\nEXTRACTED FEATURES saved at", savepath)
        return 1
    return 0


def main_extract_LRsplit(norm, overwrite=False):
    FORCE2D_FEATURECLASSES = ["glcm", "glrlm"]
    #todo: do extraction force2d split, FBW bw dependent on T1 or T2...
    path_settings = os.path.join(ftsDir, "settings", "Params_nonorm_2D.yaml")
    with open(path_settings) as file:
        settings_orig = yaml.load(file, Loader=yaml.loader.SafeLoader)
    mode = "LR split"
    savepath = os.path.join(ftsDir, "_".join([*mode.split(" "), *norm.split(" ")]) + "_extracted.csv")
    print(savepath)

    df_tot = pd.DataFrame()
    j = 1
    for experiment in find_folders(RawDir, condition="Pilot"):
        for time in find_folders(os.path.join(RawDir, experiment)):
            print("\n", experiment, time)
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            for folder in find_folders(datadir, condition="sagittal"):
                name_orig = get_name(experiment, folder, condition="sagittal")
                pbool = "p" in name_orig
                t1bool = "T1" in name_orig

                settings = settings_orig.copy()
                BW = FBW_dict_T1[norm] if t1bool else FBW_dict_T2[norm]
                settings["setting"]["binWidth"] = BW
                path_nrrd = os.path.join(nrrdDir, mode + " " + norm, experiment, time)
                # print(path_nrrd)
                # print(os.path.exists(path_nrrd))
                print(j, experiment, time, name_orig)
                j += 1

                for X in ["L", "R"]:
                    name = name_orig + "_" + X
                    path_nrrd_image = os.path.join(path_nrrd, name + "_image.nrrd")
                    path_nrrd_mask = os.path.join(path_nrrd, name + "_mask.nrrd")

                    settings["setting"]["force2D"] = False
                    extractor = featureextractor.RadiomicsFeatureExtractor(settings)
                    results1 = extractor.execute(path_nrrd_image, path_nrrd_mask)
                    df_temp1 = pd.DataFrame.from_dict(results1, orient="index")
                    for fclass in FORCE2D_FEATURECLASSES:
                        filtered = df_temp1.filter(like=fclass, axis=0)
                        df_temp1 = df_temp1.drop(index=filtered.index)

                    settings["setting"]["force2D"] = True
                    extractor = featureextractor.RadiomicsFeatureExtractor(settings)
                    results2 = extractor.execute(path_nrrd_image, path_nrrd_mask)
                    df_temp2 = pd.DataFrame.from_dict(results2, orient="index")

                    overlap = df_temp1.index.intersection(df_temp2.index)
                    df_temp2 = df_temp2.drop(index=overlap)

                    df_temp = pd.concat([df_temp1, df_temp2], verify_integrity=True)
                    df_temp = df_temp.T
                    cols = df_temp.columns
                    df_temp["name"] = name
                    df_temp["time"] = time
                    df_temp["dose"] = dose_to_name(experiment, time, name)
                    df_temp = df_temp[["name", "time", "dose", *cols]]

                    if df_tot.empty:
                        df_tot = df_temp
                    else:
                        df_tot = df_tot.append(df_temp, ignore_index=True)
                    df_tot.to_csv(savepath)

    print(len(df_tot.T), "FEATURES EXTRACTED FROM ", len(df_tot), "IMAGES.")
    if os.path.exists(savepath) and not(overwrite):
        print("PATH EXISTS - DO YOU WANT TO OVERWRITE?  y / n")
        answ = input()
        if answ == "y":
            df_tot.to_csv(savepath)
            print("\nEXTRACTED FEATURES saved at", savepath)
            return 1
        else:
            print("\nEXTRACTED FEATURES NOT SAVED.")
            return 0
    else:
        df_tot.to_csv(savepath)
        print("\nEXTRACTED FEATURES saved at", savepath)
        return 1
    return 0


def select_LRsplit_optimal_norm(save_and_overwrite=True, drop_3d_wavelet=True, LRmode="split"):
    # NORM_DICT_T1 = get_best_feature_normalization("T1", return_df=False)
    # NORM_DICT_T2 = get_best_feature_normalization("T2")
    if not LRmode in ["split", "aggregated", "average"]:
        print(">>>> INVALID LRmode", LRmode)
        return 0


    NORM_DF_T1 = get_best_feature_normalization("T1", return_df=True, LRmode=LRmode)
    NORM_DF_T2 = get_best_feature_normalization("T2", LRmode=LRmode)

    FSPS_FTS = NORM_DF_T1.index.values   # FSPS optimized features
    print("NORM DF T1:", NORM_DF_T1.shape)
    print("NORM DF T2:", NORM_DF_T2.shape)
    print("FSPS FEATURES:", len(FSPS_FTS))
    NORM_LIST = ["no norm", "nyul", "stscore"]
    savepath_t1 = os.path.join(ftsDir, f"LR_{LRmode}_FSPS_extracted_T1.csv")
    savepath_t2 = os.path.join(ftsDir, f"LR_{LRmode}_FSPS_extracted_T2.csv")

    df_fsps_t1 = pd.DataFrame()
    df_fsps_t2 = pd.DataFrame()
    DF_DICT = {}

    for j, norm in enumerate(NORM_LIST):
        if LRmode != "average":
            df_path = os.path.join(ftsDir, f"LR_{LRmode}_" + "_".join(norm.split(" ")) + "_extracted.csv")
            df_curr = pd.read_csv(df_path, index_col=0)
        else:
            from select_utils import LRsplit_to_average
            df_path = os.path.join(ftsDir, f"LR_split_" + "_".join(norm.split(" ")) + "_extracted.csv")
            df_curr = pd.read_csv(df_path, index_col=0)
            df_curr = df_curr.drop(df_curr.filter(like="diagnostic", axis=1).columns, axis=1)
            df_curr = LRsplit_to_average(df_curr, include_dose=True)

        # index_original = df_curr.index
        df_curr.index = df_curr["name"] # make names index for easy filtering

        if j == 0:
            # ADD THE FEATURES UNAFFECTED BY FSPS
            # UNAFFECTED BY NORM; THEREFORE ONLY RUN ONCE
            overlap = df_curr.columns.intersection(FSPS_FTS)
            print("Overlapping features:", len(overlap))
            df_not_fsps = df_curr.drop(overlap, axis=1)
            not_fsps_fts = df_not_fsps.columns.values
            diagnostic = df_not_fsps.filter(like="diagnostic", axis=1).columns.values
            # print(diagnostic)
            print("Not in fsps:", len(not_fsps_fts))    # 44 = 32 + 9 + 3 (3: name, dose, time)
            print("Shape fts:", df_not_fsps.filter(like="shape", axis=1).shape) # 9
            print("Diagnostic:", diagnostic.shape) # 32
            df_not_fsps = df_not_fsps.drop(diagnostic, axis=1)
            df_not_fsps_t1 = df_not_fsps.filter(like="T1", axis=0)
            df_not_fsps_t2 = df_not_fsps.drop(df_not_fsps_t1.index, axis=0)

            df_fsps_t1 = pd.concat([df_fsps_t1, df_not_fsps_t1])
            df_fsps_t2 = pd.concat([df_fsps_t2, df_not_fsps_t2])
            print("MAIN DF T1:", df_fsps_t1.shape)
            print("MAIN DF T2:", df_fsps_t2.shape)

        print(j, norm, df_curr.shape)
        df_curr_t1 = df_curr.filter(like="T1", axis=0)
        df_curr_t2 = df_curr.drop(df_curr_t1.index, axis=0)
        # print("T1:", df_curr_t1.shape)
        # print("T2", df_curr_t2.shape)
        fts_curr_t1 = NORM_DF_T1[NORM_DF_T1["norm"] == norm].index.values
        fts_curr_t2 = NORM_DF_T2[NORM_DF_T2["norm"] == norm].index.values
        print("     T1 fts:", len(fts_curr_t1), end="")
        print("     T2 fts:", len(fts_curr_t2))
        # print(df_curr[fts_curr_t1])
        df_fsps_t1 = pd.concat([df_fsps_t1, df_curr_t1[fts_curr_t1]], axis=1)
        df_fsps_t2 = pd.concat([df_fsps_t2, df_curr_t2[fts_curr_t2]], axis=1)
        # print("MAIN DF T1:", df_fsps_t1.shape)
        # print("MAIN DF T2:", df_fsps_t2.shape)

        # print(df_curr["name"].filter("T1", axis=)
        # t1_idx = ["T1" in nm for nm in df_curr["name"].values]
        # df_curr_t1 = df_curr.loc[t1_idx]
        # df_curr_t2 = df_curr.drop(df_curr_t1.index, axis=0)
        # print(df_curr_t1)
        # print(df_curr_t2)
    print()
    if drop_3d_wavelet:
        # pass
        # df_fsps_t2 = df_fsps_t2.drop(df_fsps_t2.filter(like="wavelet-HH", axis=1).columns, axis=1)
        dfl = [df_fsps_t1, df_fsps_t2]
        for i in range(2):
            dfl[i] = dfl[i].drop(dfl[i].filter(like="wavelet-HH", axis=1).columns, axis=1)
            dfl[i] = dfl[i].drop(dfl[i].filter(like="wavelet-HL", axis=1).columns, axis=1)
            dfl[i] = dfl[i].drop(dfl[i].filter(like="wavelet-LH", axis=1).columns, axis=1)
            dfl[i] = dfl[i].drop(dfl[i].filter(like="wavelet-LL", axis=1).columns, axis=1)
        df_fsps_t1, df_fsps_t2 = dfl

    print("FINAL MAIN DF T1:", df_fsps_t1.shape)
    print("FINAL MAIN DF T2:", df_fsps_t2.shape)
    if save_and_overwrite:
        df_fsps_t1.to_csv(savepath_t1)
        df_fsps_t2.to_csv(savepath_t2)
        print("SAVED BOTH T1 AND T2 FSPS DATA")
    else:
        print("NOT SAVED...")
    return 1


def plot_images_with_glcm(data, mode="center", norm="nyul otsu decile"):
    import SimpleITK as sitk
    from radiomics.glcm import RadiomicsGLCM
    BW = 950
    # BW = 1250
    fig, ax = plt.subplots(nrows=2, ncols=len(data))
    fig.tight_layout()
    fig.suptitle(f"GLCM extracted from {mode} 2D slice after {norm} and FBW discretization with BW={BW}")
    # data = np.array(data)
    # times = np.array([int(x[:-3]) for x in data[:, 1]])
    # data = data[times.argsort()]
    for i, d in enumerate(data):
        exp, time, name = d
        print("\n", exp, time, name)
        path_nrrd = os.path.join(nrrdDir, mode + " " + norm, exp, time)
        path_nrrd_image = os.path.join(path_nrrd, name + "_image.nrrd")
        path_nrrd_mask = os.path.join(path_nrrd, name + "_mask.nrrd")
        im_nrrd = sitk.ReadImage(path_nrrd_image)
        mask_nrrd = sitk.ReadImage(path_nrrd_mask)

        im_orig = sitk.GetArrayFromImage(im_nrrd)
        mask = sitk.GetArrayFromImage(mask_nrrd)
        im_orig = discretize_FBW_ISBI(im_orig, mask, bw=BW)

        glcm = RadiomicsGLCM(im_nrrd, mask_nrrd, weightingNorm=None, binWidth=BW)
        print("Ng=", glcm.coefficients["Ng"])
        glcm._initCalculation()
        print(glcm.P_glcm.shape)
        P = glcm.P_glcm[0, :, :, 0]

        print("P-shape=", P.shape)
        im = glcm.imageArray
        mask = glcm.maskArray
        im = crop_to_mask(im, mask)
        img = crop_to_mask(im_orig, mask)
        mask = crop_to_mask(mask, mask)
        roi = np.ma.masked_where(np.logical_not(mask), im)

        ax[0, i].imshow(img, "gray")
        ax[0, i].imshow(roi, "hot", alpha=0.5)
        ax[0, i].set_title(" ".join([name, time]))
        ax[1, i].imshow(P, "hot")
        ax[1, i].set_title(f"GLCM for {P.shape[1]} graylevels: $\delta, \\theta= (1, 0)$")
        # break

    plt.show()
        # plot_nrrd(path_nrrd_image, path_nrrd_mask, " ".join(d))
    pass


if __name__ == "__main__":
    print()

    # print(central_idx_dict)
    # main2d(norm="stscore image")
    # main2d_fbw(norm="nyul brain", show_sample=False, bw=512)
    # main2d_fbw(norm="nyul brain", show_sample=True, bw=0)
    # main2d_fbw(norm="nyul otsu decile", show_sample=True, bw=0)

    # MAKE FIGURE WITH ROI + FEATURE (GLCM) OVER TIME
    # figure_data = find_raw_folders_from_name("9-4", conditions=["p", "T1"], exclude_conditions=True)
    # plot_images_with_glcm(figure_data, modenorm="center nyul otsu decile")

    # plot_images_with_glcm(figure_data, mode="center", norm="nyul otsu decile")
    # figure_data = find_raw_folders_from_name("6-2", conditions=["p", "T1"], exclude_conditions=True)
    # plot_images_with_glcm(figure_data, modenorm="center nyul otsu decile")


    # EXTRACCT
    make_nrrd_LRsplit(norm="no norm", mode="LR split", n4=False)    # raw images (for comparison)
    # make_nrrd_LRsplit(norm="no norm", mode="LR split")
    # make_nrrd_LRsplit(norm="stscore", mode="LR split")
    # make_nrrd_LRsplit(norm="nyul", mode="LR split")

    # main_extract_LRsplit(norm="no norm", overwrite=True)
    # main_extract_LRsplit(norm="stscore", overwrite=True)
    # main_extract_LRsplit(norm="nyul", overwrite=True)

    # LR split --> LR aggregated
    # from select_utils import LRsplit_to_aggregated
    # for NORM in ["no_norm", "nyul", "stscore"]:
    #     print(NORM)
    #     df = pd.read_csv(os.path.join(ftsDir, f"LR_split_{NORM}_extracted.csv"))
    #     df = LRsplit_to_aggregated(df)
    #     df.to_csv(os.path.join(ftsDir, f"LR_aggregated_{NORM}_extracted.csv"))

    select_LRsplit_optimal_norm(save_and_overwrite=True, drop_3d_wavelet=True, LRmode="average")


    # main_extract(mode="center", norm="nyul otsu decile", compare_to_manual=False)
    # make_nrrd(norm="nyul otsu decile", mode="LR split")
    # make_nrrd_LRsplit(norm="nyul otsu decile", mode="LR split")
    # main_extract_LRsplit(norm="nyul otsu decile")


    # FBW DISCRETIZATION HYPERPARAMETER TUNING of BW, for T1 and T2 (separate as T1 have lower pixel variability in salivary ROI maybe?)
    # n4bool = True
    # cols = ["roi mean", "roi median", "roi std", "roi min", "roi max", "im mean", "im median", "im std", "im min", "im max"]
    # # for norm in ["no norm", "nyul otsu decile", "stscore"]:
    # for norm in ["nyul otsu decile"]:
    #     if norm == "stscore":
    #         bwvals = np.arange(0.05, 0.15, 0.025)
    #         # bwvals = [0.07]
    #     else:
    #         # bwvals = np.arange(850, 1100, 50)
    #         bwvals = np.arange(700, 850, 50)
    #         # bwvals = [950]
    #     SavePathT1 = os.path.join(RawDir, "..", "Radiomic features\discretization", f"discretization_T1_{norm}_fbw.csv")
    #     # SavePathT2 = os.path.join(RawDir, "..", "Radiomic features\discretization", f"discretization_T2_{norm}_fbw.csv")
    #     # bwresults = []
    #     bwresults_t1 = []
    #     for bw in bwvals:
    #         # bwresults.append(disc_FBW_count_bins_LRsplit(norm=norm, t1_images=False, show_sample=False, bw=bw, n4=n4bool))
    #         bwresults_t1.append(disc_FBW_count_bins_LRsplit(norm=norm, t1_images=True, show_sample=False, bw=bw, n4=n4bool))
    #     # df_t1 = pd.DataFrame(index=bwvals, data=bwresults_t1, columns=cols)
    #     # df_t1.to_csv(SavePathT1)
    #     # df_t2 = pd.DataFrame(index=bwvals, data=bwresults, columns=cols)
    #     # df_t2.to_csv(SavePathT2)
    #     print("\n\nFIXED BIN WIDTH HYPERPARAMETER TUNING: WANT # BINS IN ROI IN RANGE 30 - 130")
    #     print(f"T2 IMAGES    bw, median, std, min, max (# bins in ROI): (N={len(bwresults)})")
    #     for bw, res in zip(bwvals, bwresults):
    #         print(bw, "{}".format([round(x, 1) for x in res]))
    #     print(f"\nT1 IMAGES    bw, median, std, min, max (# bins in ROI): (N={len(bwresults_t1)})")
    #     for bw, res in zip(bwvals, bwresults_t1):
    #         print(bw, "{}".format([round(x, 1) for x in res]))

    # COMPARING NORM METHODS AT PERCENTILES / WHOLE
    # compare_norm_methods(show_sample=False, save=True, n4=True, normmodes=["raw", "stscore image", "nyul otsu decile"])
    # plot_cv_profile_norm_comparison(qcd_bool=False)
    # plot_cv_profile_norm_comparison(qcd_bool=True)
    # plot_cv_profile_norm_comparison(qcd_bool="both")
    # cv_compare_norm()


    # plot_hist_from_all(savebool=True, n4=True, norm="raw", pbool="all", t1bool=True)
    # plot_hist_from_all(savebool=True, n4=True, norm="nyul otsu decile", pbool="all", t1bool=True)
    # plot_hist_from_all(savebool=True, n4=False, norm="raw", pbool="all", t1bool=True)
    # plot_hist_from_all(savebool=True, n4=False, norm="nyul otsu decile", pbool="all", t1bool=True)

    # plot_hist_from_all(savebool=True, n4=True, norm="stscore image", pbool="all", t1bool=True)
    # plot_hist_from_all(savebool=True, n4=True, norm="stscore image", pbool="all", t1bool=False)

    # plot_hist_from_all(savebool=True, n4=True, norm="raw", pbool="all", t1bool=False)
    # plot_hist_from_all(savebool=True, n4=True, norm="nyul otsu decile", pbool="all", t1bool=False)
    # plot_hist_from_all(savebool=True, n4=False, norm="raw", pbool="all", t1bool=False)
    # plot_hist_from_all(savebool=True, n4=False, norm="nyul otsu decile", pbool="all", t1bool=False)

    # plot_values_in_roi(n4=True)
    # plot_values_in_roi(n4=False)
    # plot_hist_from_all(n4=False)
    # plot_hist_from_all(n4=True)
    # plot_hist_from_all(n4=True, norm="stscore image")
    # plot_hist_from_all(n4=True, norm="stscore ROI brain")
    # plot_hist_from_all(n4=True, pbool="all", norm="raw")
    # plot_hist_from_all(n4=True, pbool="all", norm="nyul mean decile")
    # plot_hist_from_all(n4=True, pbool="all", norm="stscore image")
    # plot_hist_from_all(n4=True, pbool="all", norm="nyul brain decile")
    # plot_hist_from_all(n4=True, pbool="all", norm="nyul brain mode")

    # plot_hist_from_all(n4=True, norm="stscore ROI saliv")
    # compare_norm_methods(show_sample=False, save=True, n4=True, normmodes=["raw", "stscore image", "stscore brain", "nyul brain decile", "nyul brain mode", "nyul mean decile", "nyul otsu decile"])
    # compare_norm_methods(show_sample=False, save=True, n4=True, stscore_first=True, normmodes=["stscore image", "stscore brain", "nyul brain decile", "nyul brain mode", "nyul mean decile", "nyul otsu decile", "nyul otsu mode"])
    # plot_cv_profile_norm_comparison(stscore_first=False)
    # plot_cv_profile_norm_comparison(stscore_first=False, qcd_bool=True)
    # plot_cv_profile_norm_comparison(stscore_first=False, qcd_bool=False)
    # cv_compare_norm(stscore_first=True)