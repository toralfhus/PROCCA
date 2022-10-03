import numpy as np
import time
import skimage.filters
from DICOM_reader import dcm_files_indexed, dcm_folder_to_pixel_matrix, find_folders, load_matrix
from preprocessing import norm_minmax_featurescaled, percentile_truncation, get_name
from watershed import browse_images_make_watershed_ROI_2D
# from sklearn.cluster import KMeans
from visualizations import plot_all_cropped_npy, plot_masked
import os
from matplotlib import pyplot as plt
import pandas as pd
import cv2
from MRI_denoising import n4correction
from extract_utils import *

def timeit(func):
    """
    Decorator for measuring function's running time.
    """
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.3e seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


# @timeit
def skimage_gaussian(m, s=1):
    return skimage.filters.gaussian(m, sigma=s)


def main(experiment, time, cropped, segfunc, params2d=(4, 5, 10, 2), kmeans=False):
    try:
        dfp = pd.read_csv(os.path.join(os.getcwd(), "..", "Segmentations", "watershed 2d man params.csv"))
        dfp = dfp.set_index(keys=dfp["ID"])
        print(dfp.T)
        print([x for x in dfp.columns if "ID." in x])
        dfp = dfp.drop([x for x in dfp.columns if "ID." in x], axis=1)
        print(dfp.T)
    except Exception as e:
        print(e.args)
        dfp = pd.DataFrame()
    if cropped:
        path_parent = os.path.normpath(os.path.join(os.getcwd(), "..", r"Segmentations\cropped_salivary", experiment, "raw", time))
        print(path_parent)
        for file in os.listdir(path_parent):
            # matr = load_matrix(os.path.join(path_parent, file))
            matr = load_matrix(os.path.join(path_parent, file), printtext=False)
            # print(file, matr.shape)
            # matr = percentile_truncation(matr, 2.5, 99.5)
            # matr = percentile_truncation(matr, 30, 100)
            folder_save = os.path.normpath(os.path.join(path_parent, "..", "..", "roi 2d watershed", time))
            # print(folder_save)
            if not os.path.exists(folder_save): os.makedirs(folder_save)
            # roiman = browse_images_make_watershed_ROI_2D(matr, title=file, time="", params=params2d, save_path=folder_save)

            # SKIP ALREADY SEGMENTED FILES
            filesavepath = os.path.join(folder_save, file)
            print(file[:-4])
            reseg = False
            if (time in ["-7day"] and file[:-4] in ["4-1__after_p_sagittal"]) or (time in ["8day"] and file[:-4] in ["1-1__after_p_sagittal"]):
                reseg = True
                print("RESEGMENT THIS MOFO")
            if (not os.path.exists(filesavepath)) or reseg:
                ROI = segfunc(matr, title=file, time="", params=params2d, save_path=folder_save)
                # ROI = segfunc(matr, title=file, time="", params=params2d, save_path=r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary\pilot1\roi 2d watershed\56day\test")  #TESTING TUPLE SAVING WHILE SAVING ROIS IN TEST FOLDER
                if ROI.any():   #save parameters as pandas dataframe csv
                    tup = ", ".join(map(str, params))
                    IDstring = experiment + " " + time + " " + file[:-4]
                    ds = pd.Series(data={"ID":IDstring, "experiment":experiment, "time":time, "name":file[:-4], "pilocarp":"p" in file[:-4], "params":tup, "kmeans":kmeans}, name=IDstring)
                    # print(ds["ID"], dfp["ID"].values)
                    # print(ds["ID"] in dfp["ID"].values and ds["time"] in dfp["time"].values)
                    notindf = True
                    try:
                        if ds["ID"] in dfp["ID"].values and ds["time"] in dfp["time"].values:
                            notindf = False
                            print("ID IS IN DF")
                            print(dfp)
                            print(ds.to_frame().T)
                            dfp.update(ds.to_frame().T)     #TODO: WHY UPDATING DATAFRAME MAKE NEW COLS ID.1.1 etc?? Maybe in to_csv()?
                            print(dfp.T)
                    except Exception as e:
                        print("ERROR IN SAVING PARAMS bro", *e.args)
                    if notindf:
                        dfp = dfp.append(ds.to_frame().T)
                        # dfp = dfp.concat(ds.to_frame().T)
                    dfp = dfp.set_index(keys=dfp["ID"])
                    print(dfp["ID"])
                    dfp = dfp.sort_index()
                    print(dfp)
                    dfp.to_csv(os.path.join(os.getcwd(), "..", "Segmentations", "watershed 2d man params.csv"))
            # elif file in ["3-2__after_p_sagittal.npy", "4-1__after_p_sagittal.npy", "1-1_sagittal.npy"]:  #specify files to resegment
            #     segfunc(matr, title=file, time="", params=params2d, save_path=folder_save)
            else:
                # print("ROI found at ", filesavepath, "\n")
                pass

    if not cropped:
        path_parent = os.path.normpath(os.path.join(os.getcwd(), "..", r"RAW DATA", experiment, time))
        for folder in find_folders(path_parent, "sagittal"):
            if experiment == "pilot1":
                name = folder
            elif experiment == "pilot2":
                name = folder[12:15] if not "p" in folder else folder[12:24]
            else:
                print(experiment, "is not a valid experiment. Try again.")
                break
            print("\n", name)
            indexed = dcm_files_indexed(os.path.join(path_parent, folder))
            matr = dcm_folder_to_pixel_matrix(indexed, os.path.join(path_parent, folder))
            # matr = percentile_truncation(matr, 2.5, 99.5)

            # plot_image(matr[14], title="RAW " + name)
            folder_save = os.path.normpath(os.path.join(path_parent, "..", "..", "..", r"Segmentations\no crop", experiment, "roi 2d watershed", time))
            if not os.path.exists(folder_save): os.makedirs(folder_save)
            # print(folder_save)
            roiman = browse_images_make_watershed_ROI_2D(matr, title=name, time="", params=params2d, save_path=folder_save)
    pass


def mainmain_cropped_2d(segfunc, params, kmeans=False):
    for pilot in ["pilot1", "pilot2"]:
        print("\n\n")
        print(pilot.upper())
        print("-"*50)
        for time in find_folders(os.path.join(os.getcwd(), "..", "Segmentations\cropped_salivary", pilot, "raw")):
            print(pilot.upper(), time)
            print("-"*50)
            main(pilot, time, cropped=True, segfunc=segfunc, params2d=params, kmeans=kmeans)
    pass


def validate_segmentations_at_slice(idx, includep=True, save=False ):
    for pilot in ["pilot1", "pilot2"]:
        print(pilot.upper())
        for time in ["-7day", "8day", "56day"] if pilot=="pilot1" else ["-7day", "8day", "26day"]:
            print(time.upper())
            if includep:
                plt = plot_all_cropped_npy(experiment=pilot, time=time, ROI=True, idx=idx)
            else:
                plt = plot_all_cropped_npy(experiment=pilot, time=time, ROI=True, exclude="_p", idx=idx)
            manager = plt.get_current_fig_manager()
            # manager.resize(*manager.window.maxsize())
            manager.window.showMaximized()
            # plt.rcParams.update({'font.size': 16})
            if save:
                savefolder = os.path.join(os.getcwd(), "..", "master_plots\segmentation\central slice", "idx" + str(idx))
                if not os.path.exists(savefolder):  os.makedirs(savefolder)
                plt.savefig(os.path.join(os.getcwd(), "..", "master_plots\segmentation\central slice", "idx"+str(idx), "".join([f"{'''''' if includep else '''no p '''}", pilot, " ", time])), dpi=600, bbox_inches="tight")
                plt.show()
                plt.close()
                print("IMAGE SAVED")
    pass


def main_segment_brain(segfunc, params2d=(4, 5, 10, 2), mice_to_resegment=[]):
    conut = 0
    import cv2
    df_seg_brain_path = os.path.join(os.getcwd(), "..", "..", r"Segmentations\brain\watershed params brain.csv")
    df_seg_brain = pd.read_csv(df_seg_brain_path, index_col=0)#, index_col="Name")
    df_idx = pd.read_csv(os.path.join(os.getcwd(), "..", "..", "Segmentations\salivary\segment params salivary.csv"), index_col=0)
    # print(df_idx)
    print(df_seg_brain)
    # print(df_seg_brain.loc[df_seg_brain["Name"] == "L2_sagittal"].loc[df_seg_brain["Time"] == "8day"])    # and df_seg_brain["Time"] == "8day"])
    # print("bad" in df_seg_brain.loc[df_seg_brain["Name"] == "L2_sagittal"].loc[df_seg_brain["Time"] == "8day"]["bad?"].values[0])

    savepath_parent = os.path.normpath(os.path.join(os.getcwd(), "..", "..", r"Segmentations\brain"))
    for experiment in ["Pilot1", "Pilot2", "Pilot3"]:
        print("\n\n")
        print(experiment.upper())
        print("-"*50)
        # print(find_folders(os.path.join(os.getcwd(), "..", "RAW DATA", experiment)))

        for time in find_folders(os.path.join(os.getcwd(), "..", "..", "RAW DATA", experiment)):
            print("\n",experiment.upper(), time)
            print("-"*50)
            datadir = os.path.normpath(os.path.join(os.getcwd(), "..", "..", r"RAW DATA", experiment, time))
            for folder in find_folders(datadir, "sagittal"):
                name = get_name(experiment, folder, condition="sagittal")
                try:
                    idx_center = df_idx.loc[df_idx["time_name"] == time + name]["idx_central"].values[0]
                except Exception as e:
                    print(e.args)
                    idx_center = 0
                print(name, " w/ center @ idx", idx_center)

                segbool = False
                folder_save = os.path.join(savepath_parent, experiment.lower(), time)
                if not os.path.exists(os.path.join(folder_save, name + ".npy")):
                    print("NO ROI EXISTS: MAKING ROI")
                    segbool = True
                    resegbool = False

                elif "very bad" in df_seg_brain.loc[df_seg_brain["time_name"] == time+name]["seg_qual"].values[0]:
                    print("ROI EXISTS, BUT BAD: MAKING NEW ROI")
                    segbool = True
                    resegbool = True
                if segbool:
                    indexed = dcm_files_indexed(os.path.join(datadir, folder), True)
                    MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder))
                    # MATR_PROC = percentile_truncation(MATR_RAW.copy(), 50, 99, settozero=True)
                    MATR_PROC = MATR_RAW.copy()
                    MATR_PROC = norm_minmax_featurescaled(MATR_PROC, 0, 255).astype("uint8")


                    print(name, MATR_RAW.shape)
                    # print()
                    im_raw = MATR_RAW[MATR_RAW.shape[0] // 2]   # CENTRAL SLICE
                    # im_disp = MATR_PROC[MATR_PROC.shape[0] // 2]
                    im_disp = MATR_PROC[idx_center]

                    im_msk = np.ma.masked_where(im_disp == 0, im_disp)  # no values instead of zero @ lower truncations
                    # plot_image(im_msk)

                    # im_eq = cv2.equalizeHist(im_msk)
                    # im_eq2 = cv2.equalizeHist(im_disp)
                    # MATR_PROC = cv2.equalizeHist(MATR_PROC)
                    for i, slice in enumerate(MATR_PROC):
                        # TODO: CHANGE THIS TO CLAHE - NO MASK NEEDED, TAKES LOCAL VARIABILITY INTO ACCOUNT!!!!!
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        cl1 = clahe.apply(slice)
                        # plt.imshow(cl1)
                        # plt.show()
                        # MATR_PROC[i] = cv2.equalizeHist(slice)
                        MATR_PROC[i] = cl1
                    # plot_image(im_eq, title=experiment+time+name)
                    # compare_images(im_msk, im_eq, suptitle=experiment+time+name, title1="mask", title2="eql mask")
                    # compare_images(im_eq, im_eq2, title1="eql mask", title2="eql trunc", suptitle=experiment+time+name)
                    # show_histogram(im_eq, bins=50, titleimg="eql msk")
                    # show_histogram(im_eq2, bins=50, titleimg="eql trunc")
                    # show_histogram(im_msk)

                    # SEGMENTATION STUFF
                    # MATR_PROC = percentile_truncation(MATR_RAW.copy(), lower=10, upper=99, settozero=True)
                    # roi = browse_images_make_watershed_ROI_2D(MATR_RAW, title=name, time="", params=params2d, save_path=folder_save, truncvals=(0, 98))
                    # print(df_seg_brain.loc[df_seg_brain["Name"] == name].loc[df_seg_brain["Time"] == time]["bad?"].values[0])
                    # print("bad" in df_seg_brain.loc[df_seg_brain["Name"] == name].loc[df_seg_brain["Time"] == time]["bad?"].values[0])
                    if not os.path.exists(os.path.join(folder_save, name + ".npy")):
                        print("ROI NOT FOUND: MAKING ROI")
                        ROI, _ = browse_images_make_watershed_ROI_2D(MATR_PROC, title=name, time="", params=params2d, save_path=folder_save, truncvals=(0, 98), contrast=False)
                        # roi = segfunc(MATR_RAW, title=name, time="", params=params2d, save_path=folder_save)
                        # roi_center = ROI[ROI.shape[0] // 2]
                        roi_center = ROI[idx_center]
                        plot_masked(im_disp, roi_center, title=time+name)
                        if ROI.any():
                            df_seg_brain = df_seg_brain.append({"time_name":time+name, "exp":experiment.lower(), "params":params2d, "seg_qual":"unknown"}, ignore_index=True)

                    elif "very bad" in df_seg_brain.loc[df_seg_brain["Name"] == name].loc[df_seg_brain["Time"] == time]["bad?"].values[0]: # Resegment BAD segmentations from segfile
                        print("BAD OLD ROI: MAKING NEW")
                        ROI_old = np.load(os.path.join(os.getcwd(), "..", "..", r"Segmentations\brain", experiment.lower(), time, name + ".npy"))
                        # roi_center_old = ROI_old[ROI_old.shape[0] // 2]
                        roi_center_old = ROI_old[idx_center]
                        plot_masked(im_disp, roi_center_old, "old ROI " + time + name)
                        ROI, _ = browse_images_make_watershed_ROI_2D(MATR_PROC, title=name, time="", params=params2d,
                                                                  save_path=folder_save, truncvals=(0, 98), contrast=False)
                        # roi_center = ROI[ROI.shape[0] // 2]
                        roi_center = ROI[idx_center]
                        plot_masked(im_disp, roi_center, title=time + name)
                        # TODO: update brain seg df
                    if ROI.any():
                        df_seg_brain.to_csv(df_seg_brain_path)
                        df_seg_brain = pd.read_csv(df_seg_brain_path, index_col=0)
                        print(df_seg_brain)
                    else:
                        print("NOTHING IN ROI --> SKIPPING")
                conut += 1
                print()
    print("FOUND", conut, "MICE IN TOTAL")
    pass


def main_segment_gland(segfunc, params2d=(4, 5, 10, 2), histeq=True, n4=False, resegnames=[]):
    conut = 0
    # import cv2
    pd.options.mode.chained_assignment = None   # Sketchy disabler for overwriting loaded DataFrame
    path_df_seg = os.path.join(os.getcwd(), "..", "..", r"Segmentations\salivary\segment params salivary.csv")
    df_seg = pd.read_csv(path_df_seg, index_col=0)
    # df_seg = pd.DataFrame({"idx":[], "time_name":[], "exp":[], "p":[], "params":[], "seg_quality":[], "num_slices_seg":[]})
    # df_seg.index = df_seg["idx"]
    print(df_seg)
    # print(df_seg.loc[df_seg["Name"] == "L2_sagittal"].loc[df_seg["Time"] == "8day"])    # and df_seg["Time"] == "8day"])
    # print("bad" in df_seg.loc[df_seg["Name"] == "L2_sagittal"].loc[df_seg["Time"] == "8day"]["bad?"].values[0])
    print("BEGINNING SEGMENTATION OF SALIVARY GLAND")
    savepath_parent = os.path.normpath(os.path.join(os.getcwd(), "..", "..", r"Segmentations\salivary"))
    # for experiment in ["Pilot1", "Pilot2", "Pilot3"]:
    for experiment in find_folders(RawDir, "Pilot"):
        print("\n\n", experiment.upper(), "\n" + "-"*50)
        for time in find_folders(os.path.join(os.getcwd(), "..", "..", "RAW DATA", experiment)):
            print("\n",experiment.upper(), time)
            print("-"*50)
            datadir = os.path.normpath(os.path.join(os.getcwd(), "..", "..", r"RAW DATA", experiment, time))
            for folder in find_folders(datadir, "sagittal"):
                name = get_name(experiment, folder, condition="sagittal")
                print(name)
                folder_save = os.path.join(savepath_parent, experiment.lower(), time)

                # segbool = False
                segbool = not os.path.exists(os.path.join(folder_save, name + ".npy"))

                # resegnames = []
                resegbool = False
                if os.path.exists(os.path.join(folder_save, name + ".npy")):
                    idx_center = df_seg.loc[df_seg["time_name"] == time + name]["idx_central"].values[0]
                    ROI_old = np.load(os.path.join(folder_save, name + ".npy"))
                    # print("BROBRO\n", time + name)
                    if not ROI_old[idx_center].any() or time+name in resegnames:
                        print("----- REMAKING ROI -----")
                        segbool = True
                        resegbool = True
                        df_seg_idx = df_seg.loc[df_seg["time_name"] == time+name].index[0]
                        print("df idx = ", df_seg_idx)
                        print(df_seg.iloc[df_seg_idx])
                        print(df_seg.iloc[df_seg_idx]["time_name"])
                # if not segbool and pd.isna(df_seg.loc[df_seg["time_name"] == time+name]["idx_central"].values):
                #     print(df_seg.loc[df_seg["time_name"] == time+name])
                #     segbool = True
                if segbool:
                    print("----- ROI NOT FOUND: MAKING ROI -----") if not os.path.exists(os.path.join(folder_save, name + ".npy")) else print("----- MAKING NEW ROI -----")
                    print(time + name)
                    indexed = dcm_files_indexed(os.path.join(datadir, folder), True)
                    MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder))
                    # MATR_PROC = percentile_truncation(MATR_RAW.copy(), 50, 99, settozero=True)
                    MATR_PROC = MATR_RAW.copy()
                    MATR_PROC = norm_minmax_featurescaled(MATR_PROC, 0, 255).astype("uint8")

                    print(name, MATR_RAW.shape)
                    # print()
                    im_raw = MATR_RAW[MATR_RAW.shape[0] // 2]  # CENTRAL SLICE
                    im_disp = MATR_PROC[MATR_PROC.shape[0] // 2]

                    im_msk = np.ma.masked_where(im_disp == 0, im_disp)  # no values instead of zero @ lower truncations
                    # plot_image(im_msk)
                    # MATR_PROC = cv2.GaussianBlur(MATR_PROC, (5, 5), 0)
                    if n4:
                        print("\t--- applying n4 correction ---")
                        print("-"*len(MATR_PROC))
                        for i in range(len(MATR_PROC)):
                            MATR_PROC[i], _, _ = n4correction(MATR_PROC[i], verbose=False)
                            print("-", end="")
                    if histeq:
                        for i, slice in enumerate(MATR_PROC):
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            cl1 = clahe.apply(slice)
                            # plt.imshow(cl1)
                            # plt.show()
                            # MATR_PROC[i] = cv2.equalizeHist(slice)
                            MATR_PROC[i] = cl1
                    if resegbool:
                        print("\nCENTRAL SLICE FOR", time+name,"=\t", central_idx_dict[time + name])
                    print(time + name)
                    ROI, idx = browse_images_make_watershed_ROI_2D(MATR_PROC, title=name, time="", params=params2d, save_path=folder_save, truncvals=(0, 98), contrast=False)
                    # roi = segfunc(MATR_RAW, title=name, time="", params=params2d, save_path=folder_save)
                    # roi_center = ROI[ROI.shape[0] // 2]
                    num_slices_segmented = 0
                    for roisl in ROI:
                        if roisl.any():
                            num_slices_segmented += 1
                    plot_masked(MATR_PROC[idx], ROI[idx], title=f"{time+name} @ slice {idx} \n{num_slices_segmented} slices segmented")
                    if num_slices_segmented != 0:
                        if not resegbool:
                            df_seg = df_seg.append({"time_name":time+name, "exp":experiment.lower(), "p":"p" in name, "params":params2d, "seg_quality":"unknown", "num_slices_seg":num_slices_segmented, "idx_central":idx}, ignore_index=True)
                            print(df_seg)
                            # df_seg.to_csv(path_df_seg)#, columns=df_seg.columns)
                            # df_seg = pd.read_csv(path_df_seg, index_col=0)
                        else:   # RESEGMENTING OLD ROI: UPDATE VALUES IN CSV INSTEAD OF APPENDING
                            # print(df_seg.iloc[df_seg_idx]["time_name"])
                            print("UPDATING DF DUE TO RESEG:")
                            # print(params2d, num_slices_segmented, idx)
                            df_seg.at[df_seg_idx, "num_slices_seg"] = num_slices_segmented
                            df_seg.at[df_seg_idx, "params"] = params2d
                            df_seg.at[df_seg_idx, "idx_central"] = idx
                            print(df_seg.iloc[df_seg_idx])
                            missingSet = set(resegnames)
                            print(timename, missingSet)
                            missingSet.remove(time + name)
                            resegnames = list(missingSet)
                            np.savetxt(X=np.array(resegnames), fname=missingPath, delimiter=", ", fmt="%s")
                            print(len(resegnames), "LEFT IN MISSING LIST FOR SEGMENTING\n")

                        df_seg.to_csv(path_df_seg)#, columns=df_seg.columns)
                        df_seg = pd.read_csv(path_df_seg, index_col=0)
                        print(df_seg)
                        print("--- DF UPDATED ---")
                # elif "very bad" in df_seg.loc[df_seg["Name"] == name].loc[df_seg["Time"] == time]["bad?"].values[0]: # Resegment BAD segmentations from segfile
                #     print("BAD OLD ROI: MAKING NEW")
                #     ROI_old = np.load(os.path.join(os.getcwd(), "..", "..", r"Segmentations\salivary", experiment.lower(), time, name + ".npy"))
                #     roi_center_old = ROI_old[ROI_old.shape[0] // 2]
                #     plot_masked(im_disp, roi_center_old, "old ROI " + time + name)
                #     ROI = browse_images_make_watershed_ROI_2D(MATR_PROC, title=name, time="", params=params2d,
                #                                               save_path=folder_save, truncvals=(0, 98), contrast=False)
                #     roi_center = ROI[ROI.shape[0] // 2]
                #     plot_masked(im_disp, roi_center, title=time + name)
                conut += 1
                print()
                # plot_image(im_eq, title=experiment+time+name)
                # compare_images(im_msk, im_eq, suptitle=experiment+time+name, title1="mask", title2="eql mask")
                # compare_images(im_eq, im_eq2, title1="eql mask", title2="eql trunc", suptitle=experiment+time+name)
                # show_histogram(im_eq, bins=50, titleimg="eql msk")
                # show_histogram(im_eq2, bins=50, titleimg="eql trunc")
                # show_histogram(im_msk)
    print("FOUND", conut, "MICE IN TOTAL")
    pass


def show_ROIs(offsetIDX=0, mode="saliv", plot=True, showtimeexp="all"):
    # evaluating at slice in center + idxshift of dcm matrix
    # b = "+"+str(offsetIDX) if offsetIDX>=0 else str(offsetIDX)
    offsetIDXorig = offsetIDX
    df_path = os.path.join(os.getcwd(), "..", "..", "Segmentations\salivary\segment params salivary.csv")
    df = pd.read_csv(df_path, index_col=0)
    # print(b)
    if offsetIDX in ["L", "R", "LR"]:
        df_lr = pd.read_csv(os.path.join(SegSalivDir, "salivary LR split indexes.csv"), index_col=0)
        offsetIDXorig = offsetIDX
        offsetIDXlist = [a for a in offsetIDX]
        print(offsetIDXlist)
        # print(df_lr)
    elif type(offsetIDXorig) == int:
        offsetIDXlist = [offsetIDXorig]
    else:
        offsetIDXlist = offsetIDXorig
    missingSliceList = []
    if mode == "brain":
        ROIpath_parent = os.path.normpath(os.path.join(os.getcwd(), "..", "..", r"Segmentations\brain"))
    elif mode in ["gland", "saliv", "salivary"]:
        ROIpath_parent = os.path.normpath(os.path.join(os.getcwd(), "..", "..", r"Segmentations\salivary"))
    else:
        print("Mode", mode, "not recognized..")
        return 0
    for experiment in ["Pilot1", "Pilot2", "Pilot3"]:
    # for experiment in ["Pilot3"]:
        # print("\n\n")
        # print(experiment.upper())
        # print("-"*50)
        for time in find_folders(os.path.join(os.getcwd(), "..", "..", "RAW DATA", experiment)):
            print("\n",experiment.upper(), time)
            print("-"*50)
            datadir = os.path.normpath(os.path.join(os.getcwd(), "..", "..", r"RAW DATA", experiment, time))
            num_mice = len(find_folders(datadir, "sagittal"))
            print(num_mice // 3)
            # todo: expand to 4 rows???
            nrows = 3 if num_mice//3 >= 4 else 2
            if num_mice // 4 > 10:
                nrows = 5
            elif num_mice // 3 > 10:
                nrows = 4
            ncols = num_mice // nrows + 1
            print(num_mice, ncols, nrows, ncols * nrows)

            isMissingSlices = False
            for offsetIDX in offsetIDXlist:
                fig, axes = plt.subplots(nrows, ncols)
                axes = axes.ravel()
                if type(offsetIDX) == str:
                    fig.suptitle(experiment +" " + time + f" ROI @ {offsetIDX} {mode}")
                else:
                    fig.suptitle(experiment +" " + time + f" @ central slice" if not offsetIDX else experiment +" " + time + f" @ central slice + {offsetIDX} idx")

                for i, folder in enumerate(find_folders(datadir, "sagittal")):
                    name = get_name(experiment, folder, condition="sagittal")
                    # print(df.loc[df["time_name"] == time+name, "idx_central"].values)

                    #TODO: add exception for when ROI is not created
                    idx_center = df.loc[df["time_name"] == time+name, "idx_central"].values[0]
                    indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
                    MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder), printbool=False)
                    # im_raw = MATR_RAW[MATR_RAW.shape[0] // 2 + offsetIDX]
                    if offsetIDXorig in ["L", "R", "LR"]:
                        dfcurr = df_lr[df_lr["time_name"] == time + name]
                        # print(dfcurr["idx_l"].values)
                        # print(dfcurr)
                        if offsetIDX.lower() == "l":
                            idx = int(dfcurr["idx_l"].values)
                        elif offsetIDX.lower() == "r":
                            idx = int(dfcurr["idx_r"].values)
                        else:
                            print("OFFSET", offsetIDX, "INVALID\n\n")
                            break
                    else:
                        idx = idx_center + offsetIDX
                    # im_raw = MATR_RAW[idx_center + offsetIDX]
                    im_raw = MATR_RAW[idx]
                    print(time + name)
                    ROI = np.load(os.path.join(ROIpath_parent, experiment, time, name + ".npy"))
                    # roi = ROI[idx_center + offsetIDX]
                    roi = ROI[idx]
                    if not roi.any():
                        isMissingSlices = True
                        timename = time + name
                        if not timename in missingSliceList:
                            missingSliceList.append(timename)
                        print("MISSING \n\n\nn")
                    im_disp = norm_minmax_featurescaled(im_raw, 0, 255)
                    im_disp = percentile_truncation(im_disp, 0, 99)

                    axes[i].imshow(im_disp, cmap="gray")
                    axes[i].imshow(roi, alpha=0.3)
                    # axes[i].set_title(name + " center=" + str(idx_center))
                    axes[i].set_title(name + " idx=" + str(idx), fontsize=7)
                for ax in axes:
                    ax.axis("off")
                # axes[i].axis("off")
                fig.tight_layout()
            print(missingSliceList)
            plt.show() if plot else plt.close()
    return missingSliceList


def show_saliv_ROI_LRsplit(plot=True, include_center=False, shift=0):
    # set shift != 0 to NOT use manually found LR split indices, but instead shift-distance from idx_center
    df_path = os.path.join(SegSalivDir, "salivary LR split indexes.csv")
    df = pd.read_csv(df_path, index_col=0)
    print(df)
    missing_roi_list = []
    for experiment in ["Pilot1", "Pilot2", "Pilot3"]:
        for time in find_folders(os.path.join(os.getcwd(), "..", "..", "RAW DATA", experiment)):
            print("\n",experiment.upper(), time)
            print("-"*50)
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            num_mice = len(find_folders(datadir, "sagittal"))
            # nrows = 3 if num_mice//3 >= 4 else 2
            # ncols = num_mice // nrows + 1
            ncols = num_mice
            nrows = 2 + include_center
            # print(num_mice, ncols, nrows, ncols * nrows)
            isMissingSlices = False
            if plot:
                fig, axes = plt.subplots(nrows, ncols)
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.01, wspace=0.1)
                for ax in axes.ravel():
                    ax.axis("off")
            for i, folder in enumerate(find_folders(datadir, "sagittal")):
                name = get_name(experiment, folder, condition="sagittal")
                # print("\n", time+name)
                idx_c = central_idx_dict[time+name]
                if shift:   # shift from center
                    idx_l = idx_c - shift
                    idx_r = idx_c + shift
                else:   # manual tabulated values
                    dfcurr = df[df["time_name"] == time + name]
                    # print(dfcurr["idx_l"].values)
                    idx_l = int(dfcurr["idx_l"].values)
                    idx_r = int(dfcurr["idx_r"].values)
                print(i, time + name, idx_l, idx_r)

                # idx_center = df.loc[df["time_name"] == time + name, "idx_central"].values[0]
                indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
                MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder), printbool=False)
                im_l = percentile_truncation(MATR_RAW[idx_l], lower=0, upper=99, verbose=False)
                im_r = percentile_truncation(MATR_RAW[idx_r], lower=0, upper=99, verbose=False)
                ROI = np.load(os.path.join(SegSalivDir, experiment, time, name + ".npy"))
                roi_l = ROI[idx_l]
                roi_r = ROI[idx_r]
                print(roi_l.any(), roi_r.any(), not(roi_l.any()) or not(roi_r.any()), not(roi_l.any() and roi_r.any()))
                if not(roi_l.any()) or not(roi_r.any()):
                    # print("MISSING ROI FOR", time + name, "\n")
                    missing_roi_list.append(time + name)

                if plot:
                    msk_l = np.ma.masked_where(np.logical_not(roi_l), im_l)
                    msk_r = np.ma.masked_where(np.logical_not(roi_r), im_r)
                    axes[0, i].imshow(im_l, cmap="gray")
                    axes[0, i].imshow(msk_l, cmap="hot")
                    axes[0, i].set_title(name + " idx_l=" + str(idx_l))
                    axes[-1, i].imshow(im_r, cmap="gray")
                    axes[-1, i].imshow(msk_r, cmap="hot")
                    axes[-1, i].set_title("idx_r=" + str(idx_r))
                    if include_center:
                        im_c = percentile_truncation(MATR_RAW[idx_c], lower=0, upper=99, verbose=False)
                        roi_c = ROI[idx_c]
                        msk_c = np.ma.masked_where(np.logical_not(roi_c), im_c)
                        axes[1, i].imshow(im_c, cmap="gray")
                        axes[1, i].imshow(msk_c, cmap="hot")
                        axes[1, i].set_title("idx_c=" + str(idx_c))
            if plot:
                fig.suptitle(time)
                plt.show()
                plt.close()
            # break
        # break
    print("MISSING ROI FOR ", len(missing_roi_list), "MICE")
    return missing_roi_list


def show_saliv_ROI_LRsplit_v2(plot=True, include_center=False, shift=0):
    # set shift != 0 to NOT use manually found LR split indices, but instead shift-distance from idx_center
    df_path = os.path.join(SegSalivDir, "salivary LR split indexes.csv")
    df = pd.read_csv(df_path, index_col=0)
    print(df)
    missing_roi_list = []
    max_cols = 4
    max_rows = 6
    experiments = list(filter(lambda f: "Pilot" in f, os.listdir(RawDir)))
    # for experiment in experiments:
    for experiment in experiments[2:]:
        for time in find_folders(os.path.join(os.getcwd(), "..", "..", "RAW DATA", experiment)):
            print("\n",experiment.upper(), time)
            print("-"*50)
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            num_mice = len(find_folders(datadir, "sagittal"))

            # ncols = num_mice
            num_plots = num_mice // (max_cols * max_rows) + 1
            num_mice_per_plot = int(np.ceil(num_mice / num_plots))
            if num_plots == 1:
                nrows = 1
                ncols = max_cols
                while nrows * ncols < num_mice_per_plot:
                    nrows += 1
                    # ncols = num_mice // 2
                    # nrows = 2
            else:
                nrows = max_rows
                ncols = max_cols

            print(num_mice, ncols, nrows, num_plots)
            # print(num_mice, ncols, nrows, ncols * nrows)
            isMissingSlices = False
            if plot:
                FIGAX = []
                for p in range(num_plots):
                    fig, axes = plt.subplots(nrows, ncols)
                    axes = axes.ravel()
                    fig.tight_layout()
                    fig.subplots_adjust(hspace=0.01, wspace=0.1)
                    for ax in axes.ravel():
                        ax.axis("off")
                    FIGAX.append((fig, axes))

            for i, folder in enumerate(find_folders(datadir, "sagittal")):
                name = get_name(experiment, folder, condition="sagittal")
                # print("\n", time+name)
                idx_c = central_idx_dict[time+name]
                if shift:   # shift from center
                    idx_l = idx_c - shift
                    idx_r = idx_c + shift
                else:   # manual tabulated values
                    dfcurr = df[df["time_name"] == time + name]
                    # print(dfcurr["idx_l"].values)
                    idx_l = int(dfcurr["idx_l"].values)
                    idx_r = int(dfcurr["idx_r"].values)
                print(i, time + name, idx_l, idx_r)

                # idx_center = df.loc[df["time_name"] == time + name, "idx_central"].values[0]
                indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
                MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder), printbool=False)
                im_l = percentile_truncation(MATR_RAW[idx_l], lower=0, upper=99, verbose=False)
                im_r = percentile_truncation(MATR_RAW[idx_r], lower=0, upper=99, verbose=False)
                ROI = np.load(os.path.join(SegSalivDir, experiment, time, name + ".npy"))
                roi_l = ROI[idx_l]
                roi_r = ROI[idx_r]
                print(roi_l.any(), roi_r.any(), not(roi_l.any()) or not(roi_r.any()), not(roi_l.any() and roi_r.any()))
                if not(roi_l.any()) or not(roi_r.any()):
                    # print("MISSING ROI FOR", time + name, "\n")
                    missing_roi_list.append(time + name)

                im_comp = np.c_[im_l, im_r]
                roi_comp = np.c_[roi_l, roi_r]
                msk_comp = np.ma.masked_where(np.logical_not(roi_comp), im_comp)
                # print(im_comp.shape)

                if plot:
                    ip = i % (ncols * nrows)
                    p = i // (ncols * nrows)
                    print(i, p, ip)
                    fig, axes = FIGAX[p]
                    axes = axes.ravel()
                    fig.tight_layout()
                    fig.subplots_adjust(hspace=0.01, wspace=0.1)
                    for ax in axes.ravel():
                        ax.axis("off")

                    axes[ip].imshow(im_comp, cmap="gray")
                    axes[ip].imshow(msk_comp, cmap="hot")
                    # title = f"{time} {name}  index R/L = {idx_l}/{idx_r}"
                    title = f"{time} {name}"
                    # axes[i].set_title()
                    axes[ip].text(0.00, 0.50, title, fontsize=10,
                                 verticalalignment="bottom", horizontalalignment="left",
                                 bbox=dict(alpha=0.5, color="white"))

            if plot:
                fig.suptitle(f"{experiment} {time}", y=-0.01)
                # fig.tight_layout()
                plt.show()
                plt.close()
            # break
        # break
    print("MISSING ROI FOR ", len(missing_roi_list), "MICE")
    return missing_roi_list



if __name__ == "__main__":

    show_saliv_ROI_LRsplit_v2()
    sys.exit()
    folder = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Pilot_LateEffects_-7day\C1_sagittal"
    # indexed_files = dcm_files_indexed(folder)
    # matrix_raw = np.array(dcm_folder_to_pixel_matrix(indexed_files, folder))
    # image_raw = np.array(pixel_matrix_raw[16])
    # image = norm_minmax_featurescaled(image_raw, lower=0, upper=255, rounded=False).astype("uint8")
    # image_ss = norm_stscore(image_raw).astype("uint8")
    # image_neighbour = neighbour_map(image)
    # experiment = "pilot1"
    experiment = "pilot2"
    # time = "-7day"
    time = "8day"
    # time = "26day"
    # time = "56day"
    # time = ""

    # idx = 4
    # plot_all_cropped_npy(experiment, time, ROI=True, idx=idx)
    # plot_all_cropped_npy(experiment, time, ROI=True, exclude="_p", idx=idx)

    #MAKE 2D roi's with watershed by manually region selection and modelling
    # params = (3, 4, 13, 2)
    params = (3, 4, 12, 3)
    # params = (4, 5, 10, 2)  # best - bad at high /low idx slices
    # params = (4, 4, 12, 2)  #   T1 off-center idx works fine???
    # params = (2, 5, 10, 2)      # FOR WORST IMAGES
    # params = (2, 5, 11, 3)      #
    # params = (4,5,2,2)      #works for after_p?
    # params = (2, 4, 10, 2)  # kinda works???
    # params = (3, 6, 6, 2)
    # params = (5, 5, 3, 4)
    # params = (2, 6, 12, 2)
    # params = (2, 8, 14, 2)
    # params = (5, 8, 12, 4)
    # params = (2, 5, 10, 2)
    # params = (2, 5, 9, 4)
    # params = (2, 8, 12, 2)      #absolute trash

    # MAIN GLAND SEGMENTATION
    # missing = show_ROIs(offsetIDX=[-2, 0, 2], mode="saliv", plot=True)
    # missing1 = ['56dayC2_sagittal', '56dayC3_sagittal', '56dayC3_sagittal_after_p', '56dayH3_sagittal', '56dayH4_male_sagittal', '56dayH4_male__sagittal_after_p', '56dayH1_sagittal_60_d', '56dayH1_sagittal_after_p', '8dayC6_male_sagittal', '8dayH1_sagittal', '8dayH4_male_sagittal', '8dayL1_sagittal_after_p_blur', '8dayL2_sagittal', '8dayC2_sagittal', '8dayC3_sagittal', '-7day1-1__after_p_sagittal', '-7day1-1_sagittal', '-7day3-2__after_p_sagittal', '-7day4-1__after_p_sagittal', '-7day1-2_sagittal', '-7day2-2__after_p_sagittal', '26day1-1__after_p_sagittal', '26day1-1_sagittal', '26day1-2_sagittal', '26day2-1_sagittal', '26day3-1__after_p_sagittal', '26day4-2_sagittal', '8day1-1__after_p_sagittal', '8day1-2__after_p_sagittal', '8day2-1__after_p_sagittal', '8day2-1_sagittal', '8day2-2__after_p_sagittal', '8day3-2__after_p_sagittal', '8day3-1__after_p_sagittal', '-7day5-1_after_p_T2_sagittal', '-7day5-1_T2_sagittal', '-7day5-2_after_p_T2_sagittal', '-7day5-3_after_p_T2_sagittal', '-7day5-3_T2_sagittal', '-7day5-4_T2_sagittal', '-7day5-5_after_p_T2_sagittal', '-7day5-5_T2_sagittal', '-7day6-1_after_p_T2_sagittal', '-7day6-1_T2_sagittal', '-7day6-2_after_p_T2_sagittal', '-7day6-3_after_p_T2_sagittal', '-7day6-4_after_p_T2_sagittal', '-7day6-5_after_p_T2_sagittal', '-7day8-2_after_p_T2_sagittal', '-7day8-2_T2_sagittal', '-7day8-4_T2_sagittal', '-7day9-3_after_p_T2_sagittal', '-7day9-3_after_p_T1_sagittal', '-7day9-5_after_p_T1_sagittal', '-7day8-3_T2_sagittal', '-7day8-5_after_p_T2_sagittal', '-7day8-5_T2_sagittal', '-7day9-1_after_p_T2_sagittal', '-7day9-4_after_p_T1_sagittal', '-7day9-5_after_p_T2_sagittal', '105day8-3_after_p_T2_sagittal', '105day8-4_after_p_T2_sagittal', '105day8-4_after_p_T1_sagittal', '105day8-4_T1_sagittal', '105day8-5_after_p_T2_sagittal', '105day8-7_after_p_T2_sagittal', '105day8-7_after_p_T1_sagittal', '105day9-1after_p__T2_sagittal', '105day9-1after_p__T1_sagittal', '105day9-1_T1_sagittal', '105day9-2_after_p_T2_sagittal', '105day9-2_after_p_T1_sagittal', '105day9-2_T2_sagittal', '105day9-2_T1_sagittal', '105day9-3_after_p_T1_sagittal', '105day9-4_after_p_T1_sagittal', '105day9-4_T2_sagittal', '105day9-4_T1_sagittal', '105day9-5_after_p_T2_sagittal', '105day9-5_after_p_T1_sagittal', '105day9-5_T1_sagittal', '105day8-5_T1_sagittal', '105day8-7_T1_sagittal', '105day9-4_after_p_T2_sagittal', '5day5-1_T2_sagittal', '5day5-2_after_p_T2_sagittal', '5day5-5_T2_sagittal', '5day6-1_T2_sagittal', '5day6-2_after_p_T2_sagittal', '5day6-2_T2_sagittal', '5day6-5_T2_sagittal', '5day8-3_T2_sagittal', '5day8-3_T1_sagittal', '5day8-4_after_p_T2_sagittal', '5day8-4_after_p_T1_sagittal', '5day8-4_T1_sagittal', '5day8-5_after_p_T2_sagittal', '5day8-6_T1_sagittal', '5day8-7_T1_sagittal', '5day9-1_after_p_T2_sagittal', '5day9-1_after_p_T1_sagittal', '5day9-2_after_p_T2_sagittal', '5day9-2_after_p_T1_sagittal', '5day9-2_T1_sagittal', '5day9-3_after_p_T2_sagittal', '5day9-3_after_p_T1_sagittal', '5day9-3_T1_sagittal', '5day9-4_after_p_T2_sagittal', '5day9-4_after_p_T1_sagittal', '5day9-5_after_p_T2_sagittal', '5day9-5_T2_sagittal', '5day9-5_T1_sagittal', '5day5-2_T2_sagittal', '5day6-3_T2_sagittal', '5day8-5_after_p_T1_sagittal', '5day8-5_T1_sagittal', '5day9-2_T2_sagittal']
    # missing2 = ['-7dayH1_sagittal', '-7dayH3_sagittal', '-7dayC2_sagittal', '-7dayL1_sagittal', '56dayC2_sagittal_after_p', '56dayC3_sagittal', '56dayC3_sagittal_after_p', '56dayH1_sagittal_60_d', '56dayH1_sagittal_after_p', '56dayH3_sagittal', '56dayH4_male_sagittal', '56dayH4_male__sagittal_after_p', '56dayL1_sagittal', '56dayL1_sagittal_after_p_blur', '8dayC2_sagittal', '8dayC2_sagittal_after_p', '8dayC6_male_sagittal', '8dayH1_sagittal', '8dayH3_sagittal', '8dayH4_male_sagittal', '8dayL1_sagittal', '8dayL1_sagittal_after_p_blur', '-7day1-1__after_p_sagittal', '-7day1-1_sagittal', '-7day1-2__after_p_sagittal', '-7day2-1_sagittal', '-7day2-2__after_p_sagittal', '-7day3-2__after_p_sagittal', '-7day4-1__after_p_sagittal', '-7day4-1_sagittal', '-7day4-2__after_p_sagittal', '-7day2-1__after_p_sagittal', '-7day3-2_sagittal', '26day1-1_sagittal', '26day1-2__after_p_sagittal', '26day2-1_sagittal', '26day2-2__after_p_sagittal', '26day2-2_sagittal', '26day3-1__after_p_sagittal', '26day3-1_sagittal', '26day3-2_sagittal', '26day4-1_sagittal', '26day4-2__after_p_sagittal', '26day4-2_sagittal', '26day1-2_sagittal', '26day1-3_during_p_sagittal', '26day2-3_during_p_sagittal', '26day3-2__after_p_sagittal', '26day4-1__after_p_sagittal', '8day1-1__after_p_sagittal', '8day1-1_sagittal', '8day1-2__after_p_sagittal', '8day1-2_sagittal', '8day2-1__after_p_sagittal', '8day2-1_sagittal', '8day2-2__after_p_sagittal', '8day3-1__after_p_sagittal', '8day3-2__after_p_sagittal', '8day4-1_sagittal', '8day4-2__after_p_sagittal', '8day2-2_sagittal', '-7day5-1_after_p_T2_sagittal', '-7day5-1_T2_sagittal', '-7day5-2_after_p_T2_sagittal', '-7day5-2_T2_sagittal', '-7day5-3_after_p_T2_sagittal', '-7day5-3_T2_sagittal', '-7day5-4_after_p_T2_sagittal', '-7day5-4_T2_sagittal', '-7day5-5_after_p_T2_sagittal', '-7day5-5_T2_sagittal', '-7day6-1_after_p_T2_sagittal', '-7day6-1_T2_sagittal', '-7day6-2_after_p_T2_sagittal', '-7day6-2_T2_sagittal', '-7day6-3_after_p_T2_sagittal', '-7day6-3_T2_sagittal', '-7day6-4_after_p_T2_sagittal', '-7day6-4_T2_sagittal', '-7day6-5_after_p_T2_sagittal', '-7day6-5_T2_sagittal', '-7day8-1_after_p_T2_sagittal', '-7day8-1_T2_sagittal', '-7day8-2_after_p_T2_sagittal', '-7day8-2_T2_sagittal', '-7day8-3_after_p_T2_sagittal', '-7day8-3_T2_sagittal', '-7day8-4_after_p_T2_sagittal', '-7day8-4_T2_sagittal', '-7day8-5_after_p_T2_sagittal', '-7day8-5_T2_sagittal', '-7day9-1_after_p_T2_sagittal', '-7day9-1_T2_sagittal', '-7day9-2_after_p_T2_sagittal', '-7day9-2_T2_sagittal', '-7day9-3_after_p_T2_sagittal', '-7day9-3_after_p_T1_sagittal', '-7day9-3_T2_sagittal', '-7day9-3_T1_sagittal', '-7day9-4_after_p_T2_sagittal', '-7day9-4_after_p_T1_sagittal', '-7day9-5_after_p_T2_sagittal', '-7day9-5_after_p_T1_sagittal', '-7day9-5_T1_sagittal', '-7day9-4_T2_sagittal', '-7day9-4_T1_sagittal', '105day8-3_after_p_T2_sagittal', '105day8-3_after_p_T1_sagittal', '105day8-3_T1_sagittal', '105day8-4_after_p_T2_sagittal', '105day8-4_T2_sagittal', '105day8-4_T1_sagittal', '105day8-5_after_p_T2_sagittal', '105day8-5_after_p_T1_sagittal', '105day8-5_T2_sagittal', '105day8-5_T1_sagittal', '105day8-7_after_p_T2_sagittal', '105day8-7_after_p_T1_sagittal', '105day8-7_T2_sagittal', '105day8-7_T1_sagittal', '105day9-1after_p__T2_sagittal', '105day9-1after_p__T1_sagittal', '105day9-1_T2_sagittal', '105day9-1_T1_sagittal', '105day9-2_after_p_T2_sagittal', '105day9-2_after_p_T1_sagittal', '105day9-2_T2_sagittal', '105day9-2_T1_sagittal', '105day9-3_after_p_T2_sagittal', '105day9-3_after_p_T1_sagittal', '105day9-3_T2_sagittal', '105day9-3_T1_sagittal', '105day9-4_after_p_T2_sagittal', '105day9-4_after_p_T1_sagittal', '105day9-4_T2_sagittal', '105day9-4_T1_sagittal', '105day9-5_after_p_T2_sagittal', '105day9-5_after_p_T1_sagittal', '105day9-5_T2_sagittal', '105day9-5_T1_sagittal', '105day8-4_after_p_T1_sagittal', '5day5-1_after_p_T2_sagittal', '5day5-1_T2_sagittal', '5day5-2_after_p_T2_sagittal', '5day5-2_T2_sagittal', '5day5-3_after_p_T2_sagittal', '5day5-4_after_p_T2_sagittal', '5day5-4_T2_sagittal', '5day5-5_after_p_T2_sagittal', '5day5-5_T2_sagittal', '5day6-1_after_p_T2_sagittal', '5day6-1_T2_sagittal', '5day6-2_after_p_T2_sagittal', '5day6-2_T2_sagittal', '5day6-3_after_p_T2_sagittal', '5day6-3_T2_sagittal', '5day6-4_after_p_T2_sagittal', '5day6-4_T2_sagittal', '5day6-5_after_p_T2_sagittal', '5day6-5_T2_sagittal', '5day8-3_after_p_T2_sagittal', '5day8-3_after_p_T1_sagittal', '5day8-3_T2_sagittal', '5day8-3_T1_sagittal', '5day8-4_after_p_T2_sagittal', '5day8-4_after_p_T1_sagittal', '5day8-4_T2_sagittal', '5day8-4_T1_sagittal', '5day8-5_after_p_T1_sagittal', '5day8-5_T2_sagittal', '5day8-5_T1_sagittal', '5day8-6_T2_sagittal', '5day8-6_T1_sagittal', '5day8-7_after_p_T2_sagittal', '5day8-7_T2_sagittal', '5day8-7_T1_sagittal', '5day9-1_after_p_T2_sagittal', '5day9-1_after_p_T1_sagittal', '5day9-2_after_p_T2_sagittal', '5day9-2_after_p_T1_sagittal', '5day9-2_T1_sagittal', '5day9-3_after_p_T2_sagittal', '5day9-3_after_p_T1_sagittal', '5day9-3_T2_sagittal', '5day9-4_after_p_T2_sagittal', '5day9-4_after_p_T1_sagittal', '5day9-4_T2_sagittal', '5day9-5_after_p_T2_sagittal', '5day9-5_T2_sagittal', '5day9-5_T1_sagittal', '5day8-5_after_p_T2_sagittal', '5day9-2_T2_sagittal', '5day9-3_T1_sagittal']


    # global missingPath, missingList
    # missingPath = os.path.join(SegSalivDir, "missing.csv")
    # missing = show_saliv_ROI_LRsplit(plot=False, include_center=True, shift=0)
    # np.savetxt(X=np.array(missing), fname=missingPath, delimiter=", ", fmt="%s")

    # missingList = np.loadtxt(missingPath, dtype=str, delimiter=", ")
    # missingList = np.array([missingList]) if type(missingList) != list else missingList
    # print(len(missingList), "LEFT IN MISSING LIST FOR SEGMENTING\n")

    # df = pd.read_csv(SegSalivDir+"\\salivary LR split indexes.csv", index_col=0)
    # timenames = df[df["idx_l"].isna()]["time_name"].values
    # print(timenames)

    # missingList = ["105day9-5_after_p_T1_sagittal"]
    missingList = ["70day11-3_T1_sagittal"]
    main_segment_gland(segfunc=browse_images_make_watershed_ROI_2D, n4=True, params2d=params, histeq=True, resegnames=missingList)
    # show_ROIs(offsetIDX="LR", mode="saliv")

    # missingList = missingList[16:]
    # main_segment_gland(segfunc=browse_images_make_watershed_ROI_2D, params2d=params, histeq=True, n4=False, resegnames=missingList[11:])
    # TODO: PICK OUT NON-SEGMENTED LR SPLIT SLICES --> SEGMENT :)

    # SEGMENTING BRAIN (PARTIALLY) FOR IMAGE NORMALIZATION
    # main_segment_brain(segfunc=browse_images_make_watershed_ROI_2D, params2d=params)
    # show_ROIs(offsetIDX=0, mode="brain")



    # mainmain_cropped_2d(segfunc=browse_images_make_watershed_ROI_2D, params=params)
    # mainmain_cropped_2d(segfunc=browse_images_make_watershed_ROI_2D_withKMEANS, params=params, kmeans=True)
    # validate_segmentations_at_slice(idx=4, includep=True, save=True)
    # validate_segmentations_at_slice(idx=4, includep=False, save=True)

    # main(experiment=experiment, time=time, cropped=True, segfunc=browse_images_make_watershed_ROI_2D, params2d=params)
    # main(experiment="pilot2", time="-7day", cropped=True, segfunc=browse_images_make_watershed_ROI_2D_withKMEANS, params2d=params)

    # time = "-7day"
    # folder_cropped_raw = r"G:\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary\raw\\" + time + r"\\"
    # folder_cropped_raw = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary\raw\\" + time + r"\\"
    # folder_save = r"G:\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary\roi man\\"
    # folder_save = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary\roi man\\"
    # params = (4, 5, 10, 2)      #best - bad at high idx slices
    # params = (2, 6, 12, 2)
    # params = (2, 8, 14, 2)
    # params = (2, 8, 12, 2)      #tjaaa
    # params = (2, 5, 10, 2)
    # params = (5, 8, 12, 4)
    # params = (4,5,2,2)
    # params = (2, 5, 10, 2)
    # for file in os.listdir(folder_cropped_raw):
    #     matr = load_matrix(folder_cropped_raw + file)
        # print(file, matr.shape)
        # matr = percentile_truncation(matr, 2.5, 99.5)
        # roiman = browse_images_make_watershed_ROI_2D(matr, title=file, time=time, params=params, save_path=folder_save)



    # matrix_raw = load_matrix(
    #     r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary\-7day\C1_sagittal.npy")
    # print_ndstats(matrix_raw, matrix)

    # matrix = percentile_truncation(matrix_raw, 10, 99)
    # matrix = norm_minmax_featurescaled(matrix, 0, 255)
    # print_ndstats(matrix_norm, matrix_smoothed)



    # idx = 4
    # p = (2, 5, 10, 2)
    # p = (4, 5, 10, 2)
    # _, _, _, labels = watershed_3d(matrix, *p)


    # compare_images(matrix[idx], labels[idx])
    #IMAGE PROCESSING: MIN-MAX NORM??, GAUSSIAN SMOOTHING, BACKGROUND SUBTRACTION?
    # matrix_smoothed = skimage_gaussian(matrix_norm, 1)
    # kmeans = KMeans(n_clusters=6, random_state=0).fit(matrix_smoothed)


    #ROI REGION GROWING
    # gland_center = [110, 140]       #for C1 sagittal i = 16
    # # gland_center = [50, 50]
    # seeds = make_seeds(gland_center, 3)
    # thresh = 175      #this works for raw
    # # thresh = 175.00000000001      #does NOT work for raw. why?
    # # roi_grown = region_growing(image, seeds, 1.5, True, region_value=150)
    # # roi_grown = region_growing(image_ss, seeds, 1, True, region_value=150)
    # roi_grown = region_growing(image_raw, seeds, thresh, True)
    # fig, ax = plt.subplots()
    # # window = (137, 139, 91, 88)
    # ax.imshow(image, "gray")#, extent=window)
    # seeds = np.array(seeds)
    # # ax.plot(seeds[:, 1], seeds[:, 0])
    # ax.imshow(roi_grown, "gray", alpha=0.3)#, extent=window)
    # plt.title("Region grown with threshold = %.2f" % (thresh))
    # plt.show()


    # circlepoints = make_circle_points(200, [110, 140], 25, 1.5)
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    # ax.plot(circlepoints[:, 0], circlepoints[:, 1], "--r", lw=2)
    # plt.show()

    # vis.show_histogram(image, bins=60, range=[0, 59])


    # vis.compare_images(image, image_otsu)
    # vis.compare_images(image, neighbourmap)
    # glcm = texture.glcm_scikit(image, [1], [0], levels=256)[:, :, 0, 0]
    # print(glcm)


    # w = 50
    # image = image_ss[110-w:110+w, 140-w:140+w]     #crop image
    # image_otsu = thres.mask_otsu(image, 40, 1)
    # image_otsu = thres.mask_otsu(image, 100, 1)  #minmax
    # vis.plot_image(image_otsu)

    # DICOM_reader.print_ndstats(image, image_raw)
    # image_canny = edg.cannyedge_cv2(image, 40, 80, 3, True)
    # disp_image = image.copy()
    # disp_image[image_otsu != 0] = 255


    # vis.compare_images(image_raw, image, title2="minmax_rounded", title1="raw")
    # vis.show_histogram(image)
    # vis.plot_image(image)
    # viz.show_histogram(image_raw)
    # viz.show_histogram(image)
    # plt.show()
    # for image in pixel_matrix_norm:
    #     otsu_thresholding.show_histogram(image)
    # plt.show()


#K-means clustering


#PCA NOT NEEDED? START WITH K-MEANS???
#I: PRINCIPAL COMPONENT ANALYSIS: dimension reduction of dataset by descending proportion of explained variance
# def num_components(pca, varlimit):     #find number of dimensions to include (# PC's) to explain some cumulative variance percentage
#     var_cum = 0
#     d = 0
#     for var in pca.explained_variance_ratio_:
#         var_cum += var
#         d += 1
#         if(var_cum > varlimit):
#             # print(var_cum)
#             break
#     return d


# d = 5 #dimensions after PCA reduction
# pca = PCA().fit(image)
# print(pca.explained_variance_ratio_.shape)
# d = num_components(pca, 0.99)
# pca_reduced = PCA(n_components=d).fit_transform(image)
# print(d, "= # of PC's", pca_reduced.shape)


#CLAHE: Contrast-limited adaptive histogram equalization
# image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# clahe = cv2.createCLAHE(clipLimit = 0.008)  #cliplimit: threshold for contrast limiting
# pca_clahe = clahe.apply(pca_reduced)
# image_clahe = clahe.apply(image)
# cv2.imshow(image_clahe)
# plt.subplot(121); plt.imshow(image); plt.title("no clahe")
# plt.subplot(122); plt.title("clahe"); plt.imshow(image_clahe); plt.show()