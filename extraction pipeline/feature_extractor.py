import pandas as pd
import numpy as np
from DICOM_reader import find_folders, print_ndstats
from preprocessing import load_matrix, norm_stscore
from matplotlib import pyplot as plt
import os
import SimpleITK as sitk
from radiomics import featureextractor
import six
import nrrd


#http://www.radiomics.io/pyradiomicsnotebook.html


class features:
    def __init__(self, dataDir, path_img, path_roi, day, plane, mouseidx, mode, norm, experiment, name):
        self.dataDir = dataDir
        self.path_img = path_img
        # self.path_roi = path_roi
        self.day = day
        self.plane = plane
        self.mode = mode
        self.norm = norm
        self.mouseidx = mouseidx
        self.experiment = experiment
        # self.mousename = self.get_mousename(self.path_img, self.plane, self.mouseidx)
        self.mousename = name
        self.savepath = os.path.normpath(os.path.join(dataDir, mode, day))
        # self.path_nrrd = os.path.normpath(os.path.join(self.dataDir, "nrrd files", self.experiment, self.norm, self.mode, self.day)) if mode == "central slice" else ""
        self.path_nrrd = os.path.normpath(os.path.join(self.dataDir, "nrrd files", self.mode + " " + self.norm, self.experiment, self.day))
        self.feats = pd.Series({"mousename":self.mousename, "mouseidx":self.mouseidx, "day":self.day[:-3]}, name="features")
        # self.feats = pd.DataFrame({"mousename": [self.mousename], "mouseidx": [self.mouseidx], "day": [self.day[:-3]]})
        # print(self.feats)
    def get_mousename(self, path_img, plane, mouseidx):
        print(self.path_img)
        for i, f in enumerate(find_folders(self.path_img, plane)):
            if i == mouseidx:
                return f[:-4]
    # def append_results(self, results):
    #     for key_, val in six.iteritems(results):
    #         print("\t%s: %s" % (key_, val))
    #         try:
    #             self.feats = self.feats.assign(key_=val)#, ignore_index=True)    #TODO: ASSIGN ALL KEYS AS HEADERS WITH VAL IN DF
    #         except Exception as e:
    #             print(e.args)
    def save_features_as_df(self, result, printfeats=False):
        # print(type(self.feats))
        # df = ft.feats.to_frame().T
        # print(df)
        if printfeats:
            for key, val in six.iteritems(result):
                print(f"{key}:\t {val}")#key, val)
        # savepath = os.path.normpath(os.path.join(self.dataDir, self.norm, self.mode, self.day))
        # savepath = os.path.normpath(os.path.join(self.dataDir, self.experiment, self.mode, self.norm, self.day))
        savepath = os.path.normpath(os.path.join(self.dataDir, self.mode + self.norm, self.experiment, self.day))

        if not os.path.exists(savepath):
            os.makedirs(savepath)
            print("New path made at", savepath)
        # print(self.feats)
        try:
            self.feats.to_csv(os.path.join(savepath, self.mousename + ".csv"))
            print(f"{len(result)} features saved in csv at ", os.path.join(savepath, self.mousename + ".csv"))
        except Exception as e:
            print(e.args)


def matrix_roi_loader_npy(path_img, path_roi, plane, printtext=False):
    num_matrices = len(find_folders(path_img, plane))
    # print(num_matrices)
    if not len(find_folders(path_roi, plane)) == num_matrices:
        print("Not enough roi's found.")
    matr_raw = []
    matr_roi = []
    for f in find_folders(path_img, plane):
        # print(f)
        current = load_matrix(os.path.join(path_img, f), printtext=printtext)
        matr_raw.append(current)
    for f in find_folders(path_roi, plane):
        current = load_matrix(os.path.join(path_roi, f), printtext=printtext)
        matr_roi.append(current)
    print(f"LOADING {len(matr_raw)} raw data matrices from {path_img}, \nand {len(matr_roi)} mask matrices from {path_roi}")
    return np.array(matr_raw), np.array(matr_roi)


def get_mousename(path_img, plane, mouseidx):
    for i, f in enumerate(find_folders(path_img, plane)):
        if i == mouseidx:
            return f[:-4]


def ndarray_to_nrrd(arr, savepath, mousename, mode="image"):  #WRITE NRRD FILE FROM NDARRAY
    # print(savepath)
    if mode == "image" or mode == "mask":
        pass
    else:
        print("mode must be image or mask, you selected mode:\t", mode)
        return 0
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print("New directory created at", savepath)
    filename = os.path.join(savepath, mousename + "_" + mode + ".nrrd")
    try:
        nrrd.write(filename, arr)
        print("ndarray converted to .nrrd ---- saved at ", filename)
    except Exception as e:
        print(e.args)
    read, header = nrrd.read(filename)
    # print(read.shape, type(read), read.dtype)
    # print(header)
    pass


def plot_nrrd(impath, maskpath, name="", printndstats=False):
    im = sitk.GetArrayFromImage(sitk.ReadImage(impath))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(maskpath))
    # print(im.shape, mask.shape)
    if printndstats:
        print_ndstats(im)
    fig, ax = plt.subplots(ncols=2);    ax1, ax2 = ax.ravel()
    ax1.imshow(im, cmap="gray")
    ax2.imshow(im, cmap="gray")
    ax2.imshow(mask, alpha=0.3)
    plt.suptitle("NRRD: " + name)
    plt.show()
    plt.close()


def main_old(num_mice, dataDir, path_img, path_roi):
    for mouseidx in range(0, num_mice+1):        #LOOP OVER ALL MICE FOR FEATURE EXTRACTION


        ft = features(dataDir=dataDir, path_img=path_img, path_roi=path_roi, day="-7day", plane="sagittal", mouseidx=mouseidx, mode="central slice")
        print(ft.mouseidx, ft.mousename)
        imagePath, maskPath = find_folders(ft.path_nrrd, condition=ft.mousename)
        imagePath = os.path.join(ft.path_nrrd, imagePath)
        maskPath = os.path.join(ft.path_nrrd, maskPath)
        print(imagePath, "\n", maskPath)
        plot_nrrd(imagePath, maskPath, ft.mousename)      #SEE IF LOADED .nrrd IMAGES ARE NOT CORRECT

        params = os.path.join(dataDir, "examples", "exampleSettings", "Params.yaml")  # path to extractor settings
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        results = extractor.execute(imagePath, maskPath)     #TODO: actual feature extracting - UNDERSTAND THIS
        ft.feats = ft.feats.append(pd.DataFrame.from_dict(results, orient="index").squeeze())       #append results to Series (therefore dataframe.squeeze)
        ft.feats = ft.feats.to_frame(name=ft.day + "_" + ft.mousename).T        #make first column values (feature names) to df columns
        ft.save_features_as_df(results, printfeats=False)
    return 0


def main_make_nrrd(dataDir, path_raw, path_roi, experiment, mode, norm):
    idx = 4     #SELECT CENTRAL SLICE
    # idx = 3
    # path_nrrd = os.path.normpath(os.path.join(dataDir, "nrrd files", experiment, norm, mode, time))
    path_nrrd = os.path.normpath(os.path.join(dataDir, "nrrd files", mode + f" {idx} " + norm, experiment, time))

    if not os.path.exists(path_nrrd):   os.makedirs(path_nrrd)
    # print(path_nrrd)
    # print(path_raw)
    numfiles = len(find_folders(path_raw))
    plot = False
    # plot = True
    if plot:
        # fig, ax = plt.subplots(ncols=int(numfiles//np.sqrt(numfiles)), nrows=int(numfiles//np.sqrt(numfiles)))
        fig, ax = plt.subplots(ncols=8, nrows=2)
        ax = ax.ravel()
        plt.suptitle(f"{numfiles} Images from {experiment} at time {time} with ROI's at central slice")

    for filenum, file in enumerate(find_folders(path_raw)):
        print("\n", filenum, file)
        file_raw = os.path.join(path_raw, file)
        file_roi = os.path.join(path_roi, file)
        im = load_matrix(file_raw)
        roi = load_matrix(file_roi)


        if norm == "stscore norm":
            im = norm_stscore(im[idx]);
        elif norm == "raw":
            im = im[idx]
        else:
            print(norm, "not valid (yet)")
        roi = roi[idx]

        # print(numfiles)
        if plot:        #SEE WHAT SLICE HAVE DECENT ROI FOR ALL IMAGES TO CONVERT TO NRRD (IF CENTRAL SLICE MODE)
            ax[filenum].imshow(im, cmap="gray")
            ax[filenum].imshow(roi, alpha=0.3)
            ax[filenum].set_title(file[:-3])
            ax[filenum].axis("off")
        # print(file[:-4])
        # name = file[:-4] if experiment == "pilot2" else file
        name = file[:-4]
        # print(path_nrrd)
        print(name)
        ndarray_to_nrrd(im, path_nrrd, name, mode="image")    #UNCOMMENT TO SAVE AS NRRD
        ndarray_to_nrrd(roi, path_nrrd, name, mode="mask")

        pass
    if plot:
        fig.tight_layout()
        plt.show()
    pass


def main(num_mice, dataDir, path_img, path_roi, time, mode, experiment, plane="sagittal", norm="stscore norm", save=False):
    for mouseidx in range(0, num_mice):        #LOOP OVER ALL MICE FOR FEATURE EXTRACTION
        # ft = features(dataDir=dataDir, path_img=path_img, path_roi=path_roi, day="-7day", plane="sagittal", mouseidx=mouseidx, mode="central slice", norm="stscore norm")
        ft = features(dataDir=dataDir, path_img=path_img, path_roi=path_roi, day=time, plane=plane, mouseidx=mouseidx, mode=mode, norm=norm, experiment=experiment)
        print(ft.mouseidx, ft.mousename)
        if "p" in ft.mousename:
            print("p")
        else:
            print("no p")
        # name_image = ft.mousename + "_image.nrrd"
        # name_mask = ft.mousename + "_mask.nrrd"
        # print(find_folders(ft.path_nrrd, condition=ft.mousename))
        # imagePath, maskPath = find_folders(ft.path_nrrd, condition=ft.mousename)
        imagePath = ft.mousename + "_image.nrrd"
        maskPath = ft.mousename + "_mask.nrrd"
        print(imagePath, maskPath)
        print(ft.path_nrrd)
        imagePath = os.path.join(ft.path_nrrd, imagePath)
        maskPath = os.path.join(ft.path_nrrd, maskPath)
        # print(imagePath.shape, maskPath.shape)
        # if "p" in imagePath:
        #     plot_nrrd(imagePath, maskPath, ft.experiment + ft.day + ft.mousename)
        # plot_nrrd(imagePath, maskPath, name=ft.mousename, printndstats=True)      #SEE IF LOADED .nrrd IMAGES ARE CORRECT

        #ACTUAL EXTRACTION
        params = os.path.join(dataDir, "settings", "Params_nonorm_2D.yaml")  # path to extractor settings
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        results = extractor.execute(imagePath, maskPath)     #TODO: actual feature extracting - UNDERSTAND THIS
        ft.feats = ft.feats.append(pd.DataFrame.from_dict(results, orient="index").squeeze())       #append results to Series (therefore dataframe.squeeze)
        ft.feats = ft.feats.to_frame(name=ft.day + "_" + ft.mousename).T        #make first column values (feature names) to df columns
        word = "lbp"
        # word = ""
        # print(ft.feats)
        print(len(np.nonzero(ft.feats.columns.str.contains(word))[0]), f"features having {word} in name.\n")
        if save:    ft.save_features_as_df(results, printfeats=False)
        else:   print("Features not saved.\n")
    return 0



if __name__ == "__main__":
    dataDir = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic features"))
    # experiment = "Pilot1"
    experiment = "pilot2"
    time = "-7day"


    # path_roi = os.path.normpath(os.path.join(os.getcwd(), "..", r"Segmentations\cropped_salivary", experiment, r"roi man very bad", time))

    plane = "sagittal"
    norm = "stscore norm"
    # norm = "raw"
    mode = "central slice"
    # print(path_raw)

    #
    # for experiment in ["pilot1", "pilot2"]:
    #     for time in ["-7day", "8day", "56day"] if experiment=="pilot1" else ["-7day", "8day", "26day"]:
    #         print(experiment, time)
    #         path_raw = os.path.normpath(
    #             os.path.join(os.getcwd(), "..", r"Segmentations\cropped_salivary", experiment, r"raw", time))
    #         path_roi = os.path.normpath(
    #             os.path.join(os.getcwd(), "..", r"Segmentations\cropped_salivary", experiment, r"roi 2d watershed",
    #                          time))
    #         print(path_raw, "\n", path_roi)
    #         # main_make_nrrd(dataDir, path_raw, path_roi, experiment=experiment, mode=mode, norm=norm) #CONVERT SEGMENTED FILES TO NRRD FILES (E.G. CENTRAL SLICE 3/ 4)
    #         num_mice = len(find_folders(path_raw))
            # main(num_mice, dataDir, path_raw, path_roi, time=time, mode="central slice 3", experiment=experiment, norm=norm, save=True)
            # main(num_mice, dataDir, path_raw, path_roi, time=time, mode="central slice 4", experiment=experiment, norm=norm, save=True)

    #TODO: FEATURE EXTRACTION FROM NRRD FILES IN PILOT 2 -7DAYS

    # num_mice = 5   #5
    #LOOP OVER ALL MICE FOR FEATURE EXTRACTION
    # path_nrrd = os.path.normpath(os.path.join(dataDir, "nrrd files\central_slice", day))

    # path_nrrd = os.path.normpath(os.path.join(dataDir, "nrrd files", experiment, norm, mode, time))
    # num_mice = len(find_folders(path_nrrd, "image"))
    # print(num_mice, path_nrrd)
    # main(num_mice, dataDir, path_raw, path_roi, time=time, mode=mode, experiment=experiment, norm=norm)


    #todo: include dcm information in nrrd file??? first: do stuff and see what breaks


    #LOAD + BROWSE NPY FILES
    # for f in find_folders(path_raw, "sagittal"):
    #     pthim = os.path.join(path_raw, f)
    #     pthroi = os.path.join(path_roi, f)
    #     print(os.path.exists(pthim), pthim)
    #     print(os.path.exists(pthroi), pthroi)
    #     m = load_matrix(pthim);     roi = load_matrix(pthroi)
    #     browse_3D_segmented(m, roi)


    #MAKE NRRD FILES FOR CENTRAL SLICES FOR ALL MICE IN day, PLANE
    # m_raw, m_roi = matrix_roi_loader_npy(path_raw, path_roi, plane)     #LOAD ALL NPY FILES OF PLANE IN FOLDERS
    # sliceidx = 5     #central slice ish

    #
    # files = find_folders(path_nrrd)
    # impath, mskpath = files[:2]
    # im1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path_nrrd, impath))).copy()
    # msk1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path_nrrd, mskpath))).copy()

    # im2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path_nrrd, "C2_sagittal_image.nrrd")))
    # print((im1 == im2).all())
    # plot_masked(im1, msk1)

    #LOCAL BINARY PATTERN ILLUSTRATION
    # N = 3
    # fig, ax = plt.subplots(ncols=2, nrows=2);    ax = ax.ravel()
    # fig.suptitle("LBP filtered image with varying sampling radius")
    # fig.tight_layout()
    # ax[0].imshow(im1, cmap="gray");  ax[0].set_title("Original")     ;ax[0].axis("off")
    # for i, rad in enumerate(range(1, N+1)):
    #     points = 8*rad
    #     LBP = local_binary_pattern(im1, P=points, R=rad, method="default")
    #     print(i, rad)
    #     ax[i+1].imshow(LBP)#, cmap="gray")
    #     ax[i+1].set_title(f"LBP: radius={rad}, n_points={points}")
    #     ax[i+1].axis("off")
    # plt.show()
    # compare_images(im1, LBP)


    # print(im1.shape, msk1.shape)
    # plot_masked(im1, msk1)
    # im = im1*msk1
    # import cv2
    # ret, thrs = cv2.threshold(im.astype("uint8"), 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plot_image(im.astype("uint8"))



    # for mouseidx in range(num_mice+1):
    #     mousename = get_mousename(path_raw, plane, mouseidx)
    #     im = m_raw[mouseidx, sliceidx]
    #     mask = m_roi[mouseidx, sliceidx]
    #     print(mouseidx, mousename, len(np.argwhere(mask != 0)))
    #     # plot_masked(im, mask, mousename)
    #     # print(im.shape, mask.shape)
    #     print_ndstats(im)
    #     im = norm_stscore(im)
    #     # print_ndstats(im)
    #     print("\n")
    #
    #     ndarray_to_nrrd(im, path_nrrd, mousename, mode="image")
    #     ndarray_to_nrrd(mask, path_nrrd, mousename, mode="mask")


    #COLLECT .nrrd FILES FOR USE IN
    # for mouseidx in range(1):
    #     print(mouseidx, mousename)
    # im_raw = sitk.GetImageFromArray(m_raw[mouseidx, sliceidx])


    #EXAMPLE CASE brain1
    # imageName, maskName = getTestCase("brain1", dataDir + r"\examples")    #paths to image and mask
    # imtest = sitk.ReadImage(imageName)
    # imtest = sitk.GetArrayFromImage(imtest)
    # print(type(imtest), type(im))

    # params = os.path.join(dataDir, "examples", "exampleSettings", "Params.yaml")    #path to extractor settings
    # extractor = featureextractor.RadiomicsFeatureExtractor(params)
    # extractor = featureextractor.RadiomicsFeatureExtractor()

     # print(extractor.enabledFeatures)

    #TODO: IS THIS 2D? 3D? in example have 3D image/ mask, what happens when putting in single slice?
    # enable shape2D (instead of shape) and set force2D=True if 3D volume is input
    # send in STRINGS - paths to PROPER FILETYPE (MAKE FROM PIXEL ARRAYS AND AGGREGATE DCM INFORMATION??)
    # result = extractor.execute(imageName, maskName)     #TODO: actual feature extracting - UNDERSTAND THIS
    # df = pd.Series({"Mousename":mousename, "day":day})                #create new pd.Series with mousename, day as first two rows
    # df = df.append(pd.DataFrame.from_dict(result, orient="index").squeeze())       #append result to pd.Series object

    #PRINT RESULTS
    # for key, val in six.iteritems(result):
    #     print("\t%s: %s" % (key, val))
    # for key, val in six.iteritems(df):
    #     print(key, val)
    # for key in result:
    #     print(key, result[key])
    # print(type(result), type(df))

    #SAVE AS PANDAS DATAFRAME
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
    #     print("new path made at", savepath)
    # try:
    #     df.to_csv(os.path.join(savepath, get_mousename(path_raw, plane, mouseidx) + ".csv"))
    #     print(f"{len(result)} features saved in csv.")
    # except Exception as e:
    #     print(e.args)


    # f = pd.DataFrame(result)
    # print(f)

    # result_path =
    # radiomics.setVerbosity(10)
    # log_file = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Radiomic features\examples\log_file.txt"


    #WHERE DO THE logging REPO COME FROM??
    # handler = logging.FileHandler(filename=log_file, mode='w')      #mode   w:overwrite   a:append

    # print(len(np.argwhere(image_in_roi != 0)), "pixels in masked image.")


    #EXPLORING EXAMPLE CASE IMAGES FOR COMPARISON
    # brain1 = sitk.ReadImage(imageName)
    # brain1 = sitk.GetArrayFromImage(brain1)
    # brainseg = sitk.ReadImage(maskName)
    # brainseg = sitk.GetArrayFromImage(brainseg)       #binary mask of values {0, 1}
    # # imagemasked = sitk.GetImageFromArray(image_in_roi)
    # print(brain1)
    # checkMask(image, roi)
    # idx = 14
    # fig, ax = plt.subplots(ncols=2); ax1, ax2 = ax.ravel()
    # ax1.imshow(brain1[idx])
    # ax1.imshow(brainseg[idx], alpha=0.3)
    # ax2.imshow(brainseg[idx])
    # # plt.imshow(image)
    # plt.show()


    #PLOT: HISTOGRAM OF INTENSITIES IN ROI
    # normed = norm_stscore(image_in_roi, ignore_zeros=True)
    # image_in_roi = im1 * msk1
    # print(np.mean(im1), np.std(im1), np.mean(image_in_roi), np.std(image_in_roi))
    # show_histogram(image_in_roi, range=(np.min(image_in_roi) + 1, np.max(image_in_roi)), log=False, bins=60, norm=False) #+1 to exclude 0's (i.e. not in masked image)
    # show_histogram(normed, range=(np.min(normed) + 1, np.max(normed)), log=False, bins=60, norm=True
    #                ,titleimg="Segmented image", titlehst=f"Histogram of standardized intensities with {60} bins.")


    # PLOT: idx slice for two mice with ROI
    # fig, ax = plt.subplots(ncols=2, nrows=1);   ax = ax.ravel();  fig.tight_layout()
    # plt.suptitle(f"Slice {sliceidx}")
    # mouseidx = 0;   name = get_mousename(path_raw, plane, mouseidx)
    # ax[0].imshow(norm_minmax_featurescaled(m_raw[mouseidx, sliceidx], 0, 255), cmap="gray", alpha=1)
    # ax[0].imshow(m_roi[mouseidx, sliceidx], alpha=0.3, cmap=plt.cm.bone)
    # ax[0].set_title(f"Mouse {name}")
    # mouseidx = 5;   name = get_mousename(path_raw, plane, mouseidx)
    # ax[1].imshow(norm_minmax_featurescaled(m_raw[mouseidx, sliceidx], 0, 255), cmap="gray")
    # ax[1].imshow(m_roi[mouseidx, sliceidx], alpha=0.3, cmap=plt.cm.bone)
    # ax[1].set_title(f"Mouse {name}")
    # plt.show()
    # plt.close()