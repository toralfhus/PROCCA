from matplotlib import pyplot as plt
import SimpleITK as sitk
import cv2
import os
import numpy as np
from DICOM_reader import find_folders, dcm_files_indexed, dcm_folder_to_pixel_matrix
# from preprocessing import get_name, norm_minmax_featurescaled, percentile_truncation
from extract_utils import *
# from skimage import io
# from scipy import ndimage as nd
# from skimage import img_as_float
# from skimage.metrics import peak_signal_noise_ratio


def n4correction(image, fwhm=0.15, bins=200, wienernoise=0.01, mask=True, close=False, verbose=True):
    print("     --- applying N4 correction ---") if verbose else 0
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetBiasFieldFullWidthAtHalfMaximum(fwhm)
    # corrector.SetMaximumNumberOfIterations(maxiter)
    # corrector.SetSplineOrder(splineorder)
    corrector.SetNumberOfHistogramBins(bins)
    corrector.SetWienerFilterNoise(wienernoise)
    im = sitk.GetImageFromArray(image)
    im = sitk.Cast(im, sitk.sitkFloat32)
    if mask:
        # msk = cv2.GaussianBlur(norm_minmax_featurescaled(image.copy(), printbool=False).astype("uint8"), ksize=(15, 15), sigmaX=0, sigmaY=0)
        msk = cv2.GaussianBlur(norm_minmax_featurescaled(image.copy(), printbool=False).astype("uint8"), ksize=(9, 9), sigmaX=0, sigmaY=0)
        msk = cv2.equalizeHist(msk)
        MI = sitk.OtsuThreshold(sitk.GetImageFromArray(msk), 0, 1, 210)
        # msk = cv2.equalizeHist(norm_minmax_featurescaled(image.copy(), printbool=False).astype("uint8"))    # OLD
        # maskImage = sitk.OtsuThreshold(sitk.GetImageFromArray(msk), 0, 1, 150)  # OLD

        # fig, ax = plt.subplots(ncols=2, nrows=2)
        # ax = ax.ravel()
        # ax[0].imshow(msk)
        # ax[1].imshow(sitk.GetArrayFromImage(maskImage))
        # ax[2].imshow(msk)
        # ax[3].imshow(sitk.GetArrayFromImage(MI))
        # plt.show()
        maskImage = MI
        if close:
            # maskImage = cv2.morphologyEx(sitk.GetArrayFromImage(maskImage), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            maskImage = cv2.morphologyEx(sitk.GetArrayFromImage(maskImage), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)))
            maskImage = sitk.GetImageFromArray(maskImage)
        else:   # if not close then open??
            # print("OPEN")
            # maskImage = cv2.morphologyEx(sitk.GetArrayFromImage(maskImage), cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
            # maskImage = cv2.morphologyEx(sitk.GetArrayFromImage(maskImage), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            # maskImage = sitk.GetImageFromArray(maskImage)
            pass
        # ax[3].imshow(sitk.GetArrayFromImage(maskImage))
        # plt.show()
        im_corrected = corrector.Execute(im, maskImage)
    else:
        im_corrected = corrector.Execute(im)
    img_corr = sitk.GetArrayFromImage(im_corrected)
    img_field = sitk.GetArrayFromImage(corrector.GetLogBiasFieldAsImage(im_corrected))
    if mask:
        return img_corr, img_field, sitk.GetArrayFromImage(maskImage)
    else:
        return img_corr, img_field, np.zeros(shape=image.shape)


if __name__ == "__main__":
    experiment = "pilot2"
    plane = "sagittal"
    # main(experiment, condition=plane) #loop over all times then send into main_time_given
    # time = "-7day"    #P1 + P2
    # time = "8day"     #P1 + P2    (W2
    time = "26day"      #P2         (W4)

    #LOOP OVER ALL RAW IMAGES TO SEE ANATOMICAL POSITION CHANGE OVER TIME
    condition = "sagittal"
    idx_shift = 3
    title="bro"
    # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(title, 800, 800)
    time_to_compare = "-7day"

    # path_cropped = os.path.join(os.getcwd(), "..", r"Segmentations\cropped_salivary\pilot1\raw\-7day\C1_sagittal.npy")
    # print(os.path.exists(path_cropped))
    # cropped_load = load_matrix(path_cropped)
    # image = cropped_load[4]
    # plt.imshow(image); plt.show()

    M = []
    for experiment in ["pilot1", "pilot2"]:
        print(experiment.upper())
        mice_path = os.path.join(os.getcwd(), "..", "..", "RAW DATA", experiment, time_to_compare)  #find all imaged mice at -7day
        micenames = [get_name(experiment, mouse_path, condition) for mouse_path in find_folders(os.path.join(mice_path), condition)]
        print(micenames)
        print(len(micenames), f"mice found with condition {condition} at time {time_to_compare}")

        for mousename, mousepath in zip(micenames, find_folders(os.path.join(mice_path), condition)):
            print(mousename)
            # print(mousepath)
            for time in ["-7day", "8day", "56day"] if experiment=="pilot1" else ["-7day", "8day", "26day"]:
                # print(time.upper())
                path = os.path.join(os.getcwd(), "..", "..", "RAW DATA", experiment, time, mousepath)
                if os.path.exists(path):
                    print(time, mousename)
                    # notskipbool = bool(mousename == "C2_sagittal" and time == "8day")
                    notskipbool = True
                    if notskipbool:
                        indexed = dcm_files_indexed(path, False)
                        MATR_RAW = dcm_folder_to_pixel_matrix(indexed, path, printbool=False)

                        # ds = ds.append(pd.Series(MATR_RAW, index=" ".join([time, mousename])))
                        M.append([MATR_RAW, " ".join([time, mousename])])
                        print(MATR_RAW.shape)
                        # idx_mid = MATR_RAW.shape[0] // 2
                        idx_mid = central_idx_dict[time + mousename]

                        im_raw = MATR_RAW[idx_mid].copy()
                        im_minmax = norm_minmax_featurescaled(MATR_RAW[idx_mid].copy(), 0, 255, printbool=False).astype("uint8")

                        # img = sitk.GetImageFromArray(im_raw)
                        # img = sitk.Cast(img, sitk.sitkFloat32)
                        # corrector = sitk.N4BiasFieldCorrectionImageFilter()
                        #
                        # # IMAGE MASKING
                        # msk = cv2.equalizeHist(norm_minmax_featurescaled(im_raw.copy()).astype("uint8"))
                        # maskImage = sitk.OtsuThreshold(sitk.GetImageFromArray(msk), 0, 1, 150)
                        # # fig, ax = plt.subplots(ncols=2)
                        # # fig.tight_layout()
                        # # [x.axis("off") for x in ax]
                        # # ax[0].imshow(msk)
                        # # ax[0].set_title("Equalized histogram")
                        # # ax[1].imshow(sitk.GetArrayFromImage(maskImage))
                        # # ax[1].set_title("Otsu threshold mask")
                        # # # fig.suptitle("Image masking for N4 bias correction")
                        # # plt.show()
                        #
                        # im_corrected = corrector.Execute(img, maskImage)
                        # img_corr = sitk.GetArrayFromImage(im_corrected)
                        # img_field = sitk.GetArrayFromImage(corrector.GetLogBiasFieldAsImage(im_corrected))

                        # img_corr, img_field, mask = n4correction(im_raw, mask=True)
                        # img_corr, img_field, mask = n4correction(im_raw, mask=True, close=False)
                        img_corr, img_field, mask = n4correction(im_raw)#, mask=True, close=False)

                        # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 4))
                        fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
                        # ax1, ax2, ax3, ax4 = axes.ravel()
                        ax1, ax2, ax4 = axes.ravel()
                        # fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
                        # ax1, ax2, ax3, ax4 = axes.ravel()
                        [ax.axis("off") for ax in axes.ravel()]
                        fig.tight_layout()
                        # fig.suptitle("Bias field correction using the N4 method")
                        ax1.imshow(mask)
                        ax1.set_title("Mask")
                        ax2.set_title("Raw image + bias field")
                        ax2.imshow(percentile_truncation(im_minmax, 0, 99.9), cmap="gray");
                        ax2.imshow(img_field, cmap="bwr", alpha=0.6)
                        # ax3.set_title("N4ITK corrected")
                        # ax3.imshow(percentile_truncation(img_corr, 0, 99.9), cmap="gray")
                        # diff = im_minmax.ravel() - img_corr.ravel()
                        diff = im_raw - img_corr

                        # print(diff.shape)
                        ax4.set_title("$\Delta$ image")
                        # im_d = ax4.imshow(percentile_truncation(diff), cmap="bwr")
                        im_d = ax4.imshow(diff, cmap="bwr")
                        from mpl_toolkits.axes_grid1 import make_axes_locatable #https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
                        divider = make_axes_locatable(ax4)
                        cax = divider.append_axes("bottom", size="5%", pad=-.10)
                        fig.colorbar(im_d, cax=cax, orientation="horizontal")
                        # diff = diff.ravel()
                        # diff = diff[np.nonzero(diff > 0.15)]
                        # print(diff.shape)
                        # ax3.hist(diff, bins=250)
                        # ax3.hist(img_corr.ravel()[np.nonzero(img_corr.ravel() > 3)], bins=150)

                        plt.show()
                        plt.close()
            print("")
            pass
        print("\n")
    cv2.destroyAllWindows()
    print(np.shape(M))