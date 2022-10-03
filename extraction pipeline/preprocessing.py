import numpy as np
# from watershed import MouseClick
import cv2
# import DICOM_reader
from DICOM_reader import find_folders, dcm_files_indexed, dcm_folder_to_pixel_matrix, load_matrix
from matplotlib import pyplot as plt
# from watershed import load_matrix
import os
from visualizations import show_histogram
# import seaborn as sns
from scipy import stats
import pandas as pd
from SimpleITK import HistogramMatchingImageFilter
from extract_utils import *
from MRI_denoising import n4correction

class MouseClick:
    def __init__(self, clicked_point=(0,0), clickval=0, sliceloc=0):
        self.clicked_point = clicked_point
        self.clickval = clickval
        self.sliceloc=sliceloc
    def click_to_point(self, event, x, y, flags, param=([], "")):
        arr, arr_name = param
        if event == cv2.EVENT_LBUTTONDOWN:
            # self.clicked_point = (x, y)#(y, x)
            self.clicked_point = (y, x)
            print("You clicked on pixel %i , %i" % (self.clicked_point))
            try:
                self.clickval = arr[self.clicked_point]
                print("%s value = " % arr_name, self.clickval)
            except Exception as e:
                print(e.args)
    def click_to_point_3D(self, event, x, y, flags, param=([], "")):
        arr, arr_name = param
        if event == cv2.EVENT_LBUTTONDOWN:
            # self.clicked_point = (x, y)#(y, x)
            self.clicked_point = (y, x)
            print("You clicked on pixel %s in slice %i" % (self.clicked_point, self.sliceloc))
            try:
                self.clickval = arr[self.sliceloc, self.clicked_point[0], self.clicked_point[1]]
                print("%s value = " % arr_name, self.clickval)
            except Exception as e:
                print(e.args)


def z(x, mu, sigma, w, mu_new):
    return w * (x - mu) / sigma + mu_new


def norm_stscore(matrix_raw, new_mean=0, weight=1, ignore_zeros=False):
    # matrix = np.array()
    if ignore_zeros:  #TODO: ADD CONDITION TO NOT INCLUDE ZEROS IN CALCULATIONS
        nonzero = matrix_raw[np.nonzero(matrix_raw)]
        print(nonzero.shape)
        mu, sigma = np.mean(nonzero), np.std(nonzero)
    else:
        mu = np.mean(matrix_raw)
        sigma = np.std(matrix_raw)
    zfunc = np.vectorize(z)
    print("Matrix of shape", matrix_raw.shape, f"with mu={mu:.2f} and std={sigma:.2f} normalized by standard z-score (to mu={new_mean}, std={weight:.2f}).")#.format(mu, sigma, weight))
    return zfunc(matrix_raw, mu, sigma, weight, new_mean) #STANDARD SCORE NORMALIZATION


def mean_centering(im, n=3, mode="image", ROI=np.zeros(0)):
    if mode == "image":
        mu = np.mean(im)
        sd = np.std(im)
    elif mode.lower() == "roi" or np.any(ROI):
        # print("ROI CENTERING NOT IMPLEMENTED YET")
        im_roi = im[ROI != 0]
        mu = np.mean(im_roi)
        sd = np.std(im_roi)
        # return 0
    else:
        print("INVALID MODE", mode)
        return 0
    func = lambda x: (x - mu) / sd + n * sd
    xfunc = np.vectorize(func)
    return xfunc(im)


# def minmax_norm(x, min, max, lower, upper, rounded):
#     a = lower + (x - min) * (upper - lower) / (max - min)
#     return round(a) if rounded else a


# def norm_minmax_featurescaled(m, lower=0, upper=255, rounded=False, printbool=True):
#     min, max = np.min(m), np.max(m)
#     func = np.vectorize(minmax_norm)
#     print("Matrix of shape", m.shape, "with range [{0:.2f},{1:.2f}] normalized by feature-scaled min-max norm to [{2},{3}].".format(min, max, lower, upper))    if printbool else 0
#     try:
#         return func(m, min, max, lower=lower, upper=upper, rounded=rounded)
#     except Exception as e:
#         print("ERROR MINMAX NORM:", e.args)
#         return m


# def percentile_truncation(m, lower=0, upper=99, settozero=False):
#     m = m.copy()
#     min, max = np.percentile(m, [lower, upper])
#     print("PERCENTILE TRUNCATION:")
#     print(f"From {len(m.ravel())} voxels, {len(m[m < min])} lower than {lower} percentile = {min} set to {0 if settozero else min}, {len(m[m > max])} above {upper} percentile = {max:.3f}.")
#     print(len(m[m < min]) + len(m[m > max]), "voxels adjusted.")
#     m[m < min] = 0 if settozero else min
#     m[m > max] = max# if not settozero else 0
#     return m


def center_and_crop(matrix, savefolder, time, title="window"):      #TODO: SAVE RAW VALUES (NOT UINT)?
    idx_max = np.shape(matrix)[0] - 1
    idx_min = 0
    idx = int(idx_max/2)
    display_matrix = norm_minmax_featurescaled(np.array(matrix).copy(), 0, 255).astype("uint8")
    cropped_disp_matrix = display_matrix.copy()
    click = MouseClick(sliceloc=idx)
    deltaz, deltax, deltay = 5, 40, 40      #cropping parameters
    deltaz = 4
    # deltaz = 3
    cropped = False
    cropped_matrix = np.empty(shape=matrix.shape)
    while True:
        print(f"Slice index {idx}")
        click.sliceloc = idx
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 800, 800)
        disp_image = display_matrix[idx].copy() if not cropped else cropped_disp_matrix[idx].copy()
        disp_image = cv2.convertScaleAbs(disp_image, alpha=2, beta=0)
        cv2.imshow(title, disp_image)
        cv2.setMouseCallback(title, click.click_to_point, (np.zeros(shape=display_matrix.shape), "empty"))
        print("Press: enter = crop + center on mouseclick, browse: ,. , c = clear crop, q = SAVE cropped as npy x = exit")
        key = cv2.waitKey(0)

        if key == 32:   #space - crop + center on mouse click
            x, y = click.clicked_point
            print(x, y)
            try:
                cropped_disp_matrix = display_matrix[idx-deltaz:idx+deltaz, x-deltax:x+deltax, y-deltay:y+deltay]
                cropped_matrix = matrix[idx-deltaz:idx+deltaz, x-deltax:x+deltax, y-deltay:y+deltay]
            except Exception as e:
                print(e.args)
            idx = deltaz - 1
            print(idx)
            idx_max = 2 * deltaz - 1; idx_min = 0
            print("Cropping image matrix.", cropped_disp_matrix.shape)
            cropped = True
            pass
        elif key == 46:
            idx += 1
            idx = idx_max if idx > idx_max else idx
        elif key == 44:
            idx -= 1
            idx = idx_min if idx < idx_min else idx
        elif key == 99:  # c = clear crop
            cropped = False
            idx_max = np.shape(matrix)[0] - 1
            idx_min = 0
            idx = int(idx_max / 2)
            cropped_disp_matrix = display_matrix.copy()
            cropped_matrix = matrix.copy()
        elif key == 113:        #q = SAVE ROI as csv
            # save_path = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary"
            print(cropped_disp_matrix.dtype, cropped_disp_matrix.shape)
            save_matrix(cropped_matrix, savefolder, time, title)
            # cv2.destroyAllWindows()
            # return
        elif key == 120:    #x
            cv2.destroyAllWindows()
            return cropped_matrix


def save_matrix(m, folder, time, title):
    save_path = folder + r"\\" + time + r"\\"
    savesave = os.path.normpath(os.path.join(save_path, title + ".npy"))
    print(os.path.exists(savesave), savesave)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        try:
            np.save(file=save_path + title, arr=m)
            print(f"FILE SAVED as {save_path + title}\n")
        except Exception as e:
            print(e.args)
    elif os.path.exists(save_path + title + ".npy"):
        while True:
            ch = input(f"File found at {save_path} \nDo you want to overwrite? y/n")
            if ch == "y":
                try:
                    np.save(file=save_path + title, arr=m)
                    print(f"FILE SAVED as {save_path + title}")
                except Exception as e:
                    print(e.args)
                break
            elif ch == "n":
                print("File not saved.")
                break
            else:
                print(ch, "is not a valid choice.")
    else:
        try:
            np.save(file=save_path + title, arr=m)
            print(f"FILE SAVED as {save_path + title}\n")
        except Exception as e:
            print(e.args)
    return 0


# def load_matrix(path, expected=np.array([]), printtext=True):
#     loaded = np.load(path)
#     # print(type(loaded))
#     if expected.any():
#         if (loaded == expected).all():
#             print("Loaded matrix is exactly the same as expected matrix.")
#         else:
#             print("Loaded matrix is not the same as expected.")
#     elif printtext:
#         print(f"\nFILE LOADED from {path}: No expected matrix to compare with. Shape = {loaded.shape}, dtype = {loaded.dtype}")
#     else:
#         pass
#     return loaded

# def get_name(experiment, folder, condition=""):
#     experiment = experiment.lower()
#     if experiment == "pilot2":
#         name = folder[:3] if not "p" in folder else folder[:12]     # differentiates small and big p!!
#         name = name + "_" + condition if bool(condition) else name
#     elif experiment == "pilot1":
#         name = folder.replace(" ", "_")
#     else:
#         print("Experiment", experiment, "not recognized.")
#     return name


def create_circular_mask(im, r, c=[0, 0]):
    '''im: 2-dimensional image
        r: pixel radius of sphere mask'''
    msk = np.zeros(shape = im.shape)
    if not any(c):
        # print("no c")
        # i0, j0 = im.shape[0] // 4, im.shape[1] // 4
        i0, j0 = 62, 81
    else:
        i0, j0 = c
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if (i - i0)**2 + (j - j0)**2 < r**2:
                msk[i, j] = 1
    return msk


# def histogram_equalization(img, )
#     #TODO: expand to 3d matrices?
#     equ = cv2.equalizeHist(img)
#     return equ

def main(experiment, condition=""):
    datadir = os.path.normpath(os.path.join(os.getcwd(), "..", "RAW DATA", experiment))
    # print(datadir)
    for time in find_folders(datadir):
        main_time_given(experiment, time, condition=condition)
    pass


def main_time_given(experiment, time, condition=""):        #read files, CENTER + CROP, NORMALIZE AFTER
    datadir = os.path.normpath(os.path.join(os.getcwd(), "..", "RAW DATA", experiment, time))
    print("Files in", datadir, ":")
    for folder in find_folders(datadir, condition):
        print("\n")
        print("folder:", folder)
        if experiment == "pilot2":
            name = folder[12:15] if not "p" in folder else folder[12:24]
            name += "_" + condition
        elif experiment == "pilot1":
            name = folder
        else:
            print("Experiment", experiment, "not recognized.")
        print("NAME = ", name)
        indexed = dcm_files_indexed(os.path.join(datadir, folder), True)
        MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder))
        # print(MATR_RAW.shape)
        #CROP AND CENTER PIXEL ARRAY
        savefolder = os.path.normpath(os.path.join(os.getcwd(), "..", r"Segmentations\cropped_salivary", experiment, "raw", time))
        print(savefolder)
        if not os.path.exists(savefolder):  os.makedirs(savefolder)
        if not os.path.exists(os.path.join(savefolder, name + ".npy")):
            center_and_crop(MATR_RAW, savefolder=savefolder, time="", title=name)   #folder for time already made, therefore time=""
            #MAKE SURE CENTERED PLANE IS AT ISH SAME ANATOMICAL POSITION IN ALL MICE
            pass
        #LOAD + SHOW CROPPED MATRIX? SEE WHATS UP
        loaded = load_matrix(os.path.join(savefolder, name + ".npy"))
        show_histogram(loaded[4], log=True, titleimg=time + " cropped_" + name)

    pass


if __name__ == "__main__":
    # folder1 = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Pilot_LateEffects_-7day\C1_sagittal"
    # folder2 = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Pilot_LateEffects_-7day\C2_sagittal"
    # # title = folder[-11:]  # XX_sagittal
    # path1 = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\watershed_2d_manual\-7days\C1_sagittal.csv"
    # path2 = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\watershed_2d_manual\-7days\C2_sagittal.csv"
    # indexed_files = DICOM_reader.dcm_files_indexed(folder1)
    # pixel_matrix_1 = DICOM_reader.dcm_folder_to_pixel_matrix(indexed_files, folder1)
    # indexed_files = DICOM_reader.dcm_files_indexed(folder2)
    # pixel_matrix_2 = DICOM_reader.dcm_folder_to_pixel_matrix(indexed_files, folder2)
    # image = pixel_matrix_1[14]
    # DICOM_reader.print_ndstats(image)

    # experiment = "pilot1"
    experiment = "pilot2"
    plane = "sagittal"
    # main(experiment, condition=plane) #loop over all times then send into main_time_given
    # time = "-7day"    #P1 + P2
    # time = "8day"     #P1 + P2    (W2
    time = "26day"      #P2         (W4)
    # time = "56day"    #P1
    # time= "W2"
    # main_time_given(experiment, time, condition=plane)   #time specified

    import pandas as pd
    # LOOP OVER ALL RAW IMAGES TO SEE ANATOMICAL POSITION CHANGE OVER TIME
    condition = "sagittal"
    idx_shift = 3
    title="bro"
    # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(title, 800, 800)
    time_to_compare = "-7day"
    ds = pd.Series()
    M = []
    pxavgvals = []
    for experiment in ["pilot1", "pilot2"]:
        print(experiment.upper())
        mice_path = os.path.join(os.getcwd(), "..", "RAW DATA", experiment, time_to_compare)  #find all imaged mice at -7day
        micenames = [get_name(experiment, mouse_path, condition) for mouse_path in find_folders(os.path.join(mice_path), condition)]
        print(micenames)
        print(len(micenames), f"mice found with condition {condition} at time {time_to_compare}")
        for mousename, mousepath in zip(micenames, find_folders(os.path.join(mice_path), condition)):
            # print(mousename)
            # print(mousepath)
            for time in ["-7day", "8day", "56day"] if experiment=="pilot1" else ["-7day", "8day", "26day"]:
                # print(time.upper())
                path = os.path.join(os.getcwd(), "..", "RAW DATA", experiment, time, mousepath)
                if os.path.exists(path):
                    print(time, mousename)
                    indexed = dcm_files_indexed(path, False)
                    MATR_RAW = dcm_folder_to_pixel_matrix(indexed, path, printbool=False)

                    # ds = ds.append(pd.Series(MATR_RAW, index=" ".join([time, mousename])))
                    M.append([MATR_RAW, " ".join([time, mousename])])
                    print(MATR_RAW.shape)
                    idx_mid = MATR_RAW.shape[0] // 2

                    # IMAGE PROCESSING
                    im_raw = MATR_RAW[idx_mid].copy()
                    pxavgvals.append(np.average(im_raw))

                    truncLOW, truncUP = 0, 99
                    im_disp = MATR_RAW[idx_mid].copy()
                    # im_disp = norm_minmax_featurescaled(MATR_RAW[idx_mid].copy(), 0, 255, printbool=False).astype("uint16")
                    im_disp = percentile_truncation(im_disp, lower=truncLOW, upper=truncUP, settozero=True)
                    # cv2.imshow(title, disp_im)
                    # cv2.waitKey(0)

                    # PLOTTING IM + HIST W/ KDE
                    nbins = 250
                    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 6)); ax1, ax2, ax3, ax4 = axes.ravel();    fig.tight_layout()
                    ax1.imshow(im_raw, cmap="gray");  ax1.set_title("RAW")
                    kde_raw = stats.gaussian_kde(im_raw.flatten())
                    # print(kde_raw.weights)
                    ax2.hist(im_raw.flatten(), density=True, bins=nbins)
                    xx = np.linspace(0, np.max(im_raw), 500)
                    ax2.plot(xx, kde_raw(xx))
                    ax2.set_title(f"px avg={np.average(im_raw):.2e}, median={np.median(im_raw):.2e} sd={np.std(im_raw):.2e}, min={np.min(im_raw):.2e}, max={np.max(im_raw):.2e}")

                    ax3.imshow(im_disp, cmap="gray");   ax3.set_title(f"Raw w/ trunc @ [{truncLOW}, {truncUP}]%")
                    kde_disp = stats.gaussian_kde(im_disp.flatten())
                    ax4.hist(im_disp.flatten(), density=True, bins=nbins)
                    xx = np.linspace(0, np.max(im_disp), 500)
                    ax4.plot(xx, kde_disp(xx))
                    ax4.set_title(f"px avg={np.average(im_disp):.2e}, median={np.median(im_disp):.2e} sd={np.std(im_disp):.2e}, min={np.min(im_disp):.2e}, max={np.max(im_disp):.2e}")

                    plt.show()
                    plt.close()
            print("")
            pass
        print("\n")
    cv2.destroyAllWindows()
    print(np.shape(M))
    plt.plot(range(len(pxavgvals)), pxavgvals)
    plt.show()


    # LOOP OVER ALL MICE, EVALUATE SD, MEAN, MEDIAN FOR VARIOUS NORMALIZATION METHODS (TO WHOLE IMAGE OR BRAIN)
    # pxvalavg_raw = []
    # pxvalsd_raw = []
    # pxvalmedian_raw = []
    #
    # pxvalavg_norm = []
    # pxvalsd_norm = []
    # pxvalmedian_norm = []
    #
    # pxvalavg_roi = []
    # pxvalsd_roi = []
    # pxvalmedian_roi = []
    # condition = "sagittal"
    # dir_roi_brain = os.path.join(os.getcwd(), "..", "..", r"Segmentations\brain")
    # dir_roi_saliv = os.path.join(os.getcwd(), "..", "..", r"Segmentations\salivary")
    # # plotimages = False
    # plotimages = True
    # df = pd.DataFrame({"name":[], "exp":[], "time":[], "p":[], "avg im":[], "median im":[], "sd im":[], "avg roi":[], "median roi":[], "sd roi":[], "stscore avg":[]})
    # df2 = pd.DataFrame({"name":[], "exp":[], "time":[], "p":[], "n4":[], "avg im":[], "median im":[], "sd im":[], "avg brain":[], "median brain":[], "sd brain":[],
    #                     "avg saliv":[], "median saliv":[], "sd saliv":[]})   # NORMALIZED DATA
    #
    # for experiment in ["pilot1", "pilot2"]:
    #     print(experiment.upper())
    #     for time in ["-7day", "8day", "56day"] if experiment=="pilot1" else ["-7day", "8day", "26day"]:
    #         datadir = os.path.normpath(os.path.join(os.getcwd(), "..", "..", "RAW DATA", experiment, time))
    #         print(time.upper())
    #         print("Files in", datadir, ":")
    #         for folder in find_folders(datadir, condition):
    #             name = get_name(experiment, folder, condition="sagittal")
    #             pbool = "p" in name
    #             ctrlBOOL = False
    #             if "1-" in name or "C" in name:
    #                 # print("CONTROL")
    #                 ctrlBOOL = True
    #             if True:
    #             # ctrlBOOL = True
    #             # if ctrlBOOL and not(pbool):
    #                 for n4bool in [False, True]:    # check differences with/ without N4 correction
    #                     indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
    #                     MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder))
    #                     # print("\nNAME = ", name)
    #                     # print("NUM SLICES = ", MATR_RAW.shape[0])
    #                     # idx_center = MATR_RAW.shape[0] // 2
    #                     idx_center = central_idx_dict[time+name]
    #                     print("\n",name, idx_center)
    #                     im_raw = MATR_RAW[idx_center].copy()
    #
    #                     if n4bool:
    #                         im_raw_corr, im_field, msk = n4correction(im_raw)
    #
    #                     # im_norm = mean_centering(im_raw.copy(), n=3)   # n=3 suggested Scalco et al. (2020)
    #
    #                     # pxvalavg_raw.append(np.average(im_raw))
    #                     # pxvalsd_raw.append(np.std(im_raw))
    #                     # pxvalmedian_raw.append(np.median(im_raw))
    #
    #
    #                     path_roi_brain = os.path.join(dir_roi_brain, experiment, time, name + ".npy")
    #                     if os.path.exists(path_roi_brain):
    #                         print("BRAIN ROI EXISTS")
    #                         roi_brain = np.load(path_roi_brain)
    #                         # roi_brain = roi_brain[roi_brain.shape[0] // 2]
    #                         roi_brain = roi_brain[idx_center]
    #                     else:
    #                         print("BRAIN ROI NOT FOUND")
    #                         # roi_brain = create_circular_mask(im_raw, c=[85, 65], r=15)  # c = [row, col] idx (y, x in imshow)
    #                         roi_brain = np.zeros(shape=im_raw.shape)
    #                     path_roi_saliv = os.path.join(dir_roi_saliv, experiment, time, name + ".npy")
    #                     if os.path.exists(path_roi_saliv):
    #                         print("SALIV ROI EXISTS")
    #                         roi_saliv = np.load(path_roi_saliv)
    #                         roi_saliv = roi_saliv[idx_center]
    #                     else:
    #                         print("SALIV ROI NOT FOUND")
    #                         roi_saliv = np.zeros(shape=im_raw.shape)
    #                     if n4bool:
    #                         # im_raw_corr
    #                         im_norm = mean_centering(im_raw_corr.copy(), mode="roi", ROI=roi_brain, n=0)
    #                         pass
    #                     else:
    #                         im_norm = mean_centering(im_raw.copy(), mode="roi", ROI=roi_brain, n=0)  # n=3 suggested Scalco et al. (2020)
    #
    #                     # if name == "C2_sagittal" and time == "8day":    #outlier...
    #                     #     fig, ax = plt.subplots(ncols=3)
    #                     #     fig.tight_layout()
    #                     #     ax[0].imshow(im_field, cmap="bwr") if n4bool else 0
    #                     #     ax[0].imshow(im_raw, cmap="gray", alpha=0.4) if n4bool else ax[0].imshow(im_raw)
    #                     #     ax[1].imshow(im_raw_corr - im_raw, cmap="bwr")
    #                     #     ax[2].imshow(im_norm, cmap="gray")
    #                     #     plt.show()
    #
    #                     # pxvalavg_norm.append(np.average(im_norm))
    #                     # pxvalsd_norm.append(np.std(im_norm))
    #                     # pxvalmedian_norm.append(np.median(im_norm))
    #
    #
    #                     # roi_brain_im = im_raw.copy()
    #                     roi_brain_im = np.ma.masked_where(roi_brain == 0, im_raw)
    #                     # print(roi_brain_im.shape)
    #                     # print(f"roi mean = {np.mean(roi_brain_im):.3e}, sd = {np.std(roi_brain_im):.3e}")
    #                     # roi = roi_brain # change what roi / image to extract mean, sd etc from
    #
    #                     im = im_norm        # change what norm im to look at
    #                     brain_vals = im[np.nonzero(roi_brain)].ravel()
    #                     saliv_vals = im[np.nonzero(roi_saliv)].ravel()
    #                     # print(roi_vals.shape)
    #                     # print(np.count_nonzero(roi_brain))
    #                     if np.count_nonzero(brain_vals) > 0 and np.count_nonzero(saliv_vals) > 0:
    #                         # roi_min = np.min(im_raw[np.nonzero(roi_brain)])
    #                         brain_min = np.min(brain_vals)
    #                         brain_max = np.max(brain_vals)
    #                         brain_avg = np.average(brain_vals)
    #                         brain_median = np.median(brain_vals)
    #                         brain_sd = np.std(brain_vals)
    #
    #                         saliv_avg = np.average(saliv_vals)
    #                         saliv_median = np.median(saliv_vals)
    #                         saliv_sd = np.std(saliv_vals)
    #
    #                         # pxvalavg_roi.append(roi_avg)
    #                         # pxvalmedian_roi.append(roi_median)
    #                         # pxvalsd_roi.append(roi_sd)
    #
    #                         # stscore = (np.average(im_raw) - roi_avg) / roi_sd + 3 * roi_sd  # ILLUSTRATIVE VALUE FOR BRAIN NORMALIZATION SHIFT
    #                         # df = df.append({"name":name, "exp":experiment, "time":time, "p":"p" in name, "avg im":np.average(im_raw), "median im":np.median(im_raw),
    #                         #                 "sd im":np.std(im_raw), "avg roi":brain_avg, "median roi":roi_median, "sd roi":roi_sd}, ignore_index=True)#, "stscore avg":stscore}, ignore_index=True)
    #                         df2 = df2.append(
    #                             {"name": name, "exp": experiment, "time": time, "p": pbool, "n4":n4bool, "avg im": np.average(im), "median im": np.median(im), "sd im": np.std(im),
    #                              "avg brain": brain_avg, "median brain": brain_median, "sd brain": brain_sd,
    #                              "avg saliv": saliv_avg, "median saliv": saliv_median, "sd saliv": saliv_sd}, ignore_index=True)
    #
    #                     else:   # Do not include roi values if roi not found...
    #                         pass
    #
    #                     # PLOTTING IM + HIST W/ KDE
    #                     if plotimages:
    #                         nbins = 250
    #                         fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(12, 6));
    #                         ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel();    fig.tight_layout()
    #                         ax1.imshow(im_raw, cmap="gray");  ax1.set_title("RAW")
    #                         kde_raw = stats.gaussian_kde(im_raw.flatten())
    #                         # print(kde_raw.weights)
    #                         ax2.hist(im_raw.flatten(), density=True, bins=nbins)
    #                         xx = np.linspace(0, np.max(im_raw), 500)
    #                         ax2.plot(xx, kde_raw(xx))
    #                         ax2.set_title(f"px avg={np.average(im_raw):.2e}, median={np.median(im_raw):.2e} sd={np.std(im_raw):.2e}, min={np.min(im_raw):.2e}, max={np.max(im_raw):.2e}")
    #
    #                         ax3.imshow(im_norm, cmap="gray");   ax3.set_title(f"Centered at mean")
    #                         kde_disp = stats.gaussian_kde(im_norm.flatten())
    #                         ax4.hist(im_norm.flatten(), density=True, bins=nbins)
    #                         xx = np.linspace(np.min(im_norm), np.max(im_norm), 500)
    #                         ax4.plot(xx, kde_disp(xx))
    #                         ax4.set_title(f"px avg={np.average(im_norm):.2e}, median={np.median(im_norm):.2e} sd={np.std(im_norm):.2e}, min={np.min(im_norm):.2e}, max={np.max(im_norm):.2e}")
    #
    #                         ax5.imshow(im_raw, cmap="gray")
    #                         ax5.imshow(roi_brain_im, alpha=0.9)
    #                         # ax5.imshow(roi_brain_im)
    #                         # ax6.hist(im_raw[np.nonzero(roi_brain)].flatten(), density=True, bins=nbins//3)
    #                         ax6.hist(brain_vals, density=True, bins=nbins//4)
    #                         # ax6.hist(roi_brain_im.flatten(), density=True, bins=nbins//3)
    #                         kde_roi = stats.gaussian_kde(brain_vals)
    #                         xx = np.linspace(np.min(brain_vals), np.max(brain_vals), 500)
    #                         ax6.plot(xx, kde_roi(xx))
    #                         # ax6.set_title(f"In ROI: avg={roi_avg:.2e}, median={roi_median:.2e}, sd={roi_sd:.2e}, min={roi_min:.2e}, max={roi_max:.2e}")
    #                         # print(np.count_nonzero(roi_brain_im), len(im_raw[np.nonzero(roi_brain)].flatten()), np.count_nonzero(roi_brain))
    #                         plt.show()
    # print(df)
    # print(df2)
    # print(np.shape(pxvalavg_raw))
    # # df_norm_melt = df2.melt(id_vars=["name", "exp", "time", "p"], value_vars=["avg im", "median im", "sd im", "avg brain", "median brain", "sd brain", "avg saliv", "median saliv", "sd saliv"])
    # df_norm_melt = df2.melt(id_vars=["name", "exp", "time", "p", "n4"], value_vars=["avg im", "median im", "sd im", "avg brain", "median brain", "sd brain", "avg saliv", "median saliv", "sd saliv"])
    # import seaborn as sns
    #
    # sns.boxplot(data=df_norm_melt, x="variable", y="value", hue="n4")
    # # plt.title("Control data (no p) after Z-score norm to ROI_brain")
    # plt.title("All data (with p) after Z-score norm to ROI_brain")
    # plt.xlabel("")
    # plt.ylabel("Px intensity / a.u.")
    # plt.grid()
    # plt.show()


    # fig, axes = plt.subplots()#, sharex=True)
    # ax1, ax2, ax3 = axes.ravel()
    # ax1 = fig.gca()
    # import seaborn as sns
    # # g = sns.PairGrid(df, vars=["avg im", "sd im", "avg roi", "sd roi", "stscore avg"], hue="p")
    # g = sns.PairGrid(df, vars=["avg im", "sd im", "avg roi", "sd roi"], hue="p")
    # g.map_upper(sns.scatterplot)
    # g.map_lower(sns.kdeplot)
    # g.map_diag(sns.kdeplot)
    # g.add_legend()
    # plt.show()
    # plt.close()

    # df_melt = df.melt(id_vars=["name", "exp", "time", "p"], value_vars=["avg im", "sd im", "avg roi", "sd roi"])
    # fig, axes = plt.subplots(ncols=3)
    # ax1, ax2, ax3 = axes.ravel()
    # [ax.grid(axis="y") for ax in axes]
    # sns.boxplot(data=df_melt, x="variable", y="value", hue="p", ax=ax1)
    # sns.boxplot(data=df_melt, x="variable", y="value", hue="time", ax=ax2)
    # sns.boxplot(data=df_melt, x="variable", y="value", hue="exp", ax=ax3)
    # ax1.set_ylabel("SD of pixel intensity / a.u.")
    # [ax.set_ylabel("") for ax in [ax2, ax3]]
    # # plt.show()
    # plt.close()

    # sns.violinplot(data=df_melt, x="variable", y="value")
    # plt.ylabel("SD of pixel intensities / a.u.")
    # plt.xlabel("")
    # plt.grid(axis="y")
    # plt.show()


    # ax1.set_xticks(range(len(datalist)), ["Avg raw", "Median raw", "sigma raw", "avg roi", "median roi", "sigma roi"])
    # "name": [], "exp": [], "time": [], "p": [], "avg im": [], "median im": [], "sd im": [], "avg roi": [], "median roi": [], "sd roi": []

    #
    #             # idx_mid = MATR_RAW.shape[0] // 2
    #             # disp_im = norm_minmax_featurescaled(MATR_RAW[idx_mid].copy(), 0, 255).astype("uint8")
    #             # cv2.imshow(title, disp_im)
    #             # cv2.waitKey(0)
    #             # cv2.destroyAllWindows()
    #             print("\n")
    #         print("\n")

    # cropped_load = load_matrix(r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary\raw\-7day\C1_sagittal.npy")
    # image = cropped_load[4]
    # # plt.imshow(image); plt.show()
    # lowerperc = 2.5; upperperc = 99.5
    # image = percentile_truncation(image, lower=lowerperc, upper=upperperc, settozero=1)
    # # image = norm_stscore(image)
    #
    # fig, axes = plt.subplots(ncols=2, nrows=2); ax=axes.ravel(); fig.tight_layout()
    # ax[0].imshow(cropped_load[4], cmap="gray")
    # ax[1].imshow(image, cmap="gray")
    # plt.show()



    #COMPARE ORDER OF PROCESSING STEPS
    # cropped_load = load_matrix(r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary\-7day\C1_sagittal.npy")
    # image = cropped_load[4]
    # lowerperc = 2.5; upperperc = 99.5
    # image_perc = percentile_truncation(image, lower=lowerperc, upper=upperperc, settozero=1)
    # image_perc = norm_stscore(image_perc)
    # image_norm = norm_stscore(image)
    # image_norm = percentile_truncation(image_norm, lower=lowerperc, upper=upperperc, settozero=1)
    #
    # fig, axes = plt.subplots(nrows=1, ncols=3);
    # ax = axes.ravel()
    # fig.tight_layout()
    # ax[0].imshow(image, cmap="gray");
    # ax[0].set_title("Raw image")
    # ax[1].imshow(image_perc, cmap="gray");  ax[1].set_title("Percentile truncation first")
    # ax[2].imshow(image_norm, cmap="gray");  ax[2].set_title("Normalization first")
    # fig.suptitle(f"Intensities truncated to percentile range: $P_{{{lowerperc:.1f}\%}}$, $P_{{{upperperc:.1f}\%}}$", size="x-large")
    # fig.subplots_adjust(top=1)
    # plt.show()