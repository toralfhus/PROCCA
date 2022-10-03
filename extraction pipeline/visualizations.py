from matplotlib.widgets import Slider
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl
# from preprocessing import norm_stscore, norm_minmax_featurescaled, percentile_truncation
import cv2
import SimpleITK as sitk
from DICOM_reader import find_folders, dcm_folder_to_pixel_matrix, dcm_files_indexed, load_matrix
from extract_utils import *
# from watershed import load_matrix, watershed_2d, watershed_3d


def show_histogram(image, range=None, bins=100, include_zero=True, norm=False, log=False, titleimg="image", titlehst=""):
    plt.figure()
    plt.subplot(121); plt.title(titleimg)
    plt.imshow(image, cmap=plt.cm.bone)
    plt.subplot(122);
    if not(titlehst):
        titlehst = "{} intensity histogram with {} bins".format(("Normalized" if norm else "Raw"), bins)
    plt.title(titlehst)
    hist = np.log10(image.ravel()) if log else image.ravel()
    plt.hist(hist, bins=bins, range=range)
    plt.grid()
    plt.ylabel("# of pixels")# if not log else plt.ylabel("log(# of pixels)")
    xlab = "intensity / [{}]".format("$\sigma$" if norm else "a.u.")
    plt.xlabel("log (" + xlab + ")" if log else xlab)
    plt.show()
    #maybe use skimage.exposure.histogram? looks way better..
    # like this
    # hist, hist_centers = histogram(image, nbins=256, normalize=False)
    # plt.plot(hist_centers, hist, lw=1)


def compare_images(image1, image2, title1="raw", title2="image 2", suptitle=""):
    # disp_image = image.copy()
    # disp_image[mask != 0] = 1
    fig = plt.figure()
    fig.tight_layout()
    plt.subplot(121); plt.title(title1)
    plt.imshow(image1, cmap=plt.cm.bone)
    # plt.colorbar()
    plt.subplot(122); plt.title(title2)
    plt.imshow(image2, cmap=plt.cm.bone)
    # plt.colorbar()
    fig.suptitle(suptitle)
    plt.show()


def plot_image(image, title=""):
    fig = plt.figure()
    fig.tight_layout()
    plt.imshow(image, cmap=plt.cm.bone)
    plt.colorbar()
    plt.title(title)
    plt.show()


def plot_masked(im, mask, title=""):
    fig, axes = plt.subplots(ncols=2)
    ax1, ax2 = axes.ravel()
    ax1.imshow(im, cmap="gray")
    ax2.imshow(im, cmap="gray")
    ax2.imshow(mask, alpha=0.3)
    plt.suptitle(title)
    plt.show()
    plt.close(fig)
    pass



def browse_3D_segmented(image_matrix, roi_matrix, title="Segmented slices", contrast=True):  #Visualize 3D ROI for each slice
    idx_max = np.shape(image_matrix)[0] - 1  # start in middle image ish
    idx = int(idx_max / 2)
    display_matrix = norm_minmax_featurescaled(np.array(image_matrix), 0, 255).astype("uint8")
    display_roi = norm_minmax_featurescaled(np.array(roi_matrix), 0, 255).astype("uint8")
    # click = MouseClick(sliceloc=idx)
    # ROI_mask = np.zeros(shape=np.shape(image_matrix))
    roivisible = True
    while True:
        print("Slice index = ", idx)
        # click.sliceloc = idx
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 800, 800)
        disp_image = display_matrix[idx]
        disp_roi = display_roi[idx]
        # disp_labels = img_as_ubyte(disp_labels)
        if contrast:
            disp_image = cv2.convertScaleAbs(disp_image, alpha=2, beta=0)
            disp_roi = cv2.convertScaleAbs(disp_roi, alpha=2, beta=0)
        #WATERSHED
        # gradient, markers, labels = watershed_2d(image_matrix[idx], mediandisksize=2, markerdisksize=5, markerthresh=10, gradientdisksize=2)
        # print("Watershed at slice %i made %i regions" % (idx, len(np.unique(labels))))
        #Visualization
        # labels = norm_minmax_featurescaled(labels, 0, 255).astype("uint8")
        # labels_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)    #kind of nice to look at
        # labels_color = cv2.applyColorMap(disp_labels, cv2.COLORMAP_HSV)      #very high contrast colors
        # disp_image = cv2.cvtColor(disp_image, cv2.COLOR_GRAY2BGR)      #to blend gray image with color channels

        # mask_image = cv2.addWeighted(disp_image, alpha1, labels_color, 1-alpha1, 0)   #blend image with labels
        # ROI_image = img_as_ubyte(np.zeros(shape=disp_image.shape))
        # for vx in np.argwhere(ROI_mask != 0):      #VISUALIZE ROI IN CURRENT SLICE
            # mask_image[vx[0], vx[1], vx[2]] = (255, 255, 255)
            # if vx[0] == idx:
            #     px = vx[1:]
            #     mask_image[px[0], px[1]] = 255
        alpha1 = 0.9
        alpha2 = 0.5
        image = cv2.addWeighted(disp_image, alpha1, disp_roi, 1 - alpha2, 0)           #blend image with ROI
        cv2.imshow(title, image) if roivisible else cv2.imshow(title, disp_image)


        # cv2.setMouseCallback(title, click.click_to_point_3D, (labels, "Labels"))
        # cv2.setMouseCallback(title, click.click_to_point_3D, (label_matrix, "Labels"))

        print("\nPress key: ,/. = browse slices")
        key = cv2.waitKey(0)
        # if key == 32:           #space = add current label to ROI   #TODO: CLICK TO ADD TO/ SUBTRACT FROM ROI
        #     for vx in np.argwhere(label_matrix == click.clickval):
        #         ROI_mask[vx[0], vx[1], vx[2]] = 255
        # #     if close:   #Morphological closing
        # #         # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # #         size = (5, 5)
        # #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
        # #         ROI_mask[idx] = cv2.morphologyEx(ROI_mask[idx], cv2.MORPH_CLOSE, kernel)
        # #
        # #         #TODO: do closing
        # #         pass
        #     print(len(np.argwhere(ROI_mask != 0)), "voxels in ROI.")
        # if key == 99:           #c = clear ROI  (ALL SLICES)
        #     ROI_mask = np.zeros(shape=np.shape(image_matrix))
        if key == 44:         #,  = prev slice
            idx -= 1
            idx = 0 if idx < 0 else idx
        elif key == 46:         #. = next slice
            idx += 1
            idx = idx_max if idx > idx_max else idx
        elif key == 115:        #s = toggle watershed labels & ROI
            roivisible = not(roivisible)
            print("ROI / mask on") if roivisible else print("ROI / mask off")
        # elif key == 113:        #q = SAVE ROI as csv
        #     save_path = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\\"
        #     print(ROI_mask.dtype, ROI_mask.shape)
        #     save_ROI(ROI_mask, save_path, title)
        #     # cv2.destroyAllWindows()
        #     # return
        elif key == 120:        #x = exit and return ROI
            cv2.destroyAllWindows()
            return 0


# def plot_all_slices(Plots, n, subplot=(111)):
#     fig, ax = plt.subplots()
#     ax.set_xlabel("Slice")
#     plt.subplots_adjust(bottom=0.25)
#     axcolor = "Black"
#     Slices = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor, valmin=0, valmax=n, valinit=0, valstep=1)
#     slice_slider = Slider(Slices, "Slice")
#     Plot, = plt.plot(Plots[i])
#     slice_slider.on_changed(slider_update)

def plot_all_cropped_npy(experiment, time, exclude="", ROI=False, roimode="roi 2d watershed", idx=3):
    # print(experiment)
    cropped_parent_im = os.path.normpath(os.path.join(os.getcwd(), "..", "Segmentations\cropped_salivary", experiment, "raw", time))
    cropped_parent_roi = os.path.normpath(os.path.join(os.getcwd(), "..", "Segmentations\cropped_salivary", experiment, roimode, time))
    excludebool = True if exclude else False
    num_files = len(find_folders(cropped_parent_im, condition=exclude, exclude_condition=excludebool))
    print(num_files)
    # row_max = 6
    row_max = 8
    if num_files <= row_max:    nr, nc = 1, num_files
    elif num_files <= 2*row_max:    nr, nc = 2, int(num_files / 2) + (num_files % 2 > 0)
    else: nr, nc = 3, int(num_files/3) + (num_files % 3 > 0)
    print(nr, nc, nr*nc)
    # idx = 3
    fig, ax = plt.subplots(nrows=nr, ncols=nc);   ax = ax.ravel()
    for i, file in enumerate(find_folders(cropped_parent_im, condition=exclude, exclude_condition=excludebool)):
        print(i+1, file)
        im = load_matrix(os.path.join(cropped_parent_im, file))[idx]
        roi = load_matrix(os.path.join(cropped_parent_roi, file))[idx] if ROI else 0
        # print(roi.shape)
        # print(im.shape, type(im))
        ax[i].imshow(im, cmap="gray")
        ax[i].imshow(roi, alpha=0.3) if ROI else 0
        ax[i].set_title(file[:-4])
    for i in range(nr*nc): ax[i].axis("off")
    # plt.suptitle(" ".join(("Cropped image at central slice for", experiment, "at time", time))) if not ROI else plt.suptitle(" ".join(("Cropped image at central slice with ROI for", experiment, "at time", time)))
    fig.tight_layout()
    plt.suptitle(f"{num_files} images from {experiment} at time {time} {'''with ROI's''' if ROI else ''} at central slice (cropped idx = {idx}) {''' excluding pilocarpine''' if excludebool else ''}")
    # plt.show()
    return plt


def load_example_image(id="pilot1:-7day:C2_sagittal", make_uint8=True):
    # load example image based on id string, default left SG index
    exp, time, name = id.split(":")
    # idx = right_idx_dict[time+name]
    idx = left_idx_dict[time+name]
    print(exp, time, name, idx)
    path_raw = os.path.join(RawDir, exp, time, name)
    files = dcm_files_indexed(path_raw)
    MTR = dcm_folder_to_pixel_matrix(files, folder_path=path_raw)
    if make_uint8:
        return norm_minmax_featurescaled(MTR[idx], lower=0, upper=255).astype("uint8")
    else:
        return MTR[idx]


def figure_histogram_equalization():
    # INSPIRED BY https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    # MAKING FIGURES FOR THESIS CHAPTER 3.2.1
    img = load_example_image()
    # img = MTR[idx].astype("uint8")
    img_eq = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(img)

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_norm_raw = cdf * float(hist.max()) / cdf.max()

    hist, bins = np.histogram(img_eq.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_norm_eq = cdf * float(hist.max()) / cdf.max()

    hist, bins = np.histogram(img_cl.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_norm_cl = cdf * float(hist.max()) / cdf.max()

    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.tight_layout()

    ax[0, 0].imshow(img, cmap="gray");  ax[0, 0].set_title("Raw image scaled and converted to uint8")
    ax[1, 0].plot(cdf_norm_raw)
    ax[1, 0].hist(img.flatten(), 256, [0, 256])

    ax[0, 1].imshow(img_eq, cmap="gray");   ax[0, 1].set_title("Image after histogram equalization")
    ax[1, 1].plot(cdf_norm_eq)
    ax[1, 1].hist(img_eq.flatten(), 256, [0, 256])

    ax[0, 2].imshow(img_cl, cmap="gray");   ax[0, 2].set_title("Image after CLAHE")
    ax[1, 2].plot(cdf_norm_cl)
    ax[1, 2].hist(img_cl.flatten(), 256, [0, 256])

    for i in range(ax.shape[1]):
        ax[0, i].axis("off")
        ax[1, i].set_xlabel("Intensity value")
        ax[1, i].set_ylabel("Pixel count")

    plt.show()
    return 0


def figure_texture_matrix(image, mask=[], disc=False, mode="glcm", plot=True):
    '''
    Make illustrations of various texture matrices in the pyradiomics package
    for thesis chapter 2.3.3
    :param image: SHOULD BE ALREADY DISCRETIZED, INTEGER VALUES
    :param mode: what texture matrix to show
    :return:
    '''

    mode = mode.lower()
    xlabel = ""
    # title = ""
    if not np.any(mask):
        mask = np.ones(np.shape(image))
    settings = {}

    import SimpleITK as sitk
    img = sitk.GetImageFromArray(image)
    msk = sitk.GetImageFromArray(mask)
    if disc:
        print("Discretization not implemented yet..")
        return 0
    settings['binWidth'] = 1

    if mode == "glcm":
        from radiomics.glcm import RadiomicsGLCM
        settings['force2D'] = True
        print("CALCULATING GLCM")
        func = RadiomicsGLCM(img, msk, **settings,  distances=[1], symmetricalGLCM=False)
        func._initCalculation()
        M = func.P_glcm[0, :, :, 0]
        mmin = np.min(M[M != 0])
        M = M * 1 / mmin    # norm to counts
        xlabel = "Intensity value $j$"
        title = r"GLCM: $\theta=0$, $\delta=1$"
    elif mode == "glszm":
        from radiomics.glszm import RadiomicsGLSZM
        title="GLSZM"
        settings['force2D'] = False
        print("CALCULATING GLSZM")
        func = RadiomicsGLSZM(img, msk, **settings)
        func._initCalculation()
        print(func.P_glszm.shape)
        M = func.P_glszm[0, :, :]
        xlabel="Zone size $j$"
    elif mode == "glrlm":
        from radiomics.glrlm import RadiomicsGLRLM
        settings['force2D'] = True
        print("CALCULATING GLRLM")
        func = RadiomicsGLRLM(img, msk, **settings)
        func._initCalculation()
        M = func.P_glrlm[0, :, :, 0]
        print(M.shape)
        xlabel="Run length $j$"
        title=r"GLRLM: $\theta=0$"
    elif mode == "ngtdm":
        from radiomics.ngtdm import RadiomicsNGTDM
        print("CALCULATING NGTDM")
        settings['force2D'] = False
        func = RadiomicsNGTDM(img, msk, **settings)
        func._initCalculation()
        M = func.P_ngtdm
        print(M.shape)
        M = M[0, :, :]
        print("i-vals?:", M[:, 2])
        print("p_i:", func.coefficients['p_i'])
        print("s_i:", func.coefficients['s_i'])
        print("Nvp:", func.coefficients['Nvp'])
        print("n_i:", func.coefficients['p_i'] * func.coefficients['Nvp'])
        # print(func.coefficients['Ngp'])
        print(M)
        Ni = M.shape[0]
        print("Ni:", Ni)
        M = np.zeros(shape=(Ni, 4))
        M[:, 0] = func.coefficients['ivector']
        M[:, 1] = func.coefficients['p_i'] * func.coefficients['Nvp']
        M[:, 2] = func.coefficients['p_i']
        M[:, 3] = func.coefficients['s_i']
        M = np.delete(M, 0, 1)  # deletes 0'th column in M (ivector)
        title="NGTDM"
        # xlabel = ["$i$", "$n_i$", "p_i", "s_i"]
    elif mode == "gldm":
        from radiomics.gldm import RadiomicsGLDM
        print("CALCULATING GLDM")
        settings['force2D'] = False
        func = RadiomicsGLDM(img, msk, **settings)
        func._initCalculation()
        M = func.P_gldm
        print(M.shape)
        M = M[0, :, :]
        title="GLDM: $\\alpha=0$, $\delta=1$"
        xlabel="Intensity dependency $j$"
    else:
        print("Mode", mode, "not implemented...")
        return 0
    Ng = func.coefficients["Ng"]
    print("Ng", Ng)
    print(np.count_nonzero(M))
    if plot:
        # print("Ng=", Ng)
        # print(M.shape)
        # print(func.imageArray)
        fig, ax = plt.subplots(ncols=2, figsize=(13, 6))
        # fig.tight_layout()
        ax1, ax2 = ax.ravel()
        ax1.imshow(image, cmap="gray")
        # ax2.imshow(M, cmap="gray")
        ax1.axis("off")

        if mode == "gldm":
            xticks = list(range(M.shape[1]))
        elif mode == "ngtdm":
            # xticks = ["$i$", "$n_i$", "$p_i$", "$s_i$"]
            xticks = ["$n_i$", "$p_i$", "$s_i$"]
        else:
            xticks = list(range(1, M.shape[1] + 1))

        yticks = list(range(1, M.shape[0] + 1))# if not mode=="ngtdm" else ax2.axis("off")

        sns.heatmap(image, ax=ax1, annot=True, linewidths=1, cmap="gray", cbar=False, annot_kws={"fontsize":16})
        sns.heatmap(M, ax=ax2, cmap="viridis", annot=True, linewidths=1, cbar=False, xticklabels=xticks, yticklabels=yticks, fmt=".3g", annot_kws={"fontsize":16})
        ax2.tick_params(axis="x", labelsize=16)
        ax2.tick_params(axis="y", labelsize=16)
        ax1.set_title("Image", fontsize=26, fontweight="bold")
        ax2.set_title(title, fontsize=26, fontweight="bold")

        ax2.set_ylabel("Intensity value $i$", fontsize=16)
        ax2.set_xlabel(xlabel, fontsize=16)
        plt.show()
    return M


def figure_filtering_morphology(make_clahe=True):
    from skimage.filters import rank
    from skimage.morphology import disk
    # MAKE FIGURES FOR MASTER THESIS CHAPTER 3.2.2
    img = load_example_image(make_uint8=True)
    if make_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    img_median = rank.median(img, disk(2))
    img_grad = rank.gradient(img, disk(2))
    img_gauss = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0, sigmaY=0)

    fig, ax = plt.subplots(ncols=3, figsize=(7, 3))

    for i in range(len(ax)):
        ax[i].axis("off")
    # ax[0].imshow(img, cmap="gray")
    # ax[0].set_title("uint8 converted image with CLAHE")
    ax[0].imshow(img_gauss, cmap="gray")
    ax[0].set_title("Gaussian image")
    ax[1].imshow(img_median, cmap="gray")
    ax[1].set_title("Median image")
    ax[2].imshow(img_grad, cmap="gray")
    ax[2].set_title("Gradient image")
    fig.suptitle("Filtering methods applied to uint8 converted image with CLAHE")
    # fig.suptitle("Median filtered and gradient image after uint8 conversion and CLAHE\nKernel = disk(2) for both the median and gradient image.")
    fig.tight_layout()
    plt.show()
    pass


def figure_otsu_tresholding():
    img = load_example_image(make_uint8=True)
    GKsz = 9
    img_blur = cv2.GaussianBlur(img, ksize=(GKsz, GKsz), sigmaX=0, sigmaY=0)
    img_he = cv2.equalizeHist(img_blur)
    # img_he = cv2.equalizeHist(img)
    # img_blur = cv2.GaussianBlur(img_he, ksize=(9, 9), sigmaX=0, sigmaY=0)

    mask_otsu = sitk.OtsuThreshold(sitk.GetImageFromArray(img_he), 0, 1, 210)
    # mask_otsu = sitk.OtsuThreshold(sitk.GetImageFromArray(img_blur), 0, 1, 128)
    mask_otsu = sitk.GetArrayFromImage(mask_otsu)
    mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)))

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # img_clahe = clahe.apply(img)
    # mask_otsu2 = sitk.OtsuThreshold(sitk.GetImageFromArray(img_clahe), 0, 1, 128)
    # img_he2 = cv2.equalizeHist(img)
    # mask_otsu2 = sitk.OtsuThreshold(sitk.GetImageFromArray(img_he2), 0, 1, 128)
    # mask_otsu2 = sitk.GetArrayFromImage(mask_otsu2)

    fig, ax = plt.subplots(ncols=2, figsize=(9, 3))
    # fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(9, 3))
    ax = ax.ravel()
    for axx in ax:
        axx.axis("off")
    # ax[0].imshow(img_blur, cmap="gray")
    # ax[0].set_title("Gaussian image \nwith $9 \\times 9$ kernel")
    ax[0].imshow(img_he, cmap="gray")
    ax[0].set_title("Gaussian blurred image ($9\\times 9$ kernel)\n with histogram equalization")
    ax[1].imshow(mask_otsu, cmap="gray")
    ax[1].set_title("Otsu tresholding")
    fig.tight_layout()
    fig.suptitle("Otsu tresholding on blurred + contrast enhanced image (left) for background identification (right)")
    # ax[2].imshow(img_he2, cmap="gray")
    # ax[3].imshow(mask_otsu2, cmap="gray")
    plt.show()


def plot_fbw_num_bins():
    folder = os.path.join(os.getcwd(), "..", "..", "Radiomic features\discretization")
    for file in os.listdir(folder)[3:]:
        # print(file.split("_"))
        _, weight, norm, discmethod = file.split("_")
        discmethod = discmethod[:-4].upper()
        normname = {"no norm":"no normalization", "nyul otsu decile":"Nyul normalization", "stscore":"standardization"}
        print(weight, norm, discmethod)
        df = pd.read_csv(os.path.join(folder, file), index_col=0)
        df_t1 = pd.read_csv(os.path.join(folder, "_".join(["discretization", "T1", norm, discmethod + ".csv"])), index_col=0)
        # print(df)
        bw = df.index.values
        bw_t1 = df_t1.index.values
        # print(all(bw == df_t1.index.values))
        # plt.plot(bw, df["roi mean"], "-x", label="roi mean")
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
        ax[0].plot(bw, df["roi median"], "--", label="roi median")
        ax[0].errorbar(bw, df["roi mean"], yerr=df["roi std"], ls="--", label="roi mean", capsize=3)
        ax[0].plot(bw, df["roi min"], ":", c="r", label="min / max")
        ax[0].plot(bw, df["roi max"], ":", c="r")
        ax[0].set_title(f"{weight} images")
        # ax[0].set_title(f"Number of bins in ROI with varying bin width\n for {weight}-weighted images after {normname[norm]} and {discmethod} discretization.")
        ax[0].set_xlabel("Bin width")
        ax[0].set_ylabel("Number of bins")
        ax[0].hlines([30, 130], xmin=min(bw), xmax=max(bw), colors="black")
        ax[0].legend()
        ax[0].grid()

        ax[1].plot(bw_t1, df_t1["roi median"], "--", label="roi median")
        ax[1].errorbar(bw_t1, df_t1["roi mean"], yerr=df_t1["roi std"], ls="--", label="roi mean", capsize=3)
        ax[1].plot(bw_t1, df_t1["roi min"], ":", c="r", label="min / max")
        ax[1].plot(bw_t1, df_t1["roi max"], ":", c="r")
        ax[1].set_title("T1 images")
        # ax[1].title(f"Number of bins in whole image with varying bin width\n for {weight}-weighted images after {normname[norm]} and {discmethod} discretization.")
        ax[1].set_xlabel("Bin width")
        ax[1].set_ylabel("Number of bins")
        ax[1].hlines([30, 130], xmin=min(bw_t1), xmax=max(bw_t1), colors="black")
        ax[1].legend()
        ax[1].grid()
        fig.suptitle(f"Number of bins in ROI with varying bin width\n after {normname[norm]} and {discmethod} discretization.")
        fig.tight_layout()
        # fig, ax = plt.subplots()
        # ax[1].plot(bw, df["im median"], "--", label="im median")
        # ax[1].errorbar(bw, df["im mean"], yerr=df["im std"], ls="--", label="im mean", capsize=3)
        # ax[1].plot(bw, df["im min"], ":", c="r", label="min / max")
        # ax[1].plot(bw, df["im max"], ":", c="r")
        # # ax[1].title(f"Number of bins in whole image with varying bin width\n for {weight}-weighted images after {normname[norm]} and {discmethod} discretization.")
        # ax[1].set_xlabel("Bin width")
        # ax[1].set_ylabel("Number of bins")
        # ax[1].hlines([30, 130], xmin=min(bw), xmax=max(bw), colors="black")
        # ax[1].legend()
        # ax[1].grid()
        plt.show()
    return 1


def figure_bias_field():
    from MRI_denoising import n4correction
    ID = "pilot1:-7day:C2_sagittal"
    # ID = "pilot1:8day:C2_sagittal"
    img = load_example_image(make_uint8=False, id=ID)
    img_corr, field, mask = n4correction(img, mask=True, close=True, verbose=True)
    # diff = img - img_corr
    diff = (img - img_corr) / np.max(img) * 100

    fig, ax = plt.subplots(ncols=3, figsize=(18, 6))
    for axx in ax:
        axx.axis("off")
    ax[0].imshow(percentile_truncation(img, 0, 98), cmap="gray")
    ax[0].imshow(mask, alpha=0.3)
    ax[1].imshow(img, cmap="gray")
    ax[1].imshow(field, cmap="bwr", alpha=0.5)
    pl = ax[2].imshow(diff, cmap="bwr")
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=-.30)
    fig.colorbar(pl, cax=cax, orientation="vertical")

    fz = 20
    ax[0].set_title("Masked foreground (otsu)", fontsize=fz)
    ax[1].set_title("Estimated bias field", fontsize=fz)
    ax[2].set_title("Raw image - corrected (% rel. to max)", fontsize=fz)
    fig.tight_layout()
    plt.show()
    return 1


def show_feature_classes_histogram(fts=[], ft_keys="idx", separate_wavelet_HL=True):
    # calculate histogram of all feature classes, image filters, etc
    # illustration for chapter 3.4 Feature extraction
    FEATURE_CLASSES = ["shape", "firstorder", "glcm", "glrlm", "gldm", "ngtdm", "glszm"]
    if separate_wavelet_HL:
        IMAGE_FILTERS = ["original", "logarithm", "squareroot", "wavelet-H", "wavelet-L", "exponential", "gradient", "lbp", "square"]
    else:
        IMAGE_FILTERS = ["original", "logarithm", "squareroot", "wavelet", "exponential", "gradient", "lbp", "square"]
    loadfts = not(any(fts))
    if loadfts:
        savepath = os.path.join(ftsDir, "LR_split_no_norm_extracted.csv")
        df = pd.read_csv(savepath, index_col=0)
        df = df.drop(["name", "dose", "time"], axis=1)
        df = df.drop(df.filter(like="diagnostics", axis=1).columns, axis=1)
        # fts = df.columns.values

        df = df.drop(df.filter(like="wavelet-LL", axis=1).columns, axis=1)
        df = df.drop(df.filter(like="wavelet-LH", axis=1).columns, axis=1)
        df = df.drop(df.filter(like="wavelet-HL", axis=1).columns, axis=1)
        df = df.drop(df.filter(like="wavelet-HH", axis=1).columns, axis=1)
        Ntot = len(df.T)
        # print(df)
    else:
        if ft_keys == "idx":
            fts_idx = fts.copy()
            from select_utils import get_feature_index_global
            index = get_feature_index_global(invert=True)
            fts = [index[idx] for idx in fts_idx]
        else:
            pass
        Ntot = len(fts)
    print("HAVING", Ntot, "FTS")
    num_in_classes = []
    for ftclass in FEATURE_CLASSES:
        if loadfts:
            df_curr = df.filter(like=ftclass, axis=1)
            Nclass = len(df_curr.T)
            print(ftclass, ":", Nclass)
        else:
            fts_red = list(filter(lambda ft: ftclass in ft, fts))
            Nclass = len(fts_red)
        num_in_classes.append(Nclass)
    num_in_filters = []
    if not loadfts:
        fts = list(filter(lambda ft: "shape" not in ft, fts))   # remove shape-features before counting filters
    for ft_filter in IMAGE_FILTERS:
        if loadfts:
            df_curr = df.filter(like=filter, axis=1)
            if ft_filter == "square":
                df_curr = df_curr.drop(df_curr.filter(like="squareroot", axis=1), axis=1)
                # print(df_curr)
            Nfilt = len(df_curr.T)
        else:
            fts_red = list(filter(lambda ft: ft_filter in ft, fts))
            Nfilt = len(fts_red)
        print("Filter", ft_filter, ":", Nfilt)
        num_in_filters.append(Nfilt)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # FEATURE_CLASSES = ["shape", "\nfirstorder", "glcm", "\nglrlm", "gldm", "\nngtdm", "glszm"]
    # IMAGE_FILTERS2 = ["original", "\nlogarithm", "\n\nsquareroot", "\nwavelet-H", "wavelet-L", "\nexponential", "\n\ngradient", "\nlbp", "square"]
    FEATURE_CLASSES2 = [f"\n{type}" if i%2 else type for i, type in enumerate(FEATURE_CLASSES)]
    IMAGE_FILTERS2 = [f"\n{filter}" if i%2 else filter for i, filter in enumerate(IMAGE_FILTERS)]

    fig, ax = plt.subplots(ncols=2)
    ax1, ax2 = ax
    ax1.bar(FEATURE_CLASSES2, num_in_classes)
    ax1.set_yticks(num_in_classes, num_in_classes)
    ax1.set_ylabel("# of features")
    ax1.grid()
    fig.tight_layout()

    ax2.bar(IMAGE_FILTERS2, num_in_filters)
    ax2.set_yticks(num_in_filters, num_in_filters)
    ax2.set_ylabel("# of features")
    ax2.grid()
    fig.tight_layout()
    plt.tight_layout()
    plt.show()
    return 0


def show_all_segmented_at_instance(id="8-3", time="105day", experiment="Pilot3", X="L", show_roi=True):

    path_time = os.path.join(RawDir, experiment, time)
    files_load = find_folders(path_time, condition=id)
    files = []
    for f in files_load:
        files.append(f) if "sagittal" in f else 0
    # print(files)

    # fig, ax = plt.subplots(nrows=2, ncols=2)
    num_img = len(files) if not X == "all" else 2 * len(files)
    if num_img < 4:
        fig, ax = plt.subplots(ncols=num_img)
    else:
        fig, ax = plt.subplots(ncols=num_img // 2, nrows=2)
        # ax = ax.ravel()

    if num_img > 1:
        ax = ax.ravel()
        [axx.axis("off") for axx in ax]
    else:
        ax.axis("off")
        ax = [ax]

    num_pixels_dict = {"8-3_T1_sagittal":1528, "8-3_T2_sagittal":1688, "8-3_after_p_T1_sagittal":1423, "8-3_after_p_T2_sagittal":1698}  # LEFT INDEX

    i = 0
    # for i, f in enumerate(files):
    for f in files:
        name_orig = get_name(experiment, f)
        print(name_orig)

        LR = ["L", "R"] if X == "all" else [X]
        for X_ in LR:
            name = name_orig + "_" + X_
            idx = right_idx_dict[time + name_orig] if X_ == "R" else left_idx_dict[time + name_orig]

            path_nrrd_image = os.path.join(nrrdDir, "LR split nyul", experiment, time, name + "_image.nrrd")
            path_nrrd_mask = os.path.join(nrrdDir, "LR split nyul", experiment, time, name + "_mask.nrrd")
            img = sitk.GetArrayFromImage(sitk.ReadImage(path_nrrd_image))
            roi = sitk.GetArrayFromImage(sitk.ReadImage(path_nrrd_mask))

            img = np.rot90(img, k=-1)
            roi = np.rot90(roi, k=-1)
            img = np.flip(img, 1)
            roi = np.flip(roi, 1)

            # img = crop_to_mask(img, roi)
            # roi = crop_to_mask(roi, roi)

            # ax[i].imshow(percentile_truncation(img, 2, 99.9), cmap="gray")
            ax[i].imshow(percentile_truncation(img, 2, 98.5), cmap="gray")
            masked = np.ma.masked_where(np.logical_not(roi), img)
            ax[i].imshow(masked, alpha=0.8, cmap="hot") if show_roi else 0
            # ax[i].set_title(f"{name}")
            try:
                ax[i].set_title(f"Having {num_pixels_dict[name_orig]} pixels in ROI", y=-0.1, fontsize=14)
                text = ax[i].text(0, -22.5, name, size=14,
                           horizontalalignment='left', verticalalignment="top")
            except Exception:
                ax[i].set_title(f"{experiment} {time} ROI @ {X_} saliv\n{name_orig} idx={idx}", fontsize=14)
                # ax[i].set_title(f"{experiment} {time} ROI @ {X_} saliv\n{name_orig}", fontsize=14)
                # ax[i].set_title(f"{name_orig} day {time[:-3]}", fontsize=14)
            i += 1
    plt.show()

    return 0


def plot_ids_over_time(df=[], title=""):
    # sort saliva values by individuals over time
    # plot control mice which not xerostomic at any point, over time
    if not df:
        from data_loader import load_saliva
        from endpoints import binary_thresh_xerostomia
        df = load_saliva(melt=True)
        df["xer"] = binary_thresh_xerostomia(df)
    # Assume df on long form with id, time as separate columns with one values each
    have_control = False
    have_xer = False
    if "ctr" in df.columns:
        have_control = True
        # df_ctrl = df["ctr"]
        ctrl_string = "ctr"
    elif "ctrl" in df.columns:
        have_control = True
        # df_ctrl = df["ctrl"]
        ctrl_string = "ctrl"
    else:   pass
    if "xer" in df.columns:
        have_xer = True
        df_xer = df["xer"]
    else:   pass
    # print(have_control, have_xer)
    id_list = np.unique(list(df["name"]))
    df_times = dict()       # {2-3: ([time1, time2], [xer1, xer2], ctrl_bool)}
    all_times = set()
    count = 0
    for nm in id_list:
        df_nm = df[df["name"] == nm]
        times = list(df_nm["time"])
        # times = np.sort(times)
        count += len(times)
        # print(nm, times)
        if not(have_control) and not(have_xer):
            df_times[nm] = times
        elif have_control and have_xer:
            xervals = list(df_nm["xer"])
            ctrl = all(list(df_nm[ctrl_string]))
            df_times[nm] = (times, xervals, ctrl)
        else:
            #todo: add some beautiful day in the future maybe
            print("not implemented")
            return 0

        all_times = all_times.union(times)
    all_times = np.sort(list(all_times))
    print(df_times)

    fig, ax = plt.subplots()
    from random import choice
    for it in df_times.items():
        if not(have_control) and not(have_xer):
            nm, times = it
        elif have_control and have_xer:
            nm, vec = it
            times, xervals, ctrl = vec
        else:
            print("not implemented")
            return 0

        if have_control and have_xer:
            for t, xer in zip(times, xervals):
                ax.plot(t, nm, "x" if ctrl else "o", c="r" if xer else "b")

        elif have_control:
            pass
        elif have_xer:
            pass
        else:
            ax.plot(times, [nm]*len(times), "x-")
    if have_control and have_xer:
        title += "\nx: control, o: irr, red: xer, blue: not xer"
    ax.set_xticks(all_times, all_times)
    fig.suptitle(title)
    ax.grid(1)
    # plt.show()
    plt.close()

    PLOT_ID_EARLY_LATE = ["11-10", "11-2", "11-4", "C2", "C3", "C4", "C5"]
    for id in PLOT_ID_EARLY_LATE:
        print(id, df_times[id])
        df_id = df[df["name"] == id]
        exp = "Pilot4" if id.split("-")[0] == "11" else "Pilot1"
        times = df_id["time"].values
        saliv_vals = df_id["val"].values
        print(df_id)
        print(exp, id, times, saliv_vals)
        T1images = []
        T1vals = []
        T2images = []
        T2vals = []
        for t, v in zip(times, saliv_vals):
            try:
                files = os.listdir(os.path.join(RawDir, exp, str(t) + "day"))
                files = list(filter(lambda f: f.split("_")[0] == id, files))
                files = list(filter(lambda f: "p" not in f, files))
                print(files)
                for f in files:
                    t1bool = "T1" in f
                    nm = get_name(exp, f)
                    print(f"{exp}:{t}day:{nm}")
                    img = load_example_image(id=f"{exp}:{t}day:{nm}")
                    img = percentile_truncation(img, lower=0.0, upper=99)
                    descr = f"day = {t}, saliva = {v} $\mu$L"
                    if t1bool:
                        T1images.append(img)
                        T1vals.append(descr)
                    else:
                        T2images.append(img)
                        T2vals.append(descr)
            except Exception as e:
                print("ERR:", *e.args)
        ncols = len(T2images)
        nrows = 2 if len(T1images) > 0 else 1
        print(id, ncols, "\n\n")
        if ncols > 0:
            fig, axes = plt.subplots(nrows=1,
                                     ncols=ncols)
            i = 0
            for img, val in zip(T1images, T1vals):
                axes[i].imshow(img, cmap="gray")
                axes[i].set_title(val)
                i += 1
            fig.suptitle(id)
            fig.tight_layout()

            fig, axes = plt.subplots(nrows=1,
                                     ncols=ncols)
            [ax.axis("off") for ax in axes]
            i = 0
            for img, val in zip(T2images, T2vals):
                axes[i].imshow(img, cmap="gray")
                axes[i].set_title(val)
                i += 1
            [ax.axis("off") for ax in axes]
            fig.suptitle(id)
            fig.tight_layout()
            plt.show()
        # break
    return 0


def illustration_inter_intra_saliva_variability():
    # x = np.linspace(-10, 20, 301)  # days
    x = np.linspace(-10, 20, 301)  # days
    print(x[x<0])
    x_baseline = x[x < 0]
    x_after = x[x >= 0]
    # print(x)
    measurement_err = 10
    amp_intra = 3
    intravar = amp_intra*np.sin(x * 2 * np.pi)  # intra-variation w period 1 day


    # Control
    intercept_a_baseline = 30
    intercept_b_baseline = 50
    slope_a_baseline = 1.2
    slope_b_baseline = 1.5

    y_a_control = intercept_a_baseline + slope_a_baseline * x
    y_b_control = intercept_b_baseline + slope_b_baseline * x

    y_a_baseline_intra = y_a_control + intravar
    y_b_baseline_intra = y_b_control + intravar

    # Irr
    measurement_err = 15
    intercept_a_baseline = 30
    intercept_b_baseline = 50
    slope_a_baseline = 1.2
    slope_b_baseline = 1.4
    slope_a_after = 0.5
    slope_b_after = 1.1

    y_a_baseline = intercept_a_baseline + slope_a_baseline * x_baseline
    y_b_baseline = intercept_b_baseline + slope_b_baseline * x_baseline
    y_a_after = y_a_baseline[-1] + slope_a_after * x_after
    y_b_after = y_b_baseline[-1] + slope_b_after * x_after

    y_a_irr = np.r_[y_a_baseline, y_a_after]
    y_b_irr = np.r_[y_b_baseline, y_b_after]
    y_a_irr_intra = y_a_irr + intravar
    y_b_irr_intra = y_b_irr + intravar


    fig, (ax, ax2) = plt.subplots(ncols=2, sharey=True)
    ax.plot(x, y_a_control, "-", c="r", label="Mouse 1 control")
    ax.plot(x, y_a_baseline_intra, "--", c="r")
    ax.plot(x, y_b_control, c="b", label="Mouse 2 control")
    ax.plot(x, y_b_baseline_intra, "--", c="b")
    ax.plot(x, y_b_control + measurement_err, "--", c="gray")
    ax.plot(x, y_a_control - measurement_err, "--", c="gray")
    # ax.set_title("Control")
    ax.legend()
    ax.set_xlabel("Day")
    ax2.set_xlabel("Day")
    ax.set_ylabel("Saliva")

    ax2.plot(x, y_a_irr, "-", c="r", label="Mouse 3 irradiated")
    ax2.plot(x, y_b_irr, "-", c="b", label="Mouse 4 irradiated")
    ax2.plot(x, y_a_irr_intra, "--", c="r")
    ax2.plot(x, y_b_irr_intra, "--", c="b")
    ax2.plot(x, y_a_irr - measurement_err, "--", c="gray")
    ax2.plot(x, y_b_irr + measurement_err, "--", c="gray")
    ax2.vlines([0], ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[-1], ls=":", color="black")
    ax2.legend()
    plt.show()
    pass


def show_texture_matrix(img_filter="logarithm", matrix="gldm", id="C2", experiment="Pilot1", weight="T2", LR="R", norm="no norm"):
    # make texture matrix of type, given images for individual
    # also show interesting feature value calc from matrix?
    import nrrd
    import radiomics
    if img_filter == "original":
        filterfunc = radiomics.imageoperations.getOriginalImage
    elif img_filter == "LoG":
        filterfunc = radiomics.imageoperations.getLoGImage
    elif img_filter == "wavelet":
        filterfunc = radiomics.imageoperations.getWaveletImage
    elif img_filter == "logarithm":
        filterfunc = radiomics.imageoperations.getLogarithmImage
    elif img_filter == "gradient":
        filterfunc = radiomics.imageoperations.getGradientImage
    elif img_filter == "square":
        filterfunc = radiomics.imageoperations.getSquareImage
    else:
        print("Filter", img_filter, "not implemented")
        return 0

    t1bool = True if weight.upper() == "T1" else False
    instances = []
    path_exp = os.path.join(RawDir, experiment)
    times = [int(t[:-3]) for t in os.listdir(path_exp)]
    times.sort()
    times = [f"{t}day" for t in times]

    for t in times:
        files = os.listdir(os.path.join(path_exp, t))
        files = list(filter(lambda f: "p" not in f, files))
        if t1bool:
            files = list(filter(lambda f: "T1" in f, files))
        else:
            files = list(filter(lambda f: "T1" not in f, files))
        fname = list(filter(lambda f: id in f, files))
        instances.append(f"{t}:{fname[0]}")
    print(instances)


    for inst in instances:
        time, name = inst.split(":")
        img_folder = os.path.join(nrrdDir, f"LR split {norm}", experiment, time)
        path_img = os.path.join(img_folder, f"{name}_{LR}_image.nrrd")
        path_mask = os.path.join(img_folder, f"{name}_{LR}_mask.nrrd")
        img, _ = nrrd.read(path_img)
        roi, _ = nrrd.read(path_mask)
        roi = resegment(img, roi)
        imgplot = percentile_truncation(img)
        # img_folder = os.path.join(path_exp, time, name)
        # files = dcm_files_indexed(img_folder, printbool=False)
        # MTR = dcm_folder_to_pixel_matrix(files, img_folder, printbool=False)
        # idx = left_idx_dict[time + name] if LR == "L" else right_idx_dict[time + name]
        # imgraw = MTR[idx]

        print(time, name, end=" ")
        print(img.shape, f"IDX_{LR}={idx}")
        imgfiltered, _, _ = next(filterfunc(sitk.GetImageFromArray(img), sitk.GetImageFromArray(roi)))
        imgfiltered = sitk.GetArrayFromImage(imgfiltered)
        print("filtered:", imgfiltered.shape)

        bw = FBW_dict_T2[norm] if weight == "T2" else FBW_dict_T1[norm]
        imgdisc = discretize_FBW_ISBI(img, roi=roi, bw=bw)
        imgfiltered_disc = discretize_FBW_ISBI(imgfiltered, roi=roi, bw=bw)

        matr = figure_texture_matrix(imgdisc, mask=roi, disc=False, mode=matrix, plot=False)
        matr_filtered = figure_texture_matrix(imgfiltered_disc, mask=roi, disc=False, mode=matrix, plot=False)
        # print(matr)
        print(matr.shape, matr_filtered.shape)

        fig, ax = plt.subplots(ncols=2, nrows=2)
        # imgplot = imgdisc
        imgplot = percentile_truncation(imgdisc)
        ax[0, 0].imshow(imgplot, cmap="gray")
        ax[0, 0].imshow(imgplot, cmap="gray")
        ax[0, 0].imshow(np.ma.masked_where(np.logical_not(roi), imgplot), cmap="hot")

        # imgfiltered = imgfiltered_disc
        imgfiltered = percentile_truncation(imgfiltered_disc)
        ax[0, 1].imshow(imgfiltered, cmap="gray")
        ax[0, 1].imshow(np.ma.masked_where(np.logical_not(roi), imgfiltered), cmap="hot")

        ax[0, 0].set_title(f"After preprocessing")
        ax[0, 1].set_title(f"{img_filter}")
        for axx in [ax[0, 0], ax[0, 1]]:
            axx.axis("off")

        transposefilter = False
        if transposefilter:
            matr = matr.T
            matr_filtered = matr_filtered.T
        ax[1, 0].imshow(matr)
        ax[1, 1].imshow(matr_filtered)

        xlabs = {"glcm":"Intensity $j$", "glrlm":"Run length $j$", "glszm":"Zone size $j$", "gldm":"Dependency $j$", "ngtdm":"$n_i$, $p_i$, $s_i$"}
        ylabs = {"glcm":"Intensity $i$", "glrlm":"Intensity $i$", "glszm":"Intensity $i$", "gldm":"Intensity $i$", "ngtdm":"Intensity $i$"}

        for axx in [ax[1, 0], ax[1, 1]]:
            if transposefilter:
                pass
                # axx.set_xlabel("Intensity $i$") # for transposed matrix
                # axx.set_ylabel("Dependency $j$")
                # axx.set_xlabel(ylabs[])
            else:
                axx.set_xlabel(xlabs[matrix])
                axx.set_ylabel(ylabs[matrix])
                # axx.set_ylabel("Intensity $i$")
                # axx.set_xlabel("Dependency $j$")

        fig.tight_layout()
        fig.suptitle(f"{id} {time} {norm}\n{matrix}")

        # fig, (ax1, ax2) = plt.subplots(ncols=2)
        # from scipy.fft import ifft2
        # ax1.imshow(np.abs(ifft2(matr)))
        # ax2.imshow(np.abs(ifft2(matr_filtered)))

    plt.show()
    pass


def plot_feature_over_time(ft="logarithm_gldm_SmallDependenceLowGrayLevelEmphasis_R", MODE="NO P", WEIGHT="T2", LRMODE="aggregated", showplot=True):
    from data_loader import load_saliva, load_fsps_data
    from name_dose_relation import is_control
    from endpoints import register_name_to_outcome, binary_thresh_xerostomia
    df = load_fsps_data(WEIGHT=WEIGHT, MODE=MODE, LRMODE=LRMODE, TRAIN_SET="all")
    print(df.shape)

    df_y = load_saliva(melt=True)
    df_y = register_name_to_outcome(df, df_y, melt=False, make_70_exception=True)
    df_y_xer = binary_thresh_xerostomia(df_y)

    df.loc[:, "name"] = [nm.split("_")[0] for nm in df["name"].values]
    df.loc[:, "time"] = [int(t.split("day")[0]) for t in df["time"].values]
    df = df[["name", "time", "dose", ft]]
    df.loc[:, "ctr"] = [is_control(nm) for nm in df["name"].values]
    print(np.unique(df["time"].values))
    df["timegroup"] = pd.cut(df["time"], [-10, 0, 12, 105], right=True, labels=["baseline",  "day 3 - 12", "day 26 - 105"])
    df["xer"] = df_y_xer
    df["saliv"] = df_y["val"]
    print(df)
    # print(df.dropna())
    # print(df_y)

    # FT CORRELATIONS TO SALIVA MEASUREMENTS
    from scipy.stats import spearmanr, pearsonr
    df_nona = df.dropna()
    rho, p = pearsonr(df_nona[ft], df_nona["saliv"])
    print(f"Pearson corr ft to saliva:")
    print(f"\tAll (N={len(df_nona)}): \t\trho={rho:.3f}, p={p:.3f}")
    df_nona_ctr = df_nona[df_nona["ctr"] == True]
    df_nona_irr = df_nona[df_nona["ctr"] == False]
    rho, p = pearsonr(df_nona_ctr[ft], df_nona_ctr["saliv"])
    print(f"\tControl (N={len(df_nona_ctr)}): \trho={rho:.3f}, p={p:.3f}")
    rho, p = pearsonr(df_nona_irr[ft], df_nona_irr["saliv"])
    print(f"\tIrradiated (N={len(df_nona_irr)}): \trho={rho:.3f}, p={p:.3f}")



    df_ctr = df[df["ctr"] == True]
    df_irr = df[df["ctr"] == False]
    print(f"Num control = {len(df_ctr)}, num irr = {len(df_irr)}")
    num_xer = len(df[df["xer"] == True])
    num_notxer = len(df[df["xer"] == False])

    times_ctr = df_ctr["time"].values
    times_irr = df_irr["time"].values
    vals_ctr = df_ctr[ft].values
    vals_irr = df_irr[ft].values
    # print(times_ctr)
    cirr, cctr = sns.color_palette()[:2]
    # plt.plot(times_ctr, vals_ctr, c="tab:orange", ls="", marker="o")
    # plt.plot(times_irr, vals_irr, c=cirr, ls="", marker="o")
    from select_utils import get_feature_index_global
    ft_index = get_feature_index_global()
    if LRMODE == "aggregated":
        ftt = "_".join(ft.split("_")[:-1])
        lr = ft.split("_")[-1]
        ft_idx = ft_index[ftt]
        ft_idx = f"{ft_idx}{lr.upper()}"
    else:
        ft_idx = ft_index[ft]
    ft_idx = f"Ft{ft_idx}"

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="timegroup", y=ft, hue="ctr", ax=ax, hue_order=[False, True])#, labels=["Control", "Irradiated"])
    # sns.swarmplot(data=df, x="timegroup", y=ft, hue="ctr", ax=ax, hue_order=[False, True])#, labels=["Control", "Irradiated"])
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[f"Irradiated (N={len(vals_irr)})", f"Control (N={len(vals_ctr)})"])
    ax.set_xlabel("")
    # ax.set_ylabel("")
    # ax.set_title(ft)
    fig.suptitle(f"FSPS {MODE} {WEIGHT} {LRMODE} {ft_idx}")

    fig, ax = plt.subplots()
    sns.swarmplot(data=df, x="timegroup", y=ft, hue="xer", ax=ax, hue_order=[False, True], palette="Set2")#, labels=["Control", "Irradiated"])
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[f"Not xerostomic (N={num_notxer})", f"Xerostomic (N={num_xer})"])
    ax.set_xlabel("")
    fig.suptitle(f"FSPS {MODE} {WEIGHT} {LRMODE} {ft_idx}")

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="ctr", y=ft, hue="xer", palette="Set2")
    # sns.swarmplot(data=df, x="ctr", y=ft, hue="xer", palette="Set2")
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[f"Not xerostomic (N={num_notxer})", f"Xerostomic (N={num_xer})"])
    ax.set_xlabel("")
    ax.set_xticks(ax.get_xticks(), ["Irradiated", "Control"])
    fig.suptitle(f"FSPS {MODE} {WEIGHT} LR-{LRMODE} {ft_idx}")

    fig, ax = plt.subplots()
    # sns.boxplot(data=df, x="xer", y=ft, hue="ctr")
    sns.boxplot(data=df, x="ctr", y=ft)
    # sns.swarmplot(data=df, x="ctr", y=ft, hue="xer", palette="Set2")
    # handles, labs = ax.get_legend_handles_labels()
    # ax.legend(handles=handles, labels=["Irradiated", "Control"])
    ax.set_xlabel("")
    # ax.set_xticks(ax.get_xticks(), ["Irradiated", "Control"])
    # ax.set_xticks(ax.get_xticks(), [f"Not xerostomic (N={num_notxer})", f"Xerostomic (N={num_xer})"])
    fig.suptitle(f"FSPS {MODE} {WEIGHT} LR-{LRMODE} {ft_idx}")
    plt.show() if showplot else 0
    pass

def plot_delta_feature(ft, LRMODE):
    from data_loader import load_delta
    from name_dose_relation import is_control
    df, y = load_delta(WEIGHT="T2", LRMODE=LRMODE, training="all", xer=True, keep_time=True, keep_names=True)


    print(df)
    df = df[["name", "time", "dose", ft]]
    df.loc[:, "name"] = [nm.split("_")[0] for nm in df["name"].values]
    df.loc[:, "ctr"] = [is_control(nm) for nm in df["name"].values]
    df.loc[:, "xer"] = y
    print(df)
    num_control = len(df[df["ctr"] == True])
    num_irr = len(df[df["ctr"] == False])
    num_xer = len(df[df["xer"] == True])
    num_notxer = len(df[df["xer"] == False])
    num_xer_ctr = len(df[(df["xer"] == True) & (df["ctr"] == True)])
    num_xer_irr = len(df[(df["xer"] == True) & (df["ctr"] == False)])
    print(f"Num xer = {num_xer}, not xer = {num_notxer}")
    print(f"Of {num_control} control: {num_xer_ctr} xer")
    print(f"Of {num_irr} irr: {num_xer_irr} xer")


    # sns.boxplot(data=df, y=ft, x="time", hue="ctr")
    # sns.swarmplot(data=df, y=ft, x="ctr")
    # fig, ax = plt.subplots()
    # sns.swarmplot(data=df, y=ft, x="time", hue="ctr")
    # fig, ax = plt.subplots()
    # sns.swarmplot(data=df, y=ft, x="time", hue="xer")
    # fig, ax = plt.subplots()
    # sns.boxplot(data=df, y=ft, x="ctr")

    fig, ax = plt.subplots()
    sns.boxplot(data=df, y=ft, x="ctr", hue="xer", palette="Set2")
    # sns.swarmplot(data=df, y=ft, x="ctr", hue="xer", palette="Set2")
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[f"Not xerostomic (N={num_notxer})", f"Xerostomic (N={num_xer})"])
    ax.set_xlabel("")
    ax.set_xticks(ax.get_xticks(), ["Irradiated", "Control"])
    ax.set_ylabel(f"$\Delta${ft}")
    fig.suptitle(f"Delta-feature LR-{LRMODE}")
    plt.show()
    pass

if __name__=="__main__":

    # plot_feature_over_time(ft="wavelet-H_gldm_LargeDependenceEmphasis", MODE="NO P", WEIGHT="T1", LRMODE="average")

    # plot_feature_over_time(ft="exponential_glszm_GrayLevelNonUniformity_L", MODE="NO P", WEIGHT="T2", LRMODE="aggregated")

    # plot_feature_over_time(ft="wavelet-L_firstorder_RobustMeanAbsoluteDeviation_R", MODE="NO P", WEIGHT="T2", LRMODE="aggregated")
    # plot_feature_over_time(ft="original_firstorder_Energy", MODE="DELTA P", WEIGHT="T2", LRMODE="average")
    # plot_feature_over_time(ft="lbp-2D_firstorder_RobustMeanAbsoluteDeviation_R", MODE="DELTA P", WEIGHT="T2", LRMODE="aggregated")

    # plot_feature_over_time(ft="logarithm_glcm_JointAverage_R", MODE="DELTA P", WEIGHT="T2", LRMODE="aggregated")
    # plot_delta_feature(ft="gradient_glcm_Correlation_R", LRMODE="aggregated")
    # plot_feature_over_time(ft="gradient_glcm_Correlation_R", MODE="NO P", WEIGHT="T2", LRMODE="aggregated")
    # plot_feature_over_time(ft="gradient_glcm_Imc1", MODE="NO P", WEIGHT="T2", LRMODE="average")

    # plot_feature_over_time(ft="square_glrlm_RunVariance_R", MODE="NO P", WEIGHT="T1", LRMODE="aggregated")
    # plot_feature_over_time(ft="original_glrlm_LongRunHighGrayLevelEmphasis", MODE="NO P", WEIGHT="T2", LRMODE="average")
    # show_texture_matrix(id="C2", LR="R", norm="no norm", img_filter="square", matrix="glrlm")
    # show_texture_matrix(id="H3", LR="R", norm="no norm", img_filter="square", matrix="glrlm")


    # plot_feature_over_time(ft="gradient_glszm_LargeAreaEmphasis_R", MODE="DELTA P", WEIGHT="T2", LRMODE="aggregated")
    # plt.show()

    # plot_feature_over_time(ft="logarithm_gldm_SmallDependenceLowGrayLevelEmphasis_R", WEIGHT="T2", LRMODE="aggregated")
    # plot_feature_over_time(ft="logarithm_gldm_SmallDependenceLowGrayLevelEmphasis_R", WEIGHT="T2", LRMODE="aggregated")

    # plot_feature_over_time(ft="original_shape2D_MajorAxisLength", MODE="DELTA P", WEIGHT="T2", LRMODE="average")
    # plot_delta_feature(ft="original_shape2D_Elongation", LRMODE="average")
    # plot_feature_over_time(ft="original_shape2D_Elongation", WEIGHT="T2", LRMODE="average")
    # plot_feature_over_time(ft="original_shape2D_Elongation", WEIGHT="T1", LRMODE="average")
    # plot_ids_over_time()
    # illustration_inter_intra_saliva_variability()
    # show_texture_matrix(id="C2", LR="R", norm="stscore")
    # show_texture_matrix(id="H3", LR="R", norm="stscore")

    # show_texture_matrix(id="C2", LR="R", norm="stscore", img_filter="gradient", matrix="glcm")
    # show_texture_matrix(id="H3", LR="R", norm="stscore", img_filter="gradient", matrix="glcm")
    # show_texture_matrix(id="H3", LR="R", norm="nyul", img_filter="gradient", matrix="glcm")

    # show_texture_matrix(id="C2", LR="R", norm="nyul", img_filter="wavelet", matrix="gldm")
    # show_texture_matrix(id="H3", LR="R", norm="nyul", img_filter="wavelet", matrix="gldm")
    # fts = [297, 12, 522, 240, 340, 635, 2, 498, 678, 216, 408, 765, 759, 1]
    # show_feature_classes_histogram(fts=fts, ft_keys="idx", separate_wavelet_HL=False)
    # show_feature_classes_histogram(fts=[], ft_keys="idx", separate_wavelet_HL=False)
    sys.exit()

    # figure_filtering_morphology()
    # figure_histogram_equalization()
    # figure_otsu_tresholding()
    # figure_bias_field()
    for exp in find_folders(RawDir, condition="Pilot"):
        for time in find_folders(os.path.join(RawDir, exp)):
            # print(exp, time)
            files = find_folders(os.path.join(RawDir, exp, time), condition="sagittal")
            ids = list(set([x.split("_")[0] for x in files]))
            print("\n", exp, time, ids)
            for id in ids:
                show_all_segmented_at_instance(id, time, exp, X="all", show_roi=False)

    # show_all_segmented_at_instance(id="C2", time="-7day", experiment="Pilot1", X="L")
    # show_all_segmented_at_instance(id="6-5", time="5day", experiment="Pilot3", X="all", show_roi=False)   # bias field + noise
    # show_all_segmented_at_instance(id="C2", time="56day", experiment="Pilot1", X="all", show_roi=False)   # horizontal line artifact

    # show_all_segmented_at_instance(id="6-1", time="-7day", experiment="Pilot3")
    # show_all_segmented_at_instance(id="9-5", time="105day", experiment="Pilot3")

    # RADIOMICS THEORY: CREATE TEXTURE MATRIX FIGURES
    # Nv = 4
    # Ng = 4
    # np.random.seed(42)
    # img = np.round(np.random.rand(Nv, Nv) * Ng) + 1
    # # img = [[1, 3, 3, 1], [4, 4, 3, 2], [2, 2, 3, 1]] # kristin master
    # # img = [[4, 3, 2, 1], [4, 3, 1, 1], [3, 2, 2, 2], [4, 1, 1, 4]] # grunbeck master
    # # img = [[1, 2, 5, 2, 3], [3, 2, 1, 3, 1], [1, 3, 5, 5, 2], [1, 1, 1, 1, 2], [1, 2, 4, 3, 5]] #pyrad GLCM
    # # img = [[5, 2, 5, 4, 4], [3, 3, 3, 1, 3], [2, 1, 1, 1, 3], [4, 2, 2, 2, 3], [3, 5, 3, 3, 2]] #pyrad GLSZM, GLRLM
    # # img = [[1, 2, 5, 2], [3, 5, 1, 3], [1, 3, 5, 5], [3, 1, 1, 1]]    #pyrad ngtdm
    #
    # print(img)
    # # figure_texture_matrix(img, mode="glcm"
    # # figure_texture_matrix(img, mode="GLRLM")
    # # figure_texture_matrix(img, mode="glszm")  # THIS NEEDS TO HAVE force2d = False !!!!!!
    # figure_texture_matrix(img, mode="gldm")
    # # figure_texture_matrix(img, mode="ngtdm")

    # DISCRETIZATION: FBW PARAM
    # plot_fbw_num_bins()

    # import os
    # folder = r"C:\Users\toral\OneDrive - Universitetet i Oslo\RAW DATA\Pilot_LateEffects_-7day\C2_sagittal"
    # cropped_path = os.path.normpath(os.path.join(os.getcwd(), "..", r"Segmentations\cropped_salivary\raw\-7day", "C2_sagittal.npy"))
    # print(cropped_path)
    # indexed_files = dcm_files_indexed(folder)
    # pixel_matrix_raw = dcm_folder_to_pixel_matrix(indexed_files, folder)


    # idx = 12  #veiny bit
    # idx = 3 #veiny bit in cropped

    # image = np.array(pixel_matrix_raw[idx])
    # cropped_matr = load_matrix(cropped_path)
    # cropped = cropped_matr[idx]        #veiny bit
    # cropped = load_matrix(cropped_path)[9]      #last slice
    # low, up = 5, 97.5
    # trunc = percentile_truncation(cropped, low, up)

    # plt.style.use("seaborn")
    #PLOT HISTOGRAMS OF RAW, CROPPED, PERCENTILE TRUNCATED IMAGE
    # fig, axes = plt.subplots(ncols=3, nrows=2)
    # ax = axes.ravel()
    # fig.suptitle("Intensity histograms for -7day C2_Sagittal")
    # # show_histogram(image)
    # # plt.figure()
    # for i in range(3,6):
    #     ax[i].grid(False)
    # ax[3].imshow(image, cmap="gray")
    # ax[0].hist(image.ravel(), bins=100, range=None)
    # ax[0].set_title("Raw image")
    # ax[4].imshow(cropped, cmap="gray")
    # ax[1].hist(cropped.ravel(), bins=100)
    # ax[1].set_title("Cropped image")
    # ax[5].imshow(trunc, cmap="gray")
    # ax[2].hist(trunc.ravel(), bins=100)
    # ax[2].set_title(f"Cropped + truncated from {low}% to {up}% percentile")
    # ax[0].set_xlabel("Pixel intensity / a.u.")
    # ax[0].set_ylabel("Pixel count")
    # ax[1].set_xlabel("Pixel intensity / a.u.")
    # ax[1].set_ylabel("Pixel count")
    # ax[2].set_xlabel("Pixel intensity / a.u.")
    # ax[2].set_ylabel("Pixel count")
    # # plt.show()
    # plt.close()

    # 2D WATERSHED PIPELINE
    # p = (2, 5, 10, 2)
    # image = cropped
    # denoised, gradient, markers, labels = watershed_2d(image, *p)
    # fig, axes = plt.subplots(ncols=4, nrows=2);  ax = axes.ravel()
    # fig.suptitle("Watershed pipeline illustration. Top: cropped. Bottom: cropped + truncated")
    # ax[0].imshow(denoised, cmap="gray")
    # ax[1].imshow(gradient, cmap="gray")
    # ax[2].imshow(image, cmap="gray")
    # ax[2].imshow(markers, cmap=plt.cm.hot, alpha=0.3)
    # ax[3].imshow(image, cmap="gray")
    # ax[3].imshow(labels, cmap=plt.cm.hot, alpha=0.3)
    # ax[0].set_title(f"Median blurring(disk({p[0]}))")
    # ax[1].set_title(f"Gradient image(disk({p[3]}))")
    # ax[2].set_title(f"Marker regions from gradient(disk({p[1]})) > {p[2]}")
    # ax[3].set_title("Watershed on (gradient, markers)")
    #
    # image = trunc
    # denoised, gradient, markers, labels = watershed_2d(trunc, *p)
    # ax[4].imshow(denoised, cmap="gray")
    # ax[5].imshow(gradient, cmap="gray")
    # ax[6].imshow(image, cmap="gray")
    # ax[6].imshow(markers, cmap=plt.cm.hot, alpha=0.3)
    # ax[7].imshow(image, cmap="gray")
    # ax[7].imshow(labels, cmap=plt.cm.hot, alpha=0.3)
    # fig.tight_layout()
    # plt.show()

    #3D
    # idx = 3
    # low, up = 45, 97.5
    # trunc_matr = percentile_truncation(cropped_matr, low, up)
    # image = cropped_matr[idx]
    # imagetrunc = trunc_matr[idx]

    # p = (2, 5, 10, 2)
    # # p = (5, 8, 12, 3)
    # matr = cropped_matr
    # denoised, gradient, markers, labels = watershed_3d(matr, *p)
    # fig, axes = plt.subplots(ncols=4, nrows=2);  ax = axes.ravel()
    # fig.suptitle("3D watershed pipeline illustration. Top: cropped. Bottom: cropped + truncated")
    # ax[0].imshow(denoised[idx], cmap="gray")
    # ax[1].imshow(gradient[idx], cmap="gray")
    # ax[2].imshow(image, cmap="gray")
    # ax[2].imshow(markers[idx], cmap=plt.cm.hot, alpha=0.3)
    # ax[3].imshow(image, cmap="gray")
    # ax[3].imshow(labels[idx], cmap=plt.cm.hot, alpha=0.3)
    # ax[0].set_title(f"Median blurring(disk({p[0]}))")
    # ax[1].set_title(f"Gradient image(disk({p[3]}))")
    # ax[2].set_title(f"Marker regions from gradient(disk({p[1]})) > {p[2]}")
    # ax[3].set_title("Watershed on (gradient, markers)")
    #
    # matr = trunc_matr
    # denoised, gradient, markers, labels = watershed_3d(matr, *p)
    # fig.suptitle("3D watershed pipeline illustration. Top: cropped. Bottom: cropped + truncated")
    # ax[4].imshow(denoised[idx], cmap="gray")
    # ax[5].imshow(gradient[idx], cmap="gray")
    # ax[6].imshow(imagetrunc, cmap="gray")
    # ax[6].imshow(markers[idx], cmap=plt.cm.hot, alpha=0.3)
    # ax[7].imshow(imagetrunc, cmap="gray")
    # ax[7].imshow(labels[idx], cmap=plt.cm.hot, alpha=0.3, norm=mpl.colors.LogNorm(vmin=labels.min(), vmax=labels.max()))
    # plt.show()

    from skimage.future import graph
    # from skimage import data, segmentation, color, filters, io
    # # image = cropped_matr[idx]
    # image = data.coffee()
    # labels = segmentation.slic(image, compactness=30, n_segments=400, start_label=1)
    # kmeans = color.label2rgb(labels, image, kind='avg', bg_label=0)
    # print(kmeans.shape, image.shape)
    # edges = filters.sobel(image)
    # g = graph.rag_boundary(kmeans, edges)
    # lc = graph.show_rag(kmeans, g, edges, img_cmap=plt.cm.hot, edge_cmap='viridis',
    #                     edge_width=1.2)
    # plt.colorbar(lc, fraction=0.03)
    # io.show()

    # print(kmeans.shape)
    # # p = (2, 5, 10, 2)
    # p = (5, 8, 12, 3)
    # fig, axes = plt.subplots(ncols=2);  ax1, ax2 = axes.ravel()
    # matr = cropped_matr
    # denoised, gradient, markers, labels = watershed_3d(matr, *p)
    # #
    # #
    # ax1.imshow(image, cmap="gray")
    # ax1.imshow(markers[idx], alpha=0.3, cmap=plt.cm.hot)
    # ax2.imshow(kmeans, cmap=plt.cm.hot)
    # plt.show()