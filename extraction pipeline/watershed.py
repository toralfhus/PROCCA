import DICOM_reader
# from DICOM_reader import center_and_crop, find_folders
# import visualizations as vis
import cv2
import numpy as np
import os
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from preprocessing import norm_minmax_featurescaled, MouseClick, save_matrix
from texturedetection import timeit
# from visualizations import plot_masked, plot_image, compare_images, show_histogram
from skimage.util import img_as_ubyte
from skimage.filters import rank, median
from skimage.morphology import disk, ball
from skimage.segmentation import watershed, slic
from skimage import color



# class MouseClick:
#     def __init__(self, clicked_point=(0,0), clickval=0, sliceloc=0):
#         self.clicked_point = clicked_point
#         self.clickval = clickval
#         self.sliceloc=sliceloc
#     def click_to_point(self, event, x, y, flags, param=([], "")):
#         arr, arr_name = param
#         if event == cv2.EVENT_LBUTTONDOWN:
#             # self.clicked_point = (x, y)#(y, x)
#             self.clicked_point = (y, x)
#             print("You clicked on pixel %i , %i" % (self.clicked_point))
#             try:
#                 self.clickval = arr[self.clicked_point]
#                 print("%s value = " % arr_name, self.clickval)
#             except Exception as e:
#                 print(e.args)
#     def click_to_point_3D(self, event, x, y, flags, param=([], "")):
#         arr, arr_name = param
#         if event == cv2.EVENT_LBUTTONDOWN:
#             # self.clicked_point = (x, y)#(y, x)
#             self.clicked_point = (y, x)
#             print("You clicked on pixel %s in slice %i" % (self.clicked_point, self.sliceloc))
#             try:
#                 self.clickval = arr[self.sliceloc, self.clicked_point[0], self.clicked_point[1]]
#                 print("%s value = " % arr_name, self.clickval)
#             except Exception as e:
#                 print(e.args)



def watershed_2d(image, mediandisksize = 2, markerdisksize = 5, markerthresh = 10, gradientdisksize=2, ordinals=False):
    # BASED ON:
    # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_marked_watershed.html?highlight=median
    image_ubyte = img_as_ubyte(image, force_copy=True)       #unsigned byte format [0, 255]
    denoised = rank.median(image_ubyte, disk(mediandisksize))
    #todo: try gaussian blur?
    gradient = rank.gradient(denoised, disk(gradientdisksize))
    markers = rank.gradient(denoised, disk(markerdisksize)) < markerthresh  #binary mask
    connectivity_structure = [[0,1,0], [1,1,1], [0,1,0]] if not ordinals else [[1,1,1], [1,1,1], [1,1,1]]
    markers = ndi.label(markers, structure=connectivity_structure)[0]       #make numbered regions if cardinally connected
    labels = watershed(gradient, markers)
    # print(f"Labelled {100 * len(np.argwhere(labels != 0)) / (30 * 256 * 256)}% of voxels, of {len(np.unique(labels))} levels.", labels.shape)
    return denoised, gradient, markers, labels


def watershed_3d(image_matrix, medianballsize=2, markerballsize=5, markerthresh=10, gradientballsize=2, close=False, closekernelsize=2):
    matrix_ubyte = img_as_ubyte(image_matrix, force_copy=True)
    shape = image_matrix.shape
    denoised = rank.median(matrix_ubyte, ball(medianballsize))
    # print("denoised", denoised.shape)
    gradient = rank.gradient(denoised, ball(gradientballsize))
    # print("gradient", gradient.shape)
    markers = rank.gradient(denoised, ball(markerballsize)) < markerthresh      #binary mask
    # print("markers binary", markers.shape)
    markers = ndi.label(markers)[0]                                             #number markers by connectivity
    # print("markers labelled", markers.shape)
    labels = watershed(gradient, markers)
    # if close:
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelsize)
    #     labels = cv2.morphologyEx(labels, cv2.MORPH_CLOSE, kernel)
    print(f"Labelled {100*len(np.argwhere(labels != 0)) / (shape[0] * shape[1] * shape[2]):.3f}% of voxels, of {len(np.unique(labels))} levels.", labels.shape)
    return denoised, gradient, markers, labels


def browse_images_make_watershed_ROI_2D(image_matrix, title="create ROI from watershed", time="", contrast=True, close=True, params=(2, 5, 10, 2), truncvals=(), save_path = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\watershed_2d_manual\\", idx=0):  #TODO: MAKE ROI ON PAROTID GLAND
    idx_max = np.shape(image_matrix)[0] - 1  # start in middle image ish
    # idx = int(idx_max / 2)
    if not idx:
        idx = image_matrix.shape[0] // 2    # Middle slice
    display_matrix = norm_minmax_featurescaled(np.array(image_matrix), 0, 255).astype("uint8")
    click = MouseClick()
    ROI_mask = np.zeros(shape=np.shape(image_matrix)).astype("uint8")
    roivisible = True
    while True:
        print("Slice index = ", idx)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 800, 800)
        disp_image = display_matrix[idx].copy()
        if contrast:
            disp_image = cv2.convertScaleAbs(disp_image, alpha=2, beta=0)

        #WATERSHED
        _, gradient, markers, labels = watershed_2d(image_matrix[idx], *params)
        print("Watershed at slice %i made %i regions" % (idx, len(np.unique(labels))))
        #Visualization
        labels = img_as_ubyte(labels)
        labels = norm_minmax_featurescaled(labels, 0, 255).astype("uint8")
        # labels_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)    #kind of nice to look at
        labels_color = cv2.applyColorMap(labels, cv2.COLORMAP_HSV)      #very high contrast colors
        disp_image = cv2.cvtColor(disp_image, cv2.COLOR_GRAY2BGR)      #to blend gray image with color channels
        alpha1 = 0.90
        mask_image = cv2.addWeighted(disp_image, alpha1, labels_color, 1-alpha1, 0)   #blend image with labels
        # ROI_image = img_as_ubyte(np.zeros(shape=disp_image.shape))
        for vx in np.argwhere(ROI_mask != 0):      #TODO: add ROI pixels to image
            if vx[0] == idx:
                px = vx[1:]
                mask_image[px[0], px[1]] = (255, 255, 255)
        # alpha2 = 0.5
        # mask_image = cv2.addWeighted(mask_image, alpha2, ROI_image, 1 - alpha2, 0) #blend maskimage with ROI
        cv2.imshow(title, mask_image) if roivisible else cv2.imshow(title, disp_image)

        #TODO: CLICK TO ADD TO/ SUBTRACT FROM ROI
        cv2.setMouseCallback(title, click.click_to_point, (labels, "Labels"))

        print("\nPress key: ,/. = browse slices, space = add to ROI, c = clear ROI (in slice), s = toggle ROI / labels, d = SHRINK ROI (in slice), q = SAVE ROI as npy and exit, x = exit")
        key = cv2.waitKey(0)
        print("you pressed key", key)
        if key == 32:           #space = add current label to ROI
            for px in np.argwhere(labels == click.clickval):
                ROI_mask[idx, px[0], px[1]] = 1
            if close:   #Morphological closing
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                size = (5, 5)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
                ROI_mask[idx] = cv2.morphologyEx(ROI_mask[idx], cv2.MORPH_CLOSE, kernel)
            print(len(np.argwhere(ROI_mask != 0)), "voxels in ROI.")
        if key == 99:           #c = clear ROI (ONLY FOR CURRENT INDEX!!!!!)
            ROI_mask[idx] = np.zeros(shape=np.shape(image_matrix)[1:])
        elif key == 100:        #d = shrink ROI
            ROI_mask[idx] = cv2.erode(ROI_mask[idx], kernel=np.ones((3,3),np.uint8))
            print("ROI SHRUNK")
        elif key == 44:         #,  = prev slice
            idx -= 1
            idx = 0 if idx < 0 else idx
        elif key == 46:         #. = next slice
            idx += 1
            idx = idx_max if idx > idx_max else idx
        elif key == 115:        #s = toggle watershed labels & ROI
            roivisible = not(roivisible)
            print("ROI / mask on") if roivisible else print("ROI / mask off")
        elif key == 113:        #q = SAVE ROI as npy and exit
            print(ROI_mask.dtype, ROI_mask.shape)
            # save_ROI(ROI_mask, save_path, title=title, time=time)
            save_matrix(ROI_mask, save_path, time, title)
            cv2.destroyAllWindows()
            return ROI_mask, idx
        elif key == 120:        #x = exit and return ROI
            cv2.destroyAllWindows()
            print("EXITING WITHOUT SAVING")
            # return np.zeros(shape=ROI_mask.shape)
            ROI_mask = np.zeros(shape=ROI_mask.shape)
            return ROI_mask, idx


def browse_images_make_watershed_ROI_2D_withKMEANS(image_matrix, title="create ROI from watershed", time="", contrast=True, close=True, params=(2, 5, 10, 2), save_path = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\watershed_2d_manual\\"):  #TODO: MAKE ROI ON PAROTID GLAND
    idx_max = np.shape(image_matrix)[0] - 1  # start in middle image ish
    idx = int(idx_max / 2)
    display_matrix = norm_minmax_featurescaled(np.array(image_matrix), 0, 255).astype("uint8")
    click = MouseClick()
    ROI_mask = np.zeros(shape=np.shape(image_matrix)).astype("uint8")
    roivisible = True
    while True:
        print("Slice index = ", idx)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 800, 800)
        disp_image = display_matrix[idx].copy()
        if contrast:
            disp_image = cv2.convertScaleAbs(disp_image, alpha=2, beta=0)

        #WATERSHED
        #TODO: IMPLEMENT KMEANS CLUSTERING INTO SEGMENTATION -- SE IF IMPROVING BAD ROI'S
        img = image_matrix[idx].copy()
        p = params
        denoised = median(img, disk(p[0]))
        grad = rank.gradient(denoised, disk(p[3]))
        # print(img.shape)
        labels = slic(denoised, compactness=.0001, n_segments=2000, start_label=1)     #KMEANS CLUSTERING
        # print(labels1.shape)
        out = color.label2rgb(labels, grad, kind='avg', bg_label=0, colors=["gray"])
        kmeans = color.rgb2gray(out)
        markers = rank.gradient(denoised, disk(p[1])) < p[2]
        # print(np.where(markers, 1, 0))
        markers = np.where(markers, 1, 0).astype("uint8")
        markers = cv2.morphologyEx(markers, cv2.MORPH_ERODE, disk(3))       #MORPHOLOGICAL OPERATION ON MARKERS (ERODE, OPEN, CLOSE, DILATE ETC)
        markers = ndi.label(markers)[0]
        labels = watershed(kmeans, markers)
        # labels = watershed(grad, markers)
        # _, gradient, markers, labels = watershed_2d(image_matrix[idx], *params)
        print("Watershed at slice %i made %i regions" % (idx, len(np.unique(labels))))
        #Visualization
        labels = img_as_ubyte(labels)
        labels = norm_minmax_featurescaled(labels, 0, 255).astype("uint8")
        # labels_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)    #kind of nice to look at
        labels_color = cv2.applyColorMap(labels, cv2.COLORMAP_HSV)      #very high contrast colors
        disp_image = cv2.cvtColor(disp_image, cv2.COLOR_GRAY2BGR)      #to blend gray image with color channels
        alpha1 = 0.90
        mask_image = cv2.addWeighted(disp_image, alpha1, labels_color, 1-alpha1, 0)   #blend image with labels
        # ROI_image = img_as_ubyte(np.zeros(shape=disp_image.shape))
        for vx in np.argwhere(ROI_mask != 0):      #TODO: add ROI pixels to image
            if vx[0] == idx:
                px = vx[1:]
                mask_image[px[0], px[1]] = (255, 255, 255)
        # alpha2 = 0.5
        # mask_image = cv2.addWeighted(mask_image, alpha2, ROI_image, 1 - alpha2, 0) #blend maskimage with ROI
        cv2.imshow(title, mask_image) if roivisible else cv2.imshow(title, disp_image)

        #TODO: CLICK TO ADD TO/ SUBTRACT FROM ROI
        cv2.setMouseCallback(title, click.click_to_point, (labels, "Labels"))

        print("\nPress key: ,/. = browse slices, space = add to ROI, c = clear ROI (in slice), s = toggle ROI / labels, x = exit")
        key = cv2.waitKey(0)
        print("you pressed key", key)
        if key == 32:           #space = add current label to ROI
            for px in np.argwhere(labels == click.clickval):
                ROI_mask[idx, px[0], px[1]] = 1
            if close:   #Morphological closing
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                size = (5, 5)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
                ROI_mask[idx] = cv2.morphologyEx(ROI_mask[idx], cv2.MORPH_CLOSE, kernel)
            print(len(np.argwhere(ROI_mask != 0)), "voxels in ROI.")
        if key == 99:           #c = clear ROI (ONLY FOR CURRENT INDEX!!!!!)
            ROI_mask[idx] = np.zeros(shape=np.shape(image_matrix)[1:])
        elif key == 44:         #,  = prev slice
            idx -= 1
            idx = 0 if idx < 0 else idx
        elif key == 46:         #. = next slice
            idx += 1
            idx = idx_max if idx > idx_max else idx
        elif key == 115:        #s = toggle watershed labels & ROI
            roivisible = not(roivisible)
            print("ROI / mask on") if roivisible else print("ROI / mask off")
        elif key == 113:        #q = SAVE ROI as csv and exit
            print(ROI_mask.dtype, ROI_mask.shape)
            # save_ROI(ROI_mask, save_path, title=title, time=time)
            save_matrix(ROI_mask, save_path, time, title)
            cv2.destroyAllWindows()
            return ROI_mask
        elif key == 120:        #x = exit and return ROI
            cv2.destroyAllWindows()
            return np.zeros(shape=ROI_mask.shape)



def browse_watershed3D_images(image_matrix, label_matrix, title="Labels from 3D watershed", contrast=True):  #Visualize 3D labels in slices
    idx_max = np.shape(image_matrix)[0] - 1  # start in middle image ish
    idx = int(idx_max / 2)
    display_matrix = norm_minmax_featurescaled(np.array(image_matrix), 0, 255).astype("uint8")
    click = MouseClick(sliceloc=idx)
    ROI_mask = np.zeros(shape=np.shape(image_matrix))
    roivisible = True
    while True:
        print("Slice index = ", idx)
        click.sliceloc = idx
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 800, 800)
        disp_image = display_matrix[idx]
        disp_labels = label_matrix[idx]
        disp_labels = img_as_ubyte(disp_labels)
        if contrast:
            disp_image = cv2.convertScaleAbs(disp_image, alpha=2, beta=0)

        #WATERSHED
        # gradient, markers, labels = watershed_2d(image_matrix[idx], mediandisksize=2, markerdisksize=5, markerthresh=10, gradientdisksize=2)
        # print("Watershed at slice %i made %i regions" % (idx, len(np.unique(labels))))
        #Visualization
        # labels = norm_minmax_featurescaled(labels, 0, 255).astype("uint8")
        # labels_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)    #kind of nice to look at
        labels_color = cv2.applyColorMap(disp_labels, cv2.COLORMAP_HSV)      #very high contrast colors
        disp_image = cv2.cvtColor(disp_image, cv2.COLOR_GRAY2BGR)      #to blend gray image with color channels
        alpha1 = 0.90
        mask_image = cv2.addWeighted(disp_image, alpha1, labels_color, 1-alpha1, 0)   #blend image with labels
        ROI_image = img_as_ubyte(np.zeros(shape=disp_image.shape))
        for vx in np.argwhere(ROI_mask != 0):      #VISUALIZE ROI IN CURRENT SLICE
            # mask_image[vx[0], vx[1], vx[2]] = (255, 255, 255)
            if vx[0] == idx:
                px = vx[1:]
                mask_image[px[0], px[1]] = (255, 255, 255)
        alpha2 = 0.5
        mask_image = cv2.addWeighted(mask_image, alpha2, ROI_image, 1 - alpha2, 0) #blend maskimage with ROI
        cv2.imshow(title, mask_image) if roivisible else cv2.imshow(title, disp_image)


        # cv2.setMouseCallback(title, click.click_to_point_3D, (labels, "Labels"))
        cv2.setMouseCallback(title, click.click_to_point_3D, (label_matrix, "Labels"))

        print("\nPress key: ,/. = browse slices")
        key = cv2.waitKey(0)
        if key == 32:           #space = add current label to ROI   #TODO: CLICK TO ADD TO/ SUBTRACT FROM ROI
            for vx in np.argwhere(label_matrix == click.clickval):
                ROI_mask[vx[0], vx[1], vx[2]] = 255
        #     if close:   #Morphological closing
        #         # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        #         size = (5, 5)
        #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
        #         ROI_mask[idx] = cv2.morphologyEx(ROI_mask[idx], cv2.MORPH_CLOSE, kernel)
        #
        #         #TODO: do closing
        #         pass
            print(len(np.argwhere(ROI_mask != 0)), "voxels in ROI.")
        if key == 99:           #c = clear ROI  (ALL SLICES)
            ROI_mask = np.zeros(shape=np.shape(image_matrix))
        if key == 44:         #,  = prev slice
            idx -= 1
            idx = 0 if idx < 0 else idx
        elif key == 46:         #. = next slice
            idx += 1
            idx = idx_max if idx > idx_max else idx
        elif key == 115:        #s = toggle watershed labels & ROI
            roivisible = not(roivisible)
            print("ROI / mask on") if roivisible else print("ROI / mask off")
        elif key == 113:        #q = SAVE ROI as csv
            save_path = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\\"
            print(ROI_mask.dtype, ROI_mask.shape)
            save_ROI(ROI_mask, save_path, title)
            # cv2.destroyAllWindows()
            # return
        elif key == 120:        #x = exit and return ROI
            cv2.destroyAllWindows()
            return ROI_mask


def test_watershed_parameters(data, vals1, vals2, vals3, vals4, mode="3D", idx=14):
    print("---- Testing parameters for %s watershed ----" % (mode))
    if mode == "3D":
        print("WATERSHED vals1 = {}".format(vals1,))
        _, _, _, labels1 = watershed_3d(data, *vals1)
        labels1 = norm_minmax_featurescaled(labels1[idx], 0, 255)
        print(len(np.unique(labels1)), "labels in ", np.unique(labels1))
        print("WATERSHED vals2 = {}".format(vals2, ))
        _, _, _, labels2 = watershed_3d(data, *vals2)
        labels2 = norm_minmax_featurescaled(labels2[idx], 0, 255)
        print(len(np.unique(labels2)), "labels in ", np.unique(labels2))
        print("WATERSHED vals3 = {}".format(vals3, ))
        _, _, _, labels3 = watershed_3d(data, *vals3)
        labels3 = norm_minmax_featurescaled(labels3[idx], 0, 255)
        print(len(np.unique(labels3)), "labels in ", np.unique(labels3))
        print("WATERSHED vals4 = {}".format(vals4, ))
        _, _, _, labels4 = watershed_3d(data, *vals4)
        labels4 = norm_minmax_featurescaled(labels4[idx], 0, 255)
        print(len(np.unique(labels4)), "labels in ", np.unique(labels4))
        # labels1, labels2, labels3, labels4 = labels1[idx], labels2[idx], labels3[idx], labels4[idx]
    elif mode == "2D":
        _, _, labels1 = watershed_2d(data, *vals1)
        _, _, labels2 = watershed_2d(data, *vals2)
        _, _, labels3 = watershed_2d(data, *vals3)
        _, _, labels4 = watershed_2d(data, *vals4)
    else:
        print("No valid mode = ", mode)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
    ax1, ax2, ax3, ax4 = axes.ravel()
    fig.suptitle(
        "%s watershed segmented regions. Parameter tuple = (mediandisksize, markerdisksize, markerthresh, gradientdisksize)" % (mode))
    ax1.imshow(disp_image, cmap="gray")
    ax1.imshow(labels1, cmap=plt.cm.nipy_spectral, alpha=0.3)
    ax1.set_title("%i labels from %s" % (len(np.unique(labels1)), vals1,))
    ax2.imshow(disp_image, cmap="gray")
    ax2.imshow(labels2, cmap=plt.cm.nipy_spectral, alpha=0.3)
    ax2.set_title("%i labels from %s" % (len(np.unique(labels2)), vals2,))
    ax3.imshow(disp_image, cmap="gray")
    ax3.imshow(labels3, cmap=plt.cm.nipy_spectral, alpha=0.3)
    ax3.set_title("%i labels from %s" % (len(np.unique(labels3)), vals3,))
    ax4.imshow(disp_image, cmap="gray")
    ax4.imshow(labels4, cmap=plt.cm.nipy_spectral, alpha=0.3)
    ax4.set_title("%i labels from %s" % (len(np.unique(labels4)), vals4,))
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.show()
    plt.close()
    pass


@timeit
def brute_force_this_bitsj(data, medians, marks, threshs, grads, idx=14):
    image = np.array(data[idx])
    disp_image = norm_minmax_featurescaled(image.copy(), 0, 255)        #normalize
    disp_image = cv2.convertScaleAbs(disp_image, alpha=3, beta=0)       #enhance contrast
    savepath = r"C:\Users\toral\OneDrive - Universitetet i Oslo\master_plots\segmentation\watershed\parameter_tuning_3D\\"
    for p1 in medians:
        for p2 in marks:
            for p3 in threshs:
                for p4 in grads:
                    print(p1, p2, p3, p4)
                    denoised, gradient, markers, labels = watershed_3d(data, p1, p2, p3, p4)
                    print("bro")
                    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
                    fig.suptitle("3D watershed with params (mediandisksize, markerdisksize, markerthresh, gradientdisksize) = %s" % ((p1, p2, p3, p4),))
                    ax2, ax3, ax4 = axes.ravel()
                    # ax1.imshow(disp_image, cmap="gray")
                    # ax1.set_title("Slice image")
                    ax2.imshow(gradient[idx], cmap="gray")
                    ax2.set_title("Gradient")
                    ax3.imshow(markers[idx], cmap=plt.cm.nipy_spectral)
                    ax3.set_title("Markers")
                    ax4.imshow(disp_image, cmap="gray")
                    ax4.imshow(labels[idx], cmap=plt.cm.nipy_spectral, alpha=0.3)
                    ax4.set_title("%i watershed labelled regions" % (len(np.unique(labels[idx]))))
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.85)
                    #ax1.get_xaxis().set_visible(False);
                    ax2.get_xaxis().set_visible(False); ax3.get_xaxis().set_visible(False); ax4.get_xaxis().set_visible(False)
                    #ax1.get_yaxis().set_visible(False);
                    ax2.get_yaxis().set_visible(False); ax3.get_yaxis().set_visible(False); ax4.get_yaxis().set_visible(False)
                    plt.savefig(savepath + str(p1) + "_" + str(p2) + "_" + str(p3) + "_" + str(p4), bbox_inches ="tight", pad_inches = 0.2)
                    plt.close()
    print("\nYou did it you beautiful son of a bitch")
    pass


def save_ROI(ROI, folder, time, title, mode="3D"):       #TODO: fix 3D array wont save?  https://www.geeksforgeeks.org/how-to-load-and-save-3d-numpy-array-to-file-using-savetxt-and-loadtxt-functions/
    save_path = folder + r"\\" + time + r"\\"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filepath = save_path + title + ".csv"
    savebool = True
    if os.path.exists(filepath):
        b = input("File already exists at " + filepath + "\n" + "overwrite? y/n \t")
        if b == "y":
            savebool = True
            # np.savetxt(filepath, ROI, delimiter=",", fmt="%d")      #TODO: fix for 3D array, else save for each sliceidx
            # print("File saved at ", filepath)
        else:
            print("File not saved.")
            savebool = False
            return 0
    if savebool:
        reshaped = ROI.reshape(ROI.shape[0], -1)
        print("ROI of shape %s reshaped to %s." % (ROI.shape, reshaped.shape))
        np.savetxt(filepath, reshaped, delimiter=",", fmt="%d")
        print("File saved as ", filepath)
        return 1
    return 0


def load_ROI(path, expectedshape=(30, 256, 256), expectedarr=np.array([])):     #read 3D ROI from csv       TODO: TEST THIS
    print("------------------\n" + "Loading ROI from path " + path)
    arr = np.loadtxt(path, delimiter=",")
    arrshape = arr.shape
    expectedshape = expectedarr.shape if expectedarr.any() else expectedshape
    # print(expectedshape)
    ROI = arr.reshape(arrshape[0], arrshape[1] // expectedshape[2], expectedshape[2])
    print("Shape of loaded array: %s reshaped to ROI of shape %s" % (arrshape, ROI.shape))
    print(len(np.argwhere(ROI != 0)), "voxels in ROI.")
    if expectedarr.any():
        if (ROI == expectedarr).all():
            print("Loaded ROI is exactly the same as expected ROI.")
        else:
            print("Loaded ROI is not the same as expected.")
    else:
        print("No expected ROI to compare with.")
    return ROI


from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def plot_ROI_3D(pixel_matrix, ROI, thresh = 0):
    p_roi = ROI.transpose(2,1,0)
    p = np.array(pixel_matrix).transpose(2, 1, 0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, thresh)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()
    pass









if __name__ == "__main__":
    # folder = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Pilot_LateEffects_-7day\C1_sagittal"
    folder = r"C:\Users\toral\OneDrive - Universitetet i Oslo\RAW DATA\Pilot_LateEffects_-7day\C2_sagittal"
    title = folder[-11:]        #XX_sagittal

    indexed_files = DICOM_reader.dcm_files_indexed(folder)
    pixel_matrix_raw = DICOM_reader.dcm_folder_to_pixel_matrix(indexed_files, folder)
    # cropped = center_and_crop(pixel_matrix_raw, time="-7day", savefolder=r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary", title=title)


    cropped_path = os.path.normpath(os.path.join(os.getcwd(), "..", r"Segmentations\cropped_salivary\raw\-7day"))
    p = (2, 5, 10, 2)

    # for i, file in enumerate(find_folders(cropped_path, "sagittal")):
    #     # print(file)
    #     if "C2_sagittal" in file:
    #         print(file)
    #         cropped_load = load_matrix(os.path.join(cropped_path, file))#, cropped)
    #
    #         idx = 4
    #         compare_images(pixel_matrix_raw[idx+9], cropped_load[idx])
        # normed = norm_stscore(cropped_load[idx])
        # compare_images(pixel_matrix_raw[idx, idx], cropped_load[idx])
        # show_histogram(cropped_load[idx])
        # roi = browse_images_make_watershed_ROI_2D(cropped_load, params=p)
        # print(roi.shape)


    # cropped_roi = load_matrix(r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary\roi man very bad\-7day")
    params = (2, 5, 10, 2)
    # params = (5, 8, 12, 4)
    # params = (4,5,2,2)
    # denoised, gradient, markers, labels = watershed_3d(cropped, *params)
    # browse_images_make_watershed_ROI_2D(cropped_load)



    # params = (5, 8, 12, 3)
    # watershed_3d(pixel_matrix_raw, *params)

    # browse_watershed3D_images(cropped, labels)

    # ROI = browse_images_make_watershed_ROI_2D(pixel_matrix_raw, close=True, title=title, time="-7days")

    # print("\n3D ROI containing %i voxels created." % (len(np.argwhere(ROI != 0))))
    # ROI_loaded = load_ROI(path=r"C:\Users\toral\OneDrive - Universitetet i Oslo\Segmentations\watershed_2d_manual\-7days\\" + title + ".csv", expectedarr=ROI)


    # params = (4, 5, 2, 2)       #tja
    # params = (2, 5, 10, 2)          #works fine iosh 3d
    # _, _, _, labels = watershed_3d(pixel_matrix_raw, *params)
    # print("\n3D labels of %i levels created." % (len(np.unique(labels))))
    # ROI = browse_watershed3D_images(pixel_matrix_raw, labels, title="-7days_C2_sagittal")
    # plot_ROI_3D(pixel_matrix_raw, ROI)

    # print("ROI made containing", len(np.argwhere(ROI != 0)), "voxels.")
    # plot_ROI_3D(pixel_matrix_raw, ROI)
    idx = 12
    image = np.array(pixel_matrix_raw[idx])
    disp_image = norm_minmax_featurescaled(image.copy(), 0, 255)        #normalize
    disp_image = cv2.convertScaleAbs(disp_image, alpha=2, beta=0)       #enhance contrast


    #WATERSHED PIPELINE ILLUSTRATION
    # params = 2, 5, 10, 2                #mediandisksize, markerdisksize, markerthresh, gradientdisksize
    # denoised, gradient, markers, labels = watershed_2d(image, *params)
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    # ax = axes.ravel()
    # # ax[0].imshow(disp_image, cmap="gray")
    # # ax[0].set_title("Raw image (contrast enhanced)")
    # ax[0].imshow(gradient, cmap="gray")
    # ax[0].set_title("Gradient image")
    # ax[1].imshow(markers, cmap=plt.cm.nipy_spectral)
    # ax[1].set_title("Marker image")
    # ax[2].imshow(disp_image, cmap="gray")
    # ax[2].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=0.3)
    # ax[2].set_title("%i watershed segmented regions"  % (len(np.unique(labels))))
    # ax[0].get_xaxis().set_visible(False); ax[1].get_xaxis().set_visible(False); ax[2].get_xaxis().set_visible(False);# ax[3].get_xaxis().set_visible(False)
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.85)
    # fig.suptitle("2D watershed segmentation illustration")
    # plt.show()
    # plt.close()


    # Plotting raw image / labelled image
    # fig, axes = plt.subplots(ncols=3, nrows=1);
    # ax1, ax2, ax3 = axes.ravel()
    # ax3.imshow(disp_image, cmap="gray")
    # ax3.imshow(labels, cmap=plt.cm.nipy_spectral, alpha=0.3)
    # # plt.imshow(disp_image, cmap="gray")
    # # plt.imshow(labels, cmap=plt.cm.nipy_spectral, alpha=0.3)
    # ax1.imshow(gradient, cmap="gray")
    # ax2.imshow(markers, cmap=plt.cm.nipy_spectral)
    # # plt.colorbar().ax.set_ylabel("Numbered labels")
    # ax1.get_xaxis().set_visible(False); ax1.get_yaxis().set_visible(False)
    # ax2.get_xaxis().set_visible(False); ax2.get_yaxis().set_visible(False)
    # ax3.get_xaxis().set_visible(False); ax3.get_yaxis().set_visible(False)
    # ax3.set_title("Watershed --> labelled regions")
    # ax1.set_title("Gradient image - disk(2)")
    # ax2.set_title("Numbered markers - gradient(disk(5)) < 10")
    # plt.tight_layout()
    # plt.show()
    # plt.close()


    #img_as_ubyte vs featurescaled(0,255).astype("uint8")
    # fig, axes = plt.subplots(ncols=2);  ax = axes.ravel()
    # ax[0].imshow(img_as_ubyte(image.copy()), cmap="gray")
    # ax[1].imshow(norm_minmax_featurescaled(image.copy(), 0, 255, rounded=True).astype("uint8"), cmap="gray")
    # plt.show()
    # plt.close()





    #3D WATERSHED
    # params = 2, 5, 10, 2        #medianballsize, markerballsize, markerthresh, gradientballsize THIS WORKS FOR 2D
    # denoised, gradient, markers, labels = watershed_3d(pixel_matrix_raw, *params)
    # browse_watershed3D_images(pixel_matrix_raw.copy(), labels.copy())

    # param1 = 2, 2, 10, 2  #mediandisksize, markerdisksize, markerthresh, gradientdisksize
    # param1 = 1, 1, 1, 1
    # param2 = 2, 2, 2, 2
    # param3 = 5, 5, 5, 5
    # param4 = 10 , 10, 10, 10
    # test_watershed_parameters(pixel_matrix_raw, param1, param2, param3, param4, mode="3D")


    #3D watershed pipeline
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
    # fig.suptitle("%i labels from 3D watershed. (mediandisksize, markerdisksize, markerthresh, gradientdisksize) = %s" % (len(np.unique(labels)), params,))
    # ax1, ax2, ax3, ax4 = axes.ravel()
    # ax1.imshow(disp_image, cmap="gray")
    # ax1.set_title("Contrast enchanced image")
    # ax2.imshow(gradient[idx], cmap="gray")
    # ax2.set_title("Gradient")
    # ax3.imshow(markers[idx], cmap=plt.cm.nipy_spectral)
    # ax3.set_title("Markers")
    # ax4.imshow(disp_image, cmap="gray")
    # ax4.imshow(labels[idx], cmap=plt.cm.nipy_spectral, alpha=0.3)
    # ax4.set_title("Watershed labelled regions")
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.90)
    # plt.show()


    #WATERSHED PIPELINE ILLUSTRATION
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    # ax = axes.ravel()
    # ax[0].imshow(image, cmap="gray")
    # ax[0].set_title("Raw image")
    # ax[1].imshow(gradient, cmap="gray")
    # ax[1].set_title("Gradient image")
    # ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
    # ax[2].set_title("Marker image")
    # ax[3].imshow(image, cmap="gray")
    # ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=0.3)
    # ax[3].set_title("Watershed segmented regions")
    # # plt.show()
    # plt.close()


    #2D WATERSHED TEST DIFFERENT PARAMETER VALUES
    # vals1 = 2, 5, 10, 1  #mediandisksize, markerdisksize, markerthresh, gradientdisksize
    # vals2 = 2, 5, 10, 2  # mediandisksize, markerdisksize, markerthresh, gradientdisksize
    # vals3 = 2, 5, 10, 5  # mediandisksize, markerdisksize, markerthresh, gradientdisksize
    # vals4 = 2, 5, 10, 10
    # test_watershed_parameters(image, vals1, vals2, vals3, vals4, mode="2D")
    # gradient1, markers1, labels1 = watershed_2d(image, *vals1)
    # gradient2, markers2, labels2 = watershed_2d(image, *vals2)
    # _, _, labels3 = watershed_2d(image, *vals3)
    # _, _, labels4 = watershed_2d(image, *vals4)
    #
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
    # ax1, ax2, ax3, ax4 = axes.ravel()
    # fig.suptitle(
    #     "Watershed segmented regions. Parameter tuple = (mediandisksize, markerdisksize, markerthresh, gradientdisksize)")
    # ax1.imshow(disp_image, cmap="gray")
    # ax1.imshow(labels1, cmap=plt.cm.nipy_spectral, alpha=0.3)
    # ax1.set_title("%i labels from %s" % (len(np.unique(labels1)), vals1,))
    # ax2.imshow(disp_image, cmap="gray")
    # ax2.imshow(labels2, cmap=plt.cm.nipy_spectral, alpha=0.3)
    # ax2.set_title("%i labels from %s" % (len(np.unique(labels2)), vals2,))
    # ax3.imshow(disp_image, cmap="gray")
    # ax3.imshow(labels2, cmap=plt.cm.nipy_spectral, alpha=0.3)
    # ax3.set_title("%i labels from %s" % (len(np.unique(labels3)), vals3,))
    # ax4.imshow(disp_image, cmap="gray")
    # ax4.imshow(labels2, cmap=plt.cm.nipy_spectral, alpha=0.3)
    # ax4.set_title("%i labels from %s" % (len(np.unique(labels4)), vals4,))
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.90)
    # plt.show()





    # print(len(np.argwhere(labels == 25)))
    # image_disp = norm.norm_minmax_featurescaled(image.copy(), 0, 255)
    # plt.imshow(image_disp, cmap="gray")
    # plt.imshow(labels, cmap=plt.cm.nipy_spectral, alpha=0.2)
    # plt.show()

    # fig, ax = plt.subplots()
    # image_disp = norm.norm_minmax_featurescaled(image, 0, 255)
    # image_disp[labels == 25] = 255
    # ax.imshow(image_disp)