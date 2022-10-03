import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import cv2
import SimpleITK as sitk
from extract_utils import *
from DICOM_reader import find_folders, dcm_files_indexed, dcm_folder_to_pixel_matrix
from preprocessing import get_name, mean_centering
from MRI_denoising import n4correction

# T1 specified, else valid for T2w images
nyul_scales_stsc = {"brain_decile_p":[1., 11.96596258, 17.86436743, 22.1925246, 26.08638543, 30.20695529, 35.13781007, 41.84636577, 50.61608983, 66.57213208, 100.],
                    "brain_decile_nop":[1., 11.8445526, 17.05404776, 20.85291031, 24.37341303, 28.22700847, 33.17585447, 41.06994313, 51.7274235, 66.15823188, 100.],
                    "brain_mode_p":[1., 25.36538799, 100.],
                    "brain_mode_nop":[1., 22.96416526, 100.],
                    "mean_decile_p":[1., 3.77301066, 7.2562269, 10.4605716, 13.57488954, 17.25096426, 22.14963388, 28.13962055, 36.44611324, 51.09277166, 100.],
                    "mean_decile_nop":[1., 5.13572619, 10.47645192, 15.16483323, 19.05131716, 23.197555, 28.94650071, 36.42937315, 46.05838229, 61.70421, 100.],
                    "otsu_decile_p":[1., 6.53998627, 10.87285376, 15.16960358, 19.54443368, 23.78397925, 28.32430218, 34.54120546, 42.93312188, 57.00577354, 100.],
                    "otsu_decile_nop":[1., 11.8445526, 17.05404776, 20.85291031, 24.37341303, 28.22700847, 33.17585447, 41.06994313, 51.7274235, 66.15823188, 100.],
                    "otsu_mode_p":[1., 16.60236451, 100.],
                    "otsu_mode_nop":[1., 24.87094098, 100.]}

nyul_scales_T2 = {"otsu decile p":  [1.00, 3.21312425e+03, 5.49598198e+03, 7.52454560e+03, 9.43351435e+03, 1.12492516e+04, 1.32660055e+04, 1.59700192e+04, 1.99794605e+04, 2.75461346e+04, 5.0e+04],   #LR AGG
                #"otsu decile p":[1.00, 3.18816033e+03, 5.50789631e+03, 7.71703853e+03, 9.85444470e+03, 1.18042000e+04, 1.39199902e+04, 1.68891067e+04, 2.10417486e+04, 2.82526160e+04, 5.0e+04], # CENTRAL SLICE
               "otsu decile nop":[1.0000, 3.30140275e+03, 6.19774690e+03, 8.95550950e+03, 1.14890908e+04, 1.37046577e+04, 1.59039525e+04, 1.88534807e+04, 2.32156111e+04, 3.09182203e+04, 5.00000000e+04],  #LR w P4
               # "otsu decile nop":[1.00, 3.21917816e+03, 5.87669897e+03, 8.46609280e+03, 1.09161936e+04, 1.31247484e+04, 1.53465674e+04, 1.83082530e+04, 2.27105574e+04, 3.04739091e+04, 5.00000000e+04],  #LR AGG
               # "otsu decile nop":[1., 3.16515549e+03, 5.74520860e+03, 8.38060838e+03, 1.10799506e+04, 1.34726630e+04, 1.58942193e+04, 1.91590519e+04, 2.37514605e+04, 3.16686883e+04, 5.00000000e+04],#CENTR
               "otsu mode p":[1.00000000e+00, 9.60745455e+03, 5.00000000e+04],
               "otsu mode nop":[1.00000000e+00, 1.15460994e+04, 5.00000000e+04]}

nyul_scales_T1 = {"otsu decile p":[1.00, 4.38750515e+03, 9.62401679e+03, 1.44893180e+04, 1.81110719e+04, 2.07968469e+04, 2.29472594e+04, 2.50843858e+04, 2.80577353e+04, 3.44509197e+04, 5.00000000e+04],#LR AGG
                # "otsu decile p":[1.00, 4.50414992e+03, 9.57243957e+03, 1.42777077e+04, 1.78379787e+04, 2.04395721e+04, 2.25400676e+04, 2.48226582e+04, 2.81647958e+04, 3.51052930e+04, 5.00000000e+04],#CENTRAL
                "otsu decile nop":[1.00, 4.42942815e+03, 1.02140688e+04, 1.63680819e+04, 2.05958617e+04, 2.35664210e+04, 2.58500104e+04, 2.80420836e+04, 3.09984782e+04, 3.69781610e+04, 5.00000000e+04],    #LR w Pilot4
                #otsu decile nop":[1.00, 5.08360083e+03, 1.05625978e+04, 1.55639517e+04, 1.92256292e+04, 2.19065584e+04, 2.40367014e+04, 2.62006043e+04, 2.92787583e+04, 3.57003411e+04, 5.00000000e+04],#LR AGG
                  # "otsu decile nop":[1., 5.26597449e+03, 1.08228019e+04, 1.61587608e+04, 2.02766842e+04, 2.31390741e+04, 2.53930946e+04, 2.77088243e+04, 3.08830019e+04, 3.69241948e+04, 5.00000000e+04],#CENTRAL
                  "otsu mode p":[1.00000000e+00, 2.10313213e+04, 5.00000000e+04],
                  "otsu mode nop":[1.00000000e+00, 2.51277791e+04, 5.00000000e+04]}



class nyul_normalizer:
    def __init__(self, pc1=5, pc2=95, s1=1, s2=100, L=[50], nbins=256, MASK_MODE="median", mapmode="linear", pbool=False, T1bool=False, n4bool=True, offsetIDX=0, verbose=False):
        self.pc1, self.pc2 = pc1, pc2   # min / max percentile to include in standard histogram!
        self.s1, self.s2 = s1, s2       # min / max intensities on the standard scale IOI (onto which p1j, p2j will be mapped)
        if any(L):
            self.L = L                      # Landmark percentile values
        else:
            self.L = [0]
        # self.nbins = nbins              # number of bins in the considered histograms ??
        self.LANDMARK_PERC = [pc1, *L, pc2]     # all landmark percentiles including pc1, pc2
        self.MASK_MODE = MASK_MODE    # how to remove background
        self.mapmode = mapmode          # how to map the subintervals of [p1j, p2j] to [s1, s2]
        self.pbool = pbool              # Whether to look at p or no p (SPLIT DATA SET INTO TWO STANDARD HISTOGRAMS)
        self.T1bool = T1bool
        self.n4bool = n4bool # Whether to N4 correct images on training set #TODO: DO FOR TRAINING, BUT SHOULD BE DONE ON IMAGES FOR TRANSFORM? OR OUTSIDE THIS NORMALIZER OBJECT?
        self.offsetIDX = offsetIDX
        self.SCALE_STD = []
        self.verbose = verbose
        # Method proposed by Nyul et al. (2000)
        # some inspiration from https://www.kaggle.com/code/arminajdehnia/brain-tumour-preprocessing/notebook


    def train_standard_histogram(self, verbose=True, plot=False, stscore_first=False):
        print(f"TRAINING STANDARD NYUL HISTOGRAM FOR {'''p''' if self.pbool else '''no p'''} DATA WITH MASK_MODE {self.MASK_MODE}")
        j = 1
        PERC = [self.pc1, *self.L, self.pc2]    # percentiles for which IOI is defined
        print("LANDMARKS = ", PERC)
        SCALE_STD = np.zeros(len(PERC))          # standard scale (to be trained)
        if not any(self.L):
            print("NO L")
            modevals = []

        # for experiment in find_folders(RawDir, condition="Pilot"):    #TODO: INCLUDE "PILOT3" AFTER SEGMENTING IS DONE!!!
        # for experiment in ["Pilot1", "Pilot2", "Pilot3"]:
        print(find_folders(RawDir, condition="Pilot"))

        for experiment in find_folders(RawDir, condition="Pilot"):
            for time in find_folders(os.path.join(RawDir, experiment)):
                print("\n", experiment, time) if verbose else 0
                datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
                # print(find_folders(os.path.join(RawDir, experiment, time), condition="sagittal"))
                for folder in find_folders(os.path.join(RawDir, experiment, time), condition="sagittal"):
                    name = get_name(experiment, folder, condition="sagittal")

                    # Filter
                    if self.pbool:
                        includebool = "p" in name
                    else:
                        includebool = not("p" in name)

                    if includebool:
                        if self.T1bool:
                            includebool = "T1" in name
                        else:
                            includebool = not("T1" in name)

                    if includebool:
                        for offsetIDX in self.offsetIDX:
                            if offsetIDX.upper() == "L":
                                IDX = left_idx_dict[time + name]
                            elif offsetIDX.upper() == "R":
                                IDX = right_idx_dict[time + name]
                            else:
                                IDX = central_idx_dict[time + name] + self.offsetIDX
                            indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
                            MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder), printbool=False)
                            print(f"\nj={j}", time, name, f"center slice + offset {self.offsetIDX} @ ", IDX, "of", MATR_RAW.shape[0]) if verbose else 0
                            im_raw = MATR_RAW[IDX]
                            # plt.imshow(im_raw)
                            # plt.show()
                            if not self.n4bool:
                                img = im_raw
                            else:
                                img = n4correction(im_raw)[0]
                                # print(img.shape)

                            if stscore_first:
                                img = mean_centering(img)
                                print("\t--- standardizing image ---")
                            else:
                                pass

                            vals = self.masked_vals(img, experiment, time, name, IDX)
                            # if self.MASK_MODE == "none":
                            #     vals = img.ravel()
                            # elif self.MASK_MODE == "mean":
                            #     vals = img[img > np.mean(img)].ravel()
                            # elif self.MASK_MODE == "median":
                            #     vals = img[img > np.median(img)].ravel()
                            # elif self.MASK_MODE == "brain":
                            #     ROI_brain = np.load(os.path.join(SegBrainDir, experiment.lower(), time, name + ".npy"))
                            #     roi_brain = ROI_brain[IDX]
                            #     vals = img[roi_brain != 0].ravel()
                            # else:
                            #     print("MASK_MODE", self.MASK_MODE, "NOT VALID. CANNOT TRAIN STANDARD HISTOGRAM...")
                            #     return 0

                            if not any(self.L):
                                print("\tNO LANDMARKS SPECIFIED: FINDING MODE IN ", self.MASK_MODE) if (verbose or self.verbose) else 0
                                mode = self.find_mode(vals, plot=False, img=img)
                                print("\tmode = ", mode)
                                modevals.append(mode)

                            print(f"\t{len(vals)} of {len(img.ravel())} px kept ({len(vals) / len(img.ravel()) * 100 :.1f}%) with MASK_MODE", self.MASK_MODE) if verbose else 0
                            p1, p2 = np.percentile(vals, [self.pc1, self.pc2])
                            m1, m2 = min(vals), max(vals)
                            # MU = np.percentile(vals, self.L)    # landmark percentile values in (masked) image
                            MU = np.percentile(vals, self.L) if any(self.L) else [mode]   # landmark values in (masked) image, either from percentiles L or mode
                            print("m1, p1, p2, m2 = {}".format([round(x, 1) for x in [m1, p1, p2, m2]])) if verbose else 0
                            print(MU) if verbose else 0
                            tau_j = interp1d([p1, p2], [self.s1, self.s2], kind=self.mapmode, fill_value="extrapolate") # map function (default: linear)
                            SCALE_STD += tau_j([p1, *MU, p2])
                            # SCALE_IM = [m1, p1, *MU, p2, m2]
                            plt.plot([p1, *MU, p2], tau_j([p1, *MU, p2]), "o:") if plot else 0
                            # plt.show()
                            j += 1
        # print(SCALE_STD)
        num_img = j - 1
        SCALE_STD = SCALE_STD / num_img     # ROUNDED MEANS YIELDS TRAINED STANDARD SCALE ON LANDMARK VALUES
        self.SCALE_STD = SCALE_STD
        print(f"TRAINING COMPLETE FOR MODE {self.MASK_MODE} with pbool={self.pbool}. STANDARD SCALE FOR {num_img} images:", SCALE_STD)
        if not any(self.L):
            print("using MODE value found at ", self.MASK_MODE, f"having median, sd, [range]: {np.median(modevals):.3g}, {np.std(modevals):.3g}, [{np.min(modevals):.3g}, {np.max(modevals):.3g}]")
        if plot:
            plt.grid()
            plt.title(f"TRAINING DATA FOR NYUL NORM W/ {self.MASK_MODE} MASK ({'''no p''' if not self.pbool else '''on p'''}, N4corr {self.n4bool}): $\\tau_j$([$p_{{1j}}$, $p_{{2j}}$]) = [s1, s2] $\\forall$ j $\in$ 1, $n$\n"
                      f"where p1, p2 corresponds to percentiles pc1, pc2 = {self.pc1, self.pc2} and s1, s2 = {self.s1, self.s2}\n"
                      f"and $\\tau$ maps all landmark percentile values for L=[{self.pc1, *self.L, self.pc2}]")
            plt.xlabel("(masked) image scale")
            plt.ylabel("standard scale")
            plt.show()
        return SCALE_STD


    def masked_vals(self, img, experiment="", time=0, name="", IDX=0, ROI=[]):
        if self.MASK_MODE == "none":
            vals = img.ravel()
        elif self.MASK_MODE == "mean":
            vals = img[img > np.mean(img)].ravel()
        elif self.MASK_MODE == "median":
            vals = img[img > np.median(img)].ravel()
        # elif self.MASK_MODE == "brain":
        #     ROI_brain = np.load(os.path.join(SegBrainDir, experiment.lower(), time, name + ".npy"))
        #     roi_brain = ROI_brain[IDX]
        #     vals = img[roi_brain != 0].ravel()
        elif self.MASK_MODE == "brain":
            if not np.any(ROI):
                ROI_brain = np.load(os.path.join(SegBrainDir, experiment.lower(), time, name + ".npy"))
                IDX = self.offsetIDX + central_idx_dict[time+name]
                roi_brain = ROI_brain[IDX]
            else:
                roi_brain = ROI
            vals = img[roi_brain != 0].ravel()
        elif self.MASK_MODE == "saliv" or self.MASK_MODE == "salivary":
            ROI_saliv = np.load(os.path.join(SegSalivDir, experiment.lower(), time, name + ".npy"))
            roi_saliv = ROI_saliv[IDX]
            vals = img[roi_saliv != 0].ravel()
        elif self.MASK_MODE == "otsu":
            # print("\n\nOTSUH\n\n")
            msk = cv2.GaussianBlur(norm_minmax_featurescaled(img.copy(), printbool=False).astype("uint8"),
                                   ksize=(9, 9), sigmaX=0, sigmaY=0)
            msk = cv2.equalizeHist(msk)
            msk = sitk.OtsuThreshold(sitk.GetImageFromArray(msk), 0, 1, 210)
            msk = sitk.GetArrayFromImage(msk)
            # print(np.count_nonzero(msk))
            vals = img[msk != 0].ravel()
        else:
            print("MASK_MODE", self.MASK_MODE, "NOT VALID. CANNOT LOAD MASKED VALUES...")
            return 0
        return vals


    def find_mode(self, vals, plot=False, img=[]):
        kde = stats.gaussian_kde(vals, bw_method="scott")
        xx = np.linspace(min(vals), max(vals), 500)
        modeheight = max(kde(xx))
        xmax = xx[np.argwhere(kde(xx) == modeheight)[0, 0]]

        print("MAXIMUM OF", self.MASK_MODE, f"HIST FOUND AT {xmax:.3g}") if self.verbose else 0
        if plot:
            # kde2 = stats.gaussian_kde(vals, bw_method="silverman")
            plt.close()
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(xx, kde(xx), "--")
            ax[0].plot(xx, kde(xx), "--", label="scott")
            # ax[0].plot(xx, kde2(xx), ":", label="silverman")
            ax[0].hist(vals, density=True, bins=freedman_diaconis_rule(vals, return_nbins=True))
            ax[0].legend();   ax[0].grid()
            # modeheight = max(kde(xx))
            # xmax = xx[np.argwhere(kde(xx) == modeheight)[0, 0]]
            buffer = xmax * 0.1
            print(xmax, buffer)
            ax[0].set_title("mode " + self.MASK_MODE + f" xmax={xmax:.2g}, xbuff={buffer:.2g}")
            ax[0].vlines(x=[xmax - buffer, xmax, xmax + buffer], ymin=min(kde(xx)), ymax=max(kde(xx)), colors="red")
            ax[1].imshow(img, cmap="gray")

            # ax[1].imshow(np.ma.masked_where(img != xmax, img), cmap="hot")
            ax[1].imshow(np.ma.masked_where(np.abs(img - xmax) > buffer, img), cmap="hot")
            ax[1].set_title("Colored pixels have intensity within xmax +- xbuff")
            plt.show()
        return xmax


    def test_standardizer(self, verbose=True, t3=False):
        print(f"\n\nTESTING NYUL STANDARDIZER FOR {'''p''' if self.pbool else '''no p'''} DATA WITH MASK_MODE {self.MASK_MODE}")
        if not any(self.SCALE_STD):
            print("STANDARD SCALE FOR TRANSFORMING IMAGE NOT TRAINED - CANNOT EXECUTE")
            return 0
        #TODO: EMPLOY TEST ON SALIV ROI FROM BRAIN TRAINING
        #j = 1
        # PERC = [self.pc1, *self.L, self.pc2]    # percentiles for which IOI is defined
        # SCALE_STD = np.zeros(len(PERC))          # standard scale (to be trained)
        test1failcount = 0
        test2failcount = 0
        test3failcount = 0
        count = 0
        firstbool = True
        for experiment in find_folders(RawDir, condition="Pilot"):
            for time in find_folders(os.path.join(RawDir, experiment)):
                print("\n", experiment, time) if verbose else 0
                datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
                # print(find_folders(os.path.join(RawDir, experiment, time), condition="sagittal"))
                for folder in find_folders(os.path.join(RawDir, experiment, time), condition="sagittal"):
                    name = get_name(experiment, folder, condition="sagittal")
                    if self.pbool:
                        includebool = "p" in name
                    else:
                        includebool = not("p" in name)
                    if includebool:
                        count += 1
                        IDX = central_idx_dict[time + name] + self.offsetIDX
                        indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
                        MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder), printbool=False)
                        print(f"\nj={count}", name, f"center slice + offset {self.offsetIDX} @ ", IDX, "of", MATR_RAW.shape[0]) if verbose else 0
                        im_raw = MATR_RAW[IDX]

                        if not self.n4bool:
                            img = im_raw
                        else:
                            img = n4correction(im_raw)[0]

                        if self.MASK_MODE == "none":
                            vals = img.ravel()
                        elif self.MASK_MODE == "mean":
                            vals = img[img > np.mean(img)].ravel()
                        elif self.MASK_MODE == "median":
                            vals = img[img > np.median(img)].ravel()
                        elif self.MASK_MODE == "brain":
                            ROI_brain = np.load(os.path.join(SegBrainDir, experiment.lower(), time, name + ".npy"))
                            roi_brain = ROI_brain[IDX]
                            vals = img[roi_brain != 0].ravel()
                        else:
                            print("MASK_MODE", self.MASK_MODE, "NOT VALID. CANNOT TRAIN STANDARD HISTOGRAM...")
                            return 0
                        if not any(self.L):
                            print("NO LANDMARKS SPECIFIED: FINDING MODE IN ", self.MASK_MODE)   if self.verbose else 0
                            mode = self.find_mode(vals, plot=False, img=img)
                            LANDMRKS = [np.percentile(vals, self.pc1), mode, np.percentile(vals, self.pc2)]
                        else:
                            LANDMRKS = np.percentile(vals, self.LANDMARK_PERC)  # landmarks in image
                        print(LANDMRKS)
                        tau = interp1d(LANDMRKS, self.SCALE_STD, kind=self.mapmode,
                                       fill_value="extrapolate")  # mapping function
                        p1, p2 = np.percentile(vals, [self.pc1, self.pc2])
                        # m1, m2 = np.min(vals), np.max(vals)
                        mu = np.percentile(vals, self.L[0]) # TODO: add loop (extend to all landmarks)
                        musd = tau(mu)
                        # print(mu, musd)
                        # musd = mu
                        if firstbool:
                            firstbool = False
                            delta_l = mu - p1
                            delta_L = mu - p1
                            delta_r = p2 - mu
                            delta_R = p2 - mu
                            mu_min = musd
                            mu_max = musd
                        else:
                            delta_l = mu - p1 if (delta_l > mu - p1) else delta_l
                            delta_L = mu - p1 if (delta_L < mu - p1) else delta_L
                            delta_r = p2 - mu if (delta_r > p2 - mu) else delta_r
                            delta_R = p2 - mu if (delta_R < p2 - mu) else delta_R
                            mu_min = musd if (mu_min > musd) else mu_min
                            mu_max = musd if (mu_max < musd) else mu_max

                        print("mumin / mumax", mu_min, mu_max)
                        print("dl, dL, dr, dR = {}".format(
                            [round(x, 1) for x in [delta_l, delta_L, delta_r, delta_R]]))
                        print("test1a: LHS=", mu_min - self.s1, "RHS=", delta_L)
                        print("test1b: LHS=", self.s2 - mu_max, "RHS=", delta_R)
                        print("test2: LSH=", self.s2 - self.s1, "RHS=", (delta_L + delta_R) * max(delta_L / delta_l, delta_R / delta_r))
                        test1 = mu_min - self.s1 >= delta_L and self.s2 - mu_max >= delta_R
                        test2 = self.s2 - self.s1 >= (delta_L + delta_R) * max(delta_L / delta_l, delta_R / delta_r)
                        if t3:
                            test3 = True
                            for x1 in vals:
                                for x2 in vals:
                                    if x1 < x2:
                                        test3 = tau(x1) < tau(x2)
                                    if not test3:
                                        break
                                if not test3:
                                    break
                            print("test3", test3)
                        else:
                            test3 = True
                        if not (test1 and test2 and test3):
                            print("TESTS NOT OK: T1, T2, T3:", test1, test2, test3)
                            test1failcount = test1failcount + 1 if not test1 else test1failcount
                            test2failcount = test2failcount + 1 if not test2 else test2failcount
                            test3failcount = test3failcount + 1 if not test3 else test3failcount

        print(f"TEST1: FAILED {test1failcount} of {count} times..")
        print(f"TEST2: FAILED {test2failcount} of {count} times..")
        print(f"TEST3: FAILED {test3failcount} of {count} times..") if t3 else print("TEST3 NOT RUN.")
        print(f"for {'''p''' if self.pbool else '''no p'''} data with MASK_MODE {self.MASK_MODE}"
              f"\nST SCALE = {self.SCALE_STD},"
              f"\nL = {self.pc1, *self.L, self.pc2},"
              f"\ns1, s2 = {self.s1, self.s2}")
        return 0


    def transform_image(self, image, experiment="", time="", name="", verbose=True, ROI=[]):
        if not any(self.SCALE_STD):
            print("STANDARD SCALE FOR TRANSFORMING IMAGE NOT TRAINED - CANNOT EXECUTE")
            return 0
        print("     --- TRANSFORMING IMAGE W NYUL ---") if verbose else 0
        # print(np.any(ROI))
        try:
            IDX = central_idx_dict[time + name] + self.offsetIDX if not np.any(ROI) else 0
        except Exception as e:
            print(*e.args)
            IDX = 0
        vals = self.masked_vals(image, experiment, time, name, IDX, ROI)

        if not any(self.L):
            print("NO LANDMARKS SPECIFIED: FINDING MODE IN ", self.MASK_MODE) if self.verbose else 0
            mode = self.find_mode(vals, plot=False, img=image)
            LANDMRKS = [np.percentile(vals, self.pc1), mode, np.percentile(vals, self.pc2)]
        else:
            LANDMRKS = np.percentile(vals, self.LANDMARK_PERC)  # landmarks in image
        # print(LANDMRKS)
        # print(self.SCALE_STD)
        tau = interp1d(LANDMRKS, self.SCALE_STD, kind=self.mapmode,
                       fill_value="extrapolate")  # mapping function
        image_norm = tau(image)
        return image_norm


    def normalize_all_images(self, verbose=False, plot=True): # helper function: looping over same images as for training
        print("--- NYUL NORMALIZING IMAGES ---")
        j = 1
        for experiment in find_folders(RawDir, condition="Pilot"):
            for time in find_folders(os.path.join(RawDir, experiment)):
                print("\n", experiment, time) if verbose else 0
                datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
                # print(find_folders(os.path.join(RawDir, experiment, time), condition="sagittal"))
                for folder in find_folders(os.path.join(RawDir, experiment, time), condition="sagittal"):
                    name = get_name(experiment, folder, condition="sagittal")
                    if self.pbool:
                        includebool = "p" in name
                    else:
                        includebool = not("p" in name)
                    if includebool:
                        IDX = central_idx_dict[time + name] + self.offsetIDX
                        indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
                        MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder), printbool=False)
                        print(f"\nj={j}", name, f"center slice + offset {self.offsetIDX} @ ", IDX, "of", MATR_RAW.shape[0]) if verbose else 0
                        im_raw = MATR_RAW[IDX]
                        if self.n4bool:
                            img = n4correction(im_raw)[0]
                        else:
                            img = im_raw
                        im_norm = self.transform_image(img, experiment=experiment, time=time, name=name, verbose=verbose)

                        fig, ax = plt.subplots(1, 2)
                        ax[0].imshow(im_raw, cmap="hot")
                        ax[1].imshow(im_norm, cmap="hot")
                        plt.show()

def nyul_initializer(norm, n4=True):
    pc1, pc2, s1, s2 = 2, 98, 1, 5e4
    if norm == "nyul otsu decile":
        return nyul_normalizer(pbool=True, MASK_MODE="otsu", n4bool=n4, L=np.arange(10, 100, 10), pc1=pc1,
                                            pc2=pc2, s1=s1, s2=s2)
    else:
        print("NORM", norm, "NOT TEMPLATED YET")
        return 0


if __name__ == "__main__":
    offsetIDX = 0
    includep = False

    # nbins = 256
    # histogram_trainer()
    L = np.arange(10, 100, 10)  # DECILES
    # L = [0]  # MODE


    # AGGREGATE L / R SMG SLICES: FIND STANDARD SCALES
    # T2 images
    # nrmny_otsu_decile = nyul_normalizer(pbool=True, MASK_MODE="otsu", offsetIDX="LR", n4bool=True, T1bool=False, L=L, pc1=2,
    #                                     pc2=98, s1=1, s2=5e4)
    # scale_p_T2 = nrmny_otsu_decile.train_standard_histogram()  # 103 images

    nrmny_otsu_decile = nyul_normalizer(pbool=False, MASK_MODE="otsu", offsetIDX="LR", n4bool=True, T1bool=False, L=L,
                                        pc1=2, pc2=98, s1=1, s2=5e4)
    scale_nop_T2 = nrmny_otsu_decile.train_standard_histogram()  # 103 images

    # T1 images
    # nrmny_otsu_decile = nyul_normalizer(pbool=True, MASK_MODE="otsu", offsetIDX="LR", n4bool=True, T1bool=True, L=L, pc1=2,
    #                                     pc2=98, s1=1, s2=5e4)
    # scale_p_T1 = nrmny_otsu_decile.train_standard_histogram()  # 103 images

    nrmny_otsu_decile = nyul_normalizer(pbool=False, MASK_MODE="otsu", offsetIDX="LR", n4bool=True, T1bool=True, L=L,
                                        pc1=2, pc2=98, s1=1, s2=5e4)
    scale_nop_T1 = nrmny_otsu_decile.train_standard_histogram()  # 103 images

    # print("\nT2 p SCALE:", scale_p_T2)
    print("\nT2 nop SCALE:", scale_nop_T2)
    # print("\nT1 p SCALE:", scale_p_T1)
    print("\nT1 nop SCALE:", scale_nop_T1)


    # CHECKING TO SEE IF L / R SMG SPLIT YIELDS DIFFERENT SCALES
    # # T2 images
    # nrmny_otsu_decile = nyul_normalizer(pbool=True, MASK_MODE="otsu", offsetIDX="L", n4bool=True, T1bool=False, L=L, pc1=2,
    #                                     pc2=98, s1=1, s2=5e4)
    # scale_p_T2_L = nrmny_otsu_decile.train_standard_histogram()  # 103 images
    #
    # nrmny_otsu_decile = nyul_normalizer(pbool=True, MASK_MODE="otsu", offsetIDX="R", n4bool=True, T1bool=False, L=L, pc1=2,
    #                                     pc2=98, s1=1, s2=5e4)
    # scale_p_T2_R = nrmny_otsu_decile.train_standard_histogram()  # 103 images
    #
    # nrmny_otsu_decile = nyul_normalizer(pbool=False, MASK_MODE="otsu", offsetIDX="L", n4bool=True, T1bool=False, L=L,
    #                                     pc1=2, pc2=98, s1=1, s2=5e4)
    # scale_nop_T2_L = nrmny_otsu_decile.train_standard_histogram()  # 103 images
    #
    # nrmny_otsu_decile = nyul_normalizer(pbool=False, MASK_MODE="otsu", offsetIDX="R", n4bool=True, T1bool=False, L=L,
    #                                     pc1=2, pc2=98, s1=1, s2=5e4)
    # scale_nop_T2_R = nrmny_otsu_decile.train_standard_histogram()  # 103 images
    #
    # # T1 images
    # nrmny_otsu_decile = nyul_normalizer(pbool=True, MASK_MODE="otsu", offsetIDX="L", n4bool=True, T1bool=True, L=L, pc1=2,
    #                                     pc2=98, s1=1, s2=5e4)
    # scale_p_T1_L = nrmny_otsu_decile.train_standard_histogram()  # 103 images
    #
    # nrmny_otsu_decile = nyul_normalizer(pbool=True, MASK_MODE="otsu", offsetIDX="R", n4bool=True, T1bool=True, L=L, pc1=2,
    #                                     pc2=98, s1=1, s2=5e4)
    # scale_p_T1_R = nrmny_otsu_decile.train_standard_histogram()  # 103 images
    #
    # nrmny_otsu_decile = nyul_normalizer(pbool=False, MASK_MODE="otsu", offsetIDX="L", n4bool=True, T1bool=True, L=L,
    #                                     pc1=2, pc2=98, s1=1, s2=5e4)
    # scale_nop_T1_L = nrmny_otsu_decile.train_standard_histogram()  # 103 images
    #
    # nrmny_otsu_decile = nyul_normalizer(pbool=False, MASK_MODE="otsu", offsetIDX="R", n4bool=True, T1bool=True, L=L,
    #                                     pc1=2, pc2=98, s1=1, s2=5e4)
    # scale_nop_T1_R = nrmny_otsu_decile.train_standard_histogram()  # 103 images
    #
    # print("\nT2 p SCALE L:", scale_p_T2_L)
    # print("\nT2 p SCALE R:", scale_p_T2_R)
    # print("\nT2 nop SCALE L:", scale_nop_T2_L)
    # print("\nT2 nop SCALE R:", scale_nop_T2_R)
    # print("\nT1 p SCALE L:", scale_p_T1_L)
    # print("\nT1 p SCALE R:", scale_p_T1_R)
    # print("\nT1 nop SCALE L:", scale_nop_T1_L)
    # print("\nT1 nop SCALE R:", scale_nop_T1_R)


    # nrmny_otsu_decile = nyul_normalizer(pbool=False, MASK_MODE="otsu", n4bool=True, T1bool=False, L=L, pc1=2,
    #                                     pc2=98, s1=1, s2=5e4)
    # scale_nop_T2 = nrmny_otsu_decile.train_standard_histogram()
    #
    # #T1 images
    # nrmny_otsu_decile = nyul_normalizer(pbool=True, MASK_MODE="otsu", n4bool=True, T1bool=True,
    #                                     L=L, pc1=2,
    #                                     pc2=98, s1=1, s2=5e4)
    # scale_p_T1 = nrmny_otsu_decile.train_standard_histogram()  # 103 images
    # nrmny_otsu_decile = nyul_normalizer(pbool=False, MASK_MODE="otsu", n4bool=True, T1bool=True,
    #                                     L=L, pc1=2,
    #                                     pc2=98, s1=1, s2=5e4)
    # scale_nop_T1 = nrmny_otsu_decile.train_standard_histogram()
    # print("\nT2 SCALE P:", scale_p_T2)
    # print("\nT2 SCALE NOP:", scale_nop_T2)
    # print("\nT1 SCALE P:", scale_p_T1)
    # print("\nT1 SCALE NOP:", scale_nop_T1)


    # nrmny = nyul_normalizer(pbool=False, MASK_MODE="brain", n4bool=True, L=[25, 50, 75], pc1=1, pc2=99, s1=1, s2=256)
    # nrmny = nyul_normalizer(pbool=False, MASK_MODE="brain", n4bool=True, L=[50], pc1=5, pc2=95, s1=1, s2=2e4)
    # nrmny.SCALE_STD = [0., 27669.82005691, 100000.]
    # nrmny = nyul_normalizer(pbool=False, MASK_MODE="brain", n4bool=True, L=np.arange(10, 100, 10), pc1=2, pc2=98, s1=1, s2=5e4)
    # nrmny.SCALE_STD = [1.0, 5.48630905e+03, 8.11602435e+03, 1.00209666e+04, 1.17940187e+04, 1.37388780e+04, 1.62289771e+04, 2.01914164e+04, 2.55586951e+04, 3.28261892e+04, 5.0e+04]


    # nrmny = nyul_normalizer(pbool=True, MASK_MODE="brain", n4bool=True, L=np.arange(10, 100, 10), pc1=2, pc2=98, s1=1, s2=5e4)
    # nrmny.SCALE_STD = [1.0, 5.51478431e+03, 8.49028859e+03, 1.06895568e+04, 1.26494816e+04, 1.47360021e+04, 1.72168060e+04, 2.06242499e+04, 2.50680921e+04, 3.31180883e+04, 5.0e+04]


    # nrmny = nyul_normalizer(pbool=False, MASK_MODE="brain", n4bool=True, L=[10, 20, 30, 40, 50, 60, 70, 80, 90], pc1=1, pc2=99, s1=1, s2=256)
    # nrmny.SCALE_STD = [1., 34.95178368, 46.19512203, 54.35308416, 61.94424755, 70.2827526, 80.98071465, 98.03055332, 121.14998526, 152.60979797, 256.]   # brain, no p, with N4

    # nrmny = nyul_normalizer(pbool=False, MASK_MODE="brain", n4bool=True, L=[10, 20, 30, 40, 50, 60, 70, 80, 90], pc1=1, pc2=99, s1=1, s2=256)


    # nrmny = nyul_normalizer(pbool=False, MASK_MODE="brain", n4bool=True, L=[], pc1=2, pc2=98, s1=1, s2=5e4)
    # nrmny.train_standard_histogram(verbose=True, plot=True)
    # nrmny.SCALE_STD = [1.0, 1.10861289e+04, 5.0e+04]

    # nrmny = nyul_normalizer(pbool=True, MASK_MODE="brain", n4bool=True, L=[], pc1=2, pc2=98, s1=1, s2=5e4)
    # nrmny.train_standard_histogram(plot=True)
    # nrmny.SCALE_STD = [1.0, 1.22860372e+04, 5.0e+04]

    # nrmny.test_standardizer(t3=True)
    # nrmny.test_standardizer(t3=False)

    # nrmny.normalize_all_images(verbose=True, plot=True)



    # for exp in ["Pilot1", "Pilot2"]:
    #     for time in os.listdir(os.path.join(RawDir, exp)):
    #         print("\n", exp, time)
    #         datadir = os.path.normpath(os.path.join(RawDir, exp, time))
    #         for i, folder in enumerate(find_folders(datadir, "sagittal")):
    #             name = get_name(exp, folder, "sagittal")
    #             pbool = "p" in name
    #
    #             if not pbool:
    #                 idx_center = central_idx_dict[time+name]
    #                 print(name)
    #                 indexed = dcm_files_indexed(os.path.join(datadir, folder), printbool=False)
    #                 MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(datadir, folder), printbool=False)
    #                 im_raw = MATR_RAW[idx_center + offsetIDX]
    #                 # ROI_saliv = np.load(os.path.join(SegSalivDir, exp.lower(), time, name + ".npy"))
    #                 # roi_saliv = ROI_saliv[idx_center + offsetIDX]
    #                 # ROI_brain = np.load(os.path.join(SegBrainDir, exp.lower(), time, name + ".npy"))
    #                 # roi_brain = ROI_brain[idx_center + offsetIDX]
    #
    #                 # vals = im_raw.ravel()
    #                 # vals = im_raw[im_raw >= np.mean(im_raw)].ravel()
    #                 vals = im_raw[im_raw >= np.median(im_raw)].ravel()
    #                 print(len(vals))
    #                 # masked = np.ma.masked_where(im_raw > np.mean(im_raw), im_raw)
    #                 # im_thresh = np.where(im_raw > np.mean(im_raw), im_raw, 0)
    #                 fig, axes = plt.subplots(ncols=2, nrows=2)
    #                 fig.tight_layout()
    #                 ax1, ax3, ax2, ax4 = axes.ravel()
    #
    #                 masked_mean = np.ma.masked_where(im_raw < np.mean(im_raw), im_raw)
    #                 masked_median = np.ma.masked_where(im_raw < np.median(im_raw), im_raw)
    #                 cmap = plt.get_cmap("gray").copy()
    #                 cmap.set_bad(color="blue")
    #                 # plt.imshow(np.where(im_raw > np.mean(im_raw), im_raw, 0), cmap="gray")
    #                 ax1.imshow(masked_mean, cmap=cmap)
    #                 ax2.imshow(masked_median, cmap=cmap)
    #                 [ax.axis("off") for ax in (ax1, ax2)]
    #                 ax1.set_title("Mean thresh")
    #                 ax2.set_title("Median thresh")
    #                 vals = im_raw[im_raw > np.mean(im_raw)].ravel()
    #                 ax3.plot(np.linspace(min(vals), max(vals), nbins), np.histogram(vals, bins=nbins)[0], label=f"{len(vals)} px in mask")
    #                 vals = im_raw[im_raw > np.median(im_raw)].ravel()
    #                 ax4.plot(np.linspace(min(vals), max(vals), nbins), np.histogram(vals, bins=nbins)[0], label=f"{len(vals)} px in mask")
    #                 # [ax.legend() for ax in (ax3, ax4)]
    #                 for ax in (ax3, ax4):   ax.legend(); ax.grid(1)
    #                 plt.show()
    #                 # hist = np.histogram(vals, bins=nbins)[0]
    #                 # plt.plot(np.linspace(min(vals), max(vals), nbins), hist)
    # plt.show()

