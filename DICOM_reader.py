import pydicom; from pydicom.fileset import FileSet
import os
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from preprocessing import center_and_crop

# print(__name__)


def dcm_files_indexed(path, printbool=True):    #return vector containing ["filename.dcm", SliceLocation] for each dcm image in path
    folder_files = os.listdir(path)                                             #get files in path as list of strings
    if printbool:
        print(len(folder_files),"files found in path: {}".format(path))
    vector = []
    for file in folder_files:
        if(file[-3:] == "dcm" and not(file[:2] == "._")):    #some MRI image files start with ._ which makes trouble :)
            # print(file)
            with pydicom.dcmread(path + r"\\" + file) as ds:
                vector.append([file, float(ds.SliceLocation)])
            # print(file, ds.SliceLocation)
    if printbool:
        print("Shape of indexed_files =", np.shape(vector), "i.e.", len(vector), "usable dcm files found.")
    return sorted(vector, key=lambda l:l[1])


def dcm_folder_to_pixel_matrix(indexed_files, folder_path):
    print("Making pixel array of {} files.".format(len(indexed_files)))
    pixel_matrix = []
    for filename in np.array(indexed_files).T[0]:
        filepath = folder_path + r"\\" + filename
        with pydicom.dcmread(filepath) as ds:
            pixel_matrix.append(ds.pixel_array)
    print("Shape of pixel array = ", np.shape(pixel_matrix))
    return np.array(pixel_matrix)


def dcm_path_to_single_pixel_matrix(path):
    print("Getting pixel array of file in path: ", path)
    with pydicom.dcmread(path) as ds:
        m = ds.pixel_array
        print("of shape", m.shape, "dtype", m.dtype)
        return ds.pixel_array


def print_ndstats(*matrices):
    for m in matrices:
        try:
            print(m.shape, "array of type", m.dtype,
                  "\t[min / max] = [{1:.2f} / {0:.2f}], \tmu / std = {2:.2f} / {3:.2f}".format(np.max(m), np.min(m), np.mean(m), np.std(m)))
            # print(f"{m.shape} array of type {m.dtype} \t with range [min / max] = [{np.min(m)} / {np.max(m)}], mu = {np.mean:.5f}, sd = {np.std(m):.5f}")
        except Exception as e:
            print("print_ndstats failed:", e.args)
        pass


def voxel_pos(ds):  #convert position info from 2D slices to 3D voxel position (append to voxel location matrix)
    im_pos = np.array(ds.ImagePositionPatient)
    spacing = np.array(ds.PixelSpacing)
    row_vec = np.array(ds.ImageOrientationPatient[:3]).astype("int")
    col_vec = np.array(ds.ImageOrientationPatient[3:]).astype("int")
    rows, columns = ds.Rows, ds.Columns
    Points = np.zeros(shape=(rows, columns))
    M = np.array([])        #put stuff in M
    # Indexes = np.array([i, j, 0, 1])    #for all i,j in Points -- no sum please
    # P = np.matmul(M, Indexes)           #linear algebra
    return M


def find_folders(main, condition=""):     #return stuff with "condition" in name as list
    stuff = os.listdir(main)
    # print(main, stuff)
    folders = []
    for s in stuff:
        if condition in s:
            # os.path.join(s)
            folders.append(s)
    return folders


def make_ndarrays_from_folders(parent, target, cond="", time="", cropped=False):
    for folder in find_folders(parent, condition=cond):
        path = parent + "\\" + folder
        print(path)
        ind = dcm_files_indexed(path)
        pxm = dcm_folder_to_pixel_matrix(ind, path)
        if cropped:
            center_and_crop(pxm, target, time=time, title=folder)
    pass


if __name__ == "__main__":
    # mainfolder = r"G:\OneDrive - Universitetet i Oslo\Pilot_LateEffects_-7day"
    # parent = r"G:\OneDrive - Universitetet i Oslo\Pilot_LateEffects_-7day"
    # target = r"G:\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary\raw"

    # time = "-7day"
    # time = "8day"
    time = "071220" #change this to correct day - 57day???
    main = os.path.normpath(os.path.join(os.getcwd(), "..", "RAW DATA\Pilot1\MRI", time))
    plane = "sagittal"
    # plane = "transverse"
    # make_ndarrays_from_folders(parent, target, cond="sagittal", time="-7day", cropped=True)
    # for folder in find_folders(main, plane):
    #     path = os.path.normpath(os.path.join(main, folder))
    #     print(path)
    #     # for file in dcm_files_indexed(path):
    #     #     print(file)
    #         # if folder == "C2_sagittal" and file[0] == "MRIm11.dcm":
    #         #     print(folder, file[0])
    #         #     filepath = os.path.join(main, folder, file[0])
    #         #     ds = pydicom.dcmread(filepath)
    #         #     print(ds)
    #     dcm0 = dcm_files_indexed(path, printbool=False)[0][0];   dcm1 = dcm_files_indexed(path, printbool=False)[1][0]
    #     ds0 = pydicom.dcmread(os.path.join(main, folder, dcm0));   ds1 = pydicom.dcmread(os.path.join(main, folder, dcm1))
    #     print(ds1.SliceLocation - ds0.SliceLocation, ds0.SliceThickness)
    #     print(ds0.PixelSpacing, ds0.ImageOrientationPatient)
        # print(ds0.ImagePositionPatient)

    #PILOT 2 DCM STUFF
    main = os.path.join(os.getcwd(), "..", "RAW DATA\Pilot2")
    for time in os.listdir(main):
        print("\n", time)
        for mouse in find_folders(os.path.join(main, time), condition="sagittal"):
            name = (mouse[12:15] + mouse[15:24]).replace("_", " ") if "p" in mouse else mouse[12:15]    #how to collect individual name (with MRI timing according to pilocarpine injection)
            dcmfiles = dcm_files_indexed(os.path.join(main, time, mouse), printbool=False)
            print(len(dcmfiles), name)

            # print(mouse[12:15])
                # print(p)
                # idx = 0
                # print(len(dcmfiles), p)

                # ds = pydicom.dcmread(os.path.join(main, time, mouse, dcmfiles[idx][0]))
                # print(ds.ImageOrientationPatient, len(dcmfiles), mouse)
                # if ds.ImageOrientationPatient == [0, 1, 0, 0, 0, -1]:
                # if ds.ImageOrientationPatient == [0, 1, 0, 0, 0, -1] and not mouse == "210315_Olga_4-1__E2_P1":
                # print(ds.ImageOrientationPatient, len(dcmfiles), mouse)
                    # p = os.path.join(main, time, mouse)
                    # os.rename(p, p + "_all_planes")
    #                 idx = 14
    #                 ds = pydicom.dcmread(os.path.join(main, time, mouse, dcmfiles[idx][0]))
    #                 fig, ax = plt.subplots()
    #                 ax.imshow(ds.pixel_array, cmap="gray")
    #                 ax.set_title(mouse)
    # plt.show()
# p = r"C:\Users\toral\OneDrive - Universitetet i Oslo\RAW DATA\Pilot2\-7day\210315_Olga_1-1__E1_P1"
# os.rename(p, p+"_sagittal")

    # for file in indexed_files:
    #     ds = pydicom.dcmread(path + r"\\" + file[0])
    #     print(file[0], ds.SpacingBetweenSlices) #ds.MagneticFieldStrength)

    #ds.SpacingBetweenSlices - distance from center to center of slices

    # image_num = 20
    # ds = pydicom.dcmread(path + r"\\" + indexed_files[image_num][0])
    # print((ds))
    # print("slice spacing =", ds.SpacingBetweenSlices, "slice thickness = ", ds.SliceThickness)
    # print(ds.NumberOfPhaseEncodingSteps)
# for file in indexed_files:
#     with pydicom.read_file(path + file[0]) as ds:
#         print(ds.ImagePositionPatient)


# files = os.listdir(path)    #get files in path as list of strings
# for f in indexed_files: print(f)
# print("shape of indexed_files =", np.shape(indexed_files))


# print("image [0] = ", ds)
# plt.imshow(ds.pixel_array, cmap=plt.cm.bone);
# print(indexed_files[image_num])
# plt.title("Image: {}, SliceLoc = {}".format(*indexed_files[image_num]))
# plt.show()


#fs = FileSet(path_mouseIR)
# for file in dirs:
#     print(file)