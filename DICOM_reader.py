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
    time = "071220" #change this to correct day - 57day???
    main = os.path.normpath(os.path.join(os.getcwd(), "..", "RAW DATA\Pilot1\MRI", time))
    plane = "sagittal"


    #PILOT 2 DCM STUFF
    main = os.path.join(os.getcwd(), "..", "RAW DATA\Pilot2")
    for time in os.listdir(main):
        print("\n", time)
        for mouse in find_folders(os.path.join(main, time), condition="sagittal"):
            name = (mouse[12:15] + mouse[15:24]).replace("_", " ") if "p" in mouse else mouse[12:15]

            #how to collect individual name (with MRI timing according to pilocarpine injection)
            dcmfiles = dcm_files_indexed(os.path.join(main, time, mouse), printbool=False)
            print(len(dcmfiles), name)

