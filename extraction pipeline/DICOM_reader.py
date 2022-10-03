from extract_utils import *
import pydicom; from pydicom.fileset import FileSet
import os
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
# from preprocessing import center_and_crop


# print(__name__)


def dcm_files_indexed(path=os.path.join(os.getcwd(), "..", ".."), printbool=False):    #return vector containing ["filename.dcm", SliceLocation] for each dcm image in path
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


def dcm_folder_to_pixel_matrix(indexed_files=dcm_files_indexed(), folder_path=os.path.join(os.getcwd(), "..", ".."), printbool=True):
    print("Making pixel array of {} files.".format(len(indexed_files))) if printbool else 0
    pixel_matrix = []
    for filename in np.array(indexed_files).T[0]:
        filepath = folder_path + r"\\" + filename
        with pydicom.dcmread(filepath) as ds:
            pixel_matrix.append(ds.pixel_array)
    print("Shape of pixel array = ", np.shape(pixel_matrix))    if printbool else 0
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


def find_folders(main, condition="", exclude_condition=False):     #return stuff with "condition" in name as list
    stuff = os.listdir(main)
    # print(main, stuff)
    folders = []
    for s in stuff:
        if condition in s and not exclude_condition:
            # os.path.join(s)
            folders.append(s)
        elif not condition in s and exclude_condition:
            folders.append(s)
    return folders


def make_ndarrays_from_folders(parent, target, cond="", time="", cropped=False):
    for folder in find_folders(parent, condition=cond):
        path = parent + "\\" + folder
        print(path)
        ind = dcm_files_indexed(path)
        pxm = dcm_folder_to_pixel_matrix(ind, path)
        if cropped:
            center_and_crop(pxm, target, time=time, title=folder)       #THIS IS PREPROCESSING
    pass


def load_matrix(path, expected=np.array([]), printtext=True):
    loaded = np.load(path)
    # print(type(loaded))
    if expected.any():
        if (loaded == expected).all():
            print("Loaded matrix is exactly the same as expected matrix.")
        else:
            print("Loaded matrix is not the same as expected.")
    elif printtext:
        print(f"\nFILE LOADED from {path}: No expected matrix to compare with. Shape = {loaded.shape}, dtype = {loaded.dtype}")
    else:
        pass
    return loaded


def count_all_data_stuff():
    RawDir = os.path.join(os.getcwd(), "..", "..", "RAW DATA")
    times = set()
    data = dict()
    slice_distances = set()
    inplane_distances = set()
    num_slices = set()
    protocol_names = set()
    import pandas as pd
    TIMES = ["-7day", "-3day", "5day", "8day", "12day", "26day", "35day", "56day", "70day", "105day"]
    ntimes = len(TIMES)
    time_string = "Day " + ", ".join([t[:-3] for t in TIMES])

    # data = pd.DataFrame(columns={"-7day", "5day", "8day", "26day", "56day", "105day"}, index={"P1/ P2/ P3", "T1 / T2", "p / no p"}, data=np.zeros(shape=(3, 6)),  dtype=int)
    # data = pd.DataFrame(columns=["-7day", "5day", "8day", "26day", "56day", "105day", "all"], index=["tot", "Exp 1/ 2/ 3", "T1 / T2", "p / no p"], data=[[0]*7,
    # data = pd.DataFrame(columns=["-7day", "-3day", "5day", "8day", "26day", "56day", "105day", "all"], index=["tot", "Exp 1/ 2/ 3/ 4", "T1 / T2", "p / no p"], data=[[0]*8, np.array([[0, 0, 0, 0]]*8), np.array([[0, 0]]*8), np.array([[0, 0]]*8)])
    data = pd.DataFrame(columns=[*TIMES, "all"], index=["tot", "Exp 1/ 2/ 3/ 4", "T1 / T2", "p / no p"], data=[[0]*(ntimes + 1), np.array([[0, 0, 0, 0]]*(ntimes + 1)), np.array([[0, 0]]*(ntimes + 1)), np.array([[0, 0]]*(ntimes + 1))])
    print(data)
    # data2 = pd.DataFrame(columns=["T1", "T2", "all"], index=["tot", "Exp 1/ 2/ 3", "p / no p"], data=[[0]*3, np.array([[0, 0, 0]]*3), np.array([[0, 0]]*3)])
    # data2 = pd.DataFrame(columns=["T1", "T2", "all"], index=["tot", "Exp 1/ 2/ 3/ 4", "p / no p", "Day -7, -3, 5, 8, 12, 26, 35, 56, 105"], data=[[0]*3, np.array([[0, 0, 0, 0]]*3), np.array([[0, 0]]*3), np.array([[0]*(ntimes)]*3)])
    data2 = pd.DataFrame(columns=["T1", "T2", "all"], index=["tot", "Exp 1/ 2/ 3/ 4", "p / no p", time_string], data=[[0]*3, np.array([[0, 0, 0, 0]]*3), np.array([[0, 0]]*3), np.array([[0]*(ntimes)]*3)])
    # print(data2.T)
    time_index = {}
    for t, i in zip(TIMES, range(ntimes)):
        time_index[t] = i
    print(time_index)
    # time_index = {"-7day":0, "-3day":1, "5day":2, "8day":3, "26day":4, "56day":5, "105day":6}

    # for nexp, exp in enumerate(["Pilot1", "Pilot2", "Pilot3"]):
    j = 0
    for nexp, exp in enumerate(find_folders(RawDir, condition="Pilot")):
        ExpDir = os.path.join(RawDir, exp)
        for time in os.listdir(ExpDir):
            # print(exp, time)
            TimeDir = os.path.join(ExpDir, time)
            # data[time] = 1
            for file in find_folders(TimeDir, condition="sagittal"):
                # print(file)
                j += 1
                path = os.path.join(TimeDir, file)
                name = get_name(exp, file, condition="sagittal")

                data["all"]["tot"] += 1
                data[time]["tot"] += 1

                data["all"]["Exp 1/ 2/ 3/ 4"][nexp] += 1
                data[time]["Exp 1/ 2/ 3/ 4"][nexp] += 1

                data2["all"]["tot"] += 1
                data2["all"]["Exp 1/ 2/ 3/ 4"][nexp] += 1
                data2["all"][time_string][time_index[time]] += 1
                indexed = dcm_files_indexed(path)
                num_slices.add(len(indexed))
                print(j, exp, time, time_index[time], name, len(indexed))
                ds = pydicom.dcmread(os.path.join(path, indexed[0][0]))
                # print(name, ds.ProtocolName)
                prot = ds.ProtocolName
                pxspacing = ds.PixelSpacing
                slice_dist = ds.SliceThickness
                slice_spacing = ds.SpacingBetweenSlices
                # print(slice_dist, pxspacing, slice_spacing)
                protocol_names.add(prot)
                slice_distances.add(slice_dist)
                slice_distances.add(slice_spacing)
                for spc in pxspacing:
                    inplane_distances.add(spc)
                # print(protocol_names)
                if "T1" in prot:
                    t1bool = True
                    # print("t1", prot)
                    data[time]["T1 / T2"][0] += 1
                    data["all"]["T1 / T2"][0] += 1

                    data2["T1"]["Exp 1/ 2/ 3/ 4"][nexp] += 1
                    data2["T1"]["tot"] += 1
                    data2["T1"][time_string][time_index[time]] += 1
                    pass
                elif "T2" in prot:
                    t1bool=False
                    data[time]["T1 / T2"][1] += 1
                    data["all"]["T1 / T2"][1] += 1
                    # print("t2", prot)
                    data2["T2"]["Exp 1/ 2/ 3/ 4"][nexp] += 1
                    data2["T2"]["tot"] += 1
                    data2["T2"][time_string][time_index[time]] += 1
                else:
                    print("WHATT\n"*50)
                if "p" in name:
                    data["all"]["p / no p"][0] += 1
                    data[time]["p / no p"][0] += 1

                    data2["all"]["p / no p"][0] += 1
                    if t1bool:
                        data2["T1"]["p / no p"][0] += 1
                    else:
                        data2["T2"]["p / no p"][0] += 1
                else:
                    data["all"]["p / no p"][1] += 1
                    data[time]["p / no p"][1] += 1

                    data2["all"]["p / no p"][1] += 1
                    if t1bool:
                        data2["T1"]["p / no p"][1] += 1
                    else:
                        data2["T2"]["p / no p"][1] += 1

                # print(cnt)
                # data["-7day"]["P1/ P2/ P3"][nexp] += 1
                # data[time]["P1/ P2/ P3"][nexp] += 1
                # print(data["all"]["P1/ P2/ P3"][nexp])
            # print(data[time]["p / no p"])
            # times.add(time)
            # print()
    print(times)
    print(data.T)
    print(data2.T)
    print(data2.drop("Exp 1/ 2/ 3/ 4", axis=0).T)
    print("Slice distances:", slice_distances)
    print("Pixel spacings:", inplane_distances)
    print("Protocols:", protocol_names)
    print("Number of slices:", num_slices)
    pass


def find_raw_folders_from_name(name_str, conditions=[], exclude_conditions=False):
    data = []
    for experiment in find_folders(RawDir, condition="Pilot"):
        # for experiment in ["Pilot2", "Pilot3"]:
        for time in find_folders(os.path.join(RawDir, experiment)):
            datadir = os.path.normpath(os.path.join(RawDir, experiment, time))
            for folder in find_folders(datadir, condition="sagittal"):
                # print(experiment, time, folder)
                if name_str in folder:
                    cond_bools = [x in folder for x in conditions] if any(conditions) else [True]
                    cond = not(any(cond_bools)) if exclude_conditions else all(cond_bools)
                    if cond:
                        # print("\n", experiment, time)
                        # print(folder)
                        name = get_name(experiment, folder)
                        # print(name)
                        data.append([experiment, time, name])
    data = np.array(data)
    times = np.array([int(x[:-3]) for x in data[:, 1]])
    data = data[times.argsort()]
    print("FOR ", name_str, "COND", conditions, f"{'''excluded''' if exclude_conditions else '''included'''}")
    print("FOUND", data)
    return data


def count_all_mice_ids():
    from name_dose_relation import dose_to_name
    count = 0
    df = pd.DataFrame()
    df_control = pd.DataFrame()
    for exp in find_folders(RawDir, "Pilot"):
        for time in find_folders(os.path.join(RawDir, exp)):
            # print(exp, time)
            for folder in find_folders(os.path.join(RawDir, exp, time), condition="sagittal"):
                count += 1
                id = folder.split("_")[0]
                # print(id, folder)
                pbool = "p" in folder
                t1bool = "T1" in folder
                df.loc[id, time] = 1
                if not time in ["-7day", "-3day"]:
                # if True:
                    df_control.loc[id, "dose"] = dose_to_name(exp, time, id)

    print(df.shape)
    print(count)
    # df["id"] = df.index.values
    id_set = set(df.index.values)
    df = df.melt()
    df = df[df["value"] == 1]   # all imaging instances (id, time) NOT COUNTING T1, AFTER-P!!
    print(df.shape)
    print("HAVING", len(id_set), "INDIVIDUALS")
    print(df_control.shape)
    print("CONTROL:", len(df_control[df_control["dose"] == 0]))
    # for id, vals in six.iteritems(df.T):
    #     print(id, vals)
    return 0


if __name__ == "__main__":
    # count_all_mice_ids()
    count_all_data_stuff()

    # mainfolder = r"G:\OneDrive - Universitetet i Oslo\Pilot_LateEffects_-7day"
    # parent = r"G:\OneDrive - Universitetet i Oslo\Pilot_LateEffects_-7day"
    # target = r"G:\OneDrive - Universitetet i Oslo\Segmentations\cropped_salivary\raw"
    # experiment = "pilot1"
    # experiment = "pilot2"
    # # time = "-7day"
    # # time = "8day"
    # time = "26day"
    # # time = "56day" #change this to correct day - 57day???
    # main = os.path.normpath(os.path.join(os.getcwd(), "..", "..", "RAW DATA", experiment, time))
    # # print(os.getcwd())
    # # print(main)
    # plane = "sagittal"
    # count_all_data_stuff()

    # find_raw_folders_from_name("6-2")
    # find_raw_folders_from_name("9-4", conditions=["p", "T1"], exclude_conditions=True)
    # find_raw_folders_from_name("9-4")#, conditions=["p", "T1"], exclude_conditions=True)


    # make_ndarrays_from_folders(parent, target, cond="sagittal", time="-7day", cropped=True)

    # for folder in find_folders(main, plane):
    #     path = os.path.normpath(os.path.join(main, folder))
    #     print(path)
    #     # print("\n")
    #     for file in dcm_files_indexed(path):
    #         filename = file[0]
    #         ds = pydicom.dcmread(os.path.join(folder, path, filename))
    #         print(ds.InstanceCreationDate, ds.InstanceCreationTime)
    #         print(ds)
    #         break
    #     # break
    #     dcm0 = dcm_files_indexed(path, printbool=False)[0][0];   dcm1 = dcm_files_indexed(path, printbool=False)[1][0]
    #     ds0 = pydicom.dcmread(os.path.join(main, folder, dcm0));   ds1 = pydicom.dcmread(os.path.join(main, folder, dcm1))
    #     print("slice thickness:", ds1.SliceLocation - ds0.SliceLocation, ds0.SliceThickness)
    #     print("pixelspacing = ", ds0.PixelSpacing, "orientation=", ds0.ImageOrientationPatient)
    #     # print(ds0.ImagePositionPatient)
    #     print(ds0.SeriesDescription)
    #     print()
    # print(ds0)


    # SORTING AND STUFF + RENAMING
    # # import shutil
    # exp = "Pilot4"
    # main = os.path.normpath(os.path.join(os.getcwd(), "..", "..", "RAW DATA", exp))
    # # main = os.path.normpath(os.path.join(os.getcwd(), "..", "..", "RAW DATA\Pilot3"))
    # l = set()
    # times_dcm = set()
    # protocols_all = set()
    # count = 0
    # pcount = 0
    # for time in os.listdir(main):
    #     # print(time)
    #     DirTime = os.path.join(main, time)
    #     times = set()   # check what times ACTUALLY in time folder
    #     for folder in os.listdir(os.path.join(main, time)):
    #         # name = folder[:5]
    #         name = get_name(exp, folder)
    #         # name = folder[12:-3]
    #         num_files = len(os.listdir(os.path.join(DirTime, folder)))
    #         count += 1
    #         # times.add(folder[:6])
    #         # if count == 0:
    #         # plt.close()
    #         # fig, ax = plt.subplots(3, 3)
    #         # fig.tight_layout()
    #         # ax = ax.ravel()
    #         # [axx.axis("off") for axx in ax]
    #         # pass
    #         # print(time, num_files, name)  # , folder)
    #         if "p" in folder:
    #             pcount += 1
    #         path_old = os.path.join(DirTime, folder)
    #         if "T1" in name:
    #             print("\n", time, num_files, name)  # , folder)
    #             indexed = dcm_files_indexed(path_old, printbool=False)
    #             protocol = set()
    #             # indexed = dcm_files_indexed(path_old)
    #             for file in np.array(indexed).T[:][0]:
    #                 ds = pydicom.dcmread(os.path.join(main, time, folder, file))
    #                 # print(ds.ProtocolName)
    #                 protocol.add(ds.ProtocolName)
    #                 print(ds)
    #                 break
    #             print(protocol)
    #             break
    #         # if "T1_RARE_coronal" in protocol:
    #         #     prot = list(protocol)[0][:-8]
    #         # else:
    #         #     prot = list(protocol)[0]
    #         # prot += "_sagittal"
    #         # # print(time, num_files, folder, name, prot)
    #         # name_new = name + "_" + prot
    #         # path_new = os.path.join(DirTime, name_new)
    #         # print(time, num_files, path_new)
    #         # # os.rename(path_old, path_new)
    # print("All:", count, "p:", pcount)

            # if num_files != 60:
            #     print(time, num_files, folder, name)
    # os.rmdir(path)
            # shutil.rmtree(path)
            # name += "_ALL_PLANES"
            #     print("???????")
            #     print(name)
            # folder_new = os.path.join(main, "..", "RAW DATA NOT IN USE", "Pilot3_" + time)
            # # print(name, os.path.normpath(folder_new))
            # print(name, path_old)
            # if not os.path.exists(folder_new):
            #     print("NEW PATH")
            #     os.makedirs(folder_new)
            # path_new = os.path.join(folder_new, name)
            # os.rename(path_old, path_new)
    #             # indexed = dcm_files_indexed(os.path.join(DirTime, folder), printbool=False)
    #             # MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(DirTime, folder), printbool=False)
    #             # ax[count].imshow(MATR_RAW[2])
    #             # ax[count].set_title(name)
    #             # count += 1
    #             # print(count, name, MATR_RAW.shape)
    #         else:
    #             # if "8-" in name or "9-" in name:
    #             #     if "E3" in name:
    #             #         name += "_T1w"
    #             for file in np.array(indexed).T[:][0]:
    #                 ds = pydicom.dcmread(os.path.join(main, time, folder, file))
    #                 protocol.add(ds.ProtocolName)
    #                 times_dcm.add(ds.InstanceCreationDate)
    #                 orientation = ds.ImageOrientationPatient
    #             # print(protocol)
    #             for p in protocol:
    #                 if "coronal" in p:
    #                     name += "_" + p[:-7] + "sagittal" # correcting name from coronal to sagittal
    #                 else:
    #                     name += "_" + p + "_sagittal"
    #                 protocols_all.add(p)
    #             print(path_old)
    #             path_new = os.path.join(main, time, name)
    #             print(path_new)
    #             print(name, protocol)
    #             os.rename(path_old, path_new)
    #             print()
    #
    #         # MATR_RAW = dcm_folder_to_pixel_matrix(indexed, os.path.join(DirTime, folder), printbool=False)
    #         # print(MATR_RAW.shape, name)
    #         # print(orientation)
    #
    #
    #             # ax[count].imshow(MATR_RAW[len(MATR_RAW) // 2])
    #             # ax[count].set_title(name)
    #             # count += 1
    #         # print(count, name, MATR_RAW.shape, protocol)
    #
    #         # print(name)#, protocol)
    #             # name += "_sagittal"
    #
    #         # if count > 8:
    #         #     count = 0
    #         #     print()
    #             # plt.show()
    #         l.add(num_files)
    #     print(times)
    # # plt.show()
    # print(l, times_dcm)
    # print(protocols_all)

    #PILOT 2 DCM STUFF
    # main = os.path.join(os.getcwd(), "..", "RAW DATA\Pilot2")
    # for time in os.listdir(main):
    #     print("\n", time)
    #     for folder in find_folders(os.path.join(main, time)):
    #         print(folder[12:])
    #         path_old = os.path.join(main, time, folder)
    #         path_new = os.path.join(main, time, folder[12:])
    #         print(path_new)
    #         # os.rename(path_old, path_new)
    #     for mouse in find_folders(os.path.join(main, time), condition="sagittal"):
    #         name = (mouse[12:15] + mouse[15:24]).replace("_", " ") if "p" in mouse else mouse[12:15]    #how to collect individual name (with MRI timing according to pilocarpine injection)
    #         dcmfiles = dcm_files_indexed(os.path.join(main, time, mouse), printbool=False)
    #         print(len(dcmfiles), name)

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