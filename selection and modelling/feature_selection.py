import os
import numpy as np
import pandas as pd
import sys
import seaborn as sns
from matplotlib import pyplot as plt
import six
import networkx as nx
from DICOM_reader import find_folders
from name_dose_relation import load_naming_data
from endpoints import load_saliva, binary_thresh_xerostomia, load_sg_area
# import researchpy as rp
import scipy.stats as stats
from select_utils import *
from data_loader import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, auc, roc_auc_score


#https://www.pythonfordatascience.org/anova-python/
#https://www.askpython.com/python/examples/correlation-matrix-in-python

class selected_features:
    def __init__(self, mode, time, condition, norm, experiment, data=[], exclude_condition=False):
        self.mode = mode
        self.time = time
        self.condition = condition
        self.exclude_condition = exclude_condition
        self.norm = norm
        self.experiment = experiment
        if not any(data):
            self.data = self.load_data()
        else:
            self.data = data

    def load_data(self):
        df = pd.DataFrame()
        if "+" in self.experiment:
            for exp in self.experiment.split(" + "):
                # path = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic features", exp, self.mode, self.norm, self.time))   #OLD
                path = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic features", " ".join([self.mode, self.norm]), exp, self.time))
                for p in find_folders(path, self.condition, exclude_condition=self.exclude_condition):
                    print(p)
                    temp = pd.read_csv(os.path.join(path, p))
                    # print(temp)
                    df = df.append(temp)
                print("Dataframe LOADED from:\t", path, f"\n{'''excluding''' if self.exclude_condition else '''including'''} {self.condition}")
        else:
            # path = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic features", self.experiment, self.mode, self.norm, self.time))
            path = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic features", " ".join([self.mode, self.norm]), self.experiment, self.time))
            for p in find_folders(path, self.condition, exclude_condition=self.exclude_condition):
                print(p, self.condition)
                temp = pd.read_csv(os.path.join(path, p))
                # print(temp)
                df = df.append(temp)
            print("Dataframe LOADED from:\t", path)
        # print(df["mouseidx"])
        # df = df.set_index(keys=df["mouseidx"])
        df = df.set_index(keys=df["Unnamed: 0"])    #set name / ID as index
        # print("Dataframe LOADED from:\t", path)
        print(df.info(verbose=False), "\n", 40*"-")
        return df

    def initial_reduction(self, floats_excluded=[], ints_included=[], remove_equal_vals=True, only_floats=False):
        print("INITIAL FEATURE REDUCTION.")
        # dropped_fts = []
        dropped_equals = []
        if only_floats:
            dcols = self.sort_cols_by_dtype()[np.dtype("float64")]  # select all columns in data of type float64
            fts = dcols.values.tolist()  # get column names in array
            fts = fts + ints_included
            self.data = self.data[fts].drop(columns=floats_excluded)  # remove mean if working with stscore norm images - this is then very close to zero by definition!
        if remove_equal_vals:   # remove ft key if all vals are the same
            count = 0
            for key, vals in six.iteritems(self.data):
                vals = list(vals)
                # print(key, "\n", vals)
                # print(vals[0])
                # if key == "lbp-2D_firstorder_90Percentile":
                    # print(vals)
                if all(val == vals[0] for val in vals):
                    # print(key, vals, "\n")
                    # print(key, "\n")
                    self.data = self.data.drop(columns=[key])
                    dropped_equals.append(key)
                    count += 1
            print(count, "features have equal values for each mouse and are thus removed.")
            print("----- DROPPED EQUAL FEAUTRES -----:\n", dropped_equals)
        print("\n", "-"*40)
        return dropped_equals

    def feature_reduction_absolute_correlation(self, thresh, save=False, savepath=""):
        fts_removed = []
        # corrmatr = corrmatr_initial.copy()
        allfeatureschecked = False
        print("REDUCING FEATURES BY AVERAGED CORRELATION UNTIL NONE ABOVE", thresh)
        # allfeatureschecked = True
        while not allfeatureschecked:
            corrmatr = self.corr_matrix()
            Cvals = pd.Series(data=None, index=corrmatr.columns, dtype="float64", name="Sum of correlation coeff")
            for ft in corrmatr.columns:
                # print(ft)
                Cvals[ft] = sum(np.abs(corrmatr[ft].values) / len(Cvals))    #CALCULATE CURRENT CORRELATION SUMS ROW-WISE FOR FEATURES LEFT IN DATAFRAME
            Cvals = Cvals.sort_values(ascending=False)      #sort features by highest average abs correlation
            print(len(Cvals), "features left in dataset.", corrmatr.shape, "\n")
            #TODO: FOR EACH FEATURE FIND WHICH CORRELATES ABOVE THRESH --> REMOVE FEATURE WITH HIGHEST C VAL
            breakbool = False
            for ft1 in Cvals.index:
                if breakbool:
                    break
                for ft2 in Cvals.index:
                    if ft1 == ft2:
                        pass
                        # print(ft1, ft2, Cvals[ft1])
                    else:
                        corr = corrmatr.loc[ft1][ft2]
                        if abs(corr) > thresh:  #FIND PAIR OF FEATURES ft1, ft2, WITH CORR > THRESH
                            # print(ft1, "\t", ft2, corrmatr.loc[ft1][ft2])
                            # print(f"Pearson corr. coeff. sum C for {ft1} = {Cvals[ft1]:.5f}, {ft2} = {Cvals[ft2]:.5f}")
                            ft_most_corr = ft1 if Cvals[ft1] > Cvals[ft2] else ft2      #FEATURE WITH HIGHEST AVERAGE ABS CORR IS SELECTED FOR REMOVAL
                            # print(ft_most_corr)
                            self.data = self.data.drop(columns=[ft_most_corr])
                            fts_removed.append(ft_most_corr)
                            print(ft_most_corr, " "*(50 - len(ft_most_corr)), f"dropped from dataset having C = {Cvals[ft_most_corr]:.5f} and corr({ft1, ft2}) = |{corrmatr.loc[ft1][ft2]:.5f}| > {thresh}")
                            # corrmatr = self.corr_matrix()
                            # print(corrmatr)
                            breakbool = True    #break both for loops, make new corrmatrix, start process over again
                            break
            #             fts_to_remove.append(ft_most_corr) if ft_most_corr not in fts_to_remove else 0
            if not breakbool:
                allfeatureschecked = True   #this is true when loop above is run without finding any features with corr > thresh

        print(f"{len(fts_removed)} FEATURES REMOVED HAVING CORR > {thresh}")
        print(fts_removed)
        # self.data = self.data.drop(columns=fts_removed)
        if save:
            if not savepath:
                savepath = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", self.experiment, self.time, "_".join((self.norm, self.time, self.mode, f"fts_remaining_after_summedcorrselection_thresh={thresh}")) + ".csv"))
            if not os.path.exists(os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", self.experiment, self.time))):
                os.makedirs(os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", self.experiment, self.time)))
            fts_remaining = self.data.columns.to_series(index=range(len(self.data.columns)), name=thresh)
            print(fts_remaining)
            fts_remaining.to_csv(savepath)
            print("Remaining features SAVED at", savepath)
        print("-"*40, "\n")
        return fts_removed


    def sigma_eval(self, dtype="float64"):
        sigma_counts = pd.Series([], dtype=dtype)
        for colnum, col in enumerate(self.data.columns):
            if self.data[col].dtype == np.dtype(dtype):
                # print(col, self.data[col])
                n, _, _ = sigma_evaluator(self.data[col])
                # print("n = ", n, "\n")
                sigma_counts = sigma_counts.add(pd.Series([1], index=[n]),
                                                fill_value=0)  # count sigma_evaluator n's for each float64 column in feats.data
        return sigma_counts

    def coeff_var(self): #calculate coefficient of variation for all features in data
        cv_list = []
        for key, vals in six.iteritems(self.data):
            try:
                mu = np.mean(vals); sd = np.std(vals)
                cv_list.append([key, float(sd / np.abs(mu))])
            except Exception as e:
                print(e.args)
        cv_list = np.array(cv_list)
        cv_list[:,1] = np.array([float(x) for x in cv_list[:,1]])
        # print(cv_list[:,1]) #TODO: why tf are these values string not float
        return pd.Series(data=cv_list[:,1], index=cv_list[:,0], name="cv")     #return pd.Series
        # return np.array(cv_list)

    def corr_matrix(self, features=[], plot=False, save=False, plotname=""):      #make correlation matrix of size nxn for n float64 features in dataset
        # features = features[500:]
        # print(self.data.loc[:, fts].dtypes)
        #TODO: IF FEATURES EMPTY MAKE CORRMATR FOR ALL DATA
        if not features:
            ds = self.data
            features = self.data.columns
        else:
            try:
                ds = self.data.loc[:, features]      #select all rows in data for each column in features
                print(ds.info)
            except Exception as e:
                print(e.args, e.__context__)
        corrmatr = ds.corr()        #pearson correlation coefficient if not else specified
        # plot=True
        if plot:
            sns.set(font_scale=0.3)
            # cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=-.4, start=0)
            # cmap = sns.color_palette("vlag", as_cmap=True)
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            # cmap = "viridis"
            sns.heatmap(corrmatr, annot=False, cmap=cmap, center=0.00)
            plt.rcParams["axes.titlesize"] = 10
            plt.title(f"Correlation matrix of {len(features)} features, selected={plotname}.")# from {norm} data.")
            if save:
                savefolder = os.path.join(os.getcwd(), "..", r"master_plots\feature selection and modelling\correlation plots", self.time + "_" + self.mode.replace(" ", "_") + "_" + self.norm)
                if not os.path.exists(savefolder):
                    os.mkdir(savefolder)
                savepath = os.path.join(savefolder, "corr_matrix_" + plotname + ".png")
                plt.savefig(savepath, dpi=300)
                print("figure saved at ", savepath)
            plt.show()
            plt.close()
        return corrmatr


    def sort_cols_by_dtype(self):
        return self.data.columns.to_series().groupby(self.data.dtypes).groups

    def find_by_dose(self, doserate=0):
        print(f"----     FINDING NAMES (INDEX) WHERE DOSERATE = {doserate:.2f} Gy  -----")
        print(f"     (dose delivered in 10 fractions over 5 days (2 fractions of {doserate:.2f} Gy per day)")
        df_dose = load_naming_data()
        # print(df_dose.columns)
        # print(df_dose.loc[df_dose["doserate"] == doserate])
        names = df_dose.loc[df_dose["doserate"] == doserate].index.values
        # print(names)
        if len(names) == 0:
            print("NO MICE HAVING DOSERATE", doserate, "FOUND. ")
        mousenames = []
        for idx in self.data.index:
            for nm in names:
                if nm in idx:
                    mousenames.append(idx)
                    break
        # print(mousenames)
        return mousenames


def sigma_evaluator(ds, print_stuff=False):        #returns n where all feature values in ds are within mean +- n*st
    vals = ds.values
    mean = np.mean(vals)
    sigma = np.std(vals)
    n = 1
    while True:
        b = np.abs(vals - mean) < n*sigma
        if b.all():
            break
        n += 1
    # print(np.abs(vals - mean), n*sigma)
    if print_stuff:
        print(f"{ds.name} feature have all values within mean +- {n}sigma = {mean:.5g} +- {n}*{sigma:.5g}={n*sigma:.5g} \n")
        # print(vals, "\n")
    # print(ds.name)
    # print(r.T.iloc[0].values)
    return n, mean, sigma


def correlation_histogram(corrmatrix, thresh, corrmode="pearson correlation coefficients"):
    counts = []
    for key, vals in six.iteritems(corrmatrix):
        if key == "Unnamed: 0":
            continue
        else:
            # print(key, vals)
            c = len(np.argwhere(np.abs(vals.values) >= thresh))
            print(c, "\t", key)
            counts.append(c)
    # sns.set_theme()
    plt.hist(counts, bins=max(counts))
    plt.grid()
    plt.title(f"Histogram of {corrmode} >= {thresh}")
    plt.xlabel(f"# of features with |corr| > {thresh}")
    plt.ylabel("# of features")
    plt.show()
    plt.close()
    return 0


def feature_correlations_sorted(corrmatr, thresh, reversed=False):
    corrfts = []
    for key, vals in six.iteritems(corrmatr):
        if key == "Unnamed: 0":
            continue
        else:
            c = len(np.argwhere(np.abs(vals.values) >= thresh))
            corrfts.append([key, c])
    return sorted(corrfts, key=lambda l:l[1], reverse=reversed)


def visualize_network(matr, thresh, keep_names=False, colored=True, save=False, savepath=""):    #https://www.python-graph-gallery.com/327-network-from-correlation-matrix
    num_edges = len(matr.columns)
    if not keep_names:      #Rename features as "ft1, ..., ftk"
        # ft_index = ["ft" + str(x) for x in range(1, len(matr.columns) + 1)]
        ft_index = list(range(1, num_edges + 1))
        # print(ft_index)
        matr.columns = ft_index;    matr.index = ft_index
    # print(matr)
    edges = matr.stack().reset_index()
    edges.columns = ["ft1", "ft2", "corr"]
    # print(edges)
    edges_filtered = edges.loc[(np.abs(edges["corr"]) > thresh) & (edges["ft1"] != edges["ft2"])]
    # print(edges_filtered)
    G = nx.from_pandas_edgelist(edges_filtered, "ft1", "ft2")
    sns.set_theme()
    plt.title(f"{num_edges} features connected by edge if |corr(ft(i), ft(k)| > {thresh}")#, fontweight="bold")
    lw = 0.05
    if colored:
        import matplotlib as mpl
        d = dict(G.degree)
        low, *_, high = sorted(d.values())
        norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.hot_r)
        # nx.draw_circular(G, with_labels=False, font_size=5, node_color=[mapper.to_rgba(i) for i in d.values()],
        #                  width=0.1, node_size=10, node_shape="x", linewidths=1)
        nx.draw_networkx_nodes(G, pos=nx.circular_layout(G), node_size=10, node_shape="o", linewidths=0.1, edgecolors="black",
                               node_color=[mapper.to_rgba(i) for i in d.values()])
        nx.draw_networkx_edges(G, pos=nx.circular_layout(G), width=lw, node_size=10, arrows=True, arrowstyle="-", connectionstyle="arc3,rad=-0.1")
        plt.colorbar(mappable=mapper, label="degree of node")
    else:
        nx.draw_circular(G, with_labels=True, font_size=10, node_color="orange", linewidths=lw)
    plt.tight_layout()
    if save:
        # savepath = os.path.join(os.getcwd(), "..", r"master_plots\feature selection and modelling\correlation plots", "correlation_graph.png")
        plt.savefig(savepath, dpi=300)
        print("Correlation graph figure saved at", savepath)
    # plt.show()
    plt.close()
    pass


def plot_reduced_AACR(experiments=["pilot1", "pilot2", "pilot1 + pilot2"], time="-7day", norm="stscore norm", mode="central slice", log=False):
    Y = []
    Y2 = []
    Yboth = []
    common1 = [];   common2 = [];   common3 = [];
    Mvals = [0.999, 0.99, 0.98, 0.97, 0.95, 0.925] + [round(x, 2) for x in np.linspace(0.9, 0.2, 15).tolist()]

    # Y_ALL = []
    # for exp in experiments:
    #     Y = []
    #     print(exp)
    #     for M in Mvals:
    #         print("M =", M)
    #         ftspath = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", exp, time, "_".join((norm, time, mode, f"fts_remaining_after_summedcorrselection_thresh={M}")) + ".csv"))
    #         fts_remaining = pd.read_csv(ftspath, index_col=0, squeeze=True)
    #         Y.append(len(fts_remaining))
    #     Y_ALL.append(Y)
    # print(np.shape(Y_ALL), len(Y_ALL))
    # sns.set_theme()
    # Mvals.reverse()
    # for Y in Y_ALL:
    #     print(Y)


    for M in Mvals:
        print("M =", M)
        ftspath_pilot1 = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", "pilot1", time, "_".join((norm, time, mode, f"fts_remaining_after_summedcorrselection_thresh={M}")) + ".csv"))
        ftspath_pilot2 = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", "pilot2", time, "_".join((norm, time, mode, f"fts_remaining_after_summedcorrselection_thresh={M}")) + ".csv"))
        ftspath_both = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", "pilot1 + pilot2", time, "_".join((norm, time, mode, f"fts_remaining_after_summedcorrselection_thresh={M}.csv"))))
        # ftspath = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", feats.experiment, feats.time, "_".join((feats.norm, feats.time, feats.mode, f"fts_remaining_after_summedcorrselection_thresh={M}")) + ".csv"))
        fts_remaining_1 = pd.read_csv(ftspath_pilot1, index_col=0, squeeze=True)
        fts_remaining_2 = pd.read_csv(ftspath_pilot2, index_col=0, squeeze=True)
        fts_remaining_both = pd.read_csv(ftspath_both, index_col=0, squeeze=True)
        # print(len(fts_remaining))
        Y.append(len(fts_remaining_1))
        Y2.append(len(fts_remaining_2))
        Yboth.append(len(fts_remaining_both))
        # feats.data = feats.data[fts_remaining]
        # print(feats.data.shape)
        overlap1 = np.intersect1d(fts_remaining_both, fts_remaining_1)  #P1 fts overlap with P1 + P2
        overlap2 = np.intersect1d(fts_remaining_both, fts_remaining_2)  #P2 fts overlap with P1 + P2
        overlap3 = np.intersect1d(fts_remaining_1, fts_remaining_2)     #P1 fts overlap with P2
        common1.append(len(overlap1));  common2.append(len(overlap2));  common3.append(len(overlap3))
        print(f"At M={M} P1 have {len(fts_remaining_1)} fts remaining, P1 + P2 have {len(fts_remaining_both)} with {len(overlap1)} in common.")
        print(f"At M={M} P2 have {len(fts_remaining_2)} fts remaining, P1 + P2 have {len(fts_remaining_both)} with {len(overlap2)} in common.")
        print(f"At M={M} P1 have {len(fts_remaining_1)} fts remaining, P2 have {len(fts_remaining_2)} with {len(overlap3)} in common.")
        print()
        # corrmatr = feats.corr_matrix(plot=True, plotname=f"thresh={M:.3f}")
        # feats.feature_reduction_absolute_correlation(thresh=M, save=True)
        # print(feats.data)
        # M_prev = M
        # break


    sns.set_theme()
    Mvals.reverse();    Y.reverse()
    Y2.reverse();   Yboth.reverse()
    common1.reverse();  common2.reverse();  common3.reverse()
    # Mvals.append(1.000);    Y.append(num_fts_original)
    # log = True
    # log = False
    if not log:
        plt.plot(Mvals, Y, "--" + "x", color="b", label="Pilot1 (6 datapoints per ft)")
        plt.plot(Mvals, Y2, "--" + "o", color="r", label="Pilot2 (8 datapoints per ft)")
        plt.plot(Mvals, Yboth, "--" + "+", color="g", label="Pilot1 + pilot2 concatenated (14)")
        plt.plot(Mvals, common1, ls=":", color="b", label="# fts shared P1 with P1 + P2")
        plt.plot(Mvals, common2, ls=":", color="r", label="# fts shared P2 with P1 + P2")
        plt.plot(Mvals, common3, ls=":", color="g", label="# fts shared P1 with P2")
    else:
        plt.semilogy(Mvals, Y, "--" + "o", color="b", label="Pilot1 (6 datapoints per ft)")
        plt.semilogy(Mvals, Y2, "--" + "x", color="r", label="Pilot2 (8 datapoints per ft)")
        plt.semilogy(Mvals, Yboth, "--" + "+", color="g", label="Pilot1 + pilot2 concatenated (14)")
        plt.semilogy(Mvals, common1, ls=":", color="b", label="# fts shared P1 with P1 + P2")
        plt.semilogy(Mvals, common2, ls=":", color="r", label="# fts shared P2 with P1 + P2")
        plt.semilogy(Mvals, common3, ls=":", color="g", label="# fts shared P1 with P2")
    plt.gca().invert_xaxis()
    # plt.title(f"Feature filtering by average absolute correlation, from {feats.experiment} with {feats.data.shape[0]} datapoints per feature.")
    plt.title(
        f"Feature filtering by average absolute correlation, for pilot1 and pilot 2 at time {time}")
    plt.xlabel("Correlation threshold M")
    plt.ylabel("# features remaining" if not log else "log(# features remaining)")
    plt.legend(loc="best")
    plt.show()
    pass


def plot_groupings_AACR(experiment="pilot1 + pilot2", time="-7day", norm="stscore norm", mode="central slice", log=False, groups=["original", "wavelet", "square", "squareroot", "logarithm", "exponential", "gradient", "lbp"], groupedby="image filter"):
    # feats = selected_features(mode, time, "p", exclude_condition=True, norm=norm, experiment=experiment)
    # for gr in groups:
    Mvals = [0.999, 0.99, 0.98, 0.97, 0.95, 0.925] + [round(x, 2) for x in np.linspace(0.9, 0.2, 15).tolist()]
    Yvals = np.zeros(shape=(len(groups), len(Mvals)))
    for i, M in enumerate(Mvals):
        ftspath = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", experiment, time, "_".join((norm, time, mode, f"fts_remaining_after_summedcorrselection_thresh={M}")) + ".csv"))
        fts_remaining = pd.read_csv(ftspath, index_col=0, squeeze=True)
        for j, g in enumerate(groups):
            fts_g = [ft for ft in fts_remaining if g in ft.lower()]
            Yvals[j, i] = len(fts_g)    #[group index, mval index]
            print(len(fts_g), f"features have {g} in name")
            print(fts_g)
        print()
    # fig, ax = plt.subplots(nrows=2, ncols=(len(groups) // 2));  ax = ax.ravel()
    sns.set_theme()
    for i, Y in enumerate(Yvals):
        print(Y.shape)
        # ax[i].plot(Mvals, Y, label=groups[i])
        plt.plot(Mvals, Y, label=groups[i]) if not log else plt.semilogy(Mvals, Y, label=groups[i])
    plt.xlabel("AACR threshold M")
    plt.ylabel("# fts remaining") if not log else plt.ylabel("log(# fts remaining)")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.title(f"Number of features remaining after AACR from {experiment} at time {time}, grouped by {groupedby}.")
    plt.show()
    pass


def load_aacr_features(experiment, time, norm, mode, thresh):
    print(f"----- LOADING AACR REDUCED FEAUTRES WITH THRESH = {thresh:.2f}     -----")
    path = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction",
                     experiment, time, "_".join((norm, time, mode,
                                                           f"fts_remaining_after_summedcorrselection_thresh={thresh}")) + ".csv"))
    df = pd.read_csv(path, index_col=0, squeeze=True)
    print("     ", len(df), "FEATURES LOADED")
    return df



def main_AACR(feats, save=False):       #absolute average correlation reduction
    Mvals = [0.999, 0.99, 0.98, 0.97, 0.95, 0.925] + [round(x, 2) for x in np.linspace(0.9, 0.2, 15).tolist()]
    for M in Mvals:
        print("M =", M)
        feats.feature_reduction_absolute_correlation(thresh=M, save=save)
    pass



def rank_features_univariate(df_orig, dfy_orig, fts, plot_regressions=False, num_bootstraps=1000, pthresh=0.05, savename=""):
    # LOOP OVER ALL FEATURES, DO REGRESSION / PEARSON / MUTUAL INFORMATION / DISTANCE CORRELATION
    # AND RANK FTS ACCORDINGLY (lin reg -> MSE / R2 / R2adj?)
    # y = df_y.values     # e.g. time, dose : R2, shape = (n, 2)
    # print(y.T[0])
    # print(kwargs) W
    # print(kwargs.get("plot_regressions"))
    # df_score = pd.DataFrame(columns=["ft", "sc", "r2adj", "pval"])
    # df_score = pd.DataFrame(columns=["ft", "s1", "s2"])
    # print(df_score)
    # print(y.shape)
    # print(df.index.values)
    # print(df_y.index.values)
    # TODO: Bootstrapping MANY times --> average feature rank?

    # df_orig = df
    # dfy_orig = df_y
    # print(df.shape, df_orig.shape)
    df_select = pd.DataFrame(index=fts, columns=["count"], data=[0]*len(fts))
    import statsmodels.api as sma
    # from scipy.stats import spearmanr
    for n in range(num_bootstraps):
        # ind_boot = resample(df_orig.index.values)
        # print(df_orig.index.values)
        # print(ind_boot)
        # df = df_orig.iloc[ind_boot]'
        print("BOOT", n)
        df, df_y = resample(df_orig, dfy_orig)
        y = df_y.values
        df_score = pd.DataFrame(columns=["ft", "sc", "r2adj", "pval"])

        # if n == 1:
        #     break
        for j, ft in enumerate(fts):
            # print("\n", j+1, ft)
            # print(d[ft])
            # break
            x = df[ft].values.reshape(-1, 1)    # feature values : R1, shape = (n, 1)
            # print(x.shape, y.shape)

            # LINEAR REGRESSION: SCORE = COEF OF DETERMINATION R^2
            reg = LinearRegression().fit(x, y)  # M: x:R1 --> y:R2
            s1 = reg.score(x, y)    # R2
            # print("Score", reg.score(x, y))
            # print("Coeffs", reg.coef_)
            # print("intecept", reg.intercept_)

            x2 = sma.add_constant(x)
            ols = sma.OLS(y, x2)
            fit = ols.fit()
            # print(fit.summary())
            # print(fit.rsquared)
            # print(fit.f_pvalue)
            # if y.shape[1] == 2:
            if np.ndim(y) == 2:
                reg_inv = LinearRegression().fit(y, x)  # M_inv y:R2 --> x:R1
                s2 = reg_inv.score(y, x)
                print("Score", reg_inv.score(y, x))
                print("Coeffs", reg_inv.coef_)
                print("intecept", reg_inv.intercept_)
                df_score = df_score.append({"ft":ft, "s1":s1, "s2":s2}, ignore_index=True)

                print(y[:, 0])
                print(y[:, 1])
                print(x.reshape(-1))
                s_krusk = stats.kruskal(x.reshape(-1), y[:, 0])
                print(s_krusk)

                s_krusk = stats.kruskal(x.reshape(-1), y[:, 1])
                print(s_krusk)
            else:
                # s_krusk = stats.kruskal(x.reshape(-1), y.reshape(-1))
                # print(s_krusk)
                # df_score = df_score.append({"ft":ft, "sc":s1, "r2":fit.rsquared, "pval":fit.f_pvalue}, ignore_index=True)
                df_score = df_score.append({"ft":ft, "sc":s1, "r2adj":fit.rsquared_adj, "pval":fit.f_pvalue}, ignore_index=True)
                pass
                # break
            # break
            # s3 = spearmanr
            # print(reg.predict(x))
            # print(*y.T)

            # plot_regressions = True if ft in ["lbp-2D_firstorder_90Percentile", "gradient_ngtdm_Busyness"] else False
            if plot_regressions:
                fig = plt.figure()
                fig.tight_layout()
                if np.ndim(y) == 2:
                    ax = fig.add_subplot(projection="3d")
                    # print(y)
                    ax.plot(y.T[0], y.T[1], x.T[0], "o", c="b")
                    y_pred = reg.predict(x)
                    ax.plot(y_pred.T[0], y_pred.T[1], x.T[0], "x", c="r")
                    x_pred = reg_inv.predict(y)
                    N = 25
                    yy1, yy2 = np.meshgrid(np.linspace(min(y.T[0]), max(y.T[0]), N),
                                           np.linspace(min(y.T[1]), max(y.T[1]), N))
                    b0, b1, b2 = reg_inv.intercept_[0], *reg_inv.coef_[0]
                    zz = b0 + yy1 * b1 + yy2 * b2
                    ax.plot_surface(yy1, yy2, zz)
                    ax.set_xlabel("time")
                    ax.set_ylabel("dose")
                    ax.set_zlabel(ft)
                    plt.show()
                else:
                    ax = fig.add_subplot()
                    ax.plot(x, y, "o", c="b", label="observed")
                    y_pred = reg.predict(x)
                    ax.plot(x, y_pred, "-x", c="r", label="predicted")
                    ax.set_xlabel(ft)
                    fig.suptitle(f"LINEAR REG: {ft} vs SALIVA\nR2={s1:.3f}, pval={fit.f_pvalue:.3f}")
                    plt.legend()
                    plt.show()
            # break
        # print(df_score)
        # print(df_score.sort_values(by=["sc"], ascending=False).head(25))
        # print(df_score.sort_values(by=["r2"], ascending=False).head(25))
        # print(df_score.sort_values(by=["pval"], ascending=False).head(25))
        # print(df_score.drop(["pval"], axis=1).sort_values(by=["sc"], ascending=False).head(25))

        # print(df_score[df_score["ft"].isin(["time", "dose"])])
        # print(df_score.drop(["sc"], axis=1).sort_values(by=["r2adj"], ascending=False).head(25))
        # print(df_score.drop(["sc"], axis=1).sort_values(by=["pval"], ascending=True).head(25))

        print("\t", len(df_score[df_score["pval"] < 0.05]["ft"].values), "fts having univariate reg p-val <", pthresh)
        for ft in df_score[df_score["pval"] < pthresh]["ft"].values:
            df_select.loc[ft, "count"] += 1
        # print(df_select.sort_values(by=["count"], ascending=False))
        if savename:
            df_select.sort_values(by=["count"], ascending=False).to_csv(os.path.join(selectDir, savename))
            print("UNIVARIATE SELECT DF SAVED")

    if plot_regressions:
        for ft in ["time", "dose"] + list(df_score.sort_values(by=["s1"], ascending=False).head(10)["ft"]):
            print(ft)
            x = df[ft].values.reshape(-1, 1)    # feature values : R1, shape = (n, 1)
            # print(x.shape, y.shape)
            # LINEAR REGRESSION: SCORE = COEF OF DETERMINATION R^2
            reg = LinearRegression().fit(x, y)  # M: x:R1 --> y:R2
            xvals = np.linspace(np.min(x), np.max(x), 50)
            yvals = reg.predict(xvals.reshape(-1, 1))
            plt.plot(x, y, "o")
            plt.plot(xvals, yvals, "--")
            plt.title(f"{ft}, score={reg.score(x, y)}")
            plt.show()

    # print(df_score.sort_values(by=["s2"], ascending=False).head(10))
    pass


def count_long_data(df):    # SEE IF DELTA-RADIOMICS IS VIABLE
    # print(df[df["time"] == -7])
    names_at_baseline = df[df["time"] == -7]["name"].values
    print(len(np.unique(names_at_baseline)))
    print(names_at_baseline)
    times = sorted(list(set(df["time"])))[1:]
    # times.pop(0)
    print(len(names_at_baseline), "at time -7")
    counted = dict({})
    for name in names_at_baseline:
        counted[name] = 0
    # print(counted)
    names_not_in_base = set()
    for time in times:
        names_in_time = df[df["time"] == time]["name"].values
        # for name in names_in_time:
        names_overlap = np.intersect1d(names_at_baseline, names_in_time)
        print(np.setdiff1d(np.intersect1d(names_at_baseline, names_in_time), names_in_time))
        for name in names_overlap:
            counted[name] += 1
        # print(np.setdiff1d(names_at_baseline, names_in_time))
        names_not_in_base.update(np.setdiff1d(names_at_baseline, names_in_time))
        # print(time, len(names_overlap), len(names_in_time))
    print(names_not_in_base)
    print(counted)
    pass


# def register_name_to_outcome(df, out, melt=True, make_70_exception=False):    # RETURN ORDERED LIST OF ENDPOINT (e.g. saliva) CORRESPONDING TO name, time IN df
#     # OUT: y = out registerred to df
#     # IF MELT = FALSE: out NEEDS TO HAVE COLUMNS name, time, value
#     y = pd.DataFrame({"idx":np.array([], dtype=int), "name":[], "time":np.array([], dtype=int), "val":[]})
#     dose_bool = "dose" in df.columns.values
#
#     if melt:
#         out = out.melt(id_vars="name", var_name="time", value_name="value")
#     # print(out)
#     # print(df["time"])
#     for j, name in enumerate(list(set(df["name"].values))):
#         # print("\n", name, df[df["name"] == name]["time"].values)
#         times, indexes = df[df["name"] == name]["time"].values, df[df["name"] == name]["time"].index.values   # TIMES FOR NAME IN INPUT (ft) DF
#         name_id = name.split("_")[0]
#         # print(name_id)
#         # out_for_name = out[[n in name for n in out["name"].values]]
#         out_for_name = out[[n == name_id for n in out["name"].values]]
#         # print(out_for_name)
#         # print("\n", indexes, df.iloc[indexes]["name"].values)
#         # print("\n", indexes, df.loc[indexes]["name"].values)    # ILOC: INTERGER BASED. LOC: INDEX (LABEL) BASED!!!!!!
#         for t, idx in zip(times, indexes):
#             t = int(t[:-3])
#             if t==70 and make_70_exception:
#                 t = 75  # Co-register day 75 saliv data to day 70 MRI
#             val = out_for_name[out_for_name["time"] == t]["value"].values
#             # print(t, idx, val)
#             if any(val) and val != "-":
#                 val = float(val)
#                 if dose_bool:
#                     dose = df.loc[idx]["dose"]
#                     y = y.append({"idx":int(idx), "name":name, "time":t, "dose":dose, "val":val}, ignore_index=True)
#                 else:
#                     y = y.append({"idx": int(idx), "name": name, "time": t, "val": val},
#                                  ignore_index=True)
#     y = y.sort_values("idx")
#     # y.index = y["idx"]
#     # print(y)
#     print(f"\tREGISTERED {len(y)} OUTCOME VALS TO DF\n")
#     return y.set_index(y["idx"]).drop(["idx"], axis=1)


def delta_fts_p_to_nop(df, relative_change=True, savename="", LR_MODE="LR split"):
    # RETURN DATA SET CONTAINING (relative) DELTA_P FEATURES: (ft_nop - ft_p) / ft_p
    # AT SAME INSTANCE (name, time)
    # print(df)
    df_p = df[["p" in x for x in df["name"].values]]
    df_nop = df.drop(df_p.index, axis=0)
    df_delta = pd.DataFrame(columns=df_p.columns)
    # print(df_delta)
    print("---- REGISTERRING DELTA-P FEATURES: P TO NO-P ---- ")
    for idx_nop, ID in zip(df_nop.index.values, df_nop[["name", "time"]].values):
        name, time = ID # identifier: USE TO COREGISTER SAME INDIVIDUAL (name) AT SAME TIME (time)
        name_orig = name
        # name = name[:3]
        name = name_orig.split("_")[0]

        X = name_orig.split("_")[-1] # L or R
        
        print("\n", idx_nop, name, time, X, name_orig)
        # cond = np.logical_and([name in nm for nm in df_p["name"].values], [t == time for t in df_p["time"].values])
        if LR_MODE == "LR split":
            cond = np.logical_and([(name in nm) and nm[-1] == X for nm in df_p["name"].values], [t == time for t in df_p["time"].values])
        else:
            cond = np.logical_and([(name in nm) for nm in df_p["name"].values], [t == time for t in df_p["time"].values])

        if np.any(cond):
            fts_p = df_p.iloc[np.where(cond)]
            fts_nop = df_nop.iloc[np.where([idx == idx_nop for idx in df_nop.index.values])]
            # print(fts_p.T)
            # print(fts_nop.T)
            dose = fts_p["dose"].values[0]
            # print(dose)

            print(fts_p[["name", "time"]].values, fts_nop[["name", "time"]].values, time, dose)
            ft_cols = fts_nop.drop(["name", "time", "dose"], axis=1).columns.values
            vals_nop = fts_nop[ft_cols].values[0]
            vals_p = fts_p[ft_cols].values[0]

            if relative_change:
                vals_delta = [0 if (not(x1 - x2) or x2 == 0) else (x2 - x1)/x1 for x1, x2 in zip(vals_nop, vals_p)]  # IF NO DIFFERENCE SET TO ZERO
            else:
                vals_delta = [x1 - x2 for x1, x2 in zip(vals_nop, vals_p)]
            df_delta = df_delta.append(pd.Series(data=[name_orig, time, dose] + list(vals_delta), index=["name", "time", "dose"] + list(ft_cols), name=name_orig), ignore_index=True)
            print(df_delta.shape)
            # break
        # break
    print(df_delta)
    print(f"DELTA-P FEATURES CALUCLATED FOR {len(df_delta)} INSTANCES HAVING {len(df_delta.columns)} FEATURES")
    if savename:
        df_delta.to_csv(os.path.join(ftsDir, savename))
        print("DELTA-P FEATURES SAVED")
    return df_delta


def plot_fts_over_time(df, fts, hue=""):
    for ft in fts:
        sns.catplot(x="time", y=ft, kind="swarm", data=df, hue=hue)
        plt.grid()
        plt.show()
    pass


# def random_forest_predict(dfx, dfy):
def random_forest_classifier(x_train, x_test, y_train, y_test, verbose=True, random_state=None):
    # from sklearn.ensemble import RandomForestClassifier
    # x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.3, random_state=42)
    # x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.3, random_state=None)
    # print(f"70 / 30 Split: N_train={len(y_train)}, N_test={len(y_test)}:")

    # random forest classifier with n_estimators=10 (default)
    # clf_rf = RandomForestClassifier(random_state=43)
    # clf_rf = RandomForestClassifier(random_state=random_state)
    clf_rf = RandomForestClassifier(criterion="entropy", min_samples_leaf=1, min_samples_split=16, n_estimators=1000, random_state=random_state)
    print("RF classifier having", clf_rf.__getattribute__("n_estimators"), "estimators")
    clr_rf = clf_rf.fit(x_train, y_train)
    y_pred = clf_rf.predict(x_test)
    ac = accuracy_score(y_test, y_pred)
    print('--- RF Accuracy is: ', ac) if verbose else 0
    # print(clr_rf.score(x_test, y_test)) # same as above!
    aucval = roc_auc_score(y_test, y_pred)
    print("\tROC AUC =\t", aucval) if verbose else 0
    cm = confusion_matrix(y_test, y_pred)
    print(cm) if verbose else 0
    # sns.heatmap(cm, annot=True, fmt="d")
    # plt.show()
    return ac, aucval, cm


def random_forest_regression(x_train, x_test, y_train, y_test, verbose=True, random_state=None):
    # rg_rf = RandomForestRegressor(random_state=random_state, max_depth=5)
    rg_rf = RandomForestRegressor(random_state=random_state, n_estimators=1000, max_depth=16, min_samples_leaf=1, min_samples_split=2, criterion="squared_error")
    rg_rf = rg_rf.fit(x_train, y_train)
    print("RF regression having", rg_rf.__getattribute__("n_estimators"), "estimators, max depth=", rg_rf.__getattribute__("max_depth"))
    # y_pred = rg_rf.predict(x_test)
    # r2 = rg_rf.score(x_train, y_train)
    # print(y_train.values)
    print('--- RF reg TRAIN R2 =', rg_rf.score(x_train, y_train)) if verbose else 0
    r2 = rg_rf.score(x_test, y_test)
    print('--- RF reg TEST R2 =', r2) if verbose else 0
    return 0


def load_delta_p_saliv_data(slice="LR", mode="disc", univar_tresh=0.80, log=False, impute_saliva=False):
    # SLICE: LR or CENTRAL / CENTER
    # MODE: DISC (binary xerostomia) OR CONT (sal. amounts)
    if slice.upper() == "LR":
        df = pd.read_csv(os.path.join(ftsDir, "LR_split_nyul_otsu_decile_extracted.csv"), index_col=0)  # LR
        df_delta_p = pd.read_csv(os.path.join(ftsDir, "LR_split_nyul_otsu_decile_delta-p.csv"), index_col=0)
        df_ranked = pd.read_csv(os.path.join(selectDir, "LR_split_nyul_otsu_decile_delta-p_univariate_reduction.csv"), index_col=0)

    elif slice.upper() == "CENTER":
        df = pd.read_csv(os.path.join(ftsDir, "center_nyul_otsu_decile_extracted.csv"), index_col=0)  # CENTER
        df_delta_p = pd.read_csv(os.path.join(ftsDir, "center_nyul_otsu_decile_delta-p.csv"), index_col=0)
        df_ranked = pd.read_csv(os.path.join(selectDir, "center_nyul_otsu_decile_delta-p_univariate_reduction.csv"), index_col=0)
    else:
        print("Slice(s)", slice, "not valid...\nTry LR or CENTER")
        return 0

    print("----- REMOVING", np.count_nonzero(list(df[["T1" in x for x in df["name"].values]].T)),
          " ROWS HAVING T1 IN NAME -----")
    df = df[[not ("T1" in x) for x in df["name"].values]]  # remove T1 images
    # TODO: TRY COL-WISE FEATURE NORM / SCALING (for all data) BEFORE SELECTION -- CHECK IF DIFFERENT SELECTED
    feats = selected_features(mode="center", time="all", condition="", norm="nyul otsu decile", experiment="all", data=df, exclude_condition=False)
    feats.initial_reduction(remove_equal_vals=True)
    df = feats.data
    print("DF DELTA P:", df_delta_p.shape)
    # LOAD / REGISTER SALIVA DATA TO INDIVIDUALS / TIMES
    if not impute_saliva:
        saliv_data = load_saliva()
        y_saliv_delta = register_name_to_outcome(df_delta_p, saliv_data)
    else:
        saliv_data = load_saliva(impute=True, melt=True)
        # print(saliv_data)
        saliv_data = saliv_data.rename(columns={"val":"value"})
        y_saliv_delta = register_name_to_outcome(df_delta_p, saliv_data, melt=False)

    y_saliv_delta_orig = y_saliv_delta
    y_saliv_delta = y_saliv_delta["val"]    # reduce to only saliva values

    df_delta_p_red = df_delta_p.loc[y_saliv_delta.index].drop(["name"], axis=1)
    fts = df_delta_p_red.columns.values

    y_saliv_delta_orig_log = y_saliv_delta_orig.copy()
    y_saliv_delta_orig_log["val"] = np.log1p(y_saliv_delta_orig["val"].values)
    y_saliv_delta_log = y_saliv_delta_orig_log["val"]

    Nboot = 1000
    include_thresh = univar_tresh
    # TODO: UNCOMMENT FUNCTION(S) BELOW TO NOT OVERWRITE!!!!!!!!!!!!!!!!!!!
    # rank_features_univariate(df_delta_p_red, y_saliv_delta, fts=fts, pthresh=0.05, plot_regressions=False, savename="center_nyul_otsu_decile_delta-p_univariate_reduction.csv")   # Center
    # rank_features_univariate(df_delta_p_red, y_saliv_delta, fts=fts, pthresh=0.05, num_bootstraps=Nboot, plot_regressions=False, savename="LR_split_nyul_otsu_decile_delta-p_univariate_reduction.csv") #LR
    # print(df_ranked[df_ranked["count"] > 500])     # FTS selected 500 / 1000 times
    df_red = df_ranked[df_ranked["count"] > Nboot * include_thresh]
    fts_red = df_red.index.values.tolist()
    print(f"----- BOOTSTRAP UNIVAR SELECTED ({include_thresh*100:.0f}% thresh):", len(df_red), "fts", f"\ttime:{'''time''' in fts_red} dose:{'''dose''' in fts_red}")
    # print(len(df_red[df_red.index.str.contains("glcm|gldm|glrlm|glszm")]), "fts having glcm / gldm / glrlm / glszm in name")
    # print(fts_red)
    df_delta_p_red_orig = df_delta_p_red.copy()
    fts_red += ["time"] if not ("time" in fts_red) else []  # INCLUDE TIME
    fts_red += ["dose"] if not ("dose" in fts_red) else []  # INCLUDE DOSE
    df_delta_p_red = df_delta_p_red[fts_red]     # INCLUDE TIME?

    print("TIME DATA", np.unique(df_delta_p_red["time"].values, return_counts=True))  # todo: maybe use times (-7, 5, 8) data (N=110) for prediction of (26, 56) days (N=28) - i.e. approx 80 / 20 split for train / test

    # TODO: FIX XER THRESHOLD STUFF
    # y_saliv_delta_orig["ctr"] = [not(d > 0) for d in y_saliv_delta_orig["dose"].values]
    # sal_med = np.median(y_saliv_delta_orig[y_saliv_delta_orig["ctr"] == True]["val"].values)
    # y_saliv_delta_orig["xer"] = [v < sal_med * 3 / 4 for v in y_saliv_delta_orig["val"].values]
    # y_saliv_delta_xer = y_saliv_delta_orig["xer"]
    y_saliv_delta_xer = binary_thresh_xerostomia(dfy=y_saliv_delta_orig, plot=True)
    print(f"Saliv: xer for {len(y_saliv_delta_xer[y_saliv_delta_xer])} of {len(y_saliv_delta_xer)}")

    # y_saliv_delta_orig_log["ctr"] = [not(d > 0) for d in y_saliv_delta_orig_log["dose"].values]
    # sal_log_med = np.median(y_saliv_delta_orig_log[y_saliv_delta_orig_log["ctr"] == True]["val"].values)
    # y_saliv_delta_orig_log["xer"] = [v < sal_log_med * 2 / 3 for v in y_saliv_delta_orig_log["val"].values]
    # y_saliv_delta_log_xer = y_saliv_delta_orig_log["xer"]
    # print(f"log(1 + saliv): xer for {len(y_saliv_delta_log_xer[y_saliv_delta_log_xer].values)} of {len(y_saliv_delta_log_xer)}")


    # BINARY CLASSIFICATION
    if mode.upper() in ["CLASSIFY", "CLASSIFICATION", "DISCRETE", "DISC"]:
        # X = df_delta_p_red.drop(["time", "dose"], axis=1)
        X = df_delta_p_red
        Y = y_saliv_delta_xer
    elif mode.upper() in ["REGRESSION", "CONTINOUS", "CONT"]:
        # X = df_delta_p_red.drop(["time", "dose"], axis=1)
        X = df_delta_p_red
        # Y = y_saliv_delta["val"]
        Y = y_saliv_delta_log if log else y_saliv_delta
    else:
        print("MODE", mode, "NOT VALID\nTRY: disc\t or \tcont")
        return 0
    print(f"---- LOADED DELTA-P DATA: X.shape={X.shape}, Y.shape={Y.shape}")
    return X, Y


def load_nop_saliv_data(slice="LR", mode="disc", univar_tresh=0.80, do_ranking=False):
    if slice.upper() == "LR":
        df = pd.read_csv(os.path.join(ftsDir, "LR_split_nyul_otsu_decile_extracted.csv"), index_col=0)  # LR
        # df_delta_p = pd.read_csv(os.path.join(ftsDir, "LR_split_nyul_otsu_decile_delta-p.csv"), index_col=0)
        df_ranked = pd.read_csv(os.path.join(selectDir, "LR_split_nyul_otsu_decile_nop_univariate_reduction.csv"), index_col=0)
    elif slice.upper() == "CENTER":
        df = pd.read_csv(os.path.join(ftsDir, "center_nyul_otsu_decile_extracted.csv"), index_col=0)  # CENTER
        # df_delta_p = pd.read_csv(os.path.join(ftsDir, "center_nyul_otsu_decile_delta-p.csv"), index_col=0)
        df_ranked = pd.read_csv(os.path.join(selectDir, "center_nyul_otsu_decile_nop_univariate_reduction.csv"), index_col=0)
    else:
        print("Slice(s)", slice, "not valid...\nTry LR or CENTER")
        return 0
    print("DATA LOADED", df.shape)
    print("----- REMOVING", np.count_nonzero(list(df[["T1" in x for x in df["name"].values]].T)), " ROWS HAVING T1 IN NAME -----")
    df = df[[not("T1" in x) for x in df["name"].values]]    # remove T1 images
    # print(df.info)
    # TODO: TRY COL-WISE FEATURE NORM / SCALING (for all data) BEFORE SELECTION -- CHECK IF DIFFERENT SELECTED
    feats = selected_features(mode="center", time="all", condition="", norm="nyul otsu decile", experiment="all", data=df, exclude_condition=False)
    feats.initial_reduction(remove_equal_vals=True)
    df = feats.data
    print("T1 DATA REMOVED:", df.shape)
    df = df[[not ("p" in x) for x in df["name"]]]  # REMOVE ALL P-DATA
    print("P-DATA REMOVED: ", df.shape)
    saliv_data = load_saliva() # RAW SALIV DATA WITH (MOSTLY) MISSING VALUES

    y_saliv_nop = register_name_to_outcome(df, saliv_data)
    y_saliv_nop_orig = y_saliv_nop.copy()
    y_saliv_nop = y_saliv_nop["val"]

    df = df.loc[y_saliv_nop.index].drop(["name"], axis=1)
    print("AFTER SALIV REG:", df.shape)

    fts = df.columns.values
    Nboot = 1000
    include_thresh = univar_tresh
    # TODO: UNCOMMENT FUNCTION(S) BELOW TO NOT OVERWRITE!!!!!!!!!!!!!!!!!!!
    # rank_features_univariate(df_delta_p_red, y_saliv_delta, fts=fts, pthresh=0.05, plot_regressions=False, savename="center_nyul_otsu_decile_nop_univariate_reduction.csv")   # Center
    # rank_features_univariate(df, y_saliv_nop, fts=fts, pthresh=0.05, num_bootstraps=Nboot, plot_regressions=False, savename="LR_split_nyul_otsu_decile_nop_univariate_reduction.csv") #LR
    # print(df_ranked[df_ranked["count"] > Nboot * include_thresh])     # FTS selected minimum e.g. 500 / 1000 times
    df_red = df_ranked[df_ranked["count"] > Nboot * include_thresh]
    fts_red = df_red.index.values.tolist()
    print(f"----- BOOTSTRAP UNIVAR SELECTED ({include_thresh*100:.0f}% thresh):", len(df_red), "fts", f"\tIncluded time={'''time''' in fts_red} dose={'''dose''' in fts_red}")
    # print(fts_red)
    fts_red += ["time"] if not("time" in fts_red) else []   # INCLUDE TIME
    fts_red += ["dose"] if not("dose" in fts_red) else []   # INCLUDE DOSE
    df_nop_red = df[fts_red] # INCLUDE TIME, DOSE
    print("AFTER BOOT:", df_nop_red.shape)
    print("TIME DATA:", np.unique(df_nop_red["time"].values, return_counts=True))

    # y_saliv_nop_orig["ctr"] = [d == 0 for d in y_saliv_nop_orig["dose"].values]
    y_saliv_nop_xer = binary_thresh_xerostomia(y_saliv_nop_orig, plot=False)
    # y_saliv_nop_xer = binary_thresh_xerostomia(plot=True)

    # BINARY CLASSIFICATION
    if mode.upper() in ["CLASSIFY", "CLASSIFICATION", "DISCRETE", "DISC"]:
        # X = df_delta_p_red.drop(["time", "dose"], axis=1)
        X = df_nop_red
        Y = y_saliv_nop_xer
    elif mode.upper() in ["REGRESSION", "CONTINOUS", "CONT"]:
        # X = df_delta_p_red.drop(["time", "dose"], axis=1)
        X = df_nop_red
        # Y = y_saliv_delta_log if log else y_saliv_delta
        Y = y_saliv_nop
    else:
        print("MODE", mode, "NOT VALID\nTRY: disc\t or \tcont")
        return 0
    print(f"---- LOADED NO-P DATA: X.shape={X.shape}, Y.shape={Y.shape}")
    return X, Y


def load_saliv_data_reg_to_future(mode="no-p"):
    # USE DATA FROM EARLY AFTER DOSE GIVEN (day 5, 8) TO PREDICT XER LATER (day 56, 105)
    # PREDICTORS: FTS + SALIV(day 5, 8)? OR ONLY FTS??? DOSE?????
    print("LOADING SALIV + FTS DATA: REGISTRATE EARLY MEASURED TO LATE")
    allowed_modes = ["delta-p", "no-p"]
    if mode.lower() == "delta-p":
        df_fts = pd.read_csv(os.path.join(ftsDir, "LR_split_nyul_otsu_decile_delta-p.csv"), index_col=0)
    elif mode.lower() == "no-p":
        df_fts = pd.read_csv(os.path.join(ftsDir, "LR_split_nyul_otsu_decile_extracted.csv"), index_col=0)  # LR
    if not(mode in allowed_modes):
        print("Mode ", mode, "not recognized - try using:", allowed_modes)
        return 0, 0
    print("FTS:", df_fts.shape)
    # saliv = load_saliva(impute=False, melt=True)
    saliv = load_saliva(melt=True, impute=False)
    print("TIMES IN SALIV:", np.unique(saliv["time"].values, return_counts=True))

    saliv_late = saliv[saliv["time"].isin([35, 56])]
    print("SALIV LATE:", saliv_late.shape)
    # print(saliv_late)
    saliv_early = saliv[saliv["time"].isin([5, 8])]
    print("SALIV EARLY:", saliv_early.shape)
    saliv_early = saliv_early[saliv_early["name"].isin(saliv_late["name"].values)]
    print("SALIV EARLY / LATE OVERLAP:", saliv_early.shape)

    saliv_early = register_name_to_outcome(df_fts, saliv_early.rename(columns={"val":"value"}), melt=False).rename(columns={"value": "val"})
    saliv_early_xer = binary_thresh_xerostomia(saliv_early, plot=True)
    print(saliv_early)
    saliv_late_xer = binary_thresh_xerostomia(saliv_late, plot=True)
    # print(saliv_late_xer)
    return 0, 0


def calculate_delta_time_LRfeatures_with_saliv(df_fts, relative_change=True, savename="", LRMODE="split"):
    # Calculate Delta feature (relative change): (Ft_afterirr - Ft_baseline) / Ft_baseline
    # For same ID: calculate Delta saliv: (Saliv_afterirr - Saliv_baseline) / Saliv_baseline
    #       AND: Register "late" saliv measurement (for xer thresholding? Regression?)
    if not LRMODE in ["split", "average", "aggregated"]:
        print(">>>>>>>LR MODE", LRMODE, "INVALID")
        return 0

    from validation_set_split import VALIDATION_SPLIT_ANALYSIS_SALIVA
    WEIGHT = "T2"   # maybe change if more T1 data comes later (have 9 now..)
    print("ORIGINAL FEATURE DF:\t", df_fts.shape)
    df_fts = df_fts[[not("p" in nm) for nm in df_fts["name"].values]]   # remove after-p
    print("\tRemoved after-p", df_fts.shape)

    ID_TRAIN, ID_VAL, x_train, x_val, y_train, y_val = VALIDATION_SPLIT_ANALYSIS_SALIVA(MODE="DELTA", WEIGHT=WEIGHT, verbose=False, plot=False, return_dataframes=True)
    df_id = pd.concat([x_train, x_val]) # MOUSE ID's WITH TIMES BASELINE + AFTER IRR
    # print(df_id)
    df_saliv_id = pd.concat([y_train, y_val])

    # MAKE DF: SALIVA TIMES + ID FOR BASELINE, AFTER IRR, LATE
    df_saliv_curr = pd.DataFrame()
    idx = 0
    for id in df_saliv_id.index.values:
        times_saliv = list(df_saliv_id.loc[id, "times"])
        times_saliv.sort()
        for t in times_saliv:
            df_saliv_curr.loc[idx, "time"] = str(t) + "day"
            df_saliv_curr.loc[idx, "name"] = id
            idx += 1

    print("ALL SALIVA TIMEPOINTS:", df_saliv_curr.shape)
    df_saliv = register_name_to_outcome(df_saliv_curr, load_saliva())
    if not(len(df_saliv) == len(df_saliv_curr)):
        print("\nMISMATCH IN SALIVA REGISTRATION -- EXITING")
        return 0

    df_main = pd.DataFrame()
    # REGISTER FTS, SALIVA, TO MRI IDS IN df_id
    j = 0
    for id in df_id.index.values:
        # id = "9-1"

        df_saliv_curr = df_saliv[df_saliv["name"] == id]
        saliv_times = list(df_saliv_curr["time"])
        id_times = list(df_id.loc[id, "times"])
        # print(id_times)
        id_times = [int(t[:-3]) for t in id_times]
        id_times.sort()
        saliv_times.sort()
        # print(id, int(id_times[0][:-3]) == saliv_times[0], id_times[0], saliv_times[0])
        if not id_times[0] == saliv_times[0]:
            print("MISMATCH ID TIMES WITH SALIV TIMES:", id_times, saliv_times, "CANNOT MAKE DELTA FEATURES...")
            return 0
        sal_vals = []
        for t_sal in saliv_times:   # FIND SALIVA VALUES FOR ID: baseline, after irr, late (all varying timepoints)
            val = df_saliv_curr[df_saliv_curr["time"] == t_sal]["val"].values[0]
            sal_vals.append(val)
        sal_baseline, sal_afterirr, sal_late = sal_vals
        sal_late_time = saliv_times[-1]
        delta_sal = (sal_afterirr - sal_baseline) / sal_baseline  # DELTA SALIVA: RELATIVE CHANGE

        print(id, id_times, saliv_times)
        df_fts_curr = df_fts[[id == name.split("_")[0] for name in df_fts["name"].values]]

        for X in ["L", "R"] if LRMODE == "split" else [""]:

            df_fts_curr_X = df_fts_curr[[X == name[-1] for name in df_fts_curr["name"].values]] if LRMODE == "split" else df_fts_curr

            # print(df_fts_curr_X)
            fts_baseline = df_fts_curr_X[df_fts_curr_X["time"] == str(id_times[0]) + "day"]
            fts_after_irr = df_fts_curr_X[df_fts_curr_X["time"] == str(id_times[1]) + "day"]

            if len(fts_baseline) == 0 or len(fts_after_irr) == 0:
                # print("Missing!")
                break

            elif len(fts_baseline) != 1 or len(fts_after_irr) != 1:
                print("FOUND MULTIPLE INSTANCES FOR ID, time, X:", id, X)
                print(f"BASELINE day {id_times[0]}:", fts_baseline)
                print(f"After irr day {id_times[1]}:", fts_after_irr)
                return 0
            dose = fts_after_irr["dose"].values[0]  # baseline dose is always 0...
            name = fts_after_irr["name"].values[0]
            if not name == fts_baseline["name"].values[0]:
                print("ERROR: DIFFERNT NAMES")
                print(name)
                print(fts_baseline["name"])
                return 0

            fts_baseline = fts_baseline.drop(["name", "time", "dose"], axis=1)
            ft_names = fts_baseline.columns.values
            fts_baseline = fts_baseline.values.flatten()
            fts_after_irr = fts_after_irr.drop(["name", "time", "dose"], axis=1).values.flatten()
            # print(len(fts_after_irr), len(fts_baseline))
            delta_features = [(after - base) / base if not base == 0 else 0 for base, after in zip(fts_baseline, fts_after_irr)]

            df_main.loc[j, ["name", "dose"]] = [name, dose]
            # df_main.loc[j, ["delta saliv", "saliv late"]] = [delta_sal, sal_late]
            print(saliv_times, sal_late_time)
            df_main.loc[j, ["delta saliv", "saliv late", "time saliv late"]] = [delta_sal, sal_late, sal_late_time]
            df_main.loc[j, ft_names] = delta_features
            print(name, j)
            j += 1

    if savename:
        df_main.to_csv(os.path.join(ftsDir, savename))
        print("DELTA FEATURES SAVED! shape=", df_main.shape)
    return 1


def make_average_fts_LRsplit(df, weight):
    df_l = df[[x[-1] == "L" for x in df["name"]]]
    df_r = df[[x[-1] == "R" for x in df["name"]]]
    print(df_r.shape)
    df_avg = pd.DataFrame()
    j = 0
    for idx, name, time, dose in zip(df_l.index.values, df_l["name"].values, df_l["time"].values, df_l["dose"].values):
        name_orig = name[:-2]   # remove _L
        print(j, name_orig, time)
        df_r_instance = df_r[np.logical_and(df_r["name"] == name_orig + "_R", df_r["time"] == time)]
        df_l_instance = df_l.loc[idx]

        if not len(df_r_instance == 1) or df_r_instance["dose"].values[0] != dose:
            print(">>>>>ERR: ", df_r_instance.shape)
            return 0
        else:
            df_r_instance = df_r_instance.iloc[0]

        df_temp = pd.DataFrame()
        df_temp.loc[j, ["name", "time", "dose"]] = [name_orig, time, dose]
        # print(df_temp)
        df_l_instance = df_l_instance.drop(["name", "time", "dose"], axis=0)
        df_r_instance = df_r_instance.drop(["name", "time", "dose"], axis=0)
        fts_names = df_l_instance.index.values
        fts_l = df_l_instance.values
        fts_r = df_r_instance.values
        # print(fts_r)
        # print(len(fts_names), len(fts_l), len(fts_r))
        fts_avg = np.average([fts_l, fts_r], axis=0)
        # print(np.shape(fts_avg))
        df_temp.loc[j, fts_names] = fts_avg
        j += 1
        df_avg = pd.concat([df_avg, df_temp], axis=0)
    print(df_avg.shape)
    savename = f"LR_average_FSPS_{weight}.csv"
    savepath = os.path.join(ftsDir, savename)
    df_avg.to_csv(savepath)
    return 1


def make_aggregate_fts_from_LRsplit(df, weight):
    # TAKE INSTANCE SPLIT INTO _L + _R, then aggregate into single row having 2x as many features

    df_l = df[[x[-1] == "L" for x in df["name"]]]
    df_r = df[[x[-1] == "R" for x in df["name"]]]
    print(df_r.shape, df_l.shape)
    df_agg = pd.DataFrame()
    j = 0
    for idx, name, time, dose in zip(df_l.index.values, df_l["name"].values, df_l["time"].values, df_l["dose"].values):
        name_orig = name[:-2]  # remove _L
        print(j, name_orig, time, dose)
        df_r_instance = df_r[np.logical_and(df_r["name"] == name_orig + "_R", df_r["time"] == time)]
        df_l_instance = df_l.loc[idx]

        if not len(df_r_instance == 1) or df_r_instance["dose"].values[0] != dose:
            print(">>>>>ERR: ", df_r_instance.shape)
            return 0
        else:
            df_r_instance = df_r_instance.iloc[0]

        df_temp = pd.DataFrame()
        df_temp.loc[j, ["name", "time", "dose"]] = [name_orig, time, dose]  # should be same values for L + R

        df_r_instance = df_r_instance.drop(["name", "time", "dose"], axis=0)
        df_l_instance = df_l_instance.drop(["name", "time", "dose"], axis=0)

        df_r_instance.index = [ind + "_R" for ind in df_r_instance.index.values]    # mark feature by L / R origins
        df_l_instance.index = [ind + "_L" for ind in df_l_instance.index.values]
        df_temp.loc[j, df_r_instance.index.values] = df_r_instance.values   # append both L + R features to same row j
        df_temp.loc[j, df_l_instance.index.values] = df_l_instance.values

        df_agg = pd.concat([df_agg, df_temp])
        # print(df_temp.shape)

        j += 1
        # break
    # print(df_agg)
    print(df_agg.shape)
    savename = f"LR_aggregated_FSPS_{weight}.csv"
    savepath = os.path.join(ftsDir, savename)
    df_agg.to_csv(savepath)
    return 1


if __name__ == "__main__":
    # UNIVAR FEATURE RANKING FOR FSPS FEATURES:
    # DO FOR NO P (T1 + T2 separately), DELTA (time), DELTA-P


    # MAKE LR AVERAGE / AGGREGATE FEATURES
    # for w in ["T1", "T2"]:
    #     df = pd.read_csv(os.path.join(ftsDir, f"LR_split_FSPS_extracted_{w}.csv"))
    #     df = df.drop(["name.1"], axis=1)
    #     print("original:", df.shape)
    #     make_aggregate_fts_from_LRsplit(df, w)
    #     # break
    #     # make_average_fts_LRsplit(df, w)
    # sys.exit()

    #Max-relevance minimum-redundancy feature selection and modelling:
    # from mrmr import mrmr_regression
    # df = load_fsps_data("T2", "NO P")
    # df_y = load_saliva(melt=False)
    # df_y = register_name_to_outcome(df, df_y, melt=True, make_70_exception=True)
    # df_y = df_y["val"]
    # df_red = df.loc[df_y.index.values].drop(["name"], axis=1)
    # df_red["time"] = [int(x[:-3]) for x in df_red["time"].values]
    #
    # mrmr_selected, scores, df = mrmr_regression(df_red, df_y, K=15, return_scores=True)
    # # _, scores2, _ = mrmr_regression(df_red, df_y, K=1, return_scores=True)
    # # print(scores2.equals(scores))
    # print(mrmr_selected)
    # print(df_red[mrmr_selected].T)

    # print(mrmr_selected)
    # print(df_mrmr)

    # sns.heatmap(np.abs(df_red.filter(mrmr_selected, axis=1).corr()), cmap="bwr")
    # plt.show()

    # MAKE DELTA-P FEATURES (LR SPLIT FSPS OPTIMIZED)
    # WEIGHT = "T2"
    # df = pd.read_csv(os.path.join(ftsDir, f"LR_split_FSPS_extracted_{WEIGHT}.csv"))
    # df = df.drop(["name.1"], axis=1)
    # delta_fts_p_to_nop(df, relative_change=True, savename="LR_split_FSPS_DELTA-P.csv")

    # DELTA-P: LR average
    # df = pd.read_csv(os.path.join(ftsDir, f"LR_average_FSPS_T2.csv"), index_col=0)
    # print(df.shape)
    # delta_fts_p_to_nop(df, relative_change=True, savename="LR_average_FSPS_DELTA-P.csv", LR_MODE="average")


    # MAKE DELTA LR average / aggregated
    LRmode = "aggregated"
    weight = "T2"
    df = pd.read_csv(os.path.join(ftsDir, f"LR_{LRmode}_FSPS_extracted_{weight}.csv"))
    # calculate_delta_time_LRfeatures_with_saliv(df, relative_change=True, savename=f"LR_{LRmode}_FSPS_DELTA-time_{weight}.csv", LRMODE=LRmode)
    delta_fts_p_to_nop(df, relative_change=True, savename=f"LR_{LRmode}_FSPS_DELTA-P.csv", LR_MODE=LRmode)
    sys.exit()


    # MAKE DELTA FEATURES (LR SPLIT FSPS)
    df = pd.read_csv(os.path.join(ftsDir, f"LR_split_FSPS_extracted_T2.csv"))
    df = df.drop(["name.1"], axis=1)
    calculate_delta_time_LRfeatures_with_saliv(df, relative_change=True, savename="LR_split_FSPS_DELTA-time.csv", LRMODE="split")
    df = load_fsps_data("T2", "DELTA", TRAIN_SET=True)  # have saliv data in df, no registration necessary at this time
    print(df.shape)


    # UNIVARIATE FEATURE RANKING ON FSPS DATA: NO P, DELTA-P, DELTA (time)
    print("UNIVARIATE RANKING FOR DELTA-time (only T2)")
    df = load_fsps_data("T2", "DELTA", TRAIN_SET=True)  # have saliv data in df, no registration necessary at this time
    dfy = df["saliv late"]
    # df = df.drop(["name", "dose", "delta saliv", "saliv late"], axis=1)
    df = df.drop(["name", "dose", "saliv late"], axis=1)    # try: keep delta saliv, check predictive power..
    fts = df.columns.values
    print("HAVE", len(fts), "FEATURES FOR RANKING FOR ", len(df), "DATAPOINTS")
    savename = "LR_split_FSPS_DELTA-time_univariate_ranking.csv"
    rank_features_univariate(df, dfy, fts=fts, savename=savename)


    for MODE in ["NO P", "DELTA P"]:
        weight_list = ["T2", "T1"] if MODE == "NO P" else ["T2"]
        for WEIGHT in weight_list:
            print("\nUNIVARIATE RANKING FOR", MODE, WEIGHT)
            df = load_fsps_data(WEIGHT, MODE, TRAIN_SET=True)
            # df = load_fsps_data(WEIGHT, MODE, TRAIN_SET=False, verbose=True)
            df_y = load_saliva(melt=False)
            df_y = register_name_to_outcome(df, df_y, melt=True, make_70_exception=True)
            df_y = df_y["val"]
            df_red = df.loc[df_y.index.values].drop(["name", "time", "dose"], axis=1)
            fts = df_red.columns.values
            print("HAVE", len(fts), "FEATURES FOR RANKING FOR ", len(df_red), "DATAPOINTS")
            savename = f"LR_split_FSPS_{'''_'''.join(MODE.split(''' '''))}_" + WEIGHT + "_univariate_ranking.csv"
            print(savename)
            rank_features_univariate(df_red, df_y, fts=fts, savename=savename)
    sys.exit()


    # df_area = load_sg_area("smg")   # val = measured area
    df_area = load_sg_area("all")   # val = measured area
    df_area["time"] = 105  # LAST IMAGING TIME FOR 9- and 8-
    df = pd.read_csv(os.path.join(ftsDir, "OLD_LR_split_nyul_otsu_decile_extracted.csv"), index_col=0)  # LR
    # df = pd.read_csv(os.path.join(ftsDir, "LR_split_FSPS_extracted_T2.csv"), index_col=0)  # LR
    # print("Times for 8- and 9-:", set(df[["8-" in x or "9-" in x for x in df["name"]]]["time"].values))
    # print(df.filter(regex="Surface", axis=1).T)
    # print(df["original_shape2D_PixelSurface"])
    df_reg = register_name_to_outcome(df=df, out=df_area, melt=False)
    df_reg["roi area"] = df["original_shape2D_PixelSurface"].loc[df_reg.index.values]   # ROI AREA VALUES
    # print(df_reg)
    # print(df["original_shape2D_PixelSurface"].loc[df_reg.index.values])
    # print(df_reg[["original_shape2D_PixelSurface", "dose", "name", "time"]])
    # names = df_reg["name"].values
    # print(names)
    df_reg.index = df_reg["name"]
    print(df_reg)
    x = 1
    names = []
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    xlabels = []
    for nm in df_area["name"]:
        df_sub = df_reg.filter(like=nm, axis=0)
        area_meas = df_sub["val"].values
        area_roi = df_sub["roi area"].values
        print(nm, len(df_sub))
        if len(df_sub) > 0:
            if x == 1:
                ln2 = ax2.plot([x] * len(area_roi), area_roi, "o", c="b", label="ROI area")
                ln1 = ax.plot([x] * len(area_meas), area_meas, "x", c="r", label="Measured area")
            else:
                ax2.plot([x]*len(area_roi), area_roi, "o", c="b")
                ax.plot([x]*len(area_meas), area_meas, "x", c="r")
            xlabels.append(nm)
            x += 1
    plt.xticks(np.linspace(1, x-1, x-1), xlabels)
    plt.xlabel("Mouse ID")
    ax.set_ylabel("Area [mm2]")
    ax2.set_ylabel("# pixels")
    plt.legend(ln1 + ln2, ["Measured area", "ROI area"])
    # plt.show()
    plt.close()
    # sys.exit()


    # df = pd.read_csv(os.path.join(ftsDir, "center_nyul_otsu_decile_extracted.csv"), index_col=0)  # CENTER
    df = pd.read_csv(os.path.join(ftsDir, "OLD_LR_split_nyul_otsu_decile_extracted.csv"), index_col=0)  # LR
    print("----- REMOVING", np.count_nonzero(list(df[["T1" in x for x in df["name"].values]].T)), " ROWS HAVING T1 IN NAME -----")
    df = df[[not("T1" in x) for x in df["name"].values]]    # remove T1 images
    # print(df.info)
    # TODO: TRY COL-WISE FEATURE NORM / SCALING (for all data) BEFORE SELECTION -- CHECK IF DIFFERENT SELECTED
    feats = selected_features(mode="center", time="all", condition="", norm="nyul otsu decile", experiment="all", data=df, exclude_condition=False)
    feats.initial_reduction(remove_equal_vals=True)
    # print(feats.data.info)
    df = feats.data
    # print(df.info)
    # delta_fts_p_to_nop(df, relative_change=True, savename="center_nyul_otsu_decile_delta-p.csv")     # RETURN DATA SET CONTAINING (relative) DELTA_P FEATURES: (ft_nop - ft_p) / ft_p
    # delta_fts_p_to_nop(df, relative_change=True, savename="LR_split_nyul_otsu_decile_delta-p.csv")    # LR SPLIT DELTA-P CALCULATION
    # df_delta_p = pd.read_csv(os.path.join(ftsDir, "center_nyul_otsu_decile_delta-p.csv"), index_col=0)  #CENTER
    df_delta_p = pd.read_csv(os.path.join(ftsDir, "OLD_LR_split_nyul_otsu_decile_delta-p.csv"), index_col=0)  #LR
    print("DF DELTA P:", df_delta_p.shape)

    # df_p = df[["p" in x for x in df["name"].values]]
    # df_nop = df.drop(df_p.index, axis=0)
    # print(f"ALL {len(df)} DATA SPLIT INTO {len(df_nop)} NOP AND {len(df_p)} P DATA")
    # fts = feats.data.drop(["name", "time", "dose"], axis=1).columns.values
    # print(f"---- Having d={len(fts)} features with n_nop={len(df_nop)}, n_p={len(df_p)} samples remaining for selection and modelling -----")
    # y_nop = df[["time", "dose"]].filter(items=df_nop.index.values, axis=0)
    # print(all(df_nop.index.values == y_nop.index.values))

    # LOAD / REGISTER SALIVA DATA TO INDIVIDUALS / TIMES
    saliv_data = load_saliva()
    print(saliv_data) # RAW SALIV DATA WITH (MOSTLY) MISSING VALUES
    print(df_delta_p)
    y_saliv_delta = register_name_to_outcome(df_delta_p, saliv_data)
    # print(df_delta_p.loc[y_saliv_delta.index])
    # sys.exit()

    y_saliv_delta_orig = y_saliv_delta
    # y_saliv_delta = y_saliv_delta.drop(["name", "time", "dose"], axis=1)    # reduce to only saliva values
    y_saliv_delta = y_saliv_delta["val"]    # reduce to only saliva values
    df_delta_p_red = df_delta_p.loc[y_saliv_delta.index].drop(["name"], axis=1)
    # fts = df_delta_p_red.columns.values
    fts = df.columns.values
    print("df shape= ", df.shape)

    # print(df_delta_p_red)
    # PLOT CORRELATION, PAIRWISE SCATTER, DISTRIBUTIONS + LOG-TRANSFORMED
    # print(y_saliv_delta)
    y_saliv_delta_orig_log = y_saliv_delta_orig.copy()
    y_saliv_delta_orig_log["val"] = np.log1p(y_saliv_delta_orig["val"].values)
    y_saliv_delta_log = y_saliv_delta_orig_log["val"]
    # y_saliv_delta_orig_log.rename(columns={"val":"log(1 + val)"}, inplace=True)   # rename val column to log(1 + val) FOR PLOTTING

    # PLOT CORR HEATMAP, PAIRGRID, HISTOGRAMS FOR SALIV + log(SALIV + 1)
    # fig, ax = plt.subplots(ncols=2)
    # sns.heatmap(y_saliv_delta_orig.corr(), annot=True, ax=ax[0])
    # sns.heatmap(y_saliv_delta_orig_log.corr(), annot=True, ax=ax[1])
    # plt.suptitle("Correlation heatmaps for saliva data (val). Left: raw val. Right: log(1 + val)")
    # # ax.plot(y_saliv_delta_orig["time"].values, y_saliv_delta_orig["val"].values, "x", label="time")
    # # ax.plot(y_saliv_delta_orig["dose"].values, y_saliv_delta_orig["val"].values, "x", label="dose")
    # # ax.legend()
    # g = sns.PairGrid(y_saliv_delta_orig)
    # g.map_offdiag(sns.scatterplot); g.map_diag(sns.histplot)
    # plt.suptitle("Saliva data")
    # g = sns.PairGrid(y_saliv_delta_orig_log)
    # g.map_offdiag(sns.scatterplot); g.map_diag(sns.histplot)
    # plt.suptitle("log(1 + saliva data)")
    #
    # fig, axx = plt.subplots(ncols=2)
    # axx[0].hist(y_saliv_delta["val"].values, bins=None)
    # axx[0].set_title("Saliva")
    # axx[1].hist(np.log1p(y_saliv_delta["val"].values), bins=None) # log(1 + x) of data values x
    # axx[1].set_title("log(1 + Saliva)")
    # plt.show()


    # UNIVARIATE FEATURE RANKING: COUNT NUMBER OF TIMES LINEAR MODEL HAVING p < 0.05
    Nboot = 1000
    include_thresh = 0.80
    # TODO: UNCOMMENT FUNCTION(S) BELOW TO NOT OVERWRITE!!!!!!!!!!!!!!!!!!!
    # print(df_delta_p_red.T.head(10))
    print(df_delta_p_red)
    # rank_features_univariate(df_delta_p_red, y_saliv_delta, fts=fts, pthresh=0.05, plot_regressions=False, savename="center_nyul_otsu_decile_delta-p_univariate_reduction.csv")   # Center
    # rank_features_univariate(df_delta_p_red, y_saliv_delta, fts=fts, pthresh=0.05, num_bootstraps=Nboot, plot_regressions=False, savename="LR_split_nyul_otsu_decile_delta-p_univariate_reduction.csv") #LR

    # df_ranked = pd.read_csv(os.path.join(selectDir, "center_nyul_otsu_decile_delta-p_univariate_reduction.csv"), index_col=0) # CENTER
    df_ranked = pd.read_csv(os.path.join(selectDir, "LR_split_nyul_otsu_decile_delta-p_univariate_reduction.csv"), index_col=0) # LR
    # print(df_ranked[df_ranked["count"] > 500])     # FTS selected 500 / 1000 times
    df_red = df_ranked[df_ranked["count"] > Nboot * include_thresh]
    print(f"----- BOOTSTRAP UNIVAR SELECTED ({include_thresh*100:.0f}% thresh):", len(df_red), "fts")
    # print(df_red)
    # print(df.T[df.columns.str.contains("gradient_n")])
    # print(df[df.index.str.contains(["glcm", "gldm", "glrlm", "glszm"])])
    print(len(df_red[df_red.index.str.contains("glcm|gldm|glrlm|glszm")]), "fts having glcm / gldm / glrlm / glszm in name")
    fts_red = df_red.index.values.tolist()
    # print(fts_red)
    df_delta_p_red_orig = df_delta_p_red.copy()
    df_delta_p_red = df_delta_p_red[fts_red + ["time", "dose"]]     # INCLUDE TIME?
    # print(df_delta_p_red)

    # PLOT DISTRIBUTIONS OF ALL FEATURES
    # count = 0
    # fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    # ax = ax.ravel()
    # for ft in df_delta_p_red.columns:
    #     if count == 9:
    #         count = 0
    #         plt.show()
    #         fig, ax = plt.subplots(3, 3)
    #         ax = ax.ravel()
    #     ax[count].hist(df_delta_p_red[ft], bins=50)
    #     ax[count].set_title(ft)
    #     count += 1
    # plt.show()

    # PLOTTING CORRELATION OF UNIVARIATE SELECTED DELTA-P FTS
    # f, ax = plt.subplots(figsize=(18, 18))
    # print(df_delta_p_red.drop(["time"], axis=1).corr())
    # sns.heatmap(df_delta_p_red.drop(["time"], axis=1).corr(), annot=False, linewidths=.125, fmt=' .1f', ax=ax)
    # plt.title(f"Heatmap of Pearson correlation between {len(df_delta_p_red.columns.values)} $\Delta_p$-features after bootstrapped univariate selection and modelling")
    # plt.show()

    # plot_fts_over_time(df_delta_p_red, fts=fts_red, hue="dose")

    # print(df_delta_p_red)
    # print(y_saliv_delta)

    print(np.unique(df_delta_p_red["time"].values, return_counts=True))  # todo: maybe use times (-7, 5, 8) data (N=110) for prediction of (26, 56) days (N=28) - i.e. approx 80 / 20 split for train / test
    y_saliv_delta_orig["ctr"] = [not(d > 0) for d in y_saliv_delta_orig["dose"].values]
    sal_med = np.median(y_saliv_delta_orig[y_saliv_delta_orig["ctr"] == True]["val"].values)
    # y_saliv_delta_orig["xer"] = [v < sal_med * 2 / 3 for v in y_saliv_delta_orig["val"].values] #todo: THIS IS WAYYYYYY WRONG -- ESTABLISH REL BETWEEN AGE AND EXPECTED SAL (TO PREDICT XER)
    y_saliv_delta_orig["xer"] = [v < sal_med * 3 / 4 for v in y_saliv_delta_orig["val"].values]
    y_saliv_delta_xer = y_saliv_delta_orig["xer"]
    # print(y_saliv_delta_xer)
    print(f"Saliv: xer for {len(y_saliv_delta_xer[y_saliv_delta_xer])} of {len(y_saliv_delta_xer)}")

    y_saliv_delta_orig_log["ctr"] = [not(d > 0) for d in y_saliv_delta_orig_log["dose"].values]
    sal_log_med = np.median(y_saliv_delta_orig_log[y_saliv_delta_orig_log["ctr"] == True]["val"].values)
    y_saliv_delta_orig_log["xer"] = [v < sal_log_med * 2 / 3 for v in y_saliv_delta_orig_log["val"].values]
    y_saliv_delta_log_xer = y_saliv_delta_orig_log["xer"]
    print(f"log(1 + saliv): xer for {len(y_saliv_delta_log_xer[y_saliv_delta_log_xer].values)} of {len(y_saliv_delta_log_xer)}")


    # BINARY CLASSIFICATION
    # X = df_delta_p_red.drop(["time", "dose"], axis=1)
    # Y = y_saliv_delta_xer
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=12)
    # rf_hyperparamtuning(x_train, y_train)
    # print(f"\n70 / 30 Split: N_train={len(y_train)}, N_test={len(y_test)}")
    # random_forest_classifier(x_train, x_test, y_train, y_test, verbose=True, random_state=None)

    # for spl in np.linspace(0.1, 0.9, 9):
    #     print("\n" + str(spl))
    #     x_train, x_test, y_train, y_test = train_test_split(df_delta_p_red.drop(["time", "dose"], axis=1), y_saliv_delta_xer, test_size=spl, random_state=12)
    #     acc, auc, cm = random_forest_classifier(x_train, x_test, y_train, y_test, random_state=None)
    #     plt.plot(spl, acc, "x", c="b")
    #     plt.plot(spl, auc, "o", c="r")
    # plt.show()

    # accmeans = []
    # accstratmeans = []
    # accsds = []
    # for k in np.arange(2, 10, 1, dtype=int):
    #     rf = RandomForestClassifier(random_state=1)
    #     cv = KFold(n_splits=k, random_state=None, shuffle=False)
    #     cvstrat = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    #     # for train, test in cv.split(X, Y):
    #     #     print(train.shape, test.shape)
    #     acc_scores = cross_val_score(rf, X, Y, cv=cv, n_jobs=-1, scoring="accuracy")
    #     auc_scores = cross_val_score(rf, X, Y, cv=cv, n_jobs=-1, scoring="roc_auc")
    #     acc_scores_strat = cross_val_score(rf, X, Y, cv=cvstrat, n_jobs=-1, scoring="accuracy")
    #     print(f"K-fold CV with k={k} having means: acc={np.mean(acc_scores):.3f}, auc={np.mean(auc_scores):.3f}")
    #     # print(acc_scores)
    #     acc_mean = np.mean(acc_scores)
    #     acc_sd = np.std(acc_scores)
    #     plt.plot(k, np.mean(acc_scores), "x", c="b")
    #     plt.errorbar(k, acc_mean, yerr=acc_sd, c="b")
    #     acc_strat_mean = np.mean(acc_scores_strat)
    #     plt.plot(k+0.1, acc_strat_mean, "x", c="r")
    #     plt.errorbar(k+0.1, acc_strat_mean, yerr=np.std(acc_scores_strat), c="r")
    # plt.show()

    # rf = RandomForestClassifier(criterion="entropy", min_samples_leaf=1, min_samples_split=16, n_estimators=1000, random_state=None)
    # # kvals = np.arange(2, 11, 1, dtype=int)
    # kvals = np.arange(5, 65, 10, dtype=int)
    # for i in range(3):
    #     accvals = []
    #     accstrat = []
    #     for k_splits in kvals:
    #         cv = KFold(n_splits=k_splits, random_state=None, shuffle=True)
    #         cvstrat = StratifiedKFold(n_splits=k_splits, random_state=None, shuffle=True)
    #         # for train, test in cv.split(X, Y):
    #         #     print(train.shape, test.shape)
    #         acc_scores = cross_val_score(rf, X, Y, cv=cv, n_jobs=-1, scoring="accuracy")
    #         # auc_scores = cross_val_score(rf, X, Y, cv=cv, n_jobs=-1, scoring="roc_auc")
    #         # print(f"{i}: K-fold CV with k={k_splits} having means: acc={np.mean(acc_scores):.3f}, auc={np.mean(auc_scores):.3f}")
    #         print(f"{i}: K-fold CV with k={k_splits} having means: acc={np.mean(acc_scores):.3f} ({np.std(acc_scores):.3f})")
    #         accvals.append(np.mean(acc_scores))
    #         # plt.plot(k_splits, np.mean(acc_scores), "x", c="b")
    #         # plt.errorbar(k_splits, np.mean(acc_scores), yerr=np.std(acc_scores), c="b")
    #         accstrat.append(np.mean(cross_val_score(rf, X, Y, cv=cvstrat, n_jobs=-1, scoring="accuracy")))
    #     plt.plot(kvals, accvals, "x-", c="r", label="No strat") if i == 0 else plt.plot(kvals, accvals, "x-", c="r")
    #     plt.plot(kvals, accstrat, "o-", c="b", label="Strat") if i == 0 else plt.plot(kvals, accstrat, "o-", c="b")
    # plt.title("RF accuracy for 10 repeated k-fold with varying k")
    # plt.ylabel("accuracy")
    # plt.xlabel("k")
    # plt.legend(loc="best")
    # plt.show()


    # # REGRESSION
    # X = df_delta_p_red.drop(["time", "dose"], axis=1)
    # X = df_delta_p_red
    # Y = y_saliv_delta_log
    # print("\nREGRESSION ON CONTINOUS DATA X, Y = ", X.shape, Y.shape)
    # todo: try regression on log(saliva val)

    ## x_train, x_test, y_train, y_test = train_test_split(df_delta_p_red.drop(["time", "dose"], axis=1), y_saliv_delta["val"], test_size=0.3, random_state=12)
    # x_train, x_test, y_train, y_test = train_test_split(df_delta_p_red, y_saliv_delta["val"], test_size=0.5, random_state=12)
    # # print(y_saliv_delta["val"])
    # print(f"\n70 / 30 Split: N_train={len(y_train)}, N_test={len(y_test)}")
    # # rf_hyperparamtuning(x_train, y_train, mode="regression")
    # from sklearn.model_selection import RepeatedKFold
    # rkf = RepeatedKFold(n_repeats=10, n_splits=5)
    # for train_idx, test_idx in rkf.split(x_train):
    #     print(len(train_idx), len(test_idx))
    #     # print(len(x_train[train_idx]))
    # random_forest_regression(x_train, x_test, y_train, y_test, verbose=True, random_state=42)

    # rf = RandomForestRegressor(random_state=None, n_estimators=1000, max_depth=16, min_samples_leaf=1, min_samples_split=2, criterion="squared_error")
    # kvals = np.arange(2, 21, 1, dtype=int)
    # Nrep = 5
    # for i in range(Nrep):
    #     r2vals = []
    #     r2strat = []
    #     for k_splits in kvals:
    #         cv = KFold(n_splits=k_splits, random_state=None, shuffle=True)
    #         # cvstrat = StratifiedKFold(n_splits=k_splits, random_state=None, shuffle=True)
    #         r2scores = cross_val_score(rf, X, Y, cv=cv, n_jobs=-1, scoring="r2")
    #         print(f"{i}: K-fold CV with k={k_splits} having means: r2={np.mean(r2scores):.3f} ({np.std(r2scores):.3f})")
    #         r2vals.append(np.mean(r2scores))
    #         # r2strat.append(np.mean(cross_val_score(rf, X, Y, cv=cvstrat, n_jobs=-1, scoring="r2")))
    #     plt.plot(kvals, r2vals, "x-", c="r", label="No strat") if i == 0 else plt.plot(kvals, r2vals, "x-", c="r")
    #     # plt.plot(kvals, r2strat, "o-", c="b", label="Strat") if i == 0 else plt.plot(kvals, r2strat, "o-", c="b")
    # plt.title(f"RF accuracy for {Nrep} repeated k-fold with varying k")
    # plt.ylabel("R2")
    # plt.xlabel("k")
    # plt.grid()
    # # plt.legend(loc="best")
    # plt.show()

    # REPEATED RF MODELLING ON Nrep TRAIN / TEST SPLTS -> AVG ACC, AUC, CM
    # Nrep = 50
    # accvals = []
    # aucvals = []
    # cmvals = []
    # x_train, x_test, y_train, y_test = train_test_split(df_delta_p_red.drop(["time", "dose"], axis=1), y_saliv_delta_xer, test_size=0.3, random_state=None)
    # print(f"\n70 / 30 Split: N_train={len(y_train)}, N_test={len(y_test)}")
    # for i in range(Nrep):
    #     x_train, x_test, y_train, y_test = train_test_split(df_delta_p_red.drop(["time", "dose"], axis=1), y_saliv_delta_xer, test_size=0.3, random_state=None)
    #     # print(f"\n70 / 30 Split: N_train={len(y_train)}, N_test={len(y_test)}")
    #     # random_forest_classifier(df_delta_p_red.drop(["time", "dose"], axis=1), y_saliv_delta_xer)
    #     acc, auc, cm = random_forest_classifier(x_train, x_test, y_train, y_test, verbose=False)
    #     print(i, f"acc={acc:.3f}, auc={auc:.3f}")
    #     accvals.append(acc)
    #     aucvals.append(auc)
    #     cmvals.append(cm)
    # print(f"Average values after {Nrep} train/ test splits:")
    # print(f"Accuracy = {np.mean(accvals):.5f}")
    # print(f"ROC AUC = {np.mean(aucvals):.5f}")
    # print(np.shape(cmvals))
    # cmavg = np.mean(cmvals, axis=0)
    # print(cmavg)
    # sns.heatmap(cmavg, annot=True)
    # plt.show()


    # random_forest_predict(df_delta_p_red, y_saliv_delta_xer)

    # sns.catplot(x="time", y="val", kind="swarm", data=y_saliv_delta_orig, hue="ctr")
    # sns.catplot(x="time", y="val", kind="swarm", data=y_saliv_delta_orig, hue="xer")
    # plt.show()




    # feats.data = df_delta_p_red
    # for M in [0.9, 0.8, 0.7, 0.6, 0.5]:
    #     # feats.feature_reduction_absolute_correlation(thresh=M, save=True, savepath=os.path.join(selectDir, "absolute average correlation reduction", f"center_nyul_otsu_decile_delta-p_thresh={M:.1f}.csv"))
    #     feats.feature_reduction_absolute_correlation(thresh=M, save=False)
    #     feats.data.corr()
    #     df = feats.data
    #     f, ax = plt.subplots(figsize=(18, 18))
    #     sns.heatmap(df.drop(["time", "dose"], axis=1).corr(), annot=True, linewidths=.125, fmt=' .1f', ax=ax)
    #     plt.title(f"Heatmap of Pearson correlation between {len(df.columns.values)-2} $\Delta_p$-features after bootstrapped univariate selection and modelling")
    #     plt.show()

    # y_nop_saliv = register_name_to_outcome(df_nop, saliv_data)    # RETURN ORDERED LIST OF ENDPOINT (e.g. saliva) CORRESPONDING TO name, time IN df
    # y_p_saliv = register_name_to_outcome(df_p, saliv_data)
    # print(all(np.equal(df_nop.loc[y_nop_saliv.index.values].index.values, y_nop_saliv.index.values)))

    # print(y_nop)
    # print(df["lbp-2D_firstorder_90Percentile"].values)
    # count_long_data(df_nop)
    # rank_features_univariate(df=df_nop, df_y=y_nop, fts=fts, plot_regressions=False)
    # print(fts)

    # print(df.T[df.columns.str.contains("original_glcm")].head(5))
    # print(df.T[df.columns.str.contains("lbp")])
    # print(df["lbp-2D_firstorder_InterquartileRange"])
    # print(np.count_nonzero(list(df[["lbp" in x for x in df.columns]])))

    # feats.corr_matrix(plot=True)
    # for key in df.drop(["name", "time", "dose"], axis=1).columns:
    #     print(key)
    #     print(df[key])
        # sns.boxplot(data=df, x="time", y=key, hue="dose")
        # sns.boxplot(data=df, x="time", y=key, hue="dose")
        # sns.histplot(data=df, x=key, hue="time", element="step")
        #         # sns.boxplot(x="time", y=key, data=timedata, color='#99c2a2', ax=ax1)
        #         # sns.swarmplot(x="time", y=key, data=timedata, color='#7d0013', ax=ax1)
        # plt.show()
        # break



    # feats = selected_features(mode=mode, time=time, condition=condition, exclude_condition=True, norm=norm, experiment=experiment)
    # feats.initial_reduction(floats_excluded=["diagnostics_Image-original_Mean"], ints_included=["diagnostics_Mask-original_VoxelNum"], remove_equal_vals=True)
    # main_AACR(feats, save=False)
    # plot_reduced_AACR(log=True)

    # M = 0.2
    # fts_AACR_path = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction",
    #                  feats.experiment, feats.time, "_".join((feats.norm, feats.time, feats.mode, f"fts_remaining_after_summedcorrselection_thresh={M}.csv"))))
    # fts_remaining = pd.read_csv(fts_AACR_path, index_col=0, squeeze=True)
    # feats.data = feats.data[fts_remaining]
    # print(feats.data)
    # print(feats.time)


    # plot_groupings_AACR(experiment=experiment, log=True, groupedby="image filter")
    # plot_groupings_AACR(groups=["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"], experiment=experiment, log=True, groupedby="feature class")


    # #ANOVA analysis of features grouped by TIME (control + all at -7day) #https://www.reneshbedre.com/blog/anova.html
    # time = "all times"
    # experiment = "pilot1 + pilot2"
    # fts_minus7day = selected_features(mode=mode, time="-7day", condition="p", exclude_condition=True, norm="stscore norm", experiment="pilot1 + pilot2")
    # fts_8day = selected_features(mode=mode, time="8day", condition="p", exclude_condition=True, norm="stscore norm", experiment="pilot1 + pilot2")
    # fts_26day = selected_features(mode=mode, time="26day", condition="p", exclude_condition=True, norm="stscore norm", experiment="pilot2")
    # fts_56day = selected_features(mode=mode, time="56day", condition="p", exclude_condition=True, norm="stscore norm", experiment="pilot1")
    # # print(fts_minus7day.data)
    # # print(fts_8day.data)
    # # print(fts_26day.data)
    # # print(fts_56day.data)
    # fts_all = selected_features(mode=mode, time="-7day", condition="p", exclude_condition=True, norm="stscore norm", experiment="pilot1 + pilot2")
    # fts_all.data = pd.concat([fts_minus7day.data, fts_8day.data, fts_26day.data, fts_56day.data])
    # # print(fts_all.data)
    # fts_all.initial_reduction(floats_excluded=["diagnostics_Image-original_Mean"], ints_included=["diagnostics_Mask-original_VoxelNum"], remove_equal_vals=True)
    # # print(fts_all.data)
    # fts_all.experiment="pilot1 + pilot2"
    # fts_all.time = "all times"
    #
    # # print(fts_all.experiment, fts_all.time)
    # # fts_all.corr_matrix(plot=True)
    # # main_AACR(fts_all, save=True)
    # # features_remaining = load_aacr_features(experiment, time, norm, mode, thresh=0.9)
    # features_remaining = load_aacr_features(experiment, time, norm, mode, thresh=1.1)   # all features basically
    # names_control = fts_all.find_by_dose(doserate=0)
    # print(names_control)
    #
    # for fts in [fts_minus7day, fts_8day, fts_26day, fts_56day, fts_all]:
    #     fts.data = fts.data[features_remaining]
    #
    #     # print(fts.data)
    # # print(fts_all.data)
    # # print(fts.data.index.values)
    #
    # names = names_control       # Choosing the control data for tests of mean, variance - "stability over time"
    # # names = fts_all.data.index
    # num = 1
    # for key, val in six.iteritems(fts_all.data):
    #     timedata = pd.DataFrame(columns=["", ""])   #make concatinated data on time, dose, etc. for SINGLE feature (then anova over time/ dose)
    #     print("\n", f"{num} / {len(features_remaining)}", key)
    #     num += 1
    #     timedata.columns=["time", key]
    #     names_with_times = []
    #     # print(timedata)
    #     # print(fts_all.data.info)
    #     for fts in [fts_minus7day, fts_8day, fts_26day, fts_56day]:
    #         df = fts.data[key].to_frame()
    #         names_in_time = list(set(names).intersection(set(df.index.values)))     # MICE WITH CORRECT DOSERATE IN CURRENT TIME
    #         names_with_times.extend(names_in_time)
    #         # print(names_in_time)
    #         df.loc[:, "time"] = fts.time
    #         # data = pd.concat([data, df])
    #         timedata = timedata.append(df)
    #
    #     #   https://www.reneshbedre.com/blog/anova.html ANOVA IN PYTHON
    #     timedata_ctrl = timedata.loc[names_with_times]
    #     # print(timedata_ctrl)
    #     # print()
    #     # print(timedata_ctrl[timedata_ctrl["time"] == "8day"][key])
    #     # print(pd.pivot(timedata_ctrl, columns="time", values=key))
    #     timedata_ctrl_time0 = timedata_ctrl[timedata_ctrl["time"] == "-7day"][key]
    #     timedata_ctrl_time1 = timedata_ctrl[timedata_ctrl["time"] == "8day"][key]
    #     timedata_ctrl_time2 = timedata_ctrl[timedata_ctrl["time"] == "26day"][key]
    #     timedata_ctrl_time3 = timedata_ctrl[timedata_ctrl["time"] == "56day"][key]
    #
    #     # VARIANCE ANALYSIS: CAN ANOVA BE USED? FIRST EVALUATE DIFFERENCE IN VARIANCE BETWEEN CONTROL GROUPS OVER TIME  https://www.statology.org/anova-unequal-sample-size/
    #     print(f"Variance for each group: {timedata_ctrl_time0.var():.3e}, {timedata_ctrl_time1.var():.3e}, {timedata_ctrl_time2.var():.3e}, {timedata_ctrl_time3.var():.3e}")
    #     chisq, pval_bart = stats.bartlett(timedata_ctrl_time0, timedata_ctrl_time1, timedata_ctrl_time2, timedata_ctrl_time3)
    #     print(f"Bartlett's test for equal variances: \tchisq = {chisq:.2f} -> pval = {pval_bart:.2f}")
    #     stat, pval_lev = stats.levene(timedata_ctrl_time0, timedata_ctrl_time1, timedata_ctrl_time2, timedata_ctrl_time3)
    #     print(f"Levene test for equal variances: \tstatistic = {stat:.2f} -> pval = {pval_lev:.2f}")
    #
    #     fval, pval = stats.f_oneway(timedata_ctrl[timedata_ctrl["time"] == "-7day"][key], timedata_ctrl[timedata_ctrl["time"] == "8day"][key],
    #                                 timedata_ctrl[timedata_ctrl["time"] == "26day"][key], timedata_ctrl[timedata_ctrl["time"] == "56day"][key])
    #     print(f"ANOVA on control data over time: \t\t\tF-value = {fval:.2f} -> pval={pval:.4g}")
    #     # stats.iqr()
    #     hval, pval_kru = stats.kruskal(timedata_ctrl_time0, timedata_ctrl_time1, timedata_ctrl_time2, timedata_ctrl_time3)
    #     print(f"Kruskal-Wallis test on control data over time: F-value = {fval:.2f} -> pval={pval:.4g}")
    #     # if pval < 0.10:
    #     # if pval_lev > 0.9:
    #     # if pval_kru > 0.95:
    #     if pval_kru < 0.05:
    #         # fig, axes = plt.subplots(1, 2, sharey=True)
    #         fig, axes = plt.subplots(1, 2, sharey=False)
    #         ax1, ax2 = axes.ravel()
    #         # sns.boxplot(x="time", y=key, data=timedata, color='#99c2a2', ax=ax1)
    #         # sns.swarmplot(x="time", y=key, data=timedata, color='#7d0013', ax=ax1)
    #         # print(timedata_ctrl)
    #         # print(key)
    #         sns.histplot(data=timedata_ctrl, x=key, ax=ax1, hue="time", multiple="stack")
    #         # sns.histplot(data=timedata_ctrl, x=key, ax=ax1, hue="time", element="step")
    #
    #         sns.boxplot(x="time", y=key, data=timedata_ctrl, color='#99c2a2', ax=ax2)
    #         sns.swarmplot(x="time", y=key, data=timedata_ctrl, color='#7d0013', ax=ax2)
    #         # sns.kdeplot(x=[-7, 8, 26, 56], y=key, data=timedata_ctrl, ax=ax2)
    #         # ax1.set_title("All data")
    #         ax2.set_title("Control")
    #         fig.suptitle(key)
    #         plt.show()
    #     # break


    # cols = [ft for ft in feats.data.columns if "lbp" in ft]   #local back pain
    # feats.data = feats.data[cols]
    # print(feats.data)
    # import researchpy as rp
    # from scipy.stats import f_oneway
    # for key, vals in six.iteritems(feats.data):
    #     print(key, vals)
    #     cont = rp.summary_cont(feats.data[key])
    #     print(cont)
    #     anova = f_oneway(vals)
    #     print(anova)
    #     break


    #SORT OUT FEATURES OF INTEREST BY DATATYPE(S)
    # print(feats.data)
    # print(feats.sort_cols_by_dtype()[np.dtype("object")])  # check dtype values to see if any are of interest
    # for ft in feats.sort_cols_by_dtype()[np.dtype("object")]:
    #     print(ft)
    #     print(feats.data[ft])
    # print(feats.data["diagnostics_Image-original_Dimensionality"])
    # dropped_equal_fts_pilot1 = ['gradient_firstorder_Entropy', 'gradient_firstorder_Uniformity', 'gradient_glcm_Autocorrelation', 'gradient_glcm_JointAverage', 'gradient_glcm_ClusterProminence', 'gradient_glcm_ClusterShade', 'gradient_glcm_ClusterTendency', 'gradient_glcm_Contrast', 'gradient_glcm_Correlation', 'gradient_glcm_DifferenceAverage', 'gradient_glcm_DifferenceEntropy', 'gradient_glcm_DifferenceVariance', 'gradient_glcm_JointEnergy', 'gradient_glcm_JointEntropy', 'gradient_glcm_Imc1', 'gradient_glcm_Imc2', 'gradient_glcm_Idm', 'gradient_glcm_Idmn', 'gradient_glcm_Id', 'gradient_glcm_Idn', 'gradient_glcm_InverseVariance', 'gradient_glcm_MaximumProbability', 'gradient_glcm_SumEntropy', 'gradient_glcm_SumSquares', 'gradient_glrlm_GrayLevelNonUniformityNormalized', 'gradient_glrlm_GrayLevelVariance', 'gradient_glrlm_HighGrayLevelRunEmphasis', 'gradient_glrlm_LowGrayLevelRunEmphasis', 'gradient_glszm_GrayLevelNonUniformityNormalized', 'gradient_glszm_GrayLevelVariance', 'gradient_glszm_HighGrayLevelZoneEmphasis', 'gradient_glszm_LowGrayLevelZoneEmphasis', 'gradient_gldm_GrayLevelVariance', 'gradient_gldm_HighGrayLevelEmphasis', 'gradient_gldm_LowGrayLevelEmphasis', 'gradient_ngtdm_Busyness', 'gradient_ngtdm_Coarseness', 'gradient_ngtdm_Complexity', 'gradient_ngtdm_Contrast', 'gradient_ngtdm_Strength', 'lbp-2D_firstorder_90Percentile', 'lbp-2D_firstorder_Maximum', 'lbp-2D_firstorder_Minimum', 'lbp-2D_firstorder_Range']
    # CV = feats.coeff_var()
    # for ft in dropped_equal_fts_pilot1:
    #     if float(CV[ft]) > 1:
    #         print(feats.data[ft])
    #         print("CV = ", CV[ft], "\n")


    # print(dropped_equals)
    # fts = feats.data.columns.values.tolist()  # remaining features (columns) in feats.data

    #STDEV & CV ANALYSIS
    # sigma_counts = feats.sigma_eval(dtype="float64")
    # print(sigma_counts)

    # cv = feats.coeff_var()
    # print(cv.sort_values())


    #CORRELATION MATRIX ANALYSIS
    # corrmatr = feats.corr_matrix(features=fts, plot=False)
    # savepath = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling",
    #                                          norm.replace(" ", "_") + "_" + time + "_" + mode.replace(" ", "_") + "_correlation_matrix" + ".csv"))
    # corrmatr.to_csv(savepath)     #SAVE AS CVS
    # corrmatr = pd.read_csv(savepath, index_col=0)  #READ FROM CSV

    # print(list(map(lambda v: v in fts_remaining2.values, fts_remaining.values)))  #SEE WHAT FEATURES ARE IN BOTH REMAINING FEATURE SERIES


    #CORR MATR SELECTION: PLOT # FEATURES SURVIVING VARIOUS ELIMINATION THRESHOLDS
    # print(feats.data)
    # num_fts_original = feats.data.shape[1]
    # feats.corr_matrix(plot=True, plotname=f"all_features")
    # print(num_fts_original)
    # feats.feature_reduction_absolute_correlation(thresh=1.1, save=True)
    #FEATURE FILTERING BY AVERAGE ABSOLUTE CORRELATION
    # Y = []
    # Y2 = []
    # Yboth = []
    # common1 = [];   common2 = [];   common3 = [];
    # plot = False
    # Mvals = [0.999, 0.99, 0.98, 0.97, 0.95, 0.925] + [round(x, 2) for x in np.linspace(0.9, 0.2, 15).tolist()]
    # for M in Mvals:
    #     print("M =", M)
    #     if plot:
    #         ftspath_pilot1 = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", "pilot1", feats.time, "_".join((feats.norm, feats.time, feats.mode, f"fts_remaining_after_summedcorrselection_thresh={M}")) + ".csv"))
    #         ftspath_pilot2 = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", "pilot2", feats.time, "_".join((feats.norm, feats.time, feats.mode, f"fts_remaining_after_summedcorrselection_thresh={M}")) + ".csv"))
    #         ftspath_both = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", "pilot1 + pilot2", feats.time, "_".join((feats.norm, feats.time, feats.mode, f"fts_remaining_after_summedcorrselection_thresh={M}.csv"))))
    #         # ftspath = os.path.normpath(os.path.join(os.getcwd(), "..", "Radiomic feature selection and modelling", "absolute average correlation reduction", feats.experiment, feats.time, "_".join((feats.norm, feats.time, feats.mode, f"fts_remaining_after_summedcorrselection_thresh={M}")) + ".csv"))
    #         fts_remaining_1 = pd.read_csv(ftspath_pilot1, index_col=0, squeeze=True)
    #         fts_remaining_2 = pd.read_csv(ftspath_pilot2, index_col=0, squeeze=True)
    #         fts_remaining_both = pd.read_csv(ftspath_both, index_col=0, squeeze=True)
    #         # print(len(fts_remaining))
    #         Y.append(len(fts_remaining_1))
    #         Y2.append(len(fts_remaining_2))
    #         Yboth.append(len(fts_remaining_both))
    #         # feats.data = feats.data[fts_remaining]
    #         # print(feats.data.shape)
    #         overlap1 = np.intersect1d(fts_remaining_both, fts_remaining_1)  #P1 fts overlap with P1 + P2
    #         overlap2 = np.intersect1d(fts_remaining_both, fts_remaining_2)  #P2 fts overlap with P1 + P2
    #         overlap3 = np.intersect1d(fts_remaining_1, fts_remaining_2)     #P1 fts overlap with P2
    #         common1.append(len(overlap1));  common2.append(len(overlap2));  common3.append(len(overlap3))
    #         print(f"At M={M} P1 have {len(fts_remaining_1)} fts remaining, P1 + P2 have {len(fts_remaining_both)} with {len(overlap1)} in common.")
    #         print(f"At M={M} P2 have {len(fts_remaining_2)} fts remaining, P1 + P2 have {len(fts_remaining_both)} with {len(overlap2)} in common.")
    #         print(f"At M={M} P1 have {len(fts_remaining_1)} fts remaining, P2 have {len(fts_remaining_2)} with {len(overlap3)} in common.")
    #         print()
    #     #     corrmatr = feats.corr_matrix(plot=True, plotname=f"thresh={M:.3f}")
    # #     feats.feature_reduction_absolute_correlation(thresh=M, save=True)
    #     # print(feats.data)
    #     # M_prev = M
    #     # break
    #
    # if plot:
    #     sns.set_theme()
    #     Mvals.reverse();    Y.reverse()
    #     Y2.reverse();   Yboth.reverse()
    #     common1.reverse();  common2.reverse();  common3.reverse()
    #     # Mvals.append(1.000);    Y.append(num_fts_original)
    #     log = True
    #     # log = False
    #     if not log:
    #         plt.plot(Mvals, Y, "--" + "x", color="b", label="Pilot1 (6 datapoints per ft)")
    #         plt.plot(Mvals, Y2, "--" + "o", color="r", label="Pilot2 (8 datapoints per ft)")
    #         plt.plot(Mvals, Yboth, "--" + "+", color="g", label="Pilot1 + pilot2 concatenated (14)")
    #         plt.plot(Mvals, common1, ls=":", color="b", label="# fts shared P1 with P1 + P2")
    #         plt.plot(Mvals, common2, ls=":", color="r", label="# fts shared P2 with P1 + P2")
    #         plt.plot(Mvals, common3, ls=":", color="g", label="# fts shared P1 with P2")
    #     else:
    #         plt.semilogy(Mvals, Y, "--" + "o", color="b", label="Pilot1 (6 datapoints per ft)")
    #         plt.semilogy(Mvals, Y2, "--" + "x", color="r", label="Pilot2 (8 datapoints per ft)")
    #         plt.semilogy(Mvals, Yboth, "--" + "+", color="g", label="Pilot1 + pilot2 concatenated (14)")
    #         plt.semilogy(Mvals, common1, ls=":", color="b", label="# fts shared P1 with P1 + P2")
    #         plt.semilogy(Mvals, common2, ls=":", color="r", label="# fts shared P2 with P1 + P2")
    #         plt.semilogy(Mvals, common3, ls=":", color="g", label="# fts shared P1 with P2")
    #     plt.gca().invert_xaxis()
    #     # plt.title(f"Feature filtering by average absolute correlation, from {feats.experiment} with {feats.data.shape[0]} datapoints per feature.")
    #     plt.title(
    #         f"Feature filtering by average absolute correlation, for pilot1 and pilot 2 at time {feats.time}")
    #     plt.xlabel("Correlation threshold M")
    #     plt.ylabel("# features remaining" if not log else "log(# features remaining)")
    #     plt.legend(loc="best")
    #     plt.show()




    # matr = corrmatr[corrmatr.columns[:50]]
    # matr = matr[:50]
    # print(matr)

    # correlation_histogram(corrmatr, thresh)
    # corrfts_sorted = feature_correlations_sorted(corrmatr, thresh)
    # for x in corrfts_sorted:
    #     print(x)
    # print(corrfts_sorted[:10])
    # savepath_correlation_graph = os.path.normpath(os.path.join(os.getcwd(), "..", r"master_plots\feature selection and modelling\correlation plots",
    #                                      norm.replace(" ", "_") + "_" + time + "_" + mode.replace(" ", "_") + "_correlation_graph" + ".png"))
    # visualize_network(corrmatr, thresh, colored=True, save=True, savepath=savepath_correlation_graph)


    #FEATURES WITH WORD IN NAME
    # word = "wavelet"
    # print(len(np.nonzero(feats.data.columns.str.contains(word))[0]), f"features having {word} in name.")
    # for c in feats.data.columns:
    #     if word in c.lower():
    #         print(c)
    # print(f"CV = {float(cv[c]):.5g}")


    # print(type(cv["wavelet-H_firstorder_RootMeanSquared"]))
    # cv = cv.sort_values(kind="stable")
    # print(cv[:5].dtype)
    # print(cv["diagnostics_Image-original_Minimum"])
    # print(cv[:,1])


    # i = 0
    # for key, vals in six.iteritems(corrmatr):
    #     # print(key)
    #     if key == "Unnamed: 0":
    #         print(vals)
    #         continue
    #     # print(key)
    #     # print(vals.values)
    #     # print(corrmatr[col])
    #     count = len(np.argwhere(vals.values >= thresh))
    #     print(f"{count}\t correlations above thresh = {thresh} for feature {key}")




#     # print(feats.data[col].shape)
    #     print(feats.data[col])
    #     if i > 5:
    #         break;
    # # for key, val in six.iteritems(feats.data):
    #     print(f"{key}:\n {val}")  # key, val)


    # path = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Radiomic features\central slice\-7day\C1_sagittal.csv"
    # path2 = r"C:\Users\toral\OneDrive - Universitetet i Oslo\Radiomic features\central slice\-7day\C2_sagittal.csv"
    # df = pd.read_csv(path)
    # df2 = pd.read_csv(path2)
    # # print(df)
    # df = df.append(df2)
    # print(df)
    # for key, val in six.iteritems(df):
    #     print(f"{key}:\t {val}")  # key, val)
    # dt = np.array(pd.read_csv(path)).T
    # print(dt, dt.shape)
    # # headers = dict(key=dt[0, :])
    # # print(headers)
    # # print(dt[0, :])
    # df = pd.DataFrame(data=dt[1,:], index=dt[0, :])#, index=range(len(dt[0])))
    # print(df.T)
    # for c in df.T:
    #     print(c)