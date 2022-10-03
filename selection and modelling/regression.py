from select_utils import *
from data_loader import *
from model_utils import *

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, RepeatedKFold, StratifiedKFold
from mrmr import mrmr_regression

PARAM_DICT_HP_RF = {"criterion": ["squared_error", "absolute_error", "poisson"],
              "min_samples_leaf": np.linspace(1, 16, 16, dtype=int),
              "min_samples_split": np.linspace(2, 64, 32, dtype=int),
              "n_estimators": np.linspace(25, 1000, 50, dtype=int), "max_depth": np.linspace(2, 64, 32, dtype=int)}


def hp_tuning(x_train, y_train, regmode="rf", n_jobs=-1, scoremode="r2"):
    # model: e.g. linear regressor or RF regressor
    # regmode: lr = linear regressor, rf = RF regressor, eln = elasticnet
    regmode = regmode.lower()
    if regmode == "rf":
        # param_grid = {"criterion": ["squared_error", "absolute_error"], "min_samples_leaf":[1, 5, 10],
        #               "min_samples_split": [2, 4, 8, 16], "n_estimators": [50, 100, 500, 1000], "max_depth":[2, 4, 8, 16, 32]}
        param_grid = {"criterion": ["squared_error", "absolute_error"], "min_samples_leaf":[1, 5, 10],
                      "min_samples_split": [2, 4, 8, 16], "n_estimators": [50, 100, 500, 1000], "max_depth":[2, 4, 8, 16, 32]}

        model = RandomForestRegressor()
    elif regmode == "lr":
        param_grid = {}
        print("NO HP TUNING NECESSARY FOR LINEAR REGRESSION????")
        return 0
    elif regmode == "eln":
        param_grid = {"alpha":[1.0], "l1_ratio":[0.5], "selection and modelling":["cyclic", "random"]}
        model = ElasticNet(tol=1e-3)    # default: 1e-4 gives convergence error in some cases
    else:
        print("NOT VALID", regmode)
        return 0
    if scoremode == "r2":
        scorename = scoremode
    elif scoremode == "mse":
        scorename = "neg_mean_squared_error"
    else:
        print("INVAlid", scoremode, "...")
        return 0

    print("N_jobs =", n_jobs, "\tEstimator =", regmode, "\tScoring =", scoremode)
    cv = RepeatedKFold(n_repeats=5, n_splits=2)
    gs = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorename, cv=cv, n_jobs=n_jobs, verbose=1)
    gs = gs.fit(x_train, y_train)

    print(gs.best_score_)
    print(gs.best_params_)
    # print(gs.cv_results_)
    print(gs.best_estimator_)
    return gs.best_params_


def load_train_test_regression(MODE="NO P", WEIGHT="T2", LRMODE="aggregated"):
    if MODE == "NO P":
        x_train, y_train = load_nop(WEIGHT=WEIGHT, LRMODE=LRMODE, training=True, xer=False)
        x_test, y_test = load_nop(WEIGHT=WEIGHT, LRMODE=LRMODE, training=False, xer=False)
    elif MODE == "DELTA P":
        x_train, y_train = load_delta_P(WEIGHT=WEIGHT, LRMODE=LRMODE, training=True, xer=False)
        x_test, y_test = load_delta_P(WEIGHT=WEIGHT, LRMODE=LRMODE, training=False, xer=False)
    elif MODE == "DELTA":
        x_train, y_train = load_delta(WEIGHT=WEIGHT, LRMODE=LRMODE, training=True, xer=False, keep_time=True)
        x_test, y_test = load_delta(WEIGHT=WEIGHT, LRMODE=LRMODE, training=False, xer=False, keep_time=True)
        x_train = x_train.drop("delta saliv", axis=1)  # using change is saliva is cheating..?
        x_test = x_test.drop("delta saliv", axis=1)  # using change is saliva is cheating..?
    else:
        print("MODE", MODE, "NOT VALID")
        sys.exit()
    return x_train, y_train, x_test, y_test


def main_regression(MODE="NO P", WEIGHT="T2", LRMODE="aggregated", num_fts=5, do_hp_tuning=True, n_jobs=-1, scoremode="r2"):
    # LOAD DATA
    x_train, y_train, x_test, y_test = load_train_test_regression(MODE=MODE, WEIGHT=WEIGHT, LRMODE=LRMODE)
    if not any(x_train):
        return 0

    df_scores = pd.DataFrame()  # scores for all models, modes, etc
    if scoremode == "r2":
        scorefunc = r2_score
    elif scoremode == "mse":
        scorefunc = mean_squared_error
    else:
        print("SCORE", scoremode, "NOT VALID")
        return 0

    fts_td = ["time", "dose"]   # split X into: having time, dose as features (td) VS remaining IMAGE features
    x_train_td, x_train = x_train[fts_td], x_train.drop(fts_td, axis=1)
    x_test_td, x_test = x_test[fts_td], x_test.drop(fts_td, axis=1)

    if not str(num_fts).lower() == "all":
        fts_best = mrmr_regression(x_train, y_train, K=num_fts)
        print(fts_best)
        x_train, x_test = x_train[fts_best], x_test[fts_best]
    else:
        print("Using all", len(x_train.columns), "features")

    print("\n----- TIME + DOSE -----")
    lr = LinearRegression()
    lr.fit(x_train_td, y_train)
    # y_pred_td = lr.predict(x_test_td)

    score_lr_train = scorefunc(y_train, lr.predict(x_train_td))
    score_lr_test = scorefunc(y_test, lr.predict(x_test_td))
    # r2_lr_train = lr.score(x_train_td, y_train)
    # r2_lr_test = lr.score(x_test_td, y_test)

    if do_hp_tuning:
        params = hp_tuning(x_train_td, y_train, regmode="eln", scoremode=scoremode)
    else:
        params = {}
    eln = ElasticNet(**params)  # linear regression with L1 / L2 regularization
    eln.fit(x_train_td, y_train)
    score_eln_train = scorefunc(y_train, eln.predict(x_train_td))
    score_eln_test = scorefunc(y_test, eln.predict(x_test_td))
    # r2_eln_train = eln.score(x_train_td, y_train)
    # r2_eln_test = eln.score(x_test_td, y_test)

    if do_hp_tuning:
        params = hp_tuning(x_train_td, y_train, regmode="rf", n_jobs=n_jobs, scoremode=scoremode)
    else:
        params = {}
    rf = RandomForestRegressor(**params)
    rf.fit(x_train_td, y_train)
    score_rf_train = scorefunc(y_train, rf.predict(x_train_td))
    score_rf_test = scorefunc(y_test, rf.predict(x_test_td))
    # r2_rf_train = rf.score(x_train_td, y_train)
    # r2_rf_test = rf.score(x_test_td, y_test)

    row = f"{MODE} {WEIGHT} td"
    df_scores.loc[row, "lr train"] = score_lr_train
    df_scores.loc[row, "eln train"] = score_eln_train
    df_scores.loc[row, "rf train"] = score_rf_train
    df_scores.loc[row, "lr test"] = score_lr_test
    df_scores.loc[row, "eln test"] = score_eln_test
    df_scores.loc[row, "rf test"] = score_rf_test
    print("Time + dose:\t", *[f"{md}={sc:.2f} " for sc, md in zip(df_scores.loc[row].values, ["LRtrain", "ELNtrain", "RFtrain", "LRtest", "ELNtest", "RFtest"])])

    print(f"\n----- {'''Top ''' + str(num_fts) if num_fts != '''all''' else '''All ''' + str(len(x_train.T))} fts -----")
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    # r2_lr_train = lr.score(x_train, y_train)
    # r2_lr_test = lr.score(x_test, y_test)
    score_lr_train = scorefunc(y_train, lr.predict(x_train))
    score_lr_test = scorefunc(y_test, lr.predict(x_test))

    if do_hp_tuning:
        params = hp_tuning(x_train, y_train, regmode="eln", n_jobs=n_jobs, scoremode=scoremode)
    else:
        params = {}
    eln = ElasticNet(**params)  # linear regression with L1 / L2 regularization
    eln.fit(x_train, y_train)
    score_eln_train = scorefunc(y_train, eln.predict(x_train))
    score_eln_test = scorefunc(y_test, eln.predict(x_test))
    # r2_eln_train = eln.score(x_train, y_train)
    # r2_eln_test = eln.score(x_test, y_test)

    if do_hp_tuning:
        params = hp_tuning(x_train, y_train, regmode="rf", n_jobs=n_jobs, scoremode=scoremode)
    else:
        params = {}
    rf = RandomForestRegressor(**params)
    rf.fit(x_train, y_train)
    # r2_rf_train = rf.score(x_train, y_train)
    # r2_rf_test = rf.score(x_test, y_test)
    score_rf_train = scorefunc(y_train, rf.predict(x_train))
    score_rf_test = scorefunc(y_test, rf.predict(x_test))

    row = f"{MODE} {WEIGHT} {num_fts}"
    df_scores.loc[row, "lr train"] = score_lr_train
    df_scores.loc[row, "eln train"] = score_eln_train
    df_scores.loc[row, "rf train"] = score_rf_train
    df_scores.loc[row, "lr test"] = score_lr_test
    df_scores.loc[row, "eln test"] = score_eln_test
    df_scores.loc[row, "rf test"] = score_rf_test

    print(f"{'''Top ''' + str(num_fts) if num_fts != '''all''' else '''All'''} fts:\t\t", *[f"{md}={sc:.2f} " for sc, md in zip(df_scores.loc[row].values, ["LRtrain", "ELNtrain", "RFtrain", "LRtest", "ELNtest", "RFtest"])])
    return df_scores


def show_results(LRMODE="aggregated", with_tuning=True, include_enet=True, scoremode="r2"):
    # loadname = f"{LRMODE}_r2scores_hptuned.csv" if with_tuning else f"{LRMODE}_r2scores.csv"
    loadname = f"{LRMODE}_{scoremode}_scores_hptuned.csv" if with_tuning else f"{LRMODE}_{scoremode}_scores.csv"
    load_path = os.path.join(ModelDir, "Regression", loadname)
    df_load = pd.read_csv(load_path, index_col=0)
    # print(df_load)
    # df_load.drop()
    if not include_enet:    # remove elastic net columns
        df_load = df_load[list(filter(lambda col: "eln" not in col, df_load.columns))]
    df_test = df_load[list(filter(lambda col: "test" in col, df_load.columns))]
    df_train = df_load[list(filter(lambda col: "train" in col, df_load.columns))]
    # print(df_test)
    # fig, axes = plt.subplots(ncols=2)
    # ax1, ax2 = axes.ravel()
    # sns.heatmap(data=df_train, annot=True, fmt=".2f", ax=ax1, cbar=False, cmap="RdYlGn", vmin=0, vmax=1)
    # sns.heatmap(data=df_test, annot=True, fmt=".2f", ax=ax2, cbar=False, cmap="RdYlGn", vmin=0, vmax=1)
    # ax1.set_title("Training data")
    # ax2.set_title("Test data")
    # ax2.get_yaxis().set_visible(False)
    # fig.tight_layout()
    # plt.show()

    print(df_load.columns.values)
    print(df_load.index.values)
    if scoremode == "r2":
        vmin, vmax = 0, 1
        cmap = "RdYlGn"
    else:
        vmin, vmax = 0, 2000
        cmap = "RdYlGn_r"

    col_names = ["Linear", "Elastic\nnet", "Random\nforest"] if include_enet else ["Linear\nregression", "Random\nforest"]
    fig, ax = plt.subplots(nrows=3, ncols=2)
    # for i, model in enumerate(["NO P T1", "NO P T2", "DELTA P T2", "DELTA T2"]):
    for i, model in enumerate(["NO P T1", "NO P T2", "DELTA P T2"]):
    # for i, model in enumerate(["No P T1", "No P T2", "Delta P T2", "Delta T2"]):
        df_test_curr = df_test.filter(like=model, axis=0)
        df_train_curr = df_train.filter(like=model, axis=0)
        model_names = df_train_curr.index.values
        # model_names = [f"{''' '''.join(nm.split(''' ''')[:-2]).capitalize()} " for nm in model_names]
        sns.heatmap(data=df_train_curr, annot=True, fmt=".2f", ax=ax[i, 0], cbar=False, cmap=cmap, vmin=vmin, vmax=vmax, yticklabels=model_names, xticklabels=col_names)
        sns.heatmap(data=df_test_curr, annot=True, fmt=".2f", ax=ax[i, 1], cbar=False, cmap=cmap, vmin=vmin, vmax=vmax, xticklabels=col_names)
        ax[i, 0].set_xticklabels(ax[i, 0].get_xticklabels(), rotation=0)

    for i in range(2):
        # ax[i, 0].axis("off")
        ax[i, 1].axis("off")
        ax[i, 0].get_xaxis().set_visible(False)
    ax[2, 1].get_yaxis().set_visible(False)
    ax[2, 1].get_xaxis().set_visible(True)
    ax[0, 0].set_title("Training data")
    ax[0, 1].set_title("Test data")
    fig.suptitle(f"{LRMODE.capitalize()}, scoring={scoremode}, hptuning={with_tuning}")
    fig.tight_layout()
    plt.show()
    pass


def calculate_r2_manually(x_train, y_train, x_test, y_test):
    from scipy.stats import pearsonr

    lm = LinearRegression()
    lm.fit(x_train, y_train)
    y_pred_train = lm.predict(x_train)
    y_pred_test = lm.predict(x_test)
    print("Multiple linear regression:")
    print(f"Sklearn score  \ttrain={lm.score(x_train, y_train):.2f}\ttest={lm.score(x_test, y_test):.2f}")
    print(f"Sklearn r2 fnc\ttrain={r2_score(y_train, y_pred_train):.2f}\ttest={r2_score(y_test, y_pred_test):.2f}")

    corr_train, _ = pearsonr(y_train, y_pred_train)
    corr_test, _ = pearsonr(y_test, y_pred_test)
    print(f"Pearson corr^2\ttrain={corr_train**2:.2f}\ttest={corr_test**2:.2e}")

    ybar_train = np.mean(y_train)
    ybar_test = np.mean(y_test)
    RSS_train = np.sum([(yi - yhat)**2 for yi, yhat in zip(y_train, y_pred_train)])
    RSS_test = np.sum([(yi - yhat)**2 for yi, yhat in zip(y_test, y_pred_test)])
    TSS_train = np.sum([(yi - ybar_train)**2 for yi in y_train])
    TSS_test = np.sum([(yi - ybar_test)**2 for yi in y_test])
    r2_generaldef_train = 1 - RSS_train / TSS_train
    r2_generaldef_test = 1 - RSS_test / TSS_test
    print(f"General def:\ttrain={r2_generaldef_train:.2f}\ttest={r2_generaldef_test:.2f}")
    pass


def find_optimal_feature_amount(MIN_FTS=2):
    from sklearn.feature_selection import RFECV, RFE
    from sklearn.model_selection import LeaveOneOut
    x, y = load_nop(WEIGHT="T1", LRMODE="aggregated", xer=False, training="all")
    # x = x[:5]
    # y = y[:5]
    # print(x.shape, y.shape)
    num_fts = len(x.T)
    num_obs = len(x)
    print(num_obs, num_fts)

    m1 = LinearRegression()
    m2 = RandomForestRegressor()
    model_names = ["Linear reg", "RF reg"]

    # rfe = RFE(estimator=, step=1, verbose=1)
    # rfe = RFECV(estimator=m, scoring="r2", cv=2, step=1, verbose=1, min_features_to_select=MIN_FTS)
    # rfe = RFECV(estimator=m, scoring="neg_mean_absolute_error", cv=RepeatedKFold(n_splits=2, n_repeats=5), step=1, verbose=1, min_features_to_select=MIN_FTS, n_jobs=1)

    fig, ax = plt.subplots()
    num_fts_list = list(range(MIN_FTS, num_fts + 1))

    best_fts = {}
    scores_avg = {}

    for m, mname in zip([m1, m2], model_names):
    # for m, mname in zip([m1], ["linreg"]):
        rfe = RFECV(estimator=m, scoring="neg_mean_absolute_error", cv=LeaveOneOut(), step=1, verbose=1, min_features_to_select=MIN_FTS, n_jobs=-1)
        rfe.fit(x, y)
        print(rfe.cv_results_.keys())
        print(rfe.get_feature_names_out())
        kbest = rfe.n_features_
        # scores_dict[mname] = np.average(rfe.)
        best_fts[mname] = rfe.get_feature_names_out()

        ax.plot(num_fts_list, np.average(rfe.grid_scores_, axis=1), "--", label=f"{mname} k={kbest}")

    print(best_fts)

    ax.set_xlabel("Num fts")
    ax.set_ylabel("CV acc")
    # ax.plot(num_fts_list, rfe.grid_scores_, ":")
    ax.set_title("RFE with LOOCV scores")
    ax.legend()
    ax.grid()
    plt.show()
    pass


def main_loocv_regression(MODE="NO P", WEIGHT="T2", LRMODE="aggregated", do_hp_tuning=False, REGRESSOR="LINREG", NFTS=5, NORMALIZE_FEATURES=False, DO_RFE=False, NREP=1, save=True):
    from sklearn.feature_selection import RFE
    savefolder = os.path.normpath(os.path.join(RegressDir, "loo"))
    if not NORMALIZE_FEATURES:
        savename = f"{LRMODE}_{MODE}_{WEIGHT}_{REGRESSOR}_nfts={NFTS}_ypreds_hptune={do_hp_tuning}.csv"
        savename_fts = f"{LRMODE}_{MODE}_{WEIGHT}_{REGRESSOR}_nfts={NFTS}_features.csv"
    else:
        from sklearn.preprocessing import StandardScaler
        savename = f"{LRMODE}_{MODE}_{WEIGHT}_{REGRESSOR}_nfts={NFTS}_stscaled_ypreds_hptune={do_hp_tuning}.csv"
        savename_fts = f"{LRMODE}_{MODE}_{WEIGHT}_{REGRESSOR}_nfts={NFTS}_stscaled_features.csv"

    if MODE == "NO P":
        x, y = load_nop(WEIGHT=WEIGHT, LRMODE=LRMODE, training="all", xer=False)
    elif MODE == "DELTA P":
        x, y = load_delta_P(WEIGHT=WEIGHT, LRMODE=LRMODE, training="all", xer=False)
    else:
        print("TRY mode: NO P, DELTA P")
        return 0
    xtd = x[["time", "dose"]]
    x = x.drop(["time", "dose"], axis=1)
    print(f"LOO REGRESSION LR{LRMODE} {MODE} {WEIGHT} {REGRESSOR} norm={NORMALIZE_FEATURES}")
    print(x.shape, xtd.shape, y.shape)
    num_obs, num_fts = x.shape
    x.index = list(range(num_obs))
    xtd.index = list(range(num_obs))
    y.index = list(range(num_obs))


    fts_all = list(x.columns)
    df_ypredvals = pd.DataFrame()
    df_featurecounts = pd.DataFrame(index=["sum", *fts_all], data=np.zeros(shape=(len(fts_all) + 1, 2)), columns=["mrmr", "signval"])
    # df_featurecounts = pd.DataFrame(index=["sum", "ft_x", *fts_all], data=np.zeros(shape=(len(fts_all) + 2, 2)), columns=["mrmr", "signval"])

    # print(df_featurecounts)

    if NORMALIZE_FEATURES: #todo: put inside LOO loop (data leakage)
        scaler = StandardScaler()
        x[fts_all] = scaler.fit_transform(x[fts_all])

    for i in range(num_obs):
        x_train = x.drop(i)
        xtd_train = xtd.drop(i)
        y_train = y.drop(i)

        x_test = x.loc[i]
        xtd_test = xtd.loc[i].values.reshape(1, -1)
        y_test = np.array([y.loc[i]]).reshape(1, -1)

        # if NORMALIZE_FEATURES:
        #     print(x_train)
        #     scaler = StandardScaler()
        #     scaler.fit(X=x_train)
        #     x_train[fts_all] = scaler.fit_transform(x_train[fts_all])
        #     print(x_train)
        #     print(x_test)
        #     x_test.loc[fts_all] = scaler.transform(x_test.values.reshape(1, -1))
        #     print(x_test)
        # return 0
        # print(x_train.shape)
        top_fts = mrmr_regression(x_train, y_train, K=5, show_progress=True)

        x_train = x_train[top_fts]
        x_test = x_test[top_fts].values.reshape(1, -1)

        print(f"\n{i+1} / {num_obs}")
        predvals = []
        predvals_td = []
        signvals = np.zeros(NFTS)

        for j in range(NREP):
            if REGRESSOR == "LINREG":
                m = LinearRegression()
                m_td = LinearRegression()
                save_beta_sign = True
            elif REGRESSOR == "RF":
                m = RandomForestRegressor()
                m_td = RandomForestRegressor()
                save_beta_sign = False
            elif REGRESSOR == "ELNET":
                alpha, l1ratio = 1.0, 0.5
                m = ElasticNet(alpha=alpha, l1_ratio=l1ratio)
                m_td = ElasticNet(alpha=alpha, l1_ratio=l1ratio)
                save_beta_sign = True
            else:
                print("TRY MODEL: LINREG, RF")
                return 0

            if do_hp_tuning:
                params = hp_tuning(x_train, y_train, regmode=REGRESSOR.lower())
                m.set_params(**params)

            m_td.fit(xtd_train, y_train)
            m.fit(x_train, y_train)

            if save_beta_sign:
                signvals += np.sign(m.coef_)
                # top_fts = ["ft_x", *top_fts]
                # signvals = [np.random.choice([-1.0, 1.0]), *signvals]
                # print(top_fts)
                # print(signvals)

            ypred_td = m_td.predict(xtd_test).reshape(-1)
            ypred = m.predict(x_test).reshape(-1)

            # y_predvals_td.append(ypred_td[0])
            # y_predvals.append(ypred_mrmr[0])

            predvals.append(ypred)
            predvals_td.append(ypred_td)

            mse_td = mean_squared_error(y_test, ypred_td)
            mse_mrmr = mean_squared_error(y_test, ypred)

            # print(".", end="")
        # end repeat loop
        pred_avg, pred_sd = np.average(predvals), np.std(predvals)
        pred_td_avg, pred_td_sd = np.average(predvals_td), np.std(predvals_td)

        # df_ypredvals.loc[i, ["gt", "td", "mrmr"]] = y_test.reshape(-1)[0], ypred_td[0], ypred[0]
        df_ypredvals.loc[i, ["gt", "td", "mrmr", "td sd", "mrmr sd"]] = y_test.reshape(-1)[0], pred_td_avg, pred_avg, pred_td_sd, pred_sd
        df_featurecounts.loc[top_fts, "mrmr"] += NREP
        df_featurecounts.loc["sum", "mrmr"] += NREP
        df_featurecounts.loc[top_fts, "signval"] += signvals
        print(df_featurecounts.loc[top_fts])
        # break
    print(df_featurecounts)
    print(df_ypredvals)

    if save:
        df_ypredvals.to_csv(os.path.join(savefolder, savename))
        df_featurecounts.to_csv(os.path.join(savefolder, savename_fts))
        print("SAVED DFs")
    else:
        print("NOT SAVED....")
    return 1


def show_loocv_results(show_rfe=False, show_scaled=False):
    folder = os.path.join(RegressDir, "loo")
    files = os.listdir(folder)
    files = list(filter(lambda f: "ypreds" in f, files))
    files = list(filter(lambda f: "hptune" in f, files))
    if not show_scaled:
        files = list(filter(lambda f: "scaled" not in f, files))

    print(files)
    df_r2scores = pd.DataFrame()
    df_r2annot = pd.DataFrame()

    df_msescores = pd.DataFrame()
    # files = ["aggregated_DELTA P_T2_LINREG_nfts=5_ypreds_hptune=False.csv"]

    for f in files:
        ff = f.split("_")
        # print(ff)
        lrmode, mode, weight, regressor, nfts = ff[:5]
        nfts = nfts.split("=")[-1]
        scaled = " stscaled" if ff[-2] == "stscaled" else ""
        print(lrmode, mode, weight, regressor, nfts, scaled)

        df_ypred = pd.read_csv(os.path.join(folder, f), index_col=0)
        # loadname_fts = "_".join(ff[:-1]) + "_features.csv"
        # df_fts = pd.read_csv(os.path.join(folder, loadname_fts), index_col=0)
        # df_fts = df_fts[(df_fts.T != 0).any()]  # drop non-selected features

        # print(df_ypred)
        # print(df_fts)
        y_true = df_ypred["gt"]
        ypred_td = df_ypred["td"]
        ypred_mrmr = df_ypred["mrmr"]
        ypred_rfe = df_ypred["rfe"] if show_rfe else 0

        r2_td = r2_score(y_true, ypred_td)
        r2_mrmr = r2_score(y_true, ypred_mrmr)
        r2_rfe = r2_score(y_true, ypred_rfe) if show_rfe else 0

        if any(["sd" in c for c in df_ypred.columns.values]):
            ypred_td_sd = np.array(df_ypred["td sd"])
            ypred_mrmr_sd = np.array(df_ypred["mrmr sd"])

            ypred_td_lower = np.array(ypred_td) - ypred_td_sd
            ypred_td_upper = np.array(ypred_td) + ypred_td_sd
            ypred_mrmr_lower = np.array(ypred_mrmr) - ypred_mrmr_sd
            ypred_mrmr_upper = np.array(ypred_mrmr) + ypred_mrmr_sd

            r2_td_lower = r2_score(y_true, ypred_td_lower)
            r2_td_upper = r2_score(y_true, ypred_td_upper)
            r2_mrmr_lower = r2_score(y_true, ypred_mrmr_lower)
            r2_mrmr_upper = r2_score(y_true, ypred_mrmr_upper)

            r2_delta_td = np.average(np.abs([r2_td - r2_td_lower, r2_td - r2_td_upper]))
            r2_delta_mrmr = np.average(np.abs([r2_mrmr - r2_mrmr_lower, r2_mrmr - r2_mrmr_upper]))
            print("TD:", r2_td_lower - r2_td, r2_td - r2_td_upper, r2_delta_td)
            print("FTS:", r2_mrmr_lower - r2_mrmr, r2_mrmr - r2_mrmr_upper, r2_delta_mrmr)
        else:
            r2_delta_td, r2_delta_mrmr = 0, 0

        mse_td = mean_squared_error(y_true, ypred_td)
        mse_mrmr = mean_squared_error(y_true, ypred_mrmr)
        mse_rfe = mean_squared_error(y_true, ypred_rfe) if show_rfe else 0

        col = f"LR {lrmode} {regressor}"

        row = f"{mode} {weight} td"
        df_r2scores.loc[row, col] = r2_td
        df_r2scores.loc[row, "num fts"] = 2
        df_r2annot.loc[row, col] = f"{r2_td:.2f}$\pm${r2_delta_td:.2f}"
        df_r2annot.loc[row, "num fts"] = 2
        df_msescores.loc[row, col] = mse_td
        df_msescores.loc[row, "num fts"] = 2

        # row = f"{mode} {weight} mrmr{scaled}"
        row = f"{mode} {weight} {nfts} fts{scaled}"
        df_r2scores.loc[row, col] = r2_mrmr
        df_r2scores.loc[row, "num fts"] = int(nfts)
        df_r2annot.loc[row, col] = f"{r2_mrmr:.2f}$\pm${r2_delta_mrmr:.2f}"
        df_r2annot.loc[row, "num fts"] = int(nfts)
        df_msescores.loc[row, col] = mse_mrmr
        df_msescores.loc[row, "num fts"] = int(nfts)

        # row = f"{mode} {weight} rfe{scaled}" if show_rfe else 0
        # df_r2scores.loc[row, col] = r2_rfe if show_rfe else 0
        # df_r2scores.loc[row, "num fts"] = int(nfts) if show_rfe else 0
        # df_msescores.loc[row, col] = mse_rfe if show_rfe else 0
        # df_msescores.loc[row, "num fts"] = int(nfts) if show_rfe else 0

    print(df_r2annot)
    df_r2scores = df_r2scores.sort_values(by="num fts").drop("num fts", axis=1)
    df_r2annot = df_r2annot.sort_values(by="num fts").drop("num fts", axis=1)
    df_msescores = df_msescores.sort_values(by="num fts").drop("num fts", axis=1)

    print(df_r2scores)
    print(df_r2annot)
    # if not show_rfe:
    #     df_r2scores = df_r2scores.drop(0)
    #     df_r2annot = df_r2annot.drop(0)
    #     df_msescores = df_msescores.drop(0)

    fig, ax = plt.subplots()
    # sns.heatmap(data=df_r2scores, annot=True, fmt=".2f", cbar=False, vmax=1, cmap="RdYlGn", ax=ax)
    sns.heatmap(data=df_r2scores, annot=df_r2annot, fmt="", cbar=False, vmax=1, cmap="RdYlGn", ax=ax)
    cols = df_r2scores.columns.values
    cols = [f"R2\n{''' '''.join(c.split(''' ''')[:-1])}\n{c.split(''' ''')[-1]}" for c in cols]
    ax.set_xticks(ax.get_xticks(), cols, rotation=0)


    fig, ax = plt.subplots()
    sns.heatmap(data=df_msescores, annot=True, fmt=".2f", cbar=False, cmap="RdYlGn_r", ax=ax)
    cols = df_msescores.columns.values
    ax.set_xticks(ax.get_xticks(), cols, rotation=0)

    plt.show()

    pass


def calculate_scores_for_hp(predmode="simult", REGRESSOR="RF", LRMODE="aggregated", num_fts=5, repeats=5):
    if REGRESSOR == "RF":
        param_dict = {"criterion": ["squared_error", "absolute_error", "poisson"], "min_samples_leaf":np.linspace(1, 16, 16, dtype=int),
                      "min_samples_split":np.linspace(2, 64, 32, dtype=int), "n_estimators": np.linspace(25, 1000, 50, dtype=int), "max_depth":np.linspace(2, 64, 32, dtype=int)}
        # del param_dict["criterion"]
    else:
        print("try model RF")
        return 0

    if predmode == "simult":
        x_t1_train, y_t1_train = load_nop(WEIGHT="T1", LRMODE=LRMODE, training=True, xer=False)
        x_t1_test, y_t1_test = load_nop(WEIGHT="T1", LRMODE=LRMODE, training=False, xer=False)
        x_t1_train, x_t1_test = x_t1_train.drop(["time", "dose"], axis=1), x_t1_test.drop(["time", "dose"], axis=1)

        x_t2_train, y_t2_train = load_nop(WEIGHT="T2", LRMODE=LRMODE, training=True, xer=False)
        x_t2_test, y_t2_test = load_nop(WEIGHT="T2", LRMODE=LRMODE, training=False, xer=False)
        x_t2_train, x_t2_test = x_t2_train.drop(["time", "dose"], axis=1), x_t2_test.drop(["time", "dose"], axis=1)

        x_dp_train, y_dp_train = load_delta_P(WEIGHT="T2", LRMODE=LRMODE, xer=False, training=True)
        x_dp_test, y_dp_test = load_delta_P(WEIGHT="T2", LRMODE=LRMODE, xer=False, training=False)
        x_dp_train, x_dp_test = x_dp_train.drop(["time", "dose"], axis=1), x_dp_test.drop(["time", "dose"], axis=1)

        fts_t1 = mrmr_regression(x_t1_train, y_t1_train, K=num_fts)
        fts_t2 = mrmr_regression(x_t2_train, y_t2_train, K=num_fts)
        fts_dp = mrmr_regression(x_dp_train, y_dp_train, K=num_fts)
        print("T1:", fts_t1)
        print("T2:", fts_t2)
        print("Delta-P:", fts_dp)

        x_t1_train, x_t1_test = x_t1_train[fts_t1], x_t1_test[fts_t1]
        x_t2_train, x_t2_test = x_t2_train[fts_t2], x_t2_test[fts_t2]
        x_dp_train, x_dp_test = x_dp_train[fts_dp], x_dp_test[fts_dp]
    else:
        return 0

    for param, vals in param_dict.items():
        df_results = pd.DataFrame()
        savename = f"hpscores_{LRMODE}_{REGRESSOR}_{param}_r2.csv"
        print("\n", param, vals)
        for v in vals:
            print(param, v, end="\t")
            R2VALS = []
            for i in range(repeats):
                print(".", end=" ")
                params = {param:v}
                rf1 = RandomForestRegressor(**params, oob_score=False)
                rf1.fit(x_t1_train, y_t1_train)
                rf2 = RandomForestRegressor(**params, oob_score=False)
                rf2.fit(x_t2_train, y_t2_train)
                rf3 = RandomForestRegressor(**params, oob_score=False)
                rf3.fit(x_dp_train, y_dp_train)

                pred_t1_train = rf1.predict(x_t1_train)
                pred_t1_test = rf1.predict(x_t1_test)
                pred_t2_train = rf2.predict(x_t2_train)
                pred_t2_test = rf2.predict(x_t2_test)
                pred_dp_train = rf3.predict(x_dp_train)
                pred_dp_test = rf3.predict(x_dp_test)

                r2_t1_train = r2_score(y_t1_train, pred_t1_train)
                r2_t1_test = r2_score(y_t1_test, pred_t1_test)
                r2_t2_train = r2_score(y_t2_train, pred_t2_train)
                r2_t2_test = r2_score(y_t2_test, pred_t2_test)
                r2_dp_train = r2_score(y_dp_train, pred_dp_train)
                r2_dp_test = r2_score(y_dp_test, pred_dp_test)

                R2VALS.append([r2_t1_train, r2_t1_test, r2_t2_train, r2_t2_test, r2_dp_train, r2_dp_test])
                # R2VALS.append([x + i for x in range(6)])
            print()
            R2MEANS = np.mean(R2VALS, axis=0)
            R2STD = np.std(R2VALS, axis=0)

            for r2mean, r2std, col in zip(R2MEANS, R2STD,
                               ["T1 train", "T1 test", "T2 train", "T2 test", "delta-p train", "delta-p test"]):

                df_results.loc[v, f"{col} mu"] = r2mean
                df_results.loc[v, f"{col} std"] = r2std


        df_results.to_csv(os.path.join(RegressDir, savename))
        # break
    return 1


def plot_hpparam_results():
    files = os.listdir(RegressDir)
    files = list(filter(lambda f: "hpscores" in f, files))
    print(files)
    CMAP = {"T1":"r", "T2":"b", "delta-p":"y"}
    files_agg = list(filter(lambda f:"aggregated" in f, files))
    files_avg = list(filter(lambda f:"average" in f, files))

    for f_agg in files_agg:
        ff = f_agg.split("_")
        # print(ff)
        lrmode, regressor = ff[1:3]
        scorer = ff[-1].split(".")[0]
        param = "_".join(ff[3:-1])
        print(lrmode, regressor, scorer, param)
        # print(ff)
        f_avg = "_".join([ff[0], "average", *ff[2:]])

        df_agg = pd.read_csv(os.path.join(RegressDir, f_agg), index_col=0)
        df_avg = pd.read_csv(os.path.join(RegressDir, f_avg), index_col=0)

        xvals = df_agg.index.values
        cols_agg = df_agg.columns.values
        cols_agg_mu = list(filter(lambda c: "mu" in c, cols_agg))
        cols_agg_sd = list(filter(lambda c: "std" in c, cols_agg))
        cols_agg_mu.sort()
        cols_agg_sd.sort()
        cols_avg = df_avg.columns.values
        cols_avg_mu = list(filter(lambda c: "mu" in c, cols_avg))
        cols_avg_sd = list(filter(lambda c: "std" in c, cols_avg))
        cols_avg_mu.sort()
        cols_avg_sd.sort()

        print(cols_avg_mu)
        print(cols_agg_mu)
        if not cols_avg_mu == cols_agg_mu:
            print("!!!")
            return 0

        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
        for cmu_agg, csd_agg, cmu_avg, csd_avg in zip(cols_agg_mu, cols_agg_sd, cols_avg_mu, cols_avg_sd):
            # print(cmu, csd)
            mode = cmu_agg.split(" ")[0]
            modename = " ".join(cmu_agg.split(" ")[:-1])
            if not mode == csd_agg.split(" ")[0]:
                print("ERR mu sd")
                return 0

            muvals_agg = df_agg[cmu_agg].values
            sdvals_agg = df_agg[csd_agg].values
            muvals_avg = df_avg[cmu_avg].values
            sdvals_avg = df_avg[csd_avg].values
            c = CMAP[mode]
            mrk = "x" if "train" in modename else "o"
            ax1.plot(xvals, muvals_agg, linestyle="--", marker=mrk, c=c, label=modename)
            ax1.errorbar(xvals, muvals_agg, yerr=sdvals_avg, linestyle="", c=c)
            ax2.plot(xvals, muvals_avg, linestyle="--", marker=mrk, c=c, label=modename)
            ax2.errorbar(xvals, muvals_avg, yerr=sdvals_avg, linestyle="", c=c)

        ax1.grid(1);    ax2.grid(1)
        ax1.set_title("LRaggregated")
        ax2.set_title("LRaverage")
        ax1.set_ylabel("R2")
        ax1.set_xlabel(param)
        ax2.set_xlabel(param)
        ax1.legend(loc="best")


        plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(vals, SCORES1_TRAIN, marker="o", linestyle="--", label="No-p T1 train", c="r")
    # ax.plot(vals, SCORES1_TEST, marker="x", linestyle="--", label="No-p T1 test", c="r")
    # ax.plot(vals, SCORES2_TRAIN, marker="o", linestyle="--", label="No-p T2 train", c="b")
    # ax.plot(vals, SCORES2_TEST, marker="x", linestyle="--", label="No-p T2 test", c="b")
    # ax.plot(vals, SCORES3_TRAIN, marker="o", linestyle="--", label="delta-p train", c="y")
    # ax.plot(vals, SCORES3_TEST, marker="x", linestyle="--", label="delta-p test", c="y")
    #
    # ax.legend()
    # ax.grid(1)
    # ax.set_title(f"{REGRESSOR} {LRMODE} ")
    # ax.set_xlabel(param)
    # plt.show()
    pass


def plot_hpparam_results_timedose():
    folder = os.path.join(RegressDir, "timedose all")
    files = os.listdir(folder)
    files = list(filter(lambda f: "hpscores" in f, files))
    print(files)
    for f in files:
        ff = f.split("_")
        param = "_".join([*ff[4:-1], ff[-1].split(".")[0]])
        df = pd.read_csv(os.path.join(folder, f), index_col=0)
        print(df.columns.values)
        fig, (ax, ax2, ax3) = plt.subplots(nrows=3)
        xvals = df.index.values
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for k in range(1, 6):
            r2train = df[f"train fold {k}"]
            r2test = df[f"test fold {k}"]
            r2cv = df[f"cv train fold {k}"]
            # ax.plot(xvals, r2train, "--o", c=colors[k])
            ax.plot(xvals, r2test, "--o", c=colors[k])
            ax2.plot(xvals, r2train, "--o", c=colors[k], label=f"Fold {k}")
            ax3.plot(xvals, r2cv, "--o", c=colors[k], label=f"Fold {k}")

        ax.set_ylabel("R2")
        ax.set_title(f"5-fold test results param {param}")
        ax.grid()
        ax2.set_ylabel("R2")
        ax2.set_title(f"5-fold training results param {param}")
        ax2.grid()
        ax2.legend(loc="best")
        ax3.set_title(f"5-fold results 5rep2fold cv param {param}")
        ax3.grid()
        ax3.legend(loc="best")
        ax3.set_ylabel("R2")
        ax3.set_xlabel(param)
        plt.show()
        print(df)
        print(param)
    pass


def get_best_fts(MODE, WEIGHT, NUMFTS, LRMODE, REGRESSOR, loocv=True, pick_thresh=0.5, plot=True):
    filename = "_".join([LRMODE, MODE, WEIGHT, REGRESSOR, f"nfts={NUMFTS}"]) + "_features.csv"
    if loocv:
        path = os.path.join(RegressDir, "loo", filename)
        df = pd.read_csv(path, index_col=0)
        Nfts_tot = df.shape[0]
        # df = df[df["mrmr"] > 0]
        Nfts_picked = df[df["mrmr"] > 0].shape[0]
        # df = df.sort_values(by="mrmr", ascending=False)
        df = df.rename(columns={"mrmr":"picks"})
    else:
        return 0
    print(f"Picked: {Nfts_picked} of {Nfts_tot} tot fts")
    N_picks_tot = df.loc["sum", "picks"] # number of times picks was done (models * repeats)
    df = df.drop("sum")
    # df.loc[:, :] = np.array(df.values) / N_picks_tot
    df.loc[:, "picks rel"] = np.array(df["picks"].values) / N_picks_tot
    df_red = df[df["picks rel"] >= pick_thresh]
    N_above_thresh = df_red.shape[0]
    print(f"Picked >{pick_thresh*100:.0f}%: {N_above_thresh}")

    # print(df)
    # print(df_red)

    fts_all = df.index.values
    fts_red = df_red.index.values
    if plot:
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        x1 = list(range(len(fts_all)))
        y1 = list(df["picks rel"])
        ax1.vlines(x1, ymin=0, ymax=y1, linestyles="-", linewidths=1, colors="orange")
        ax1.plot(x1, y1, "o", color="black", markersize=1.0)
        # ax1.hlines([pick_thresh], xmin=0, xmax=np.max(x1), linestyles="--", linewidths=0.5, colors="black")
        ax1.plot(x1, np.repeat(pick_thresh, len(x1)), "--", color="black")
        ax1.set_ylabel("Fraction of times a feature was selected")
        ax1.set_xlabel("Feature number")

        # x2 = fts_red
        x2 = np.argwhere([ft in fts_red for ft in df.index.values]).reshape(-1)

        y2 = list(df_red["picks rel"])
        ax2.vlines(x2, ymin=0, ymax=y2, linestyles="-", linewidths=1.0, colors="orange")
        ax2.plot(x2, y2, "o", color="black", markersize=1.0)
        # ax2.set_xticks(ax2.get_xticks(), x2, rotation=10)
        # ax2.set_xticks(x2, x2, rotation=10)
        ax2.set_xlabel("Feature number")
        fig.suptitle(f"{filename}\nTot fts: {Nfts_tot}, picked: {Nfts_picked}, picked > {pick_thresh} for {N_picks_tot} models: {N_above_thresh}")
        plt.show()
    return fts_red


def regression_timedose_saliva(do_hp_tuning=False, k_splits=5, n_repeats=25, calc_RFhp_results=False, save=True, only_ols_all_data=False):

    savefolder = os.path.join(RegressDir, "timedose all")

    saliv = load_saliva(melt=True).reset_index(drop=True)
    for i in saliv.index.values:
        name, time = saliv.loc[i, ["name", "time"]]
        dose = dose_to_name(name=name, time=time, ignore_exp=True)
        # print(name, time, dose)
        saliv.loc[i, "dose"] = dose

    saliv = shuffle(saliv, random_state=1995)
    # print(saliv)
    Ntot = len(saliv)
    Nctr = len(saliv[saliv["ctr"] == True])
    Nirr = len(saliv[saliv["ctr"] == False])
    print(f"Sample size: {Ntot}, {Nctr} control ({Nctr / Ntot * 100:.0f}%), {Nirr} irr")

    y = saliv[["val", "ctr"]]
    x = saliv[["time", "dose"]]
    print(saliv)
    print(x.shape, y.shape)

    yval = y["val"]
    # Regression on all data
    rf = RandomForestRegressor()
    lr = LinearRegression()
    rf.fit(x, yval)
    lr.fit(x, yval)
    print(f"\nRegression to ALL data")
    print(f"\tR2: RF={rf.score(x, yval):.2f}, LR={lr.score(x, yval):.2f}")
    print(f"\tMSE: RF={mean_squared_error(yval, rf.predict(x)):.2f}, LR={mean_squared_error(yval, lr.predict(x)):.2f}")
    print("\tLR:", f"{lr.intercept_:.2f} + {lr.coef_[0]:.2f} time + {lr.coef_[1]:.2f} dose")
    print("\tRF feature importance:", rf.feature_names_in_, rf.feature_importances_)

    from statsmodels.formula.api import ols
    res_ols = ols('val ~ time + dose', data=saliv).fit()
    print(res_ols.summary())

    res_ols = ols('val ~ time + dose + time:dose', data=saliv).fit()    # with interactions time:dose
    print(res_ols.summary())

    if only_ols_all_data:
        return 0
    # HP values k-fold CV:
    if calc_RFhp_results:
        repkcv = RepeatedKFold(n_splits=2, n_repeats=5)
        for param, vals in PARAM_DICT_HP_RF.items():
            df_results_r2 = pd.DataFrame()
            savename_r2 = f"hpscores_r2_RF_timedose_{param}.csv"
            print(param, np.shape(vals))
            kfold = StratifiedKFold(n_splits=k_splits)
            print(f"\nTrain / test for {k_splits} folds:")
            k = 0
            for idx_train, idx_test in kfold.split(x, y["ctr"]):
                k += 1
                print(f"\nPARAM {param} FOLD {k}")
                x_train = x.loc[idx_train]
                y_train = y.loc[idx_train]
                x_test = x.loc[idx_test]
                y_test = y.loc[idx_test]

                for v in vals:
                    # print(".", end="")
                    print(param, v)
                    prm = {param: v}
                    rf = RandomForestRegressor(**prm)
                    rf.fit(x_train, y_train)
                    r2_train_rf = rf.score(x_train, y_train)
                    r2_test_rf = rf.score(x_test, y_test)

                    rf = RandomForestRegressor(**prm)
                    r2_cvscores = cross_val_score(estimator=rf, X=x_train, y=y_train, cv=repkcv)    # same as is done when hp tuning
                    print(r2_cvscores)
                    print(f"R2 train = {r2_train_rf:.2f}, test = {r2_test_rf:.2f}, 5-repeated 2-fold on train = {np.average(r2_cvscores):.2} +- {np.std(r2_cvscores):.2f}")

                    col = f"train fold {k}"
                    df_results_r2.loc[v, col] = r2_train_rf
                    col = f"test fold {k}"
                    df_results_r2.loc[v, col] = r2_test_rf
                    col = f"cv train fold {k}"
                    df_results_r2.loc[v, col] = np.mean(r2_cvscores)

            print(df_results_r2)
            df_results_r2.to_csv(os.path.join(savefolder, savename_r2)) if save else 0
        return 0

    # Train / test on K folds
    kfold = StratifiedKFold(n_splits=k_splits)
    k = 1
    i = 0
    df_results_r2 = pd.DataFrame()
    df_results_mse = pd.DataFrame()

    savename_r2 = f"{k_splits}fold_r2scores_timedose.csv"
    savename_mse = f"{k_splits}fold_mse_timedose.csv"
    print(f"\nTrain / test for {k_splits} folds:")
    for idx_train, idx_test in kfold.split(x, y["ctr"]):
        x_train = x.loc[idx_train]
        y_train = y.loc[idx_train]
        x_test = x.loc[idx_test]
        y_test = y.loc[idx_test]
        Ntest = len(y_test)
        Ntest_ctr = len(y_test[y_test["ctr"] == True])
        Ntest_irr = len(y_test[y_test["ctr"] == False])
        print(f"\n\nFold {k}: num test = {Ntest} ({Ntest_ctr} control ({Ntest_ctr / Ntest*100:.0f}%), {Ntest_irr} irr)")
        y_train, y_test = y_train["val"], y_test["val"]

        if do_hp_tuning:
            params = hp_tuning(x_train, y_train, regmode="rf", scoremode="r2")
        else:
            params = {}

        for j in range(n_repeats):
            rf = RandomForestRegressor(**params)
            lr = LinearRegression()

            rf.fit(x_train, y_train)
            lr.fit(x_train, y_train)

            ypred_rf_train = rf.predict(x_train)
            ypred_rf_test = rf.predict(x_test)
            ypred_lr_train = lr.predict(x_train)
            ypred_lr_test = lr.predict(x_test)

            r2_rf_train = r2_score(y_train, ypred_rf_train)
            r2_rf_test = r2_score(y_test, ypred_rf_test)
            r2_lr_train = r2_score(y_train, ypred_lr_train)
            r2_lr_test = r2_score(y_test, ypred_lr_test)

            mse_rf_train = mean_squared_error(y_train, ypred_rf_train)
            mse_rf_test = mean_squared_error(y_test, ypred_rf_test)
            mse_lr_train = mean_squared_error(y_train, ypred_lr_train)
            mse_lr_test = mean_squared_error(y_test, ypred_lr_test)

            # print(f"\tR2 train: RF={r2_rf_train:.2f}, LR={r2_lr_train:.2f}\ttest: RF={r2_rf_test:.2f}, LR={r2_lr_test:.2f}")
            # for r2, col in zip([r2_rf_train, r2_rf_test, r2_lr_train, r2_lr_test], ["rf train", "rf test", "lr train", "lr test"]):
            #     df_results_k.loc[j, col] = r2
            df_results_r2.loc[i, "fold"] = k
            df_results_r2.loc[i, "repeat"] = j
            df_results_mse.loc[i, "fold"] = k
            df_results_mse.loc[i, "repeat"] = j

            for r2, col in zip([r2_rf_train, r2_rf_test, r2_lr_train, r2_lr_test], ["RF train", "RF test", "LR train", "LR test"]):
                df_results_r2.loc[i, col] = r2

            for mse, col in zip([mse_rf_train, mse_rf_test, mse_lr_train, mse_lr_test], ["RF train", "RF test", "LR train", "LR test"]):
                df_results_mse.loc[i, col] = mse

            print("\tLR:", f"{lr.intercept_:.2f} + {lr.coef_[0]:.2f} time + {lr.coef_[1]:.2f} dose")
            print(f"\tLR R2 train: {r2_lr_train:.2f}, test: {r2_lr_test:.2f}")
            print(f"\tRF R2 train: {r2_rf_train:.2f}, test: {r2_rf_test:.2f}")
            print("\tRF feature importance:", rf.feature_names_in_, rf.feature_importances_)

            res_ols = ols('val ~ time + dose', data=saliv.loc[idx_train]).fit()
            print(res_ols.summary())

            i += 1
        # print(df_results_r2)
        k += 1
    print(df_results_r2)
    if save:
        df_results_r2.to_csv(os.path.join(savefolder, savename_r2))
        df_results_mse.to_csv(os.path.join(savefolder, savename_mse))
    pass


def show_timedose_results(metric="r2scores"):
    # saliv = load_saliva(melt=True)
    # saliv["dose"] = [dose_to_name(nm, time, ignore_exp=True) for nm, time in zip(saliv["name"].values, saliv["time"].values)]
    # print(saliv)
    # fig, ax = plt.subplots()
    # sns.jointplot(saliv, x="time", y="dose", hue="ctr")

    folder = os.path.join(RegressDir, "timedose all")
    df = pd.read_csv(os.path.join(folder, f"5fold_{metric}_timedose.csv"), index_col=0).drop("repeat", axis=1)
    # print(df)
    print(df.groupby("fold").mean())
    print(df.groupby("fold").std())
    print(f"AVERAGES {metric}:\n", df.mean())
    print(f"SD {metric}:\n", df.std())

    df_annot = pd.DataFrame()
    k = 1
    cols = df.drop("fold", axis=1).columns.values
    for means_k, stds_k in zip(df.groupby("fold").mean().values, df.groupby("fold").std().values):
        # print(means_k, stds_k)
        df_annot.loc[k, cols] = [f"{mu:.2f}$\pm${sd:.2f}" for mu, sd in zip(means_k, stds_k)]
        k += 1
    # print(df_annot)
    cols_new = ["LR train", "RF train", "LR test", "RF test"]
    df_annot, df_means = df_annot[cols_new], df.groupby("fold").mean()[cols_new]
    fig, ax = plt.subplots()
    sns.heatmap(df_means, annot=df_annot, fmt="", cbar=False, cmap="RdYlGn", vmax=1, ax=ax)
    # sns.scatterplot(data=df.drop(["RF train", "LR train"], axis=1).melt(id_vars=["fold"]), x="fold", y="value", hue="variable")
    ax.set_yticks(ax.get_yticks(), list(range(1, 6)), rotation=0)
    plt.show()
    return 1


def loocv_timedose_saliv(do_hp_tuning=True, Nrep=5):
    # for each left out: repeat fit / prediction NREP times for RF, once for LINREG
    savefolder = os.path.join(RegressDir, "timedose all")
    saliv = load_saliva(melt=True).reset_index(drop=True)
    for n in saliv.index.values:
        name, time = saliv.loc[n, ["name", "time"]]
        dose = dose_to_name(name=name, time=time, ignore_exp=True)
        # print(name, time, dose)
        saliv.loc[n, "dose"] = dose

    # saliv = shuffle(saliv, random_state=1995)
    # print(saliv)
    Nobs = len(saliv)
    Nctr = len(saliv[saliv["ctr"] == True])
    Nirr = len(saliv[saliv["ctr"] == False])
    print(f"Sample size: {Nobs}, {Nctr} control ({Nctr / Nobs * 100:.0f}%), {Nirr} irr")

    # y = saliv[["val", "ctr"]]
    y = saliv["val"]
    x = saliv[["time", "dose"]]
    print(x.shape, y.shape)

    df_results = pd.DataFrame()
    i = 0
    for n in range(Nobs):
        x_train = x.drop(n)
        y_train = y.drop(n)
        x_test = x.loc[n].values.reshape(1, -1)
        y_test = np.array([y.loc[n]]).reshape(1, -1)
        print(f"\nLOO {n} / {Nobs}")
        if do_hp_tuning:
            params = hp_tuning(x_train, y_train, regmode="rf")
        else:
            params = {}
        for r in range(Nrep):
            print(".", end="")
            rf = RandomForestRegressor(**params)
            lr = LinearRegression()
            rf.fit(x_train, y_train)
            lr.fit(x_train, y_train)

            pred_rf = rf.predict(x_test)
            pred_lr = lr.predict(x_test)

            df_results.loc[i, "idx"] = n
            df_results.loc[i, "rep"] = r
            df_results.loc[i, "y_gt"] = y_test
            df_results.loc[i, "pred lr"] = pred_lr
            df_results.loc[i, "pred rf"] = pred_rf
            i += 1
        # print(df_results)
    print(df_results)
    df_results.to_csv(os.path.join(savefolder, f"loocv_predictions_timedose_nrep={Nrep}.csv"))
    pass


def show_loocv_timedose_results(nrep=5):
    filename = f"loocv_predictions_timedose_nrep={nrep}.csv"
    df = pd.read_csv(os.path.join(RegressDir, "timedose all", filename), index_col=0)
    # print(df)
    # print(df.groupby("idx").mean())
    df_sd = df.groupby("idx").std().reset_index(drop=True)
    df = df.groupby("idx").mean().drop(["rep"], axis=1).reset_index(drop=True)
    print(df)
    print(df_sd)
    ytrue = df["y_gt"]
    pred_lr = df["pred lr"]
    pred_rf = df["pred rf"]
    pred_rf_upper = pred_rf + df_sd["pred rf"]
    pred_rf_lower = pred_rf - df_sd["pred rf"]

    r2_lr = r2_score(ytrue, pred_lr)
    r2_rf = r2_score(ytrue, pred_rf)
    r2_rf_upper = r2_score(ytrue, pred_rf_upper)
    r2_rf_lower = r2_score(ytrue, pred_rf_lower)
    mse_lr = mean_squared_error(ytrue, pred_lr)
    mse_rf = mean_squared_error(ytrue, pred_rf)
    mse_rf_upper = mean_squared_error(ytrue, pred_rf_upper)
    mse_rf_lower = mean_squared_error(ytrue, pred_rf_lower)
    print(f"MSE: lr={mse_lr:.2f}, rf={mse_rf:.2f}")
    print(f"R2: lr={r2_lr:.2f}, rf={r2_rf:.2f}")
    print(f"R2 RF lower / upper: [{r2_rf_lower:.2f}, {r2_rf_upper:.2f}]")
    print(f"MSE RF lower / upper: [{mse_rf_lower:.2f}, {mse_rf_upper:.2f}]")
    pass


def loocv_combined_saliv(MODE="NO P", WEIGHT="T2", LRMODE="aggregated", REGRESSOR="LINREG", NREP=100, do_hp_tuning=False, save=True):
    # Given mode + weight + LRmode: Have k best features: selected > 50% of times in previous LOOCV analysis
    # have saliva data corresponding to
    # for each left out:
    #   for each rep:
            # calculate pred td, pred td + ft1, .., pred td + ftk
            # save ytrue + predictions
    savename = f"loocv_combined_{MODE}_{WEIGHT}_LR{LRMODE}_{REGRESSOR}_nrep={NREP}_hptune={do_hp_tuning}.csv"
    if not REGRESSOR in ["LINREG", "RF"]:
        print("Try regressor: LINREG, RF")
        return 0

    if MODE == "NO P":
        x_orig, y_orig = load_nop(WEIGHT, LRMODE, training="all", xer=False)
    elif MODE == "DELTA P":
        x_orig, y_orig = load_delta_P(WEIGHT, LRMODE, training="all", xer=False)
    else:
        return 0
    x_orig = x_orig.reset_index(drop=True)
    y_orig = y_orig.reset_index(drop=True)
    # print(y_orig)

    td = ["time", "dose"]
    best_fts = get_best_fts(MODE, WEIGHT, 5, LRMODE, REGRESSOR, plot=False)
    print(best_fts)

    # Xlist = [x_orig[td], x_orig[best_fts], *[x_orig[ft, "time", "dose"] for ft in best_fts]]
    Xlist = [x_orig[td], x_orig[best_fts]]
    for ft in best_fts:
        Xlist.append(x_orig[[*td, ft]])
    MODEL_DESCRIPTORS = ["td", "fts", *[f"td+ft{i+1}" for i in range(len(best_fts))]]
    print(MODEL_DESCRIPTORS)

    df_results = pd.DataFrame()
    df_results.loc[-1, "idx"] = -1
    df_results.loc[-1, "y_gt"] = -1
    for c, ft in zip(MODEL_DESCRIPTORS, ["time dose", "all fts", *best_fts]):
        df_results.loc[-1, c] = ft
    print(df_results)

    Nobs = len(x_orig)
    i = 0
    for n in range(Nobs):
        print(f"\nLOOCV {n} / {Nobs}")
        # x_train, xtd_train, y_train = x_orig.drop(i), xtd.drop(i), y_orig.drop(i)
        # x_test, xtd_test, y_test = np.reshape([x_orig.loc[i]], (-1, 1)), np.reshape([xtd.loc[i]], (-1, 1)), np.reshape([y_orig.loc[i]], (-1, 1))
        y_train, y_test = y_orig.drop(n), np.reshape([y_orig.loc[n]], (-1, 1))
        # print(y_train.shape, y_test.shape)

        XTRAIN = [X.drop(n) for X in Xlist]
        XTEST = [np.reshape([X.loc[n]], (1, -1)) for X in Xlist]

        if do_hp_tuning and REGRESSOR == "RF":
            print("NOT IMPLEMENTED")
            return 0
            # params = hp_tuning(x_train[1], y_train, regmode="rf")
        else:
            params = {}

        for r in range(NREP):
            predictions = []
            coefs = []  # coefficient of ft_i in all k td + ft_i models
            print(".", end="")
            for x_train, x_test in zip(XTRAIN, XTEST):
                # print(x_train.shape, x_test.shape)
                md = RandomForestRegressor(**params) if REGRESSOR == "RF" else LinearRegression()
                md.fit(x_train, y_train)
                if REGRESSOR == "LINREG" and len(x_train.T) == 3:
                    # print(md.feature_names_in_)
                    # print(md.coef_)
                    coef_fti = md.coef_[-1]
                    coefs.append(coef_fti)
                pred = md.predict(x_test)
                predictions.append(pred[0])
            y_gt = y_test.reshape(-1)[0]
            # print(y_gt, predictions)

            df_results.loc[i, "idx"] = n
            df_results.loc[i, "y_gt"] = y_gt

            k = 0
            for c, pred in zip(MODEL_DESCRIPTORS, predictions):
                df_results.loc[i, c] = pred
                if REGRESSOR == "LINREG" and c not in ["td", "fts"]:
                    df_results.loc[i, f"{c} coef"] = coefs[k]
                    k += 1
            i += 1

    print(df_results)
    if save:
        df_results.to_csv(os.path.join(RegressDir, "td fts combined", savename))
        print("DF SAVED")
    else:
        print("DF NOT SAVED...")
    return 1


def show_loocv_combined_results():
    from scipy.stats import ttest_rel
    folder = os.path.join(RegressDir, "td fts combined")
    files = os.listdir(folder)
    files = list(filter(lambda f: "loocv_combined" in f, files))
    print("FOUND", len(files), "LOOCV COMBINED FILES")

    files_include = ["loocv_combined_DELTA P_T2_LRaggregated_LINREG_nrep=1_hptune=False.csv",
                     "loocv_combined_DELTA P_T2_LRaverage_LINREG_nrep=1_hptune=False.csv",
                     "loocv_combined_NO P_T2_LRaggregated_LINREG_nrep=1_hptune=False.csv"]
    # files = list(filter(lambda f: f in files_include, files))
    # print(files)

    df_sign_models = pd.DataFrame(index=files, columns=["num sign"], data=np.zeros(len(files)))
    print(df_sign_models)
    tot_tests = 0
    for f in files:
        mode, weight, lrmode, regressor, nrep, hptune = f.split("_")[2:]
        print("\n", f)
        print(mode, weight, lrmode, regressor, nrep, hptune)
        df = pd.read_csv(os.path.join(folder, f), index_col=0)
        fts_best = df.loc[-1].drop(["idx", "y_gt", "td", "fts"]).dropna().values
        df = df.drop(-1).astype("float64")
        ft_id_dict = get_feature_index_fsps(LRMODE=lrmode[2:], MODE=mode, WEIGHT=weight)
        print(fts_best)
        ft_keys = [ft_id_dict.get(ft) for ft in fts_best]
        print(ft_keys)
        n_fits = len(df)
        df_mean = df.groupby("idx").mean()  # mean of repeated estimates per left-out (idx)
        df_sd = df.groupby("idx").std()
        n_obs = len(df_mean)
        y_true = df_mean["y_gt"]

        cols_new = ["y_gt", "td", *[f"td+{idx}" for idx in ft_keys]]
        cols_old = df_mean.columns.values
        # cols_new = cols_old
        # print(cols_new)

        df_squared_error = pd.DataFrame(columns=cols_new)
        # df_squared_error = pd.DataFrame(columns=df_mean.columns)
        # for c in df_mean.drop(["y_gt"], axis=1).columns.values:
        for c_new, c_old in zip(cols_new, cols_old):
            df_squared_error[c_new] = [(ygt - v)**2 for ygt, v in zip(y_true, df_mean[c_old])]

        df_stats = pd.DataFrame(columns=["MSE"], data=df_squared_error.sum() / n_obs)
        # print(df_stats)
        td_sqe = df_squared_error["td"]
        ft_idx_rows = df_squared_error.drop(["y_gt", "td"], axis=1).columns.values
        for c in ft_idx_rows:
            tot_tests += 1
            sqe_vals = df_squared_error[c].values
            t, p = ttest_rel(td_sqe, sqe_vals)
            df_stats.loc[c, "p-val equal"] = p
            t, p = ttest_rel(td_sqe, sqe_vals, alternative="greater")
            df_stats.loc[c, "p-val greater"] = p

        if any(["coef" in c for c in df.columns]):
            df_coef = df.filter(like="coef", axis=1)
            avg_coefs = df_coef.mean()
            sign_rel_coefs = df_coef.apply(lambda x: np.sign(x)).sum() / n_fits
            cv_coefs = df_coef.apply(lambda x: np.std(x) / np.mean(x))
            # print(cv_coefs)
            # print(df_coef.apply(lambda x: np.mean(x)))
            # print(avg_coefs)
            # print(sign_rel_coefs)
            for c, c_stats in zip(avg_coefs.index, ft_idx_rows):
                df_stats.loc[c_stats, "coef sign frac"] = sign_rel_coefs.loc[c]
                df_stats.loc[c_stats, "coef avg"] = avg_coefs.loc[c]
                df_stats.loc[c_stats, "coef cv"] = cv_coefs.loc[c]
        print(df_stats)
        df_stats_sign = df_stats[df_stats["p-val greater"] < 0.05]
        print(df_stats_sign)
        num_sign = len(df_stats_sign)
        df_sign_models.loc[f, "num sign"] = num_sign

    print(df_sign_models)
    print(df_sign_models[(df_sign_models.T != 0).any()])
    print("Total tests performed:", tot_tests)
    pass


if __name__ == "__main__":
    # find_optimal_feature_amount(MIN_FTS=2)

    # plot_best_fts("NO P", "T2", 5, "aggregated", "LINREG")
    # plot_best_fts("DELTA P", "T2", 5, "average", "RF")

    # calculate_scores_for_hp(LRMODE="average")
    # calculate_scores_for_hp(LRMODE="aggregated")
    # plot_hpparam_results()
    # plot_hpparam_results_timedose()

    # regression_timedose_saliva(do_hp_tuning=False, n_repeats=1, calc_RFhp_results=False, save=False, only_ols_all_data=True)
    # show_timedose_results("r2scores")
    # show_timedose_results("mse")

    # show_loocv_results(show_rfe=False)
    # loocv_timedose_saliv(do_hp_tuning=False, Nrep=100)
    # show_loocv_timedose_results(nrep=10)
    # show_loocv_timedose_results(nrep=100)

    show_loocv_combined_results()
    # loocv_combined_saliv(MODE="NO P", WEIGHT="T2", LRMODE="aggregated", REGRESSOR="LINREG", NREP=1, save=True)
    # loocv_combined_saliv(MODE="DELTA P", WEIGHT="T2", LRMODE="aggregated", REGRESSOR="LINREG", NREP=1, save=True)
    # loocv_combined_saliv(MODE="DELTA P", WEIGHT="T2", LRMODE="average", REGRESSOR="LINREG", NREP=1, save=True)

    sys.exit()

    tune_hp = False
    REG = "RF"
    Nrep = 100
    for LRMODE in ["aggregated", "average"]:
    # for LRMODE in ["average"]:
        main_loocv_regression("NO P", "T1", LRMODE, REGRESSOR=REG, NORMALIZE_FEATURES=False, do_hp_tuning=tune_hp, NREP=Nrep)
        main_loocv_regression("NO P", "T2", LRMODE, REGRESSOR=REG, NORMALIZE_FEATURES=False, do_hp_tuning=tune_hp, NREP=Nrep)
        main_loocv_regression("DELTA P", "T2", LRMODE, REGRESSOR=REG, NORMALIZE_FEATURES=False, do_hp_tuning=tune_hp, NREP=Nrep)

    # main_loocv_regression("NO P", "T1", "average", REGRESSOR="LINREG", NORMALIZE_FEATURES=False, save=False)
    sys.exit()


    # COMPARE DESICION BOUNDARY FOR TIME, DOSE vs INTERPOLATED SALIVA VALUES
    # mode, wgt, lrmode = "DELTA", "T2", "aggregated"
    # x_train, y_train, x_test, y_test = load_train_test_regression(MODE=mode, WEIGHT=wgt, LRMODE=lrmode)
    # x_train, x_test = x_train[["time", "dose"]], x_test[["time", "dose"]]
    #
    # lm = LinearRegression()
    # rf = RandomForestRegressor()
    # rf.fit(x_train, y_train)
    # lm.fit(x_train, y_train)
    # print(f"R2: lr={lm.score(x_test, y_test):.2f}, rf={rf.score(x_test, y_test):.2f}")
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # title = f"{mode} {wgt}, LR{lrmode}"
    # show_decision_boundary_regression(rf, x_train, y_train, x_test, y_test, Nsteps=25, title=f"Random forest model, {title}", xlab="")
    # show_decision_boundary_regression(lm, x_train, y_train, x_test, y_test, Nsteps=25, title=f"Multiple linear model, {title}", xlab="")

    # MANUAL CALCULATION OF WEIRD R2 VALUES
    # x_train, y_train, x_test, y_test = load_train_test_regression(MODE="NO P", WEIGHT="T2", LRMODE="aggregated")
    # x_train, y_train, x_test, y_test = load_train_test_regression(MODE="DELTA P", WEIGHT="T2", LRMODE="aggregated")
    # x_train, x_test = x_train.drop(["time", "dose"], axis=1), x_test.drop(["time", "dose"], axis=1)
    # calculate_r2_manually(x_train, y_train, x_test, y_test)
    # sys.exit()

    # LRMODE = "split"
    # LRMODE = "average"
    LRMODE = "aggregated"
    show_results(with_tuning=True, include_enet=False, LRMODE=LRMODE, scoremode="r2")
    sys.exit()

    scoremode = "mse"
    # scoremode = "r2"
    do_hp_tuning = True
    n_jobs = -1
    NUMFTS_LIST = [5, 10, 15, "all"]
    # for LRMODE in ["aggregated", "average", "split"]:
    for LRMODE in ["aggregated"]:
        df_main = pd.DataFrame()
        savename = f"{LRMODE}_{scoremode}_scores_hptuned.csv" if do_hp_tuning else f"{LRMODE}_{scoremode}_scores.csv"
        save_path = os.path.join(ModelDir, "Regression", savename)

        for MODE in ["NO P", "DELTA P", "DELTA"]:
            for WEIGHT in ["T1", "T2"] if MODE == "NO P" else ["T2"]:
                for num_fts in NUMFTS_LIST:
                    print("\n\n------- New regression models for:", LRMODE, MODE, WEIGHT, num_fts, "fts -------")
                    df_curr = main_regression(MODE=MODE, WEIGHT=WEIGHT, LRMODE=LRMODE, num_fts=num_fts,
                                              do_hp_tuning=do_hp_tuning, n_jobs=n_jobs, scoremode=scoremode)
                    # if str(num_fts) != "all":
                    if str(num_fts) != str(NUMFTS_LIST[0]):
                        df_curr = df_curr.drop(f"{MODE} {WEIGHT} td")
                    # print(df_curr)
                    df_main = pd.concat([df_main, df_curr], axis=0)
                    print(df_main)
                    # print(df_main.shape)
                    df_main.to_csv(save_path)  # save here if pc stops working or whatever
                    # break
                # break
            # break