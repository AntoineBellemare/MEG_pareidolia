import pandas as pd
import seaborn as sbn
import numpy as np
import sys

sys.path.insert(0, "C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions")
import MEG_pareidolia_utils
from MEG_pareidolia_utils import merge_multi_GLM, get_pareidolia_bids
from ML_functions import *
from PARAMS import FOLDERPATH
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

################################CLASSIFICATORS#########################################################


savepath = "../../../OUTPUTS/ML_accuracy_scores/"
rng_seed = 1
kernel = 1.0 * RBF(1.0)
min_obs = 30
data = pd.read_csv(
    "../../../Merged_dataframes/df_ALL_metadata_MEG_sub00to11_epo_RT_ALL_new.csv"
)
list_participants = [0, 1, 2, 4, 6, 8, 10, 11]
list_participants = range(0, 11, 1)
# df_p = []
# for p in list_participants:
#    df_p.append(data.loc[data["participant"] == p])

# df_new = pd.concat(df_p)
# df_new["oneVSmany"] = df_new["n_obj_class"] - 1
# print(df_new["oneVSmany"])
# df_new = df_new[df_new.oneVSmany != -1]
TO_PREDICT = "parei"
savename = "epo_RT_StratGroupKfold_BAcc_minObs30_single_feature_parei"
variables = [
    "delta",
    "theta",
    "alpha",
    "low_beta",
    "high_beta",
    "gamma1",
    "gamma2",
    "LZ",
    "exp",
]
n_permutations = 100

participants = list(data["participant"].unique())
df_list = []

for variable in variables:
    df_ = data.copy()

    elec_min = np.min(df_["electrodes"].unique())
    elec_max = np.max(df_["electrodes"].unique()) + 1
    for elec in range(elec_min, elec_max):
        df_ml = df_.loc[df_["electrodes"] == elec]
        dfs = []
        for v in [variable]:
            dfs.append(df_ml[[v]])
        dfs.append(df_ml[[TO_PREDICT, "participant"]])
        df_final = pd.concat(dfs, axis=1)

        R_frame = df_final.copy()

        # Balance classes
        parei = R_frame.loc[R_frame[TO_PREDICT] == 1]
        NOparei = R_frame.loc[R_frame[TO_PREDICT] == 0]

        """if len(parei)>len(NOparei):
            parei = resample(parei,
                         replace=True,
                         n_samples=len(NOparei),
                         random_state=42)
        if len(parei) < len(NOparei):
            NOparei = resample(NOparei,
                         replace=True,
                         n_samples=len(parei),
                         random_state=42)"""
        # print(participant, 'N observations : ', len(parei))
        # if len(parei) >= min_obs:
        R_frame = pd.concat([parei, NOparei])
        ##Keep only specific FDs
        # R_frame = R_frame[(R_frame.loc[:, 'FD'] == 1.3) | (R_frame.loc[:, 'FD'] > 1.6)]
        # R_frame = R_frame[R_frame['FD'] > 1.5]
        R_frame.index = pd.RangeIndex(len(R_frame.index))

        # Keep relevant variables
        # R_frame = R_frame.loc[:, [c1, c2, TO_PREDICT, 'participant']]

        groups = R_frame.loc[:, "participant"].values

        # R_frame = R_frame.drop('participant', axis=1)
        # Création du dataset
        X = R_frame[variable].values
        X = X.reshape(-1, 1)
        ## Encodage des espèces en valeurs numériques pour la coloration
        le = LabelEncoder()
        le.fit(R_frame[TO_PREDICT].unique())
        y = le.transform(R_frame[TO_PREDICT])
        R_frame = [X, y]

        ##Calculate the mean Accuracy scores on 9 classifiers for a definite set of Random Seeds

        Acc_scores = []
        Score_multiclass = []
        clfs = [
            LogisticRegression(
                random_state=rng_seed,
                max_iter=1000,
                penalty="l2",
                C=0.01,
                solver="liblinear",
            ),
            SVC(kernel="rbf", C=1, gamma=0.2, max_iter=1000),
        ]
        # DecisionTreeClassifier(max_depth=None, random_state=rng_seed),
        # RandomForestClassifier(max_depth=None, n_estimators=100, max_features=2, random_state=rng_seed, n_jobs=5)
        # ]
        # GaussianProcessClassifier(kernel=kernel),
        # GradientBoostingClassifier(n_estimators=15, learning_rate=0.5, subsample=1, max_depth=2, random_state=rng_seed),
        # ]
        # Liste des noms associés
        clf_names = [
            "Régression Logistique",
            "SVM",
        ]  # ,'Processus Gaussien','Gradient Boosting'   ]
        df = pd.DataFrame()
        score_ = []
        pval_ = []
        classifier_ = []
        # Assuming `participant` is the column in your dataframe that contains the participant IDs.
        participants = list(data["participant"].unique())
        n_splits = len(participants)  # Number of participants for leave-one-subject-out

        cv = StratifiedGroupKFold(n_splits=n_splits)
        object = StandardScaler()
        X = object.fit_transform(X)
        for cnt, clf in enumerate(clfs):
            (
                score,
                permutation_scores,
                pvalue,
            ) = sklearn.model_selection.permutation_test_score(
                clf,
                X,
                y,
                cv=cv,
                groups=groups,
                n_permutations=n_permutations,
                scoring="balanced_accuracy",
            )
            score_.append(score)
            pval_.append(pvalue)
            classifier_.append(clf_names[cnt])
            print("VARIABLE", variable, "ELECTRODE:", elec)
            print("SCORE : ", score, "\nCLASSIFIER", clf_names[cnt], "pval", pvalue)
        df["score"] = score_
        df["pvalue"] = pval_
        df["classifier"] = classifier_
        df["electrodes"] = elec
        df["feature"] = variable
        df_list.append(df)

if len(df_list) > 0:
    df_Acc = pd.concat(df_list)
    df_Acc.to_csv(
        "../../../OUTPUTS/ML_accuracy_scores/classifiers_{}_{}_{}.csv".format(
            savename, variable, str(n_permutations)
        ),
        index=False,
    )
