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
from sklearn.utils import resample

################################CLASSIFICATORS#########################################################
from sklearn.utils import resample

##DIMENSIONALITY REDUCTION TO PREDICT QUESTIONNAIRE GROUPS

rng_seed = 1
kernel = 1.0 * RBF(1.0)

data = pd.read_csv(
    "../../../Merged_dataframes/df_ALL_metadata_MEG_sub00to11_epo_RT_before.csv"
)
TO_PREDICT = "parei"

min_obs = 50
# n_seeds = 50
savename = "epo_RT_before_allSpectral"
variables = ["delta", "alpha", "theta", "gamma2", "low_beta", "high_beta", "gamma1"]
n_permutations = 100

participants = list(data["participant"].unique())
print("PARTICIPANTS", participants)
all_df_Acc = []
for participant in participants[:]:
    df_ = data.loc[data["participant"] == participant]

    df_list = []
    elec_min = np.min(df_["electrodes"].unique())
    elec_max = np.max(df_["electrodes"].unique()) + 1
    for elec in range(elec_min, elec_max):
        # for elec in range(10):
        df_ml = df_.loc[df_["electrodes"] == elec]

        dfs = []
        for v in variables:
            dfs.append(df_ml[[v]])
        dfs.append(df_ml[[TO_PREDICT, "participant"]])
        df_final = pd.concat(dfs, axis=1)

        X2 = df_ml.drop(TO_PREDICT, axis=1).values
        y2 = df_ml[TO_PREDICT].values

        # X=X.values
        # y=y.values
        n_samples, n_features = X2.shape
        # n_neighbors = 15
        n_components = 2

        R_frame = df_ml.copy()

        # Balance classes
        parei = R_frame.loc[R_frame[TO_PREDICT] == 1]
        NOparei = R_frame.loc[R_frame[TO_PREDICT] == 0]

        if len(parei) > len(NOparei):
            parei = resample(
                parei, replace=True, n_samples=len(NOparei), random_state=42
            )
        if len(parei) < len(NOparei):
            NOparei = resample(
                NOparei, replace=True, n_samples=len(parei), random_state=42
            )
        print(participant, "N observations : ", len(parei))
        if len(parei) >= min_obs:
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
            X = R_frame[variables].values

            ## Encodage des espèces en valeurs numériques pour la coloration
            le = LabelEncoder()
            le.fit(R_frame.parei.unique())
            y = le.transform(R_frame.parei)
            R_frame = [X, y]

            ##Calculate the mean Accuracy scores on 9 classifiers for a definite set of Random Seeds

            Acc_scores = []
            Score_multiclass = []
            clfs = [
                # KNeighborsClassifier(n_neighbors=3, weights='uniform', leaf_size=30),
                LogisticRegression(
                    random_state=rng_seed,
                    max_iter=1000,
                    penalty="l2",
                    C=0.01,
                    solver="liblinear",
                ),
                SVC(kernel="rbf", C=1, gamma=0.2, max_iter=1000),
                DecisionTreeClassifier(max_depth=30, random_state=rng_seed),
                RandomForestClassifier(
                    max_depth=None,
                    n_estimators=15,
                    max_features=2,
                    random_state=rng_seed,
                ),
                # AdaBoostClassifier(n_estimators=15, learning_rate=0.5,random_state=rng_seed),
                # MLPClassifier(hidden_layer_sizes=(5,), activation='relu', alpha=0.01, max_iter=10000, random_state=rng_seed)]
                # GaussianProcessClassifier(kernel=kernel),
                # GradientBoostingClassifier(n_estimators=15, learning_rate=0.5, subsample=1, max_depth=2, random_state=rng_seed),
            ]
            # Liste des noms associés
            clf_names = [
                "Régression Logistique",
                "SVM",
                "Arbre de décision",
                "Forêt aléatoire",
            ]  # ,'Processus Gaussien','Gradient Boosting'   ]
            df = pd.DataFrame()
            score_ = []
            pval_ = []
            classifier_ = []
            for cnt, clf in enumerate(clfs):
                (
                    score,
                    permutation_scores,
                    pvalue,
                ) = sklearn.model_selection.permutation_test_score(
                    clf, X, y, groups=groups, n_permutations=n_permutations
                )
                score_.append(score)
                pval_.append(pvalue)
                classifier_.append(clf_names[cnt])
                print("SCORE : ", score, "\nCLASSIFIER", clf_names[cnt], "pval", pvalue)
            df["score"] = score_
            df["pvalue"] = pval_
            df["classifier"] = classifier_
            df["electrodes"] = elec
            df["participant"] = participant
            df_list.append(df)

    if len(df_list) > 0:
        df_Acc = pd.concat(df_list)
        df_Acc.to_csv(
            "../../../OUTPUTS/ML_accuracy_scores/classifiers_{}_{}_-p{}_{}.csv".format(
                savename, variables, str(participant), str(n_permutations)
            ),
            index=False,
        )
        all_df_Acc.append(df_Acc)
df_Acc_final = pd.concat(all_df_Acc)
df_Acc_final.to_csv(
    "../../../OUTPUTS/ML_accuracy_scores/classifiers_{}_{}_{}.csv".format(
        savename, variables, str(n_permutations)
    ),
    index=False,
)
