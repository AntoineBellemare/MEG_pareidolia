import pandas as pd
import seaborn as sbn
import numpy as np
import sys
sys.path.insert(0, 'C:/Users/Antoine/github/MEG_pareidolia/python_scripts/Functions')
import MEG_pareidolia_utils
from MEG_pareidolia_utils import merge_multi_GLM, get_pareidolia_bids
from ML_functions import *
from PARAMS import FOLDERPATH
import sklearn
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

################################CLASSIFICATORS#########################################################
from sklearn.utils import resample

##DIMENSIONALITY REDUCTION TO PREDICT QUESTIONNAIRE GROUPS
savepath = '../../../OUTPUTS/ML_accuracy_scores/'
rng_seed = 1
kernel = 1.0 * RBF(1.0)

data = pd.read_csv('../../../Merged_dataframes/df_ALL_metadata_MEG_sub00to11_epo_RT_before.csv')
list_participants = [0, 1, 2, 4, 6, 8, 10, 11]
df_p = []
for p in list_participants:
    df_p.append(data.loc[data['participant'] == p])

df_new = pd.concat(df_p)
TO_PREDICT = 'parei'
savename='epo_RT_before_StratKfold_BAcc_minObs30'
variables = ['delta', 'alpha', 'theta', 'low_beta', 'high_beta', 'gamma1', 'gamma2']
n_permutations = 100

participants = list(data['participant'].unique())
list_df = []
for elec in df_new['electrodes'].unique():
    df_ml = df_new.loc[df_new['electrodes'] == elec]
    dfs = []
    for v in variables:
        dfs.append(df_ml[[v]])
    dfs.append(df_ml[[TO_PREDICT, 'participant']])
    df_final = pd.concat(dfs, axis=1)

    X2 = df_ml.drop(TO_PREDICT, axis=1).values
    y2 = df_ml[TO_PREDICT].values

    n_samples, n_features = X2.shape

    R_frame = df_ml.copy()


    #Balance classes
    #parei = R_frame.loc[R_frame[TO_PREDICT] == 1]
    #NOparei = R_frame.loc[R_frame[TO_PREDICT] == 0]


    #parei = resample(parei,
    #             replace=True,
    #             n_samples=len(NOparei),
    #             random_state=42)

    #R_frame = pd.concat([parei, NOparei])

    R_frame.index = pd.RangeIndex(len(R_frame.index))

    #Keep relevant variables
    #R_frame = R_frame.loc[:, [c1, c2, TO_PREDICT, 'participant']]

    groups = R_frame.loc[:, 'participant'].values

    #R_frame = R_frame.drop('participant', axis=1)
    # Création du dataset
    X = R_frame[variables].values
    object= StandardScaler()
    X = object.fit_transform(X)
    ## Encodage des espèces en valeurs numériques pour la coloration
    le = LabelEncoder()
    le.fit(R_frame.parei.unique())
    y = le.transform(R_frame.parei)
    R_frame = [X,y]


    ##Calculate the mean Accuracy scores on 9 classifiers for a definite set of Random Seeds

    Acc_scores = []

    clfs = [
                    KNeighborsClassifier(n_neighbors=3, weights='uniform', leaf_size=10),
                    SVC(kernel='rbf', C=1,gamma=.2,max_iter=1000),
                    DecisionTreeClassifier(max_depth=30, random_state=rng_seed),
                    RandomForestClassifier(max_depth=None, n_estimators=15, max_features=3, random_state=rng_seed)
    ]

    clf_names = [
        'k-NN',
        'SVM',
        'Arbre de décision','Forêt aléatoire']

    #cv = StratifiedGroupKFold(n_splits=3)
    '''BAcc_all = []
    for train_idxs, test_idxs in cv.split(X, y, groups):

        X_train = X[train_idxs]
        X_test = X[test_idxs]
        y_train = y[train_idxs]
        y_test = y[test_idxs]
        BAcc_fold = []
        for cnt, clf in enumerate(clfs):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            BAcc = balanced_accuracy_score(y_test, y_pred)
            BAcc_fold.append(BAcc)
        BAcc_all.append(BAcc_fold)


    BAcc_all = np.array(BAcc_all)'''
    #cv = StratifiedGroupKFold(n_splits=3)


    BAccs = []
    pvals = []
    classifiers_ = []
    df = pd.DataFrame()
    for cnt, clf in enumerate(clfs):

        score, permutation_scores, pvalue = sklearn.model_selection.permutation_test_score(clf, X, y,
                                                                                               groups=groups, n_permutations=n_permutations,
                                                                                               cv=5, scoring='balanced_accuracy')
        BAccs.append(score)
        pvals.append(pvalue)
        classifiers_.append(clf_names[cnt])
        print('SCORE : ', score, '/nCLASSIFIER', clf_names[cnt], 'Pval', pvalue)
    df['BAcc'] = BAccs
    df['pvalue'] = pvals
    df['classifier'] = classifiers_
    df['electrode'] = elec
    list_df.append(df)

if len(list_df) > 0:
    df_Acc = pd.concat(list_df)
    df_Acc.to_csv(savepath+'classifiers_{}_participants_{}_{}.csv'.format(str(len(participants)), savename, str(n_permutations)), index=False)
