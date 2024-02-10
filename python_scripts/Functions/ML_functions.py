import pandas as pd
import seaborn as sbn
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from time import time
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pylab
from sklearn.model_selection import train_test_split
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import random


def strat_group_tt_split(X, y, groups, p_per_class=1, rnd_state=1):
  # Split observations with stratification at the "group" level.
  # This is to be used when we want to predict the subject's class from individual 
  # observations, in case we have multiple observations per subject.
  # In this example, "group" refer to the subject ID (it is a "group" of observations)
  # and "class" refers to the subject's experimental group.
  # This type of split is needed to avoid training and testing on different observations
  # from each subject, which would bias the classifier to recognize the subject himself
  # instead of his experimental group.
  
  
  #### NON-EXHAUSTIF  
  #### FOR UNSTRATIFIED ----> use sklearn.model_selection.GroupShuffleSplit
  
    groups_by_class = []
    groups_lab, groups_lab_idx = np.unique(groups, return_index=True)
    classes_lab = np.unique(y)
    print('GROUPS_BY_CLASS', groups_lab_idx)
    random.seed(rnd_state)
  
  #stocker les index (sur groups_lab) des participants appartenant à chaque classe
    for class_toclassif in classes_lab:
        groups_by_class.append([i for i in range(len(groups_lab)) if y[groups_lab_idx[i]] == class_toclassif])

  #choisir au hasard un ptcp dans chaque classe
    group_to_keep = []
    for class_name, class_content in enumerate(groups_by_class):
        tokeep_idx = random.sample(class_content, p_per_class)
        group_to_keep.append(groups_lab[tokeep_idx])
    
  #créer test_set à partir des participants choisis
    X_test = X[[i for i in range(len(X)) if groups[i] in np.asarray(group_to_keep).flatten()]]
    y_test = y[[i for i in range(len(y)) if groups[i] in np.asarray(group_to_keep).flatten()]]
    groups_test = groups[[i for i in range(len(groups)) if groups[i] in np.asarray(group_to_keep).flatten()]]
  
  #créer train_set
    X_train = X[[i for i in range(len(X)) if not groups[i] in np.asarray(group_to_keep).flatten()]]
    y_train = y[[i for i in range(len(y)) if not groups[i] in np.asarray(group_to_keep).flatten()]]
    groups_train = groups[[i for i in range(len(groups)) if not groups[i] in np.asarray(group_to_keep).flatten()]]
  
  #print(classes_lab)
    return X_train, y_train, groups_train, X_test, y_test, groups_test


def creationMesh(X):
    """
    Crée un grille sur un espace bidimensionnel. Prends le min et le max de chaque dimension et calcule la grille avec une résolution de 0.02. 
    X: un vecteur à deux colonnes de données. 
    """
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx,yy

def AccScores(clf,data, groups=None, rdn=1):
  
    X, y = data
    X = StandardScaler().fit_transform(X)
    # Séparation des données en TRAIN - TEST
    if groups == None:
        print(rdn)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rdn)
    if groups is not None:
        X_train, y_train, groups_train, X_test, y_test, groups_test = strat_group_tt_split(X, y, groups, p_per_class=5, rnd_state=rng_seed_p)
    
    # entrainement du classificateur et calcul du score final (accuracy)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    return score
  
def plotClassifierOnData(name,clf,data,i=3,n=1,multi=False, groups=None, rdn=55):
    """
    Pour Afficher les récultat d'un classificateur sur un dataset
    name : le titre du graphique
    clf : le classificateur à utiliser
    data : les données à utiliser
    i : Le ième graphique sur n à afficher (pour afficher 3 graphiques par ligne)
    n : Le nombre total de graphiques à afficher
    multi: détermine si on affiche juste la frontière de décision (true) ou 
           le score/proba de chaque point de l'espace car on ne peut afficher le score en multiclasse.
    """
   
    # Préparation rapide des données : 
    # normalisation des données 
    X, y = data
    X = StandardScaler().fit_transform(X)
    # Séparation des données en TRAIN - TEST
    if groups == None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rdn)
    if groups is not None:
        X_train, y_train, groups_train, X_test, y_test, groups_test = strat_group_tt_split(X, y, groups, p_per_class=5, rnd_state=rng_seed_p)
    
   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.10, random_state=rng_seed)
    # Pour la visualisation des régions et calcul des bornes 
    xx,yy = creationMesh(X)

    # creation du bon nombre de figures à afficher (3 par lignes)
    ax = plt.subplot(n/3,3,i)
    
    # entrainement du classificateur et calcul du score final (accuracy)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    # Pour afficher les frontières de décision on va choisir une color pour 
    # chacun des points x,y du mesh [x_min, x_max]x[y_min, y_max].

    # Si on est en multiclasse (2 ou +) on affiche juste les frontières
    if multi:
         Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    else:# sinon on peut afficher le gradient du score
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # On affiche le mesh de décision
    Z = Z.reshape(xx.shape)
    test = ax.contourf(xx, yy, Z, 100, cmap=cm, alpha=.8)

    #On affiche la légende
    cbar = plt.colorbar(test)
    cbar.ax.set_title('score')
    
    # On affiche les points d'entrainement
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_points,
               edgecolors='k',s=100)
    # Et les points de test
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_points, 
               edgecolors='k',marker='X',s=100)

    # on définit les limites des axes et autres gogosses
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    ax.set_title(name,fontsize=22)
    # dont le score en bas à droite
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')

    return X_train, y_train, X_test, y_test, score

# On définit une fonction pour adfficher la projection
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y2[i]),
                 color=plt.cm.Set1((y2[i]+1) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)