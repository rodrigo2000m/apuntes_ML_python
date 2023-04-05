# PCA

import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

if __name__ == "__main__":
    df_heart = pd.read_csv("./data/heart.csv")
    #print(df_heart.head())

    df_features = df_heart.drop(["target"], axis=1)
    df_target = df_heart["target"]

    #normalización de los datos
    df_features = StandardScaler().fit_transform(df_features)

    x_train, x_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=7)


    #PCA
    pca = PCA(n_components=3)
    pca.fit(x_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(x_train)

    #plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    #plt.show()

    logitic = LogisticRegression(solver="lbfgs")

    df_train = pca.transform(x_train)
    df_test = pca.transform(x_test)

    logitic.fit(df_train, y_train)

    print("SCORE PCA:", logitic.score(df_test, y_test))


    df_traini = ipca.transform(x_train)
    df_testi = ipca.transform(x_test)
    logitic.fit(df_traini, y_train)
    print("SCORE IPCA: ", logitic.score(df_testi, y_test))
