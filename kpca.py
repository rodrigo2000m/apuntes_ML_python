# KPCA

import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

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



    kpca = KernelPCA(n_components=4, kernel="poly")

    kpca.fit(x_train)


    df_train = kpca.transform(x_train)
    df_test = kpca.transform(x_test)

    logistic = LogisticRegression(solver="lbfgs")

    logistic.fit(df_train, y_train)
    print("SCORE KPCA: ", logistic.score(df_test, y_test))