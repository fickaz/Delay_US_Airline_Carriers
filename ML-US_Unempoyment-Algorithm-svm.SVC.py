#MACHINE LEARNING - SVC Linear kernel

from sklearn import svm, preprocessing, cross_validation
import pandas as pd
import numpy as np


#Economic_data = pd.read_pickle('Economic_Data.pickle')
Unemployment_Data = pd.read_pickle('Unemployment_Data.pickle')

Unemployment_Data['National_Unemployment_Future']=Unemployment_Data['National_Unemployment_Rate'].shift(-1)
Unemployment_Data.replace([np.inf, -np.inf],np.nan, inplace=True)
Unemployment_Data.dropna(inplace=True)
print(Unemployment_Data.head())

def create_labels(cur_unemp, fut_unemp):
    if fut_unemp < cur_unemp:
        return 1
    else:
        return 0

Unemployment_Data['label']=list(map(create_labels,
                                    Unemployment_Data['National_Unemployment_Rate'],
                                    Unemployment_Data['National_Unemployment_Future']))



X = np.array(Unemployment_Data.drop(['label','National_Unemployment_Future'],1))
##print(X)
X = preprocessing.scale(X)

y = np.array(Unemployment_Data['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                                     y,
                                                                     test_size=0.1)

clf = svm.SVC(kernel='linear', C=0.1)
clf.fit(X_train,y_train)
print('Linear kernel: ',clf.score(X_test, y_test))


clf2 = svm.SVC(kernel='rbf', C=0.1, gamma=3)
clf2.fit(X_train,y_train)
print('RBF kernel: ', clf2.score(X_test,y_test))

clf3 = svm.SVC(kernel='poly', C=1.0, degree=3)
clf3.fit(X_train, y_train)
print('Polynomial kernel: ',clf3.score(X_test, y_test))





