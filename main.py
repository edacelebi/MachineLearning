from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matrix_curve import matrix_curve
from matplotlib import pyplot as plt

veri= pd.read_csv("music_genre.csv")
veri.drop(['instance_id',"artist_name","track_name","obtained_date"],axis=1,inplace=True)
veri1 = pd.DataFrame(veri)
veri1['tempo'] = pd.to_numeric(veri1['tempo'],errors='coerce')

veri1['music_genre']=veri1['music_genre'].replace({'Electronic':0,'Anime':1,'Jazz':2,'Alternative':3,'Country':4, 'Rap': 5,
                                                   'Blues': 6, 'Rock': 7,'Classical': 8, 'Hip-Hop': 9})
veri1['key']=veri1['key'].replace({"A": 1, "A#": 2, "B": 3, "B#": 4, "C": 5, "C#": 6,
                        "D": 7, "D#": 8, "E": 9, "E#": 10, "F": 11, "F#": 12, "G": 13, "G#": 14})
veri1['mode'] = veri1['mode'].replace({"Minor": 0, "Major": 1})

veri1=veri1.drop_duplicates()
veri1.dropna(inplace=True)

X=veri1.drop("music_genre",axis=1)
y=veri1["music_genre"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,
                             random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#LOGİSTİC REGRESSİON
lg_model=LogisticRegression(max_iter=len(X_train),multi_class='multinomial')
lg_model.fit(X_train,y_train)
y_pred = lg_model.predict(X_test)
print("Logistic regression Accuracy:", accuracy_score(y_test, y_pred))

#matrix_curve().plot_cofusion_matrix(y_test, y_pred, "Logistic Regression")
#matrix_curve().plot_traning_curves(X, y, lg_model, "Logistic Regression")

#SUPPORT VECTOR MACHİNE
svm_model =svm.SVC(kernel='rbf')
svm_model.fit(X_train,y_train)
y_pred = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))

#matrix_curve().plot_cofusion_matrix(y_test, y_pred, "Support Vector Machine")
#matrix_curve().plot_traning_curves(X, y, svm_model, "Support Vector Machine")



#KNN ALGORİTHM
knn_model=KNeighborsClassifier(n_neighbors=27)
knn_model.fit(X_train,y_train)
y_pred=knn_model.predict(X_test)
print("KNN Algorithm Accuracy:", accuracy_score(y_test, y_pred))

#error = list()
#for i in range(1, 40):
#    knn = KNeighborsClassifier(n_neighbors=i)
#    knn.fit(X_train, y_train)
#    pred_i = knn.predict(X_test)
#    error.append(np.mean(pred_i != y_test))

#plt.figure(figsize=(12, 6))
#plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
#        markerfacecolor='blue', markersize=10)
#plt.title('Error Rate K Value')
#plt.xlabel('K Value')
#plt.ylabel('Mean Error')
#plt.show()


#matrix_curve().plot_cofusion_matrix(y_test, y_pred, "KNN")
#matrix_curve().plot_traning_curves(X, y, knn_model, "KNN")





