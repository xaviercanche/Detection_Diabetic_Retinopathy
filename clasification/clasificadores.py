# -*- coding: utf-8 -*-
"""
    Algoritmo para Clasificar Exudados Duros
    Created on Tue Oct 14 19:12:16 2014
    @author: Mario Xavier Canche Uc
"""

import numpy as np
# Librerias para la Clasificacion
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.metrics import roc_curve, auc

from math import sqrt
import csv
    
# Leemos los nombres de las Imagenes
reader = csv.reader(open('../exudatesDIARETDB1.csv','rb'))
    
# Leemos la base de datos
X = []
y = []
for index, name in enumerate(reader):
    y = y + list(np.load('../Features/ResultadosE/'+name[0]+'.npy'))
    X = X + list(np.load('../Features/ResultadosF/' +name[0]+'.npy'))

# Convertimos de 'list' a 'array'
X = np.array(X)
""" 0: Imagenes de Retina Sanos 
    1: Imagenes de Retina con Exudados Duros    """
y = np.array(y)

# Seleccion de Caracteristicas
X = X[:,[0,1,4,6,7,8,9,11,17,18,19,21,22,23]]
#X = X[:,[0, 1, 2, 5, 12, 13, 17, 22]] # RBF SVM
#X = X[:,[2, 5, 12, 13, 17, 21, 22, 23]] # RBF SVM

# Estandarizamos la base de Datos
X = StandardScaler().fit_transform(X)

print 'Total de Caracteristicas: ', X.shape[1]
print '       Total de Muestras: ', y.shape[0]
print ' Total de Regiones Sanas: ', (y==0).sum()
print '       Regiones No sanas: ', (y==1).sum()



# Escogemos los Clasificadores a utilizar
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

# Creamos la lista para guardar los Resultados por Clasificador
Resultado = [[],[],[],[],[],[],[],[],[]]

# Realizamos 30 Iteraciones y promediamos el resultado
for Iteracion in range(100):
    
    # Creamos los Datos de Entrenamiento y de Validación
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
        
    # Iteramos sobre la lista de Clasificadores    
    for nombre, clasificador, i in zip(names, classifiers,range(9)):
        
        # Entrenamos el Clasificador con nuestros Datos de Entrenamiento
        clasificador.fit(X_train, y_train)
    
        # Prediccion del clasificador
        P = clasificador.predict(X_test)
        # Tabla de Confusion
        TP = (P*y_test).sum()
        FP = P.sum() - TP
        FN = y_test.sum() - TP
        TN = y_test.shape[0]- y_test.sum() - FP
    
        SN = float(TP)/(TP+FN)
        SP = float(TN)/(TN+FP)
        PPV = float(TP)/(TP+FP)
        ACC = float(TP+TN)/(y_test.shape[0])

	Fm =  float(2.0*PPV*SN)/float(PPV+SN)
	GM = sqrt((float(TP)/float(TP+FN))*(float(TN)/float(FP+TN)))
	if SN > 0:
		AGM = float(float(GM+SP*(FP+TN))/float(1.0+FP+TN))
	else:
		AGM = 0
    
        # Determinamos el false positive and true positive rates
        fpr, tpr, _ = roc_curve(y_test, clasificador.predict_proba(X_test)[:,1])
        # Calculamos el Area bajo la Curva ROC
        AUC = auc(fpr, tpr)

        # Aplicamos una validacion cruzada
        cross = cross_val_score(clasificador,X,y,cv=10, n_jobs=-1, scoring='roc_auc').mean()

        

        # Guardamos los resultados de la Iteracion
        Resultado[i] += [[ACC,SP,SN,PPV,AUC,Fm,GM,AGM,cross]]
        
    # Liberamos memoria
    del X_test, y_test


# Convertimos a Matriz para poder sacar la media de las iteraciones
R = np.array(Resultado)


# Imprimimos los Resultados
print '   ACC       SP        SN        PPV'
for nombre, i in zip(names,range(9)):
    print ' {0:4.4f}    {1:4.4f}    {2:4.4f}    {3:4.4f}    {4:1s}'.format(R[i,:,0].mean(),R[i,:,1].mean(),R[i,:,2].mean(),R[i,:,3].mean(),nombre)

print ' '
print '   AUC       Fm        GM        AGM'
for nombre, i in zip(names,range(9)):
    print ' {0:4.4f}    {1:4.4f}    {2:4.4f}    {3:4.4f}    {4:1s}'.format(R[i,:,4].mean(),R[i,:,5].mean(),R[i,:,6].mean(),R[i,:,7].mean(),nombre)

print ' '
print '   Cross-Validation'
for nombre, i in zip(names,range(9)):
    print ' {0:4.4f}    {1:1s}'.format(R[i,:,8].mean(),nombre)


