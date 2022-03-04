# -*- coding: utf-8 -*-
"""
    Algoritmo para exportar las caracteristicas a MatLab
    Created on Wed Jul 08 10:41:16 2015
    @author: Mario Xavier Canche Uc
"""

import csv
import numpy as np
import scipy.io as sio

# Librerias para la Clasificacion
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# Leemos los nombres de las Imagenes
reader = csv.reader(open('../exudatesUVA_ALL.csv','rb'))
    
# Leemos la base de datos
X = []
y = []
for index, name in enumerate(reader):
    y = y + list(np.load('../Features_02/ResultadosE/'+name[0]+'.npy'))
    X = X + list(np.load('../Features_02/ResultadosF/' +name[0]+'.npy'))

# Convertimos de 'list' a 'array'
X = np.array(X)
""" 0: Imagenes de Retina Sanos 
    1: Imagenes de Retina con Exudados Duros    """
y = np.array(y)


# Seleccion de Caracteristicas
X = X[:,[0,1,4,6,7,8,9,11,17,18,19,21,22,23]]
#X = X[:,[0, 1, 2, 5, 12, 13, 17, 22]] # RBF SVM
#X = X[:,[2, 5, 12, 13, 17, 21, 22, 23]] # RBF SVM

print 'Total de Caracteristicas: ', X.shape[1]
print '       Total de Muestras: ', y.shape[0]
print ' Total de Regiones Sanas: ', (y==0).sum()
print '       Regiones No sanas: ', (y==1).sum()



# Estandarizamos la base de Datos
X = StandardScaler().fit_transform(X)

# Creamos los Datos de Entrenamiento y de Validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

# Guardamos los Datos        
sio.savemat('featuresTrain.mat',{'X_train':X_train})
sio.savemat('featuresTest.mat',{'X_test':X_test})
sio.savemat('EtiquetasTrain.mat',{'y_train':y_train})
sio.savemat('EtiquetasTest.mat',{'y_test':y_test})

# Liberamos memoria
del X_test, y_test, X_train, y_train


# Guardamos para la Selección de Caracteristicas
#np.save('features',X)
#np.save('Etiquetas',y)
#sio.savemat('features_total.mat',{'X':X})
#sio.savemat('Etiquetas_total.mat',{'Y':y})




