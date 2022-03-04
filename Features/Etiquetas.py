# -*- coding: utf-8 -*-
"""
    Algoritmo para etiquetar los Exudados Duros
    Created on Fry Jul 03 07:50:36 2015
    @author: Mario Xavier Canche Uc
"""

import numpy as np
from skimage import io as Image
from skimage.morphology import label, disk, dilation
from skimage.measure import regionprops


def etiquetas( name ):

    # Nivel de confianza
    NIVEL = 0.5

    E = [] # Etiquetas    
    
    # Leemos la Imagen Segmentada
    I = Image.imread('../Segmentation/resultImage_Seg/'+name)
    # Leemos el Groundtruth de los Exudados Duros
    I2 = Image.imread('../../UVA_ALL/Resize/resultImage_marcadas/'+name)
    Mask = I2[:,:,2]
    # Leemos la Imagen Maskara del Disco Optico
    I3 = Image.imread('../OpticDisk/Resultados/'+name)
    
    # Invertimos la Maskara del Disco Optico
    OD = np.array(I3==0)
    # Eliminamos el Disco Optico de la Segmentacion
    I = I*OD;

    # Realizamos una Apertura Morfologica a la Segmentacion para reducir ruido
    I = dilation(I,disk(2))


    label_I = label(I)
    propiedades = regionprops(label_I)

    for i in range(len(propiedades)):
        if Mask[np.split(np.array(propiedades[i].centroid).astype(int),2)] >= NIVEL*255:
            E += [1] # Exudado Duro
        else:
            E += [0] # No Exudado Duro


    # Liberamos Memoria
    del I, Mask, label_I, propiedades
    return E



def  example(NAME):
    
    print NAME
    # Extraemos las caracteristicas
    labels =  etiquetas(NAME+'.png')
    np.save('ResultadosE/'+NAME, labels)
    del labels


if __name__ == '__main__':

    from multiprocessing import Pool
    import csv
    
    # Leemos los nombres de las Imagenes
    reader = csv.reader(open('../exudatesUVA_ALL.csv','rb'))
    
    # Creamos la lista de Nombres
    NAMES = []
    for index, name in enumerate(reader):
        NAMES = NAMES + name

    # Extraemos las Caracteristicas
    pool = Pool()
    pool.map(example,NAMES)
