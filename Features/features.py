# -*- coding: utf-8 -*-
"""
    Algoritmo de Extracción de Características
    Created on Tue Sep 23 17:35:36 2014
    @author: Mario Xavier Canche Uc
    
"""

import numpy as np
from skimage import io as Image
from skimage.morphology import dilation, square, label, disk
from skimage.measure import regionprops
from skimage.filter import prewitt
from skimage.feature import greycomatrix

def ExtractFeatures( name ):
    # Leemos la Imagen Segmentada
    I = Image.imread(str('../Segmentation/resultImage_Seg/')+name)
    # Leemos la Imagen RGB
    I2 = Image.imread(str('../../UVA_ALL/Resize/resultImage_originales/')+name)
    # Leemos la Imagen Maskara del Disco Optico
    I3 = Image.imread(str('../OpticDisk/Resultados/')+name)
    
    
    # Invertimos la Maskara del Disco Optico
    OD = np.array(I3==0)
    # Eliminamos el Disco Optico de la Segmentacion
    I = I*OD;

    # Realizamos una Apertura Morfologica a la Segmentacion para reducir ruido
    I = dilation(I,disk(2))

    
    R = I2[:,:,0]
    G = I2[:,:,1]
    B = I2[:,:,2]

    label_I = label(I)

    propiedades_R = regionprops(label_I, properties={'mean_intensity','coords','centroid','area','perimeter'}, intensity_image=R, cache=True)
    propiedades_G = regionprops(label_I, properties={'mean_intensity'}, intensity_image=G, cache=True)
    propiedades_B = regionprops(label_I, properties={'mean_intensity'}, intensity_image=B, cache=True)

    f1 = [] # Media dentro de la Region. Canal Rojo
    f2 = [] # Media dentro de la Region. Canal Verde
    f3 = [] # Media dentro de la Region. Canal Azul

    f4 = [] # Desviacion Estandar dentro de la Region. Canal Rojo
    f5 = [] # Desviacion Estandar dentro de la Region. Canal Verde
    f6 = [] # Desviacion Estandar dentro de la Region. Canal Azul

    f7 = [] # Media alrededor de la Region. Canal Rojo
    f8 = [] # Media alrededor de la Region. Canal Verde
    f9 = [] # Media alrededor de la Region. Canal Azul

    f10 = [] # Desviacion Estandar alrededor de la Region. Canal Rojo
    f11 = [] # Desviacion Estandar alrededor de la Region. Canal Verde
    f12 = [] # Desviacion Estandar alrededor de la Region. Canal Azul

    f13 = [] # Valor de la Intensidad en el Centroide. Canal Rojo
    f14 = [] # Valor de la Intensidad en el Centroide. Canal Verde
    f15 = [] # Valor de la Intensidad en el Centroide. Canal Azul

    f16 = [] # Tamaño de la Region
    f17 = [] # Compacidad de la Region
    f18 = [] # Fuerza de los Borde en la Region

    f19 = [] # Homogeneidad. Canal Rojo
    f20 = [] # Homogeneidad. Canal Verde
    f21 = [] # Homogeneidad. Canal Azul

    f22 = [] # Diferencia de Color. Canal Rojo
    f23 = [] # Diferencia de Color. Canal Verde
    f24 = [] # Diferencia de Color. Canal Azul
    
    print 'Total de Regiones: ', len(propiedades_R)
    for i in range(len(propiedades_R)):
        # Separamos Individualmente cada Region    
        Region = np.zeros(I.shape,dtype=np.uint8)
        Region[np.split(np.array(propiedades_R[i].coords),2,axis=1)] = 1    
        # Aplicamos una Dilatacion a cada Region
        RegionDil_R = regionprops(dilation(Region,square(10))-Region, properties={'mean_intensity','coords'}, intensity_image=R, cache=True)
        RegionDil_G = regionprops(dilation(Region,square(10))-Region, properties={'mean_intensity'}, intensity_image=G, cache=True)
        RegionDil_B = regionprops(dilation(Region,square(10))-Region, properties={'mean_intensity'}, intensity_image=B, cache=True)    
        # Detectamos Bordes con el Operador de Prewitt
        Edge = prewitt(Region.astype(float))
        # Calculamos la Matriz de Co-oncurrencia
        MatrixCo_R = greycomatrix(Region*R,[1],[0],normed=True)
        MatrixCo_G = greycomatrix(Region*G,[1],[0],normed=True)
        MatrixCo_B = greycomatrix(Region*B,[1],[0],normed=True)

    
        # Propiedades del Canal Rojo
        f1 += [propiedades_R[i].mean_intensity];
        f4 += [np.std( R[np.split(np.array(propiedades_R[i].coords),2,axis=1)].astype(float) )]
        f7 += [RegionDil_R[0].mean_intensity] # Calculamos la Media alrededor de la Region
        f10 += [np.std( R[np.split(np.array(RegionDil_R[0].coords),2,axis=1)].astype(float) )] # Calculamos la Desviacion Estandar alrededor de la Region
        f13 += list(R[np.split(np.array(propiedades_R[i].centroid).astype(int),2)])
        f19 += [-(np.extract(MatrixCo_R[:,:,0,0]>0,MatrixCo_R[:,:,0,0])*np.log(np.extract(MatrixCo_R[:,:,0,0]>0,MatrixCo_R[:,:,0,0]))).sum()] # Entropia de Shannon
        f22 += [f1[i]/f7[i]]


        # Propiedades del Canal Verde
        f2 += [propiedades_G[i].mean_intensity];
        f5 += [np.std( G[np.split(np.array(propiedades_R[i].coords),2,axis=1)].astype(float) )]
        f8 += [RegionDil_G[0].mean_intensity] # Calculamos la Media alrededor de la Region
        f11 += [np.std( G[np.split(np.array(RegionDil_R[0].coords),2,axis=1)].astype(float) )] # Calculamos la Desviacion Estandar alrededor de la Region
        f14 += list(G[np.split(np.array(propiedades_R[i].centroid).astype(int),2)])
        f20 += [-(np.extract(MatrixCo_G[:,:,0,0]>0,MatrixCo_G[:,:,0,0])*np.log(np.extract(MatrixCo_G[:,:,0,0]>0,MatrixCo_G[:,:,0,0]))).sum()] # Entropia de Shannon
        f23 += [f2[i]/f8[i]]


        # Propiedades del Canal Azul
        f3 += [propiedades_B[i].mean_intensity];
        f6 += [np.std( B[np.split(np.array(propiedades_R[i].coords),2,axis=1)].astype(float) )]
        f9 += [RegionDil_B[0].mean_intensity] # Calculamos la Media alrededor de la Region
        f12 += [np.std( B[np.split(np.array(RegionDil_R[0].coords),2,axis=1)].astype(float) )] # Calculamos la Desviacion Estandar alrededor de la Region
        f15 += list(B[np.split(np.array(propiedades_R[i].centroid).astype(int),2)])
        f21 += [-(np.extract(MatrixCo_B[:,:,0,0]>0,MatrixCo_B[:,:,0,0])*np.log(np.extract(MatrixCo_B[:,:,0,0]>0,MatrixCo_B[:,:,0,0]))).sum()] # Entropia de Shannon
        f24 += [f3[i]/f9[i]]


        # Propiedades que no toman en cuenta el canal
        f16 += [propiedades_R[i].area]
        f17 += [(propiedades_R[i].perimeter**2)/(propiedades_R[i].area)]
        f18 += [np.extract(Edge>0,Edge).mean()]            
    
    
        del Region, RegionDil_R, RegionDil_G, RegionDil_B, Edge, MatrixCo_R, MatrixCo_G, MatrixCo_B

    return np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24],dtype=float).T



def  example(NAME):
    # Extraemos las caracteristicas
    print NAME
    features =  ExtractFeatures(NAME+'.png')
    np.save('ResultadosF/'+NAME, features)
    del features



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

        





















