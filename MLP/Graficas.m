function Graficas()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Algoritmo para usar Redes Neuronales Artificiales
%       a partir de la libreria NetLab.
%
%       Clasificacion de las 24 Caracteristicas para Detectar
%       Exudados Duros.
%       Mario Xavier Canche Uc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
    close all
%%%%%%%%%%%%%%%%%%%   Obtenemos los datos   %%%%%%%%%%%%%%%%%%%%%%%%%
    load ACCURACY.mat
    load SN.mat
    load PPV.mat
    
    figure, imagesc(ACCURACY)
    figure, imagesc(SN)
    figure, imagesc(PPV)
end
