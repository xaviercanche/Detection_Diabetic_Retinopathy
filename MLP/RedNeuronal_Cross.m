function [CV] = RedNeuronal_Cross()
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
    load featuresTrain.mat
    load featuresTest.mat
    load EtiquetasTrain.mat
    load EtiquetasTest.mat

    % Asignamos los datos de Entrenamiento
    traindata = X_train;
    traintarget = double(y_train');

    % Asignamos los datos de Validacion
    testdata = X_test;
    testtarget = double(y_test');

    [n1,n2] = size(traindata);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    X = [traindata;testdata];
    y = [traintarget;testtarget];

    % Aplicamos la particion para el Cross-Validation
    indices = crossvalind('Kfold',y,10);



    % Aplicamos la Red Neuronal 10 veces
    Media = [];
    for i = 1:10
	test = (indices == i);
	train =~ test; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%%%%%%%%%%%%%%%%%%   Fase de Entrenamiento  %%%%%%%%%%%%%%%%%%%%%%%%

	% Fix the seeds
        rand('state', 434);
	randn('state', 434);

	% Parametros de la Red Neuronal
        nhidden = 87;
	nout = 1;
        v = 1;	% Weight decay
	ncycles = 100;	% Number of training cycles. 

	% Creamos la Red Neuronal MLP
        net = mlp(n2, nhidden, nout, 'logistic', v);

	% Entrenamos usando gradiente conjugado.
        options = zeros(1,18);
	options(1) = 0;                 % Print out error values
	options(14) = ncycles;

	[net] = netopt(net, options, X(train,:), y(train,:), 'conjgrad');

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%%%%%%%%%%%%%%%%%%%   Fase de Validacion  %%%%%%%%%%%%%%%%%%%%%%%%%%

        % Compute network classification
        yt = mlpfwd(net, X(test,:));
            
       	% Convert single output to posteriors for both classes
        testpost = [yt 1-yt];
       	[C,trate]=confmat(testpost,[y(test,:)==1 y(test,:)~=1]);
	    
	% Calculamos el Area Bajo la Curva ROC
        [x1,y1,th,AUC] = perfcurve(y(test,:),yt,1);

	% Imprimimos los resultados
        disp(['          AUC :  ' num2str(AUC)])

	Media = [Media,AUC];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
	CV = mean(Media);
end
