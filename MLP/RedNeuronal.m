function [ACC,SN,SP,PPV,Fm,GM,AGM,AUC] = RedNeuronal()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Algoritmo para usar Redes Neuronales Artificiales
%       a partir de la libreria NetLab.
%
%       Clasificacion de las 24 Caracteristicas para Detectar
%       Exudados Duros.
%       Mario Xavier Canche Uc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Leemos la Matriz de Caracteristicas desde Python
    system('python FeaturesExport.py')
 
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
    
    % Paralelizamos el proceso
    %matlabpool open
    
    x = 434;
    for j = x
        %for i = 1:100
        for i = 87
    %%%%%%%%%%%%%%%%%%%   Fase de Entrenamiento  %%%%%%%%%%%%%%%%%%%%%%%%

		% Fix the seeds
        	%rand('state', 434);
	        %randn('state', 434);
	        rand('state', j);
        	randn('state', j);

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

	        [net] = netopt(net, options, traindata, traintarget, 'conjgrad');

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %%%%%%%%%%%%%%%%%%%%   Fase de Validacion  %%%%%%%%%%%%%%%%%%%%%%%%%%

	        % Compute network classification
	        yt = mlpfwd(net, testdata);
            
        	% Convert single output to posteriors for both classes
	        testpost = [yt 1-yt];
        	[C,trate]=confmat(testpost,[testtarget==1 testtarget~=1]);
	    
        	TP = C(1,1);
	        FP = C(1,2);
        	TN = C(2,2);
	        FN = C(2,1);
        	P = TP + FN;
	        N = FP + TN;
            
            
            ACC = (TP+TN)/(P+N);
            SN = TP/P;
            SP = TN/N;
            PPV = TP/(TP+FP);
            
            Fm = (2*PPV*SN)/(PPV+SN);
            GM = sqrt((TP/P)*(TN/N));
            if SN > 0
                AGM = (GM+SP*N)/(1+N);
            else
                AGM = 0;
            end
            [x1,y1,th,AUC] = perfcurve(testtarget,yt,1);
            
        	disp(['Network Confusion Matrix (' num2str(trate(1)) '%)'])
        	disp(['      Accuracy:  ' num2str(ACC)])
	        disp(['   Sensitivity:  ' num2str(SN)])
            disp(['Especificidad :  ' num2str(SP)])
        	disp(['          PPV :  ' num2str(PPV)])
            
            disp(['           Fm :  ' num2str(Fm)])
            disp(['           GM :  ' num2str(GM)])
            disp(['          AGM :  ' num2str(AGM)])
            disp(['          AUC :  ' num2str(AUC)])
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end
    
%    matlabpool close
end
