function boosted(Name)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Algoritmo para realizar Segmentacion Suave y Segmentacion Gruesa
%   Mario Xavier Canche Uc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Ruta de los datos
    path_O = '../../UVA_ALL/Resize/resultMat_originales/';
    path_BSS_I = 'resultImage_BSS/';
    path_BSS_M = 'resultMat_BSS/';
    path_Seg_I = 'resultImage_Seg/';
    path_Seg_M = 'resultMat_Seg/';
    
    % Leemos la Imagen RGB desde el archivo .mat
    link = [path_O, Name, '.mat'];
    load(link)
    RGB = ScaleRetina;

    % Obtenemos la Maskara del area de la Retina
    Mask = RGB(:,:,2) > 0.0196;
    
 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                   Espacios de Color                       %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Obtenemos el Canal Verde
    Ig = RGB(:,:,2).*Mask;
    dimX = size(RGB);

	Ig = enhancement(Ig,Mask);

   % Obtenemos la Maskara de las Marcas de Exudados
    label = im2bw( Ig , 0.6 );
	%figure, imshow(label)

	    Ig(Ig>1) = 1;
	    Ig(Ig<0) = 0;
%	figure, imshow(Ig)


    % Convertimos del Espacio de Color RGB a LUV
    C1 = makecform('srgb2xyz');
    C2 = makecform('xyz2uvl');
    XYZ = applycform(RGB,C1);
    LUV = applycform(XYZ,C2);
    % Obtenemos el Canal de Luminicencias
    Il = LUV(:,:,3).*Mask;
    MAX = max(max(Il));
    MIN = min(min(Il));
    Il = Il.*Mask;

	Il = enhancement(Il,Mask);
	    Il(Il>1) = 1;
	    Il(Il<0) = 0;
%	figure, imshow(Il)


    % Convertimos del Espacio de Color RGB a CMYK
    C = makecform('srgb2cmyk');
    CMYK = applycform(RGB,C);
    % Obtenemos el Canal Yellow
    Iy = CMYK(:,:,2).*Mask;
    MAX = max(max(Iy));
    MIN = min(min(Iy));
    Iy = 1 - Iy;
    Iy = Iy.*Mask;

	Iy = enhancement(Iy,Mask);
	    Iy(Iy>1) = 1;
	    Iy(Iy<0) = 0;
%	figure, imshow(Iy)




    % Guardamos los Resultados
    NameR = [path_BSS_I,Name,'_RGB.png'];
    imwrite(Ig,NameR);

    % Guardamos los Resultados
    NameR = [path_BSS_I,Name,'_LUV.png'];
    imwrite(Il,NameR);

    % Guardamos los Resultados
    NameR = [path_BSS_I,Name,'_CMYK.png'];
    imwrite(Iy,NameR);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                 Boosted Soft Segmentation                  %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % INPUT: Datos de Entrada
    X(:,:,1) = Ig;
    X(:,:,2) = Il;
    X(:,:,3) = Iy;
    
    X2 = reshape(X,[dimX(1)*dimX(2) dimX(3)]);
    dimX2 = size(X2,1);
    y = double(reshape(label,[dimX2 1]));
    y(y==0) = -1;
    
    
    % PASO 1:  Actualizamos los Parametros para el Weak Learn 
    % Dividimos los datos en dos clases 1 y -1
    C1 = X2(y>0,:);
    C2 = X2(y<0,:);
    % Calculamos los parametros de cada clase
    MeanC1 = ones(dimX2,1) *mean(C1,1);
    VarC1 =  ones(dimX2,1)*var( C1 ,0);
    MeanC2 = ones(dimX2,1) *mean(C2,1);
    VarC2 =  ones(dimX2,1)*var( C2 ,0);
    
    
    % PASO 2:  Inicializamos las Variables
    D = ones(dimX(1)*dimX(2),1)/(dimX(1)*dimX(2));   % Pixels Weights
    W = zeros(dimX(3),1);
    U = zeros(dimX(3),1);
    Error = zeros(dimX(3),1);
    
    % Paso 3:  Inicializamos las iteraciones
    for t = 1:3
        
        % PASO 4:  Aplicamos el Clasificador debil
        hk = sign( exp((X2-MeanC1).^2./(2*VarC1))-exp((X2-MeanC2).^2./(2*VarC2)) );
        % Calculamos el Error minimo de cada Weak Learn
        err1 = [sum( D.*(y~=hk(:,1)) ),sum( D.*(y~=hk(:,2)) ),sum( D.*(y~=hk(:,3)) )];
        err2 = 1 - err1;
        Error(err1<err2) = err1(err1<err2);
        Error(err1>err2) = err2(err1>err2);        
        

        % PASO 5:  Calculamos el parametro de polaridad U
        U(err1<err2) = 1;
        U(err1>err2) = -1;
        % Calculamos el Error minimo y el indice del Weak learn
        [err,i] = min(Error);

        
        % PASO 6:  Calculamos el valor de confianza
        alpha = 0.5*log((1-err)/err);
        
        
        % PASO 7:  Actualizmos la Distribucion de Pesos
        D = D.*exp( -alpha*y.*hk(:,i).*U(i));
        D = D/sum(D);
        
        % PASO 8:  Agregamos el Weak Learn optimo
       	W(i) = W(i) + alpha*U(i);


	    %fprintf('%f   %f   %f\n', W(1),W(2),W(3))
    end
    
    %W'
    % OUTPUT:  Proyectamos los datos
    Y = (W'*X2')';
    BSS = reshape(Y,[dimX(1) dimX(2)]);
    MAX = max(Y);   MIN = min(Y);
    %[MAX MIN]
    BSS = (BSS - MIN)./(MAX - MIN);
    BSS = BSS.*Mask;


    % Guardamos los Resultados
    NameR = [path_BSS_I,Name,'.png'];
    imwrite(BSS,NameR);
	%figure, imshow(BSS)
	%title('BSS')
    
    NameR = [path_BSS_M,Name,'.mat'];
    save( NameR,'BSS');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                  Segmentacion Gruesa                       %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Aplicamos un Multi-Scale Median Filter Bank
%    BSS_70 = medfilt2(real(BSS),[90 90]);
    BSS_70 = medfilt2(real(BSS),[120 120]);
%	kernel = fspecial('gaussian',[1000 1000],500)
%	BSS_70 = imfilter(BSS,kernel);
	


	%figure, imshow(BSS_70)
	%title('BSS_70')
    
    % Restamos el mapa de confidencialidad de el background
    ValueMAX = BSS_70-BSS;


	%figure, imshow(ValueMAX)
	%title('ValueMAX')
    
    % Umbralizacion Otsu
%	level = graythresh(ValueMAX);
%	U = ValueMAX > level;
	U = ValueMAX > 0.1;
%	U = ValueMAX > 0.2;
        U = U.*Mask;
	%figure, imshow(U.*Mask)
	%title('Otsu') 

	SE = strel('diamond',1);
	U = imopen(U,SE);
	%figure, imshow(U.*Mask)
	%title('Otsu 2') 


    % Guardamos los Resultados
    Segmentation = U;
    NameR = [path_Seg_I,Name,'.png'];
    imwrite(Segmentation,NameR);
    
    NameR = [path_Seg_M,Name,'.mat'];
    save( NameR,'Segmentation');
    
end
