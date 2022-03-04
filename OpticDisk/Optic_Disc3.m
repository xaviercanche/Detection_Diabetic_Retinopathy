function Optic_Disc3(name)

%%%%%%%% Deteccion de Disco Optico en Imagenes de Retina   %%%%%%%%%%%
%
%	Algoritmo para la deteccion del Disco Optico
%	en Imagenes de Retina.
%
%	Input:
%		name: 	Nombre de la Imagen de Retina
%
%	Output:
%		    I:	Imagen de Retina con el Disco Optico Seleccionado
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	clear All
    close all
    
    % Leemos la Imagen desde el archivo .png
    link = ['../../UVA_ALL/Resize/resultMat_originales/',name,'.mat'];
    load(link);
    I = ScaleRetina;
    %I = im2double(imread(link));
    
	% Extraemos el Canal Verde de la Imagen
	Green = I(:,:,2);
    Blue = Green;
%	figure, imshow(Blue)

    % Eliminamos el Ruido con un filtro de Mediana de 20x20
	Blue = medfilt2(Blue,[20 20]);
%	figure, imshow(Blue)

	% Leemos la Maskara
    Retina = Blue > 0.03;
    % Eliminamos los pixels negros sobre zona blanca
    SE = strel('disk',5);
    mask = imclose(Retina,SE);


%%%%%%%%%%%%%%%%%%%%%%%  Pre-Procesamiento %%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% Mejoramos el contraste y la luminosidad
	Blue = enhancement(Blue,mask);
%	figure, imshow(Blue)

	% Aplicamos Operaciones Morfologicas
	struct = strel('disk',35); % Elemento estructurante disco r=23
	BlueM = imclose(Blue,struct); % Cerradura Morfologica
	BlueM = BlueM.*mask;
%	figure, imshow(BlueM)

%%%%%%%%%%%%%%%%%%%%%%%  Segmentacion %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for THA = 0.97:-0.01:0.95
	% Calculamos un Umbral
	level = stretchlim(BlueM,[0 THA]);

	% Segmentamos de acuerdo al Umbral
	BW = BlueM>level(2);
	BW = imfill(BW,'holes');
%    figure, imshow(BW)

    % Aplicamos Operaciones Morfologicas
    struct = strel('disk',40); % Elemento estructurante disco r=40
    BW = imerode(BW,struct); % Apertura Morfologica
    struct = strel('disk',45); % Elemento estructurante disco r=45
    BW = imdilate(BW,struct); % Apertura Morfologica
    %BW = imopen(BW,struct); % Apertura Morfologica
    %figure, imshow(BW)

	% Extraemos las Caracteristicas
	[L,N] = bwlabel(BW); % Enlistamos las Regiones
    %figure, imshow(BW)

    dim = size(Blue);
	features = regionprops(L,Green,'Area','Perimeter','MajorAxisLength','MinorAxisLength','BoundingBox','MeanIntensity'); % Propiedades a las Regiones
    for i = 1:N
		% Calculamos las Caracteristicas
		A = features(i).Area;   % Area
		R = features(i).MajorAxisLength/features(i).MinorAxisLength;
		C = 4*pi*A/(features(i).Perimeter^2);    % Compactness
        ML = features(i).MeanIntensity;
        MG = sum(Green(:))/sum(mask(:));

		%fprintf('\n%10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  \n', i, A, R, C, THA, ML, MG)

		% Eliminamos las regiones que no cumplen como Disco Optico
		%if A<10700 || C < 0.6
        if A<15255 || C < 0.7 || R > 3 || ML < MG
			lim = round(features(i).BoundingBox);
            if lim(1) < 0, a = 1; else a = lim(1); end
            if lim(2) < 0, b = 1; else b = lim(2); end
            if lim(2)+lim(4) > dim(1), c = dim(1); else c = lim(2)+lim(4); end
            if lim(1)+lim(3) > dim(2), d = dim(2); else d = lim(1)+lim(3); end
            BW(b:c,a:d) = 0;
        end
    end
    %return
    % Aplicamos Operaciones Morfologicas
	struct = strel('disk',5); % Elemento estructurante disco r=23
	BW = imopen(BW,struct); % Apertura Morfologica
%	figure, imshow(BW)
%    title('Primera Opcion')
    
    [L,N] = bwlabel(BW); % Enlistamos las Regiones
    features = regionprops(L,Green,'Area','Perimeter','MajorAxisLength','MinorAxisLength','MeanIntensity'); % Propiedades a las Regiones
    for i = 1:N
        % Calculamos las Caracteristicas
		A = features(i).Area;   % Area
		R = features(i).MajorAxisLength/features(i).MinorAxisLength;
		C = 4*pi*A/(features(i).Perimeter^2);    % Compactness

		fprintf('\n%10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f\n', i, A, R, C, THA, features(i).MeanIntensity, sum(Green(:))/sum(mask(:)))
    end
    
    % Probar con otro Umbral
    if N ~= 1
        continue
    else
        break
    end
end
    
%%%%%%%%%%%%%%%%%%%%%%%  Post-Procesamiento %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
	% Extendemos el Area de la Region
	delta = 20;
	Box = regionprops(BW,'BoundingBox'); % Calculamos el BoudingBox
    %figure, imshow(BW)
    
    % Verificamos que exista Bouding Box
    if size(Box,1) ~=1
%        fig = figure;
%        imshow(I)
        % Guardamos las Imagenes
%        print(fig,'-dpng',['NEW5/',name,'.png']);
        %clf(fig)

	BW = zeros(size(Green));
	% Guardamos la maskara del Disco Optico
	NameR = ['Resultados/',name,'.png'];
	imwrite(BW,NameR);
        return
    end

	lim = round(Box.BoundingBox);
    if lim(1)-delta < 0, a = 1; else a = lim(1)-delta; end
    if lim(2)-delta < 0, b = 1; else b = lim(2)-delta; end
    if lim(2)+lim(4)+delta > dim(1), c = dim(1); else c = lim(2)+lim(4)+delta; end
    if lim(1)+lim(3)+delta > dim(2), d = dim(2); else d = lim(1)+lim(3)+delta; end
	BW(b:c,a:d) = 1;

    
	% SubImagen
	%SubImage = Blue(lim(2)-delta:lim(2)+lim(4)+delta,lim(1)-delta:lim(1)+lim(3)+delta);
    SubImage = Blue(b:c,a:d);
	%figure, imshow(SubImage)
NameR = ['Resultados/',name,'_3.png'];
imwrite(I(lim(2):lim(2)+lim(4),lim(1):lim(1)+lim(3),:),NameR);
NameR = ['Resultados/',name,'_4.png'];
imwrite(I(b:c,a:d,:),NameR);
    
	level = stretchlim(SubImage,[0 0.8]);
	BW = BlueM.*BW > level(2);
	%figure, imshow(BW)
NameR = ['Resultados/',name,'_5.png'];
imwrite(BW,NameR);

	% Rellenamos los hoyos
	BW = imfill(BW,'holes');
NameR = ['Resultados/',name,'_6.png'];
imwrite(BW,NameR);
    [L,N] = bwlabel(BW); % Enlistamos las Regiones
    if N > 1
        struct = strel('disk',15); % Elemento estructurante disco 20
        BW = imerode(BW,struct);
        struct = strel('disk',17); % Elemento estructurante disco 20
        BW = imdilate(BW,struct);
        
        [L,N] = bwlabel(BW); % Enlistamos las Regiones
        if N > 1
%            fig = figure;
%            imshow(I)
            % Guardamos las Imagenes
%            print(fig,'-dpng',['NEW5/',name,'.png']);
            %clf(fig)

	    BW = zeros(size(Green));
	    % Guardamos la maskara del Disco Optico
	    NameR = ['Resultados/',name,'.png'];
	    imwrite(BW,NameR);
            return
        end
    end
    
%    figure, imshow(BW)
    
	% Dibujamos el circulo
    %fig = figure;
	%imshow(I)
	%[L,N] = bwlabel(BW); % Enlistamos las Regiones
	Box = regionprops(BW,'BoundingBox','Centroid'); % Calculamos el BoudingBox
	%rectangle('Position',Box.BoundingBox,'Curvature',[1,1],'EdgeColor','b','LineWidth',2)

	% Devolvemos la Maskara del Disco Optico
	BW = zeros(size(BW));
	BW( round(Box.Centroid(2)), round(Box.Centroid(1)) ) = 1;

	d = round(Box.BoundingBox);
	lim = max(d(4),d(3));
	struct = strel('disk',round(lim/2)); % Elemento estructurante disco 20
	BW = imdilate(BW,struct);
	%figure, imshow(BW)

	% Guardamos las Imagenes
	%print(fig,'-dpng',['NEW5/',name,'.png']);
	%clf(fig)

    % Guardamos la maskara del Disco Optico
    NameR = ['Resultados/',name,'.png'];
    imwrite(BW,NameR);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
