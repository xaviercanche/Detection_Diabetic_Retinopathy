%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Ejemplo de Preprocesado de las Imagenes de Retina
%       Mario Xavier Canche Uc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close All

% Leemos el archivo con la Informacion de los datos
Date = textread('../exudatesUVA_ALL.csv','%s');
n = length(Date); % Longitud de los datos

% Paralelizamos el proceso
matlabpool open

% Leemos las Imagenes
parfor i = 1:n
%for i = 1:n
    % Nombre de la Imagen a Procesar
    Name = char(Date(i)); % Convertimos a cadena la linea
    fprintf('%6s \n',Name);
    
    % Aplicamos la Segmentacion Suave y Gruesa
    boosted(Name);
end
matlabpool close

