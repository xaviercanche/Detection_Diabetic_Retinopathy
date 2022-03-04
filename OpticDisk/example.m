%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Ejemplo de Detección del Disco Optico
%       Mario Xavier Canche Uc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close All
warning('off','all')

% Leemos el archivo con la Informacion de los datos
Date = textread('../exudatesUVA_ALL.csv','%s');
n = length(Date); % Longitud de los datos

% Paralelizamos el proceso
%matlabpool open

% Leemos las Primeras 9 Imagenes
for i = 1:n
%parfor i = 1:n
    % Nombre de la Imagen a Procesar
    Name = char(Date(i)); % Convertimos a cadena la linea
    fprintf('%6s \n',Name);
    
    %try
    Optic_Disc3(Name);
    %catch
    %    continue
    %end
    break
end
%matlabpool close

