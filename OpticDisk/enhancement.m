function U = enhancement(IGreen,Mask)

    % Obtenemos el Canal Verdes
    dimension = size(IGreen);

    % Estimamos los puntos del Background
    N = ones(dimension);
    h1 = ones(125); % Vecindad de 125x125
    denominador = imfilter( N , h1 ).*Mask;
    
    % Media
    numerador = imfilter( IGreen , h1 ).*Mask;
    media = numerador./denominador;
    media(isnan(media)) = 0;
    
    % Desviacion Estandar
    numerador = imfilter( IGreen.*IGreen , h1 ).*Mask;
    media2 = numerador./denominador;
    media2(isnan(media2)) = 0;
    DesvStd = sqrt( media2 - (media.*media) );
    
    % Calculamos la Distancia de Mahalanobis
    d = abs( (IGreen.*Mask - media)./(DesvStd + 0.01) );
    d(isnan(d)) = 0;
    % Umbralizamos para obtener el background
    Background = (d < 0.7).*Mask;

    % Ahora Estimamos L(x,y) y C(x,y)
    h1 = ones(50); % Vecindad de 50x50
    denominador = imfilter( N.*Background , h1 ).*Mask;
    
    % Media
    numerador = imfilter( IGreen.*Background , h1 ).*Mask;
    media = numerador./denominador;
    media(isnan(media)) = 0;

    % Creamos la Imagen Mejorada
    L = media;
    U = 0.5*(IGreen.*Mask )./(L + 0.01);
    U(isnan(U)) = 0;
    U(U<0) = 0;

end
