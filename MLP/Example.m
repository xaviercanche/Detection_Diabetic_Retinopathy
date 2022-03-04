% Media de los Resultados obtenidos con las Redes Neuronales

Media = zeros(1,8);

for i = 1:100
    [ACC,SN,SP,PPV,Fm,GM,AGM,AUC] = RedNeuronal();
    Media = Media + [ACC,SN,SP,PPV,Fm,GM,AGM,AUC];
end

disp('Media:')
Media./100

disp('Media:')
[CV] = RedNeuronal_Cross();
disp(['           CV :  ' num2str(CV)])
