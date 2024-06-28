
function [avg_MAPE, avg_RMSE, avg_MPE, avg_MSE, avg_CorrCoef, avg_accuracy] = nnetbist(input, target, training_rate, n1)
    veri = xlsread("Iris.xlsx");
    input = veri(:, 1:4);
    target = veri(:, 6);
    noofdata = size(input, 1);
    training_rate = 0.70;
    n1 = 20;
    ntd = round(noofdata * training_rate);
    random_indices = randperm(noofdata);
    xt = input(random_indices(1:ntd), :);
    xv = input(random_indices(ntd+1:end), :);
    yt = target(random_indices(1:ntd));
    yv = target(random_indices(ntd+1:end));
    xt = xt';
    xv = xv';
    yt = yt';
    yv = yv';
    k = 5;
    kf = cvpartition(noofdata, 'KFold', k);

    ort_test=0;
    ort_egitim=0;

    for i = 1:k
        trIdx = kf.training(i);
        teIdx = kf.test(i);
        xt = input(trIdx, :)';
        xv = input(teIdx, :)';
        yt = target(trIdx)';
        yv = target(teIdx)';
        xtn = mapminmax(xt);
        xvn = mapminmax(xv);
        [ytn, ps] = mapminmax(yt);
        net = newff(xtn, ytn, n1, {'tansig', 'purelin'}, 'trainlm');
        net.trainParam.epoch = 1500;
        net.trainParam.goal = 0.01;
        net.trainParam.min_grad = 1e-9; 
        net.trainParam.max_fail = 6;
        net.trainParam.show = NaN;
        net.PerformFcn = 'mse';
        net = train(net, xtn, ytn);


        yen = sim(net, xvn);
        ye = mapminmax('reverse', yen, ps);
        ye = ye';
        yv = yv';
        fark = ye - yv;
        farkkare = (ye - yv).^2;
        farktop = sum(farkkare);
        MSE_test = mean(farktop);
        MAPE_test = mean((abs(ye - yv)) ./ yv);
        RMSE_test = sqrt(MSE_test);
        MPE_test = mean((ye - yv) ./ yv);
        CorrCoef_test = corrcoef(ye, yv);
        CorrCoef_test = CorrCoef_test(1, 2);
        predicted_labels = round(ye); 
        true_labels = round(yv); 
        accuracy_test = sum(predicted_labels == true_labels) / numel(true_labels);


        yen_train = sim(net, xtn);
        ye_train = mapminmax('reverse', yen_train, ps);
        ye_train = ye_train';
        yt = yt';
        fark_train = ye_train - yt;
        farkkare_train = (ye_train - yt).^2;
        farktop_train = sum(farkkare_train);
        MSE_train = mean(farktop_train);
        MAPE_train = mean((abs(ye_train - yt)) ./ yt);
        RMSE_train = sqrt(MSE_train);
        MPE_train = mean((ye_train - yt) ./ yt);
        CorrCoef_train = corrcoef(ye_train, yt);
        CorrCoef_train = CorrCoef_train(1, 2);
        predicted_labels = round(ye_train); 
        true_labels = round(yt); 
        accuracy_train= sum(predicted_labels == true_labels) / numel(true_labels);



        disp(['Orijinal Veriler İle Eğitim #' num2str(i) ' Test Seti Performans Değeri:']);
        disp(['MAPE: ' num2str(MAPE_test)]);
        disp(['RMSE: ' num2str(RMSE_test)]);
        disp(['MPE: ' num2str(MPE_test)]);
        disp(['MSE: ' num2str(MSE_test)]);
        disp(['Korelasyon Katsayısı: ' num2str(CorrCoef_test)]);
        disp(['Test Doğruluk Yüzdesi: ' num2str(accuracy_test) '%']);
        disp(' ');

        disp(['Orijinal Veriler İle Eğitim #' num2str(i) ' Eğitim Seti Performans Değeri']);
        disp(['MAPE: ' num2str(MAPE_train)]);
        disp(['RMSE: ' num2str(RMSE_train)]);
        disp(['MPE: ' num2str(MPE_train)]);
        disp(['MSE: ' num2str(MSE_train)]);
        disp(['Korelasyon Katsayısı: ' num2str(CorrCoef_train)]);
        disp(['Eğitim Doğruluk Yüzdesi: ' num2str(accuracy_train) '%']);

        ort_test=ort_test+ accuracy_test;
        ort_egitim=ort_egitim + accuracy_train;

    end

    ort_test=ort_test/k;
    ort_egitim=ort_egitim/k;
    disp(["Orijinal verilerin ortalama test doğruluk değeri: " num2str(ort_test)]);
    disp(["Orijinal verilerin ortalama eğitim doğruluk değeri: " num2str(ort_egitim)]);

    main(veri);


end



function main(veri)
    sifreli_veri_seti = simetrik_sifreleme(veri);

    disp("Orjinal Veri Seti:");
    disp(veri);

    disp("Şifreli Veri Seti:");
    disp(sifreli_veri_seti);
    egitim(sifreli_veri_seti);

end



function [sifreli_veri, ascii_degerleri] = simetrik_sifreleme(veri)
    [m, n] = size(veri);
    sifreli_veri = zeros(m, n);
    ascii_degerleri = nan(m, n);
    sifrelenmis=nan(m,n);
    karakterler = strings(m, n); 

    characters = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '.', '/', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z', '[', '\', ']', '^', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'];

    values = 0:0.1:6.9;

    for i = 1:m
        for j = 1:n
            if ~isnan(veri(i, j)) && (j <= 4 || j == 6)
            
                val = veri(i, j);

                [~, idx] = min(abs(values - val));
                char = characters(idx);

                ascii_val = double(char);
                ascii_degerleri(i, j) = ascii_val;
                karakterler(i, j) = char; 
                
                sifreli_veri(i, j) = (ascii_val * 5 + 5);
                sifrelenmis(i,j)=sifreli_veri(i,j);
                sifreli_veri(i,j)=sifreli_veri(i,j)/100;
           
            else
              
                sifreli_veri(i, j) = NaN;
                ascii_degerleri(i, j) = NaN;
                karakterler(i, j) = "";
            end
        end
    end



    disp('Karşılık Gelen Karakterler:');
    disp(karakterler);
    disp(' ');
    disp('ASCII Değerleri:');
    disp(ascii_degerleri);
    disp('Şifreli değerler:');
    disp(sifrelenmis);
    
end


function egitim = egitim(sifreli_veri)
    veri = sifreli_veri;
    input = veri(:, 1:4);
    target = veri(:, 6);
    noofdata = size(input, 1);
    training_rate = 0.70;
    n1 = 20;
    ntd = round(noofdata * training_rate);
    random_indices = randperm(noofdata);
    xt = input(random_indices(1:ntd), :);
    xv = input(random_indices(ntd+1:end), :);
    yt = target(random_indices(1:ntd));
    yv = target(random_indices(ntd+1:end));
    xt = xt';
    xv = xv';
    yt = yt';
    yv = yv';
    k = 5;
    kf = cvpartition(noofdata, 'KFold', k);

    ort_test=0;
    ort_egitim=0;

    for i = 1:k
        trIdx = kf.training(i);
        teIdx = kf.test(i);
        xt = input(trIdx, :)';
        xv = input(teIdx, :)';
        yt = target(trIdx)';
        yv = target(teIdx)';
        xtn = mapminmax(xt);
        xvn = mapminmax(xv);
        [ytn, ps] = mapminmax(yt);
        net = newff(xtn, ytn, n1, {'tansig', 'purelin'}, 'trainlm');
        net.trainParam.epoch = 1500;
        net.trainParam.goal = 0.01;
        net.trainParam.min_grad = 1e-9; 
        net.trainParam.max_fail = 6;
        net.trainParam.show = NaN;
        net.PerformFcn = 'mse';
        net = train(net, xtn, ytn);


        yen = sim(net, xvn);
        ye = mapminmax('reverse', yen, ps);
        ye = ye';
        yv = yv';
        fark = ye - yv;
        farkkare = (ye - yv).^2;
        farktop = sum(farkkare);
        MSE_test = mean(farktop);
        MAPE_test = mean((abs(ye - yv)) ./ yv);
        RMSE_test = sqrt(MSE_test);
        MPE_test = mean((ye - yv) ./ yv);
        CorrCoef_test = corrcoef(ye, yv);
        CorrCoef_test = CorrCoef_test(1, 2);
        predicted_labels = round(ye); 
        true_labels = round(yv); 
        accuracy_test = sum(predicted_labels == true_labels) / numel(true_labels);


        yen_train = sim(net, xtn);
        ye_train = mapminmax('reverse', yen_train, ps);
        ye_train = ye_train';
        yt = yt';
        fark_train = ye_train - yt;
        farkkare_train = (ye_train - yt).^2;
        farktop_train = sum(farkkare_train);
        MSE_train = mean(farktop_train);
        MAPE_train = mean((abs(ye_train - yt)) ./ yt);
        RMSE_train = sqrt(MSE_train);
        MPE_train = mean((ye_train - yt) ./ yt);
        CorrCoef_train = corrcoef(ye_train, yt);
        CorrCoef_train = CorrCoef_train(1, 2);
        predicted_labels = round(ye_train); 
        true_labels = round(yt); 
        accuracy_train= sum(predicted_labels == true_labels) / numel(true_labels);



        disp(['Şifrelenmiş Veriler İle Eğitim #' num2str(i) ' Test Seti Performans Değeri:']);
        disp(['MAPE: ' num2str(MAPE_test)]);
        disp(['RMSE: ' num2str(RMSE_test)]);
        disp(['MPE: ' num2str(MPE_test)]);
        disp(['MSE: ' num2str(MSE_test)]);
        disp(['Korelasyon Katsayısı: ' num2str(CorrCoef_test)]);
        disp(['Test Doğruluk Yüzdesi: ' num2str(accuracy_test) '%']);
        disp(' ');

        disp(['Şifrelenmiş Veriler İle Eğitim #' num2str(i) ' Eğitim Seti Performans Değeri']);
        disp(['MAPE: ' num2str(MAPE_train)]);
        disp(['RMSE: ' num2str(RMSE_train)]);
        disp(['MPE: ' num2str(MPE_train)]);
        disp(['MSE: ' num2str(MSE_train)]);
        disp(['Korelasyon Katsayısı: ' num2str(CorrCoef_train)]);
        disp(['Eğitim Doğruluk Yüzdesi: ' num2str(accuracy_train) '%']);
        disp(' ')
        ort_test=ort_test+ accuracy_test;
        ort_egitim=ort_egitim + accuracy_train;

    end

    ort_test=ort_test/k;
    ort_egitim=ort_egitim/k;
    disp(["Şifrelenmiş verilerin ortalama test doğruluk değeri: " num2str(ort_test)]);
    disp(["Şifrelenmiş verilerin ortalama eğitim doğruluk değeri: " num2str(ort_egitim)]);


end

