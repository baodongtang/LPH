clc;
clear;

load wikiData.mat;
fname = 'wikiData';
name = strcat(fname,'.txt');
% make the training/test data zero-mean
I_te = bsxfun(@minus, I_te, mean(I_tr, 1));     %693*128
I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));     %
T_te = bsxfun(@minus, T_te, mean(T_tr, 1));     %693*10
T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));


X_train = I_tr';
Y_train = T_tr';
Tr_label = L_tr;

X_test = I_te';
Y_test = T_te';
Te_label = L_te;

[d_I_tr,n_I_tr] = size(X_train);
[d_T_tr,n_T_tr] = size(Y_train);
[d_I_te,n_I_te] = size(X_test);
[d_T_te,n_T_te] = size(Y_test);
cc = unique(L_te);

S1 = constructW(X_train', struct('k', 5));
S2 = constructW(Y_train', struct('k', 5));
S1 = (S1+S1')/2;
S1 = max(S1,0);
S2 = (S2+S2')/2;
S2 = max(S2,0);
opt.S1 = S1;
opt.S2 = S2;

% opt.S1 = ones(n_I_tr);
% opt.S2 = ones(n_I_tr);


c = length(unique(L_te));   %ÀàÊý
bits = [32];  %[16,32,64,128]
lambda1 = 1e-5;
lambda2 = 1e-5;
lambda3 = 1e-5;

%test wiki
L = zeros(c,n_I_tr);
for i = 1:n_I_tr
    a = Tr_label(i);
    L(a,i) = 1;
end

TL = zeros(c,n_T_te);
for i = 1:n_T_te
    a = Te_label(i);
    TL(a,i) = 1;
end

% % mirflicker25
% % L = L_tr';
% % TL = L_te';


I_cateTrainTest = zeros(n_I_tr,n_I_te);

for i = 1:n_I_te
    a = Te_label(i);
    idx = find(a==Tr_label);
    I_cateTrainTest(idx,i) = 1;
end

T_cateTrainTest = zeros(n_I_tr,n_T_te);

for i = 1:n_T_te
    a = Te_label(i);
    idx = find(a==Tr_label);
    T_cateTrainTest(idx,i) = 1;
end


fin_result = cell(length(bits)*length(lambda1)*length(lambda2)*length(lambda3),1);
i_result = 1;

for bi = 1:length(bits)
    fid = fopen(name,'a+');
    fprintf(fid,'%5s\r\n',' ');
    fprintf(fid,'%5s\t','-------------------- bits =');
    fprintf(fid,'%8g\r\n',bits(bi));
    fclose(fid);
    for i = 1:length(lambda1)
        for j = 1:length(lambda2)
            for k = 1:length(lambda3)
                opt.lambda1 = lambda1(i);
                opt.lambda2 = lambda2(j);
                opt.lambda3 = lambda3(k);
                opt.bits = bits(bi);
                opt.maxItr = 5;
                [W, U, V, B, obj] = Train(X_train, Y_train, L, opt);
                [result] = evaluation(U,V,X_test,Y_test,Tr_label,Te_label,B,I_cateTrainTest);
                fid = fopen(name,'a+');
                fprintf(fid,'%5s','lambda1 =');
                fprintf(fid,'%8.5g\t',lambda1(i));
                fprintf(fid,'%5s','lambda2 =');
                fprintf(fid,'%8.5g\t',lambda2(j));
                fprintf(fid,'%5s','lambda3 =');
                fprintf(fid,'%8.5g\t',lambda3(k));
                fprintf(fid,'%5s','I_Pre =');
                fprintf(fid,'%8.5g\t',result.I_Pre);
                fprintf(fid,'%5s','I_Rec =');
                fprintf(fid,'%8.5g\t',result.I_Rec);
                fprintf(fid,'%5s','I2T_MAP =');
                fprintf(fid,'%8.5g\t',result.I2T_MAP);
                fprintf(fid,'%5s','T_Pre =');
                fprintf(fid,'%8.5g\t',result.T_Pre);
                fprintf(fid,'%5s','T_Rec =');
                fprintf(fid,'%8.5g\t',result.T_Rec);
                fprintf(fid,'%5s','T2I_MAP =');
                fprintf(fid,'%8.5g\t\n',result.T2I_MAP);
                fclose(fid);
                result.bits = bits(bi);
                result.lambda1 = lambda1(i);
                result.lambda2 = lambda2(j);
                result.lambda3 = lambda3(k);
                fin_result{i_result,1} = result;
                i_result = i_result + 1;
            end 
        end
    end
    name = strcat(fname,'.txt');
    fid = fopen(name,'a+');
    fprintf(fid,'%5s\t','-------------------- end time =');
    fprintf(fid,'%8s\r\n',datestr(now,31));
    fclose(fid);
end

save('fin_result','fin_result');
plot(1:length(obj),obj);
