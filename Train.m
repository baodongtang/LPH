function [W, U, V, B, obj] = Train(X, Y, L, opt)

lambda1 = opt.lambda1;
lambda2 = opt.lambda2;
lambda3 = opt.lambda3;
S1 = opt.S1;
S2 = opt.S2;

maxItr = opt.maxItr;
l = opt.bits;
[d1,n1] = size(X);
[d2,n2] = size(Y);
I1 = eye(l);
I2 = eye(d1);
I3 = eye(d2);
B = rand(l,n1);
i = 0;

D1 = diag(sum(S1,1)+eps);
D2 = diag(sum(S2,1)+eps);
tol = 0.000001;

while i < maxItr
    i = i + 1;
    
    % update w
    W = pinv(B*B'+lambda3*I1)*B*L';
 
 
    % update U
    UA = X*D1*X' + (lambda3/lambda1+tol)*I2;
    U = pinv(UA)*(X*S1'*B');
    
    %update V
    VA = Y*D2*Y' + (lambda2/lambda1+tol)*I3;
    V = pinv(VA)*(Y*S2'*B');
    
    %update B
    BA = W*W';
    BB = lambda1*D1+lambda2*D2;
    BC = W*L+lambda1*U'*X*S1'+lambda2*V'*Y*S2';
    B = lyap(BA,BB,-BC); 
    
    obj1 = norm((L-W'*B),'fro');
    obj2 = sum(diag(abs(B*D1*B' - 2*B*S1*X'*U))) + sum(diag(X*D1*X'));
    obj3 = sum(diag(abs(B*D2*B' - 2*B*S2*Y'*V))) + sum(diag(Y*D2*Y'));
    obj4 = norm(W,'fro') + norm(U,'fro') + norm(V,'fro');

    obj(i) = obj1 + lambda1*obj2 + lambda2*obj3 + lambda3*obj4;
    
    disp(i);
end


