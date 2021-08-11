clear;

hbar = 1;
m = 1;

X = 10;      % 长度 
N = 200;    % 格点数
dx = 2*X/N; % 空间步长
En = 9;     % 能级个数

x = linspace(-X,X,N);

% 势能矩阵
v = 1/2*x.^2;
##v = x;
##I = find(v>-5 & v<5);
##v(I) = 0;
##I = find(v~=0);
##v(I) = 20;
V = spdiags(v',0,N,N); % NxN

% Hermitian矩阵
D = ones(N,1);
H = spdiags([1*D -2*D 1*D],-1:1,N,N);
H = (-(hbar^2)/(2*m))*(1/dx^2)*H;

A = H + V;

% 求特征值和特征向量
[Vec, Val] = eigs(A,En,0);

% 画图
for i = 1:En
    psi = Vec(:,i);
    E = Vec(i,i);
    subplot(3,3,i);
    [hAx,hL1,hL2] = plotyy(x,psi.^2/sum(psi.^2*dx),x,v);
    xlabel("x");
    ylabel(hAx(1),"|\Psi(x)|^2","fontsize",10);
    ylabel(hAx(2),"V(x)","fontsize",10);
    title_info = ["n=",mat2str(i)," E=" mat2str(E,3)];
    title(title_info);
    axis on;
    grid on;
end
