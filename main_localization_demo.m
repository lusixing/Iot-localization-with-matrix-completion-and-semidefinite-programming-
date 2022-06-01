%-----------------------------------------------
% Sensor localization from partial range data based on low rank matrix
% completion and semidefinite programming  
% Code written by Sixing Lu, email:562237037@qq.com
%--------------------------------------------------
clear
clc

u = [-20, -20, 20, 20, 0;     %anchor locations
    -20, 20, 20, -20, 0];

x_min = -30;
x_max = 30;
y_min = -30;
y_max = 30;

figure(1)
clf
scatter(u(1,:),u(2,:),'ro')
hold on
axis([-40,40,-40,40])

ratio_noise = 0.1;
alpha = 0.1;  %lost ratio

d = 2; %problem dimensions 
N_a = size(u,2); %number of anchors
N_s = 20;  %number of sensor nodes

x1 = 2*x_max*rand(1,N_s)-x_max;
x2 = 2*y_max*rand(1,N_s)-y_max;
x=[x1;x2];    %randomly generated unknow iot device locations

scatter(x1,x2,'bo')
hold on

Duu = zeros(N_a,N_a);
for n1 =1:N_a
    for n2 =1:N_a
        Duu(n1,n2) = sum((u(:,n1)-u(:,n2)).^2);
    end
end

Dxx = zeros(N_s,N_s);
for n1 =1:N_s
    for n2 =1:N_s
        Dxx(n1,n2) = sum((x(:,n1)-x(:,n2)).^2);
    end
end

Dxu = zeros(N_s,N_a);
for n1 =1:N_s
    for n2 =1:N_a
        Dxu(n1,n2) = sum((x(:,n1)-u(:,n2)).^2);
    end
end

D1 = [Duu, Dxu';    %[U,sig,V]=svd(D1) rank(D1)
       Dxu, Dxx];
   
Noise = randn(N_a +N_s,N_a +N_s);
for i=1:N_a +N_s
    for j=1:N_a +N_s
        if (i<=N_a && j<=N_a) || i==j
            Noise(i,j) = 0;
        end
    end
end

Noise = ratio_noise * norm(D1,'fro')/norm(Noise,'fro')*Noise;
Do = D1;

for i1 = N_a+1:N_a+N_s
   for i2 = 1:i1
       if rand()<alpha
           Do(i1,i2) = 0;
           Do(i2,i1) = 0;
       end
   end
end
Lost_mask = Do~=D1;  %bool matrix where the entries of D are lost

Do =Do + Noise;
n2 = N_s +N_a;

cvx_begin sdp quiet
variable D_hat(n2,n2);  
variable W1(n2,n2) ; 
variable W2(n2,n2) ;

minimize(trace(W1) +trace(W2))
subject to
sym_cvx([W1, D_hat;D_hat',W2])>=0
for i=1:n2
    for j=1:n2
        if ~Lost_mask(i,j)
            D_hat(i,j) == Do(i,j);
        end
    end
end
cvx_end

nmse_D = norm(D1-D_hat,'fro')/norm(D1,'fro');
fprintf("NMSE of the recovered complete range matrix: %.2f\n",nmse_D);

%% recover location using D_hat

h = zeros(N_s+d,N_s, N_a +N_s);
Z = [x'*x,x';x,eye(2)];

for i=1:N_s
    idx1 = i;
    for j=1:N_s +N_a
        h(idx1,i,j)=1;
        idx2 = j;
        if j<=N_a   %to anchor node
            anchor_idx = j;
            h(N_s+1:N_s+d,i,j) = -u(:,anchor_idx);           
        else
            node2_idx = j-N_a;
            if node2_idx~=i
                h(node2_idx,i,j) = -1;
            else
                h(idx1,i,j) = 0;
            end
        end
        assert(h(:,i,j)'*Z*h(:,i,j) - D1(N_a+i,j)<1e-10)
    end
end


cvx_begin sdp quiet
variable X_hat(d,N_s) 
variable Y(N_s,N_s) symmetric
variable t(N_s*(N_s+N_a),1)
expression Z(N_s+d,N_s+d)

Z =[Y,X_hat'; X_hat,eye(d)];
Z = sym_cvx(Z);

minimize ( norm( t, 2 ) )
subject to
    Z>=0
    for i=1:N_s
           for j=1:N_s+N_a
               t_idx =(i-1)*(N_s+N_a) + j;
               t(t_idx) >=abs( h(:,i,j)'*Z*h(:,i,j) - D_hat(N_a+i,j));
           end
    end
cvx_end

nmse_X = norm(x-X_hat,'fro')/norm(x,'fro');
fprintf("NMSE of the recovered loacation matrix with SDP: %.2f\n",nmse_X);

scatter(X_hat(1,:),X_hat(2,:),'kx')
hold on

%% alternative localization by Triangulation
X_hat2 = zeros(d,N_s);

for s=1:N_s
    A = zeros(N_a-1, 2);
    b = zeros(N_a-1, 1);
    
    for i=1:N_a-1
        A(i,1) = 2*(u(1,N_a)-u(1,i));
        A(i,2) = 2*(u(2,N_a)-u(2,i));
        b(i) = D_hat(s+N_a,i) - D_hat(s+N_a,N_a) - u(1,i)^2 - u(2,i)^2 + u(1,N_a)^2 + u(2,N_a)^2;
    end
    X_hat2(:,s) = (A'*A)^-1*A'*b;
end

scatter(X_hat2(1,:),X_hat2(2,:),'gv')
hold on
xlabel("X axis(meters)")
ylabel("Y axis(meters)")
legend("anchor nodes","x locations ground truth","x locations obtain by MC+SDP","x locations obtain by MC+triangulation")
ttl = strcat("IoT localization with lost ratio =",num2str(alpha)," noise ratio=",num2str(ratio_noise));
title(ttl)

nmse_X_tri = norm(x-X_hat2,'fro')/norm(x,'fro');
fprintf("NMSE of the recovered loacation matrix with triangulation method: %.2f\n",nmse_X_tri);




