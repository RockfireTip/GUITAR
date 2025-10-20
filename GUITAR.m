function [labels,Z,converge_Z,converge_Z_G,E,Sigma,Y,A,X]= GUITAR(lambda1,lambda2,lambda3,delta,anc,X,N_v,Wm,cls_num,X_mis_ind,X_com_ind,dimen)
    
m = length(X);
N = size(X{1},2);
t=anc;%anchors
nC=cls_num;
s=nC;
K_Lv=4;
K_Ls=1;
rank_fun = 1; % 1:HPl_\delta
%% ======================= Dimension reduction=================
for v=1:m 
    valid_indices = X_com_ind{v};
    X_valid = X{v}(:, valid_indices); 
    %% pca
    [~, score, ~] = pca(X_valid');

    X_reduced{v} = score(:,1:min(min(size(X_valid',1)-1,size(X_valid',2)),dimen(v)))';

    X_restored = zeros(size(X_reduced{v}, 1), N);
    X_restored(:, valid_indices) = X_reduced{v};
    X{v} = X_restored;
end 
%% ======================= Initialization ====================
for v =1:m
    L{v}=eye(N,N);
    Z{v} = ones(t,N);
    E{v} = zeros(size(X{v},1),N_v(v));
    A{v} = zeros(size(X{v},1),t);
    Q{v} = zeros(t,N);
    G{v} = zeros(t,N);
    Y{v} = zeros(size(X{v},1),N);
    Sigma{v} = repmat(eye(size(X{v},1)), [1, 1, 1]);
end

Ls=eye(N,N);

Isconverg = 0;
iter = 0;
max_iter=200;
epson = 1e-7;
mu = 0.0001; max_mu = 10e12; eta_mu = 1.2;
rho = 0.0001; max_rho = 10e12; eta_rho =1.2;
         
sX = [t, N, m];
converge_Z=[];
converge_Z_G=[];

%% ================================ Upadate ===============================
while(Isconverg == 0) 
%% ============================== Upadate A^k =============================
    for v = 1:m
        M{v}=(Y{v}+mu*X{v}-mu*E{v})*Z{v}';
        [UA,~,VA] = svd(M{v},'econ');
        A{v} = UA*VA';
    end


%% ============================== Upadate E^k =============================
    for v = 1:m
       D_v = size(X{v}, 1);
        for n = 1:N_v(v)
            first_term = lambda1*inv(Sigma{v}) + mu * eye(D_v);  
            second_term = Y{v}(:, n) - mu * (A{v} * Z{v}(:, n)  - X{v}(:, n));
            E{v}(:, n) = first_term\ second_term;
        end
    end

%% ============================== Upadate \Sigma^v_k ========================
for v = 1:m
   D_v = size(X{v}, 1);
   Sig_temp = zeros(D_v, D_v); 
   for n = 1:N_v(v)
       Sig_temp=Sig_temp+E{v}(:,n)*E{v}(:,n)';
   end 
   Sigma{v}=(Sig_temp+epson*eye(size(X{v},1)))/N_v(v);
end
%% ====================== Update X ========================================
for v=1:m
    X{v}(:,X_mis_ind{v})=0;
    X{v}=X{v}+A{v}*Z{v}*Wm{v};
end

%% ============================== Upadate Y^k =========================
for v=1:m
   Y{v} = Y{v} + mu*(X{v}-A{v}*Z{v}-E{v});
end

%% ============================= Upadate G^k ==============================
Z_tensor = cat(3, Z{:,:});
Q_tensor = cat(3, Q{:,:});
 %nonconvex function
G_tensor = solve_G_fun(Z_tensor + 1/rho*Q_tensor,rho,sX,delta,rank_fun);
for v =1:m
    G{v} = G_tensor(:,:,v);
end
%% ============================== Upadate Q ===============================
Q_tensor=Q_tensor+rho*(Z_tensor-G_tensor);
for v=1:m
    Q{v} = Q_tensor(:,:,v);
end

%% ============================== Upadate Z^k =============================
for v = 1:m
    sum_Lv=zeros(N,N);
    for vi=1:m
        if vi ~= v
            sum_Lv=sum_Lv+L{vi};
        end
    end
    temp=rho*G{v}-Q{v}+A{v}'*Y{v}+mu*A{v}'*X{v}-mu*A{v}'*E{v};
    pre_Z{v}=Z{v};
    Z{v} = temp/((rho+mu)*eye(N,N)+2*lambda2*sum_Lv+2*lambda3*Ls);
end
%% ===============================L^v======================================
for v = 1:m
    S = zeros(N, N);
    norm_Zv = sqrt(sum(Z{v}.^2, 1)); 
    Zv_normalized = Z{v} ./ norm_Zv; 
    cos_sim = Zv_normalized' * Zv_normalized;
    
    for i = 1:N
        sim_i = cos_sim(i, :);
        sim_i(i) = -inf;
        [~, idx] = maxk(sim_i, K_Lv);
        S(i, idx) = cos_sim(i, idx);
    end
    
    S = max(S, S');
    
    Sv{v}=S;

    D_inv_sqrt = diag(1./sqrt(sum(S, 2)));
    L{v} = eye(N) - D_inv_sqrt * S * D_inv_sqrt;
end

%% ===============================Ls======================================
S_shared = zeros(N, N);
for v = 1:m
    norm_Zv = sqrt(sum(Z{v}.^2, 1));
    Zv_normalized = Z{v} ./ norm_Zv;
    cos_sim = Zv_normalized' * Zv_normalized;

    S_v = zeros(N, N);
    for i = 1:N
        sim_i = cos_sim(i, :);
        sim_i(i) = -inf;
        [~, idx] = maxk(sim_i, K_Ls);
        S_v(i, idx) = cos_sim(i, idx);
    end
    S_v = max(S_v, S_v');
    S_shared = S_shared + S_v;
end

S_shared = S_shared / m;

D_shared_inv_sqrt = diag(1./sqrt(sum(S_shared, 2)));
Ls = eye(N) - D_shared_inv_sqrt * S_shared * D_shared_inv_sqrt;

%% ============================= Upadate penalty parameters ===============
mu = min(mu*eta_mu, max_mu);
rho = min(rho*eta_rho, max_rho);
%% ====================== Checking Coverge Condition ======================
max_Z=0;
max_Z_G=0;
history = struct();
Isconverg = 1;
for v=1:m
    first_norm=norm(X{v}-A{v}*Z{v}-E{v},inf);
    if (first_norm>epson)
        history.norm_Z = first_norm;
        Isconverg = 0;
        max_Z=max(max_Z,history.norm_Z );
    end

    sec_norm=norm(Z{v}-G{v},inf);
    if (sec_norm>epson)
        history.norm_Z_G = sec_norm;
        Isconverg = 0;
        max_Z_G=max(max_Z_G, history.norm_Z_G);
    end
end
converge_Z=[converge_Z max_Z];
converge_Z_G=[converge_Z_G max_Z_G];
fprintf('iter = %d, max_Z = %f, max_Z_G = %f\n', iter, max_Z, max_Z_G);


if (iter>max_iter)
    Isconverg  = 1;
end

iter=iter+1;
end
Sbar=[];
for i = 1:m
    Sbar=cat(1,Sbar,Z{i});
    
end

[F,~,~] = mySVD(Sbar',nC); 

for i = 1:N
    F(i,:) = F(i,:) ./ norm(F(i,:)+eps);%
end
labels=litekmeans(F, nC, 'MaxIter', 100,'Replicates',10);

end