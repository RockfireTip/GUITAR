clear 
clc

cur_folder = pwd;
addpath([cur_folder, '\funs']);
addpath([cur_folder, '\dataset']);
Dataname = ["COIL20MV"];
name = 'COIL20MV';
percentDel = [0.5];
%% =====================Parameter setting===============================
result=[];
max_val=0;
ii=0;
mean_time=0;
times=5;

%% ===================== Optimization ==================================
for it_name = 1:length(Dataname)
    for j = 1:length(percentDel)
        %% =====================Load Datatset========================
        dataPath = sprintf('dataset\\%s.mat', name);
        load(dataPath);
        for v=1:3
            X{v} = X{v}';
        end
        gt = double(gt);
        truthF = gt;

        foldFile = sprintf('dataset\\%s_folds_%.1f.mat', name, percentDel(j));
        load(foldFile);  
        ind_folds=folds;

        cls_num = length(unique(truthF));
        anchor=[cls_num];
        for iv = 1:length(X)
            %% =====================Constructing a matrix of missing data========================
            X1 = X{iv}';
            X1 = NormalizeFea(X1,0);
            ind_0 = find(ind_folds(:,iv) == 0);
            ind_1 = find(ind_folds(:,iv) == 1);
            X1(:,ind_0) = 0;
            X_mis{iv} = X1;
            X_mis_ind{iv} = ind_0 ;
            X_com_ind{iv}=ind_1;
            %% ================dimen=============================================================
            dimen(iv)=min(20,round(size(X1,1)/4));
            %% ===================== Constructing indicator matrix Wm ===============================
            linshi_W = eye(size(X1,2));
            linshi_W(ind_1,ind_1) = 0;
            Wm{iv} = linshi_W;%shape:N x N
            N_v(iv) =size(X1,2); %length(ind_0);%N_v
            clear linshi_W
        end
        %% ===================== Hyperparameters============================
        lambda1 = 10^(-3);
        lambda2 = 10^(-1);
        lambda3=1;
        delta=10^(-2);
        anc = anchor(1);
        for ii = 1:times
            tic;
            [y,Z,~,~,E,Sigma,Y,A,X] = GUITAR(lambda1,lambda2,lambda3,delta,anc,X_mis,N_v,Wm,cls_num,X_mis_ind,X_com_ind,dimen);
            time = toc;
            mean_time=mean_time+time;
            result(ii,:) = ClusteringMeasure(truthF, y);
            %% =====================Saving the best results====================
            if result(ii,1) > max_val
                max_val = result(ii,1);
                record = [lambda1,lambda2,lambda3,delta,anc,time];
                record_result = result(ii,:);
                record_c ={Z};
                record_time = time;
            end
        end
        mean_time=mean_time/times;
        disp('The result of '+Dataname(it_name)+', missing rate:'+num2str(percentDel(j)))
        max_val;
        mean_acc=sum(result(:,1))/size(result,1);
        mean_nmi=sum(result(:,2))/size(result,1);
        mean_pur=sum(result(:,3))/size(result,1);
        s_acc = std(result(:,1));
        s_nmi =std(result(:,2));
        s_purity = std(result(:,3));
        fprintf('mean_acc = %.4f, mean_nmi = %.4f, mean_pur = %.4f, std_acc = %.4f, std_nmi = %.4f, std_pur = %.4f\n', ...
    mean_acc, mean_nmi, mean_pur, s_acc, s_nmi, s_purity);
        save(strcat('.\result\GUITAR_',num2str(percentDel(j)),'_',Dataname(it_name),'.mat'),'result','record','max_val','record_result','time','record_c','mean_acc','mean_nmi','mean_pur','s_acc','s_nmi','s_purity','mean_time')
    end
end