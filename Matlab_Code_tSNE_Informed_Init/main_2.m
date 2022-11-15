% This Matlab code is used for demonstration of using t-SNE with Isolation kernel.
% The results are similar to Table 1 in the JAIR paper:
% Zhu, Y. and Ting, K.M., 2021, July. Improving the Effectiveness and
% Efficiency of Stochastic Neighbour Embedding with Isolation Kernel. 
% Journal of Artificial Intelligence Research.

clear
clc
close all

%%
% Read Spike Sequence Data

data = csvread("E:/RA/IJCAI/Dataset/Original/first_kmers_freq_vec_seq_data_7000.csv");

class = csvread("E:/RA/IJCAI/Dataset/Original/first_variant_names_spike_7000.csv");


%%
kmers_kernel = csvread("E:/RA/IJCAI/Dataset/Original/First__dataset_Alligned_with_variants_Kernel_k3_m0_Alphabet_Size21_trial_1.csv");

%% create 5 subspace clusters

% n=250;
% d=10;
% b=zeros(5*n,5*d);
% class=b(:,1);
% for i=1:3
%    b(n*(i-1)+1:n*i,(i-1)*d+1:d*i)=(randn(n,d)*i^4);    
%    class(n*(i-1)+1:n*i)=i;
% end
% for i=4:5
%    b(n*(i-1)+1:n*i,(i-1)*d+1:d*i)=(randn(n,d)*i^4)+100*i;    
%    class(n*(i-1)+1:n*i)=i;
% end
% b(1,:)=b(1,:)-b(1,:); % add the origin point 

%% t-SNE with Gaussian kernel

n=250;
d=10;

b = data;
per=250; 
tic
D=pdist2(b,b); 
[P B] = d2p(D .^ 2, per, 1e-5); 
time_isolated = toc;
disp(strcat("Time consumed by Gaussian kernel: ", num2str(time_isolated)));

tic
ydata1 = tsne_p(P, class, 2);  
ydata11=normalize(ydata1);
time_gaussian = toc;
disp(strcat("Time consumed by t-SNE with gaussian kernel: ", num2str(time_gaussian)));

%%
ydata1_knn = tsne_p(P, class, 50);  
Idx = knnsearch(ydata1_knn,ydata1_knn,'K',10);
disp("done");

%%
csvwrite("E:/RA/IJCAI/Dataset/Original/gaussian_kernel_mat_orig_before_tsne.csv",P);

%%
figure
gscatter(ydata11(:,1),ydata11(:,2),class)
% hold on
% scatter(ydata1(1,1),ydata1(1,2),400,'LineWidth',1.2)
% legend('1','2','3','4','5','O')
legend('off')
% title(['Gaussian kernel with perplexity=' num2str(per)])
set(gcf,'color','w');
set(gca,'linewidth',1,'fontsize',20,'fontname','Times');
% xlim([-2.5 2.5]);
% ylim([-2.5 2.5]);

 %% t-SNE with Isolation kernel
 
psi=per;
tic
D=pdist2(b,b); 
[ ~, sim ] = aNNE (D, psi, 200);
for i=1:size(b,1)
    sim(i,i)=0;
    sim(i,:)=sim(i,:)./sum(sim(i,:));
end
time_isolated = toc;
disp(strcat("Time consumed by isolated kernel: ", num2str(time_isolated)));

% csvwrite("E:/RA/IJCAI/Dataset/Original/isolated_kernel_mat_orig_before_tsne.csv",sim);


tic
ydata = tsne_p(sim, class, 50);
csvwrite("E:/RA/IJCAI/Dataset/Original/isolated_tsne_50_dim.csv",ydata);

%%
ydata=normalize(ydata); 
time_isolated = toc;
disp(strcat("Time consumed by aNNE t-SNE: ", num2str(time_isolated)));

%%
figure
gscatter(ydata(:,1),ydata(:,2),class)
% hold on
% scatter(ydata(1,1),ydata(1,2),400,'black','x','LineWidth',1.2)
% legend('1','2','3','4','5','O')
legend('off')
% title(['Isolation kernel with \psi=' num2str(psi)])
set(gcf,'color','w');
set(gca,'linewidth',1,'fontsize',20,'fontname','Times'); 
xlim([-2.5 2.5]);
ylim([-2.5 1.5]);

 %% t-SNE with approximate kernel
tic
ydata_approx = tsne_p(kmers_kernel, class, 2);
% csvwrite("E:/RA/IJCAI/Dataset/Original/approximate_kernel_tsne_50_dim.csv",ydata_approx);

% ydata_approx=normalize(ydata_approx); 
time_isolated = toc;
disp(strcat("Time consumed by approximate t-SNE: ", num2str(time_isolated)));


figure
gscatter(ydata_approx(:,1),ydata_approx(:,2),class)
% hold on
% scatter(ydata(1,1),ydata(1,2),400,'black','x','LineWidth',1.2)
% legend('1','2','3','4','5','O')
legend('off')
% title(['Isolation kernel with \psi=' num2str(psi)])
set(gcf,'color','w');
set(gca,'linewidth',1,'fontsize',20,'fontname','Times'); 
% xlim([-2.5 2.5]);
% ylim([-2.5 1.5]);

