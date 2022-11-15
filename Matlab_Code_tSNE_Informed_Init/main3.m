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
clear
clc
close all
data = csvread("E:/RA/IJCAI/Dataset/Original/first_kmers_freq_vec_seq_data_7000.csv");
class = csvread("E:/RA/IJCAI/Dataset/Original/first_variant_names_spike_7000.csv");
kmers_kernel = csvread("E:/RA/IJCAI/Dataset/Original/First__dataset_Alligned_with_variants_Kernel_k3_m0_Alphabet_Size21_trial_1.csv");
n=250;
d=10;
b = data;
per=250; 

%% t-SNE with Gaussian kernel


tic
D=pdist2(b,b); 
[P B] = d2p(D .^ 2, per, 1e-5); 
csvwrite("E:/RA/IJCAI/Dataset/Original/gaussian_kernel_mat_orig_before_tsne.csv",P);

time_isolated = toc;
disp(strcat("Time consumed by Gaussian kernel: ", num2str(time_isolated)));

tic
ydata1 = tsne_p(P, class, 2);  
csvwrite("E:/RA/IJCAI/Dataset/Original/gaussian_kernel_tsne_2_dim.csv",ydata1);

disp("Gaussian Done");

 %% t-SNE with Isolation kernel
clear
clc
close all
data = csvread("E:/RA/IJCAI/Dataset/Original/first_kmers_freq_vec_seq_data_7000.csv");
class = csvread("E:/RA/IJCAI/Dataset/Original/first_variant_names_spike_7000.csv");
kmers_kernel = csvread("E:/RA/IJCAI/Dataset/Original/First__dataset_Alligned_with_variants_Kernel_k3_m0_Alphabet_Size21_trial_1.csv");
n=250;
d=10;
b = data;
per=250; 

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

csvwrite("E:/RA/IJCAI/Dataset/Original/isolated_kernel_mat_orig_before_tsne.csv",sim);


tic
ydata = tsne_p(sim, class, 2);
csvwrite("E:/RA/IJCAI/Dataset/Original/isolated_tsne_2_dim.csv",ydata);

disp("Isolated Done");

 %% t-SNE with approximate kernel
% clear
% clc
% close all
% data = csvread("E:/RA/IJCAI/Dataset/Original/first_kmers_freq_vec_seq_data_7000.csv");
% class = csvread("E:/RA/IJCAI/Dataset/Original/first_variant_names_spike_7000.csv");
% kmers_kernel = csvread("E:/RA/IJCAI/Dataset/Original/First__dataset_Alligned_with_variants_Kernel_k3_m0_Alphabet_Size21_trial_1.csv");
% n=250;
% d=10;
% b = data;
% per=250; 
% 
% tic
% ydata_approx = tsne_p(kmers_kernel, class, 2);
% csvwrite("E:/RA/IJCAI/Dataset/Original/approximate_kernel_tsne_2_dim.csv",ydata_approx);
% 
% disp("Approximate Done");