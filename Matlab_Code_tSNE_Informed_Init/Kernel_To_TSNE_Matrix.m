% This Matlab code is used for computing 2-D matrix representation for t-SNE using different kernels.

%%
% Read Spike Sequence Data
clear
clc
close all
% data = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/data/kmers_Frequency_Vectors_7000.csv");
% data = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/host_data/Spike2Vec_on_unaligned_for_Host_Classification_Data_5558_seq.csv");
% class = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/host_data/attributes.csv");
% class = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/genome_data/all_attributes_8220.csv");
% data = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/circle/tSne_circle_data.csv");
% % % kmers_kernel = csvread("E:/RA/IJCAI/Dataset/Original/First__dataset_Alligned_with_variants_Kernel_k3_m0_Alphabet_Size21_trial_1.csv");
% data = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/shortRead_data/kmers_frequency_vector_0_10181.csv");
% 
% n=250;
% d=10;
% b = data;
% per=250; 
% 
% % t-SNE with Gaussian kernel
% disp("Starting Gaussian Kernel");
% gaussian_time_all = strings([74,1]);
% j = 1;
% for i= 0:100:10200
%     tic
%     if i == 10200
%         i = 10181;
%     end
%     x = b(1:i,:);
%     D=pdist2(x,x); 
%     [P B] = d2p(D .^  2, per, 1e-5); 
%     time_gaussian = toc;
% %     csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/genome/gaussian_kernel_matrix_Matlab.csv",P);
%     csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/shortRead/gaussian_kernel_matrix_Matlab.csv",P);
% %     csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/one_hot/gaussian_kernel_matrix_Matlab.csv",P);
%     
%     disp(j);
%     disp(strcat("Time consumed by Gaussian kernel: ", num2str(time_gaussian)));
%     
%     gaussian_time_all(j) = num2str(time_gaussian);
%     j = j+1;
%     
%     
% end
% 
% disp("final ans");
% disp(gaussian_time_all);


% n=1:7000
% sequence =  n
% 
% disp(sequence(1:700));

% tic
% D=pdist2(b,b); 
% [P B] = d2p(D .^ 2, per, 1e-5); 
% 
% 
% time_isolated = toc;
% % csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/one_hot/gaussian_kernel_matrix_Matlab.csv",P);
% 
% disp(strcat("Time consumed by Gaussian kernel: ", num2str(time_isolated)));
% gaussian_kernel = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/shortRead/gaussian_kernel_matrix_Matlab.csv");
% disp("Gaussian Started");
% gaussian_kernel = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/shortRead/gaussian_kernel_matrix_Matlab.csv");
% 
% tic
% ydata1 = tsne_p(gaussian_kernel, 2);  
% csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_shortRead_data/Gaussian/random_init_tsne_2_dim.csv",ydata1);
% 
% disp("Gaussian Done");

% kmers_gaussian_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/kmer/gaussian_kernel_matrix_Matlab.csv");
% 
% tic
% y_gaussian_kernel = tsne_p(kmers_gaussian_kernel, class, 3);
% 
% time_gaussian = toc;
% disp(strcat("Time consumed for gaussian kernel to tSNE: ", num2str(time_gaussian)));
% 
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/kmer/gaussian_kernel_tsne_3_dim.csv",y_gaussian_kernel);

 %% t-SNE with Isolation kernel
% 
% psi=per;
% tic
% D=pdist2(b,b); 
% [ ~, sim ] = aNNE(D, psi, 200);
% for i=1:size(b,1)
%     sim(i,i)=0;
%     sim(i,:)=sim(i,:)./sum(sim(i,:));
% end
% time_isolated = toc;
% disp(strcat("Time consumed by isolated kernel: ", num2str(time_isolated)));
% 
% % csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/one_hot/isolated_kernel_matrix_Matlab.csv",sim);
% csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/shortRead/isolated_kernel_matrix_Matlab.csv",sim);
disp("Approximate Started");
isolated_kernel = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/shortRead/approximate_kernel_matrix.csv");

tic
ydata_isolated = tsne_p(isolated_kernel, 2);
csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_shortRead_data/Approximate/random_init_tsne_2_dim.csv",ydata_isolated);

disp("Isolated Done");


% disp("Isolated Started");
% isolated_kernel = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/shortRead/isolated_kernel_matrix_Matlab.csv");
% 
% tic
% ydata_isolated = tsne_p(isolated_kernel, 2);
% csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_shortRead_data/Isolated/random_init_tsne_2_dim.csv",ydata_isolated);
% 
% disp("Isolated Done");

 %% t-SNE with approximate kernel
% clear
% clc
% close all
% data = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/kmers_Frequency_Vectors_7000.csv");
% class = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/first_variant_names_spike_7000.csv");
% % kmers_kernel = csvread("E:/RA/IJCAI/Dataset/Original/First__dataset_Alligned_with_variants_Kernel_k3_m0_Alphabet_Size21_trial_1.csv");
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

%% t-SNE with RBF kernel

% data = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/kmers_Frequency_Vectors_7000.csv");
% class = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/first_variant_names_spike_7000.csv");
% kmers_rbf_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/minimizer/rbf.csv");
% 
% tic
% y_kmers_rbf_kernel = tsne_p(kmers_rbf_kernel, class, 2);
% 
% time_rbf = toc;
% disp(strcat("Time consumed for rbf kernel to tSNE: ", num2str(time_rbf)));
% 
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/minimizer/rbf_tsne_2_dim.csv",y_kmers_rbf_kernel);
% 
% disp("RBF Done");

%% t-SNE with RBF kernel

% data = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/kmers_Frequency_Vectors_7000.csv");
% class = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/first_variant_names_spike_7000.csv");
% kmers_rbf_numexpr_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/minimizer/rbf_numexpr.csv");
% 
% tic
% y_kmers_rbf_numexpr_kernel = tsne_p(kmers_rbf_numexpr_kernel, class, 2);
% 
% time_rbf_numexpr = toc;
% disp(strcat("Time consumed for rbf numexpr kernel to tSNE: ", num2str(time_rbf_numexpr)));
% 
% 
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/minimizer/rbf_numexpr_tsne_2_dim.csv",y_kmers_rbf_numexpr_kernel);
% 
% disp("RBF numexpr Done");
% 
% %% t-SNE with additive_chi2 kernel
% 
% % data = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/kmers_Frequency_Vectors_7000.csv");
% % class = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/first_variant_names_spike_7000.csv");
% % 
% % n=250;
% % d=10;
% % b = data;
% % per=250; 
% % 
% % 
% kmers_additive_chi2_kernel = csvread("C:/Users/pchourasia1/Desk top/tSNE-Evaluation/Kernel_Matrix/minimizer/additive_chi2.csv");
% 
% tic
% y_additive_chi2_kernel = tsne_p(kmers_additive_chi2_kernel, class, 3);
% 
% time_additive_chi2 = toc;
% disp(strcat("Time consumed for cosine kernel to tSNE: ", num2str(time_additive_chi2)));
% 
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/minimizer/additive_chi2_tsne_3_dim.csv",y_additive_chi2_kernel);
% 
% 
% disp("additive_chi2 Done");

%% t-SNE with cosine kernel

% kmers_cosine_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/one_hot/cosine.csv");
% 
% tic
% y_cosine_kernel = tsne_p(kmers_cosine_kernel, class, 3);
% 
% time_cosine = toc;
% disp(strcat("Time consumed for cosine kernel to tSNE: ", num2str(time_cosine)));
% 
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/one_hot/cosine_tsne_3_dim.csv",y_cosine_kernel);
% 
% 
% disp("cosine Done");

% %% t-SNE with gaussian kernel
% 
% kmers_gaussian_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/kmer/gaussian.csv");
% 
% tic
% y_gaussian_kernel = tsne_p(kmers_gaussian_kernel, class, 3);
% 
% time_gaussian = toc;
% disp(strcat("Time consumed for gaussian kernel to tSNE: ", num2str(time_gaussian)));
% 
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/kmer/gaussian_tsne_3_dim.csv",y_gaussian_kernel);


% disp("gaussian Done");
% 
% %% t-SNE with laplacian kernel
% 
% kmers_laplacian_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/minimizer/laplacian.csv");
% kmers_laplacian_kernel = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/host/laplacian.csv");
% 
% disp("Laplacian Started Done");
% tic
% y_laplacian = tsne_p(kmers_laplacian_kernel, 2);
% 
% time_laplacian = toc;
% disp(strcat("Time consumed for laplacian kernel to tSNE: ", num2str(time_laplacian)));
% 
% csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_host_data/Laplacian/random_init_tsne_2_dim.csv",y_laplacian);
% 
% 
% disp("laplacian Done");

% %% t-SNE with linear kernel
% 
% % data = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/kmers_Frequency_Vectors_7000.csv");
% % class = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/first_variant_names_spike_7000.csv");
% % 
% % n=250;
% % d=10;
% % b = data;
% % per=250; 
% 
% kmers_linear_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/minimizer/linear.csv");
% 
% tic
% y_linear_kernel = tsne_p(kmers_linear_kernel, class, 2);
% 
% time_linear = toc;
% disp(strcat("Time consumed for linear kernel to tSNE: ", num2str(time_linear)));
% 
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/minimizer/linear_tsne_2_dim.csv",y_linear_kernel);
% 
% disp("linear Done");
% 
% %% t-SNE with polynomial kernel
% 
% kmers_polynomial_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/minimizer/polynomial.csv");
% 
% tic
% y_polynomial_kernel = tsne_p(kmers_polynomial_kernel, class, 2);
% 
% time_poly = toc;
% disp(strcat("Time consumed for polynomial kernel to tSNE: ", num2str(time_poly)));
% 
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/minimizer/polynomial_tsne_2_dim.csv",y_polynomial_kernel);
% 
% disp("polynomial Done");
%  
% %% t-SNE with sigmoid kernel
%  
% kmers_sigmoid_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/minimizer/sig.csv");
% 
% tic
% y_sigmoid_kernel = tsne_p(kmers_sigmoid_kernel, class, 2);
% 
% time_sig = toc;
% disp(strcat("Time consumed for sigmoid kernel to tSNE: ", num2str(time_sig)));
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/minimizer/sigmoid_tsne_2_dim.csv",y_sigmoid_kernel);
% disp("sigmoid Done");
% 
% %% t-SNE with chi squared kernel
% 
% % data = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/kmers_Frequency_Vectors_7000.csv");
% % class = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/first_variant_names_spike_7000.csv");
% % 
% % n=250;
% % d=10;
% % b = data;
% % per=250; 
% % 
% kmers_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/kmer/chi_squared.csv");
% 
% tic
% ydata = tsne_p(kmers_kernel, class, 3);
% 
% time_linear = toc;
% disp(strcat("Time consumed for chi squared kernel to tSNE: ", num2str(time_linear)));
% 
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/kmer/chi_squared_tsne_3_dim.csv",ydata);
% 
% disp("chi squared Done");