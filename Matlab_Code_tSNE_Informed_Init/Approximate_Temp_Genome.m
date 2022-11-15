% This Matlab code is used for computing 2-D matrix representation for t-SNE using different kernels.

%%
% Read Spike Sequence Data
clear
clc
close all
% data = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/data/kmers_Frequency_Vectors_7000.csv");
% class = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/first_variant_names_spike_7000.csv");
% % % % kmers_kernel = csvread("E:/RA/IJCAI/Dataset/Original/First__dataset_Alligned_with_variants_Kernel_k3_m0_Alphabet_Size21_trial_1.csv");
% n=250;
% d=10;
% b = data;
% per=250; 

% % t-SNE with Gaussian kernel
% gaussian_time_all = strings([74,1]);
% gaussian_tsne_time_all  = strings([74,1]);
% j = 1;
% for i= 100:100:7001
%     tic
%     x = b(1:i,:);
%     D=pdist2(x,x); 
%     [P B] = d2p(D .^  2, per, 1e-5); 
%     time_gaussian = toc;
% %     % csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/one_hot/gaussian_kernel_matrix_Matlab.csv",P);
%     
% %     disp(j);
%     disp(strcat("Time consumed by Gaussian kernel: ", num2str(time_gaussian)));
%     
%     gaussian_time_all(j) = num2str(time_gaussian);
%     j = j+1;
%     
%     tic
%     ydata1 = tsne_p(P, class, 2);  
%     time_gaussian_tsne = toc;
%     disp(strcat("tsne gaussian kernel: ", num2str(time_gaussian_tsne)));
%     gaussian_tsne_time_all(j) = num2str(time_gaussian_tsne);
%     
% end
% 
% disp("final ans");
% disp(gaussian_tsne_time_all);


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

% tic
% ydata1 = tsne_p(P, class, 2);  
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/one_hot/gaussian_kernel_tsne_2_dim.csv",ydata1);

disp("Gaussian Done");

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

% isolation_time_all = strings([74,1]);
% isolation_tsne_time_all = strings([74,1]);
% j = 1;
% for k= 300:100:7001
%     psi=per;
%     tic
%     z = b(1:k,:);
%     D=pdist2(z,z); 
%     [ ~, sim ] = aNNE(D, psi, 200);
%     for i=1:size(z,1)
%         sim(i,i)=0;
%         sim(i,:)=sim(i,:)./sum(sim(i,:));
%     end
% 
%     time_isolation = toc;
% %     % csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/one_hot/gaussian_kernel_matrix_Matlab.csv",P);
%     
% %     disp(j);
%     disp(strcat("Time consumed by Isolation kernel: ", num2str(time_isolation)));
%     
%     isolation_time_all(j) = num2str(time_isolation);
%     j = j+1;
% 
%     tic
%     ydata = tsne_p(sim, class, 2);
%     time_isolation_tsne = toc;
%     isolation_tsne_time_all(j) = num2str(time_isolation_tsne);
%     
% end
% 
% disp("final ans");
% disp(isolation_tsne_time_all);


% psi=per;
% tic
% z = b(1:1000,:);
% D=pdist2(z,z); 
% [ ~, sim ] = aNNE(D, psi, 200);
% for i=1:size(b,1)
%     sim(i,i)=0;
%     sim(i,:)=sim(i,:)./sum(sim(i,:));
% end
% time_isolated = toc;
% disp(strcat("Time consumed by isolated kernel: ", num2str(time_isolated)));

% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/one_hot/isolated_kernel_matrix_Matlab.csv",sim);
% 
% 
% tic
% ydata = tsne_p(sim, class, 2);
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/one_hot/isolated_tsne_2_dim.csv",ydata);

disp("Isolated Done");

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

% data = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/data/Minimizer_Frequency_Vector_7000.csv");
% class = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/first_variant_names_spike_7000.csv");
% kmers_rbf_numexpr_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/kmer/cosine.csv");
% 
% 
% rbf_numexpr_tsne_time_all = strings([74,1]);
% j = 1;
% for i= 100:100:7001
%     z = kmers_rbf_numexpr_kernel(1:i,1:i);
%     tic
%     y_kmers_rbf_numexpr_kernel = tsne_p(z, class, 2);
% 
%     time_rbf_numexpr = toc;
%     disp(strcat("Time consumed for rbf numexpr kernel to tSNE: ", num2str(time_rbf_numexpr)));
%     rbf_numexpr_tsne_time_all(j) = num2str(time_rbf_numexpr);
%     j = j+1;
% end
% disp("RBF numexpr Done");
% disp(rbf_numexpr_tsne_time_all)
% csvwrite("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/tSne_Matrix/minimizer/rbf_numexpr_tsne_3_dim.csv",y_kmers_rbf_numexpr_kernel);

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
% disp("Starting Isolation - ICA");
% kmers_isolated_kernel = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/circle/isolated_kernel_matrix_Matlab.csv");
% initializaed_ica = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_circle_data/Isolated/ica_2d.csv");
% 
% for i= 100:100:2001
%     tic
%     y_isolated_kernel = tsne_p(kmers_isolated_kernel, 2, i, initializaed_ica);
%     
%     time_isolation = toc;
%     disp(strcat("Time consumed for isolation kernel to tSNE: ", num2str(time_isolation)));
%     
%     csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_ica/circle/Isolation/isolation_tsne_kmer_matlab_2d_"+i+"_.csv",y_isolated_kernel);
% 
% end
% 
% disp("Starting Isolation - PCA");
% initializaed_pca = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_circle_data/Isolated/pca_2d.csv");
% for i= 100:100:2001
%     tic
%     y_isolated_kernel = tsne_p(kmers_isolated_kernel, 2, i, initializaed_pca);
%     
%     time_isolation = toc;
%     disp(strcat("Time consumed for isolation kernel to tSNE: ", num2str(time_isolation)));
%     
%     csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_pca/circle/Isolation/isolation_tsne_kmer_matlab_2d_"+i+"_.csv",y_isolated_kernel);
% end
% 
% disp("Starting Isolation - Ensemble");
% initializaed_ensemble = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_circle_data/Isolated/ensemble_2d.csv");
% for i= 100:100:2001
%     tic
%     y_isolated_kernel = tsne_p(kmers_isolated_kernel, 2, i, initializaed_ensemble);
%     
%     time_isolation = toc;
%     disp(strcat("Time consumed for isolation kernel to tSNE: ", num2str(time_isolation)));
%     
%     csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_ensemble/circle/Isolation/isolation_tsne_kmer_matlab_2d_"+i+"_.csv",y_isolated_kernel);
% end
% 
% disp("Starting Isolation - Random");
% initializaed_ensemble = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_circle_data/Isolated/random_init_tsne_2_dim.csv");
% for i= 100:100:2001
%     tic
%     y_isolated_kernel = tsne_p(kmers_isolated_kernel, 2, i, initializaed_ensemble);
%     
%     time_isolation = toc;
%     disp(strcat("Time consumed for isolation kernel to tSNE: ", num2str(time_isolation)));
%     
%     csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_random/circle/Isolation/isolation_tsne_kmer_matlab_2d_"+i+"_.csv",y_isolated_kernel);
% end

% disp("Starting Laplacian");
% kmers_laplacian_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/circle/laplacian.csv");
% 
% for i= 100:100:2001
%     tic
%     y_laplacian_kernel = tsne_p(kmers_laplacian_kernel, class, 2, i, initializaed_ica);
%     
%     time_laplacian = toc;
%     disp(strcat("Time consumed for laplacian kernel to tSNE: ", num2str(time_laplacian)));
%     
%     csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_ica/circle/Laplacian/isolation_tsne_kmer_matlab_2d_"+i+"_.csv",y_laplacian_kernel);
% 
% end
% disp("Starting Gaussian");
% kmers_gaussian_kernel = csvread("C:/Users/pchourasia1/Desktop/tSNE-Evaluation/Kernel_Matrix/circle/gaussian_kernel_matrix_Matlab.csv");
% 
% for i= 100:100:2001
%     tic
%     y_gaussian_kernel = tsne_p(kmers_gaussian_kernel, class, 2, i, initializaed_ica);
%     
%     time_gausssian = toc;
%     disp(strcat("Time consumed for Gaussian kernel to tSNE: ", num2str(time_gausssian)));
%     
%     csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_ensemble/circle/Gaussian/gaussian_tsne_kmer_matlab_2d_"+i+"_.csv",y_gaussian_kernel);
% 
% end

% disp("Starting Gaussian - ICA");
% gaussian_kernel = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/shortRead/gaussian_kernel_matrix_Matlab.csv");
% initializaed_ica = csvread("C:/Users/pchourasia1/Desktop/tSNE_Informed_Init/tSNE_matrix_2d_shortRead_data/ica_2d.csv");

% for i= 100:100:2001
%     tic
%     y_gaussian_kernel = tsne_p(gaussian_kernel, 2, i, initializaed_ica);
%     
%     time_isolation = toc;
%     disp(strcat("Time consumed for Gaussian kernel to tSNE: ", num2str(time_isolation)));
%     
%     csvwrite("C:/Users/pchourasia1/Desktop/tSNE_Informed_Init/tSNE_matrix_ica/shortRead/Gaussian/gaussian_tsne_kmer_matlab_2d_"+i+"_.csv",y_gaussian_kernel);
% 
% end
% 
% disp("Starting Gaussian - PCA");
% initializaed_pca = csvread("C:/Users/pchourasia1/Desktop/tSNE_Informed_Init/tSNE_matrix_2d_shortRead_data/pca_2d.csv");
% for i= 100:100:2001
%     tic
%     y_gaussian_kernel = tsne_p(gaussian_kernel, 2, i, initializaed_pca);
%     
%     time_isolation = toc;
%     disp(strcat("Time consumed for Laplacian kernel to tSNE: ", num2str(time_isolation)));
%     
%     csvwrite("C:/Users/pchourasia1/Desktop/tSNE_Informed_Init/tSNE_matrix_pca/shortRead/Gaussian/gaussian_tsne_kmer_matlab_2d_"+i+"_.csv",y_gaussian_kernel);
% end

% disp("Starting Gaussian - Ensemble");
% initializaed_ensemble = csvread("C:/Users/pchourasia1/Desktop/tSNE_Informed_Init/tSNE_matrix_2d_shortRead_data/ensemble_2d.csv");
% for i= 100:100:2001
%     tic
%     y_gaussian_kernel = tsne_p(gaussian_kernel, 2, i, initializaed_ensemble);
%     
%     time_isolation = toc;
%     disp(strcat("Time consumed for Gaussian kernel to tSNE: ", num2str(time_isolation)));
%     
%     csvwrite("C:/Users/pchourasia1/Desktop/tSNE_Informed_Init/tSNE_matrix_ensemble/shortRead/Gaussian/gaussian_tsne_kmer_matlab_2d_"+i+"_.csv",y_gaussian_kernel);
% end

% disp("Starting Gaussian - Random");
% initializaed_ensemble = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_shortRead_data/Gaussian/random_init_tsne_2_dim.csv");
% for i= 100:100:2001
%     tic
%     y_gaussian_kernel = tsne_p(gaussian_kernel, 2, i, initializaed_ensemble);
%     
%     time_isolation = toc;
%     disp(strcat("Time consumed for Gaussian kernel to tSNE: ", num2str(time_isolation)));
%     
%     csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_random/shortRead/Gaussian/gaussian_tsne_kmer_matlab_2d_"+i+"_.csv",y_gaussian_kernel);
% end


disp("Starting Approx - ICA");
gaussian_kernel = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/kmer/First__dataset_Alligned_with_variants_Kernel_k3_m0_Alphabet_Size21_trial_1.csv");
initializaed_ica = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_spike_data/ica_2d.csv");

for i= 100:100:2001
    tic
    y_gaussian_kernel = tsne_p(gaussian_kernel, 2, i, initializaed_ica);
    
    time_isolation = toc;
    disp(strcat("Time consumed for Gaussian kernel to tSNE: ", num2str(time_isolation)));
    
    csvwrite("C:/Users/pchourasia1/Desktop/tSNE_Informed_Init/tSNE_matrix_ica/spike/Approximate/approximate_tsne_kmer_matlab_2d_"+i+"_.csv",y_gaussian_kernel);

end

disp("Starting Approx - PCA");
initializaed_pca = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_spike_data/pca_2d.csv");
for i= 100:100:2001
    tic
    y_gaussian_kernel = tsne_p(gaussian_kernel, 2, i, initializaed_pca);
    
    time_isolation = toc;
    disp(strcat("Time consumed for Gaussian kernel to tSNE: ", num2str(time_isolation)));
    
    csvwrite("C:/Users/pchourasia1/Desktop/tSNE_Informed_Init/tSNE_matrix_pca/spike/Approximate/approximate_tsne_kmer_matlab_2d_"+i+"_.csv",y_gaussian_kernel);
end

disp("Starting Approx - Ensemble");
initializaed_ensemble = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/tSNE_matrix_spike_data/ensemble_2d.csv");
for i= 100:100:2001
    tic
    y_gaussian_kernel = tsne_p(gaussian_kernel, 2, i, initializaed_ensemble);
    
    time_isolation = toc;
    disp(strcat("Time consumed for Gaussian kernel to tSNE: ", num2str(time_isolation)));
    
    csvwrite("C:/Users/pchourasia1/Desktop/tSNE_Informed_Init/tSNE_matrix_ensemble/spike/Approximate/approximate_tsne_kmer_matlab_2d_"+i+"_.csv",y_gaussian_kernel);
end