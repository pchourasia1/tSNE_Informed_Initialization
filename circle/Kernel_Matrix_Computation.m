% This Matlab code is used for computing kernel matrix using different kernels.

%%
% Read Data
clear
clc
close all
data = csvread("C:/Users/pchourasia1/Desktop/tSne_extension/circle/tSne_circle_data.csv");
n=250;
d=10;
b = data;
per=250; 

%% t-SNE with Gaussian kernel
gaussian_time_all = strings([74,1]);
j = 1;
for i= 100:100:7001
    tic
    x = b(1:i,:);
    D=pdist2(x,x); 
    [P B] = d2p(D .^  2, per, 1e-5); 
    time_gaussian = toc;
    csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/circle_data/gaussian_kernel_matrix_Matlab.csv",P);
    
    disp(j);
    disp(strcat("Time consumed by Gaussian kernel: ", num2str(time_gaussian)));
    
    gaussian_time_all(j) = num2str(time_gaussian);
    j = j+1;    
end

disp("final ans");
disp(gaussian_time_all);

 %% t-SNE with Isolation kernel
psi=per;
tic
D=pdist2(b,b); 
[ ~, sim ] = aNNE(D, psi, 200);
for i=1:size(b,1)
    sim(i,i)=0;
    sim(i,:)=sim(i,:)./sum(sim(i,:));
end
time_isolated = toc;
disp(strcat("Time consumed by isolated kernel: ", num2str(time_isolated)));
csvwrite("C:/Users/pchourasia1/Desktop/tSne_extension/kernel_matrix/circle_data/isolated_kernel_matrix_Matlab.csv",P);

disp("Isolated Done");

