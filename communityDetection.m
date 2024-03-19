
clear
clc 

corr_g1 = permute(cell2mat(struct2cell(load('/Users/ismaila/Documents/C-Codes/SCI_GraphTheory/sci_data/SCI/fc/corr_hc.mat'))), [2 3 1]);
corr_g2 = permute(cell2mat(struct2cell(load('/Users/ismaila/Documents/C-Codes/SCI_GraphTheory/sci_data/SCI/fc/corr_sci.mat'))), [2 3 1]);

tic
n = 1000;

count=1;

gamma = 1.21; % 1.16 1.18 1.2 1.21
% log_omega = -1; % 
omega = 0.1; % 0.0, 0.1, 0.5, 1.0

m_g1 = zeros(1, n);
m_g2 = zeros(1, n);
S_g1_temp = zeros(200, 76, 1000);
S_g2_temp = zeros(200, 61, 1000);

parfor i = 1:n
    % omega = power(10, temp_y);
    A_g1 = squeeze(num2cell(corr_g1, [1 2]));
    A_g2 = squeeze(num2cell(corr_g2, [1 2]));
    N = length(A_g1{1});
    T_g1 = size(corr_g1, 3);
    T_g2 = size(corr_g2, 3);
    
    [B_g1, twom_g1] = multicat(A_g1, gamma, omega);
    PP_g1 = @(S_g1)postprocess_categorical_multilayer(S_g1, T_g1);
    [S_g1, Q_g1] = iterated_genlouvain(B_g1, 10000, 0, 1, 'moverandw', [], PP_g1);
    Q_g1 = Q_g1 / twom_g1;
    
    S_g1 = reshape(S_g1, N, T_g1);
    C_g1 = mode(S_g1, 2);
    K_g1 = max(S_g1, [], 'all');
    
    [B_g2, twom_g2] = multicat(A_g2, gamma, omega);
    PP_g2 = @(S_g2)postprocess_categorical_multilayer(S_g2, T_g2);
    [S_g2, Q_g2] = iterated_genlouvain(B_g2, 10000, 0, 1, 'moverandw', [], PP_g2);
    Q_g2 = Q_g2 / twom_g2;
    
    S_g2 = reshape(S_g2, N, T_g2);
    C_g2 = mode(S_g2, 2);
    K_g2 = max(S_g2, [], 'all');
    
    m_g1(i) = Q_g1;
    m_g2(i) = Q_g2; 

    S_g1_temp(:, :, i) = S_g1;
    S_g2_temp(:, :, i) = S_g2;

end 
 
fprintf('Parfor OK!\n');

% % Example 3D array (replace this with your actual data)
% % array = rand(200, 76, 50);
% 
% % Reshape the array to size 200x50x76 for G1
% reshaped_array1 = reshape(S_g1_temp, [200, 50, 76]);
% reshaped_array2 = reshape(S_g2_temp, [200, 50, 61]);
% 
% % Initialize arrays to store max and min values
% max_values_g1 = zeros(1, 76);
% min_values_g1 = zeros(1, 76);
% 
% max_values_g2 = zeros(1, 61);
% min_values_g2 = zeros(1, 61);
% 
% max_min_index_G1 = zeros(3, 76);
% max_min_index_G2 = zeros(3, 61);
% 
% % Iterate over the second dimension
% for i = 1:size(reshaped_array1, 3)
%     % Get the slice along the first and second dimensions
%     slice1 = reshaped_array1(:,:,i);
%     % slice2 = reshaped_array2(:,:,i);
% 
%     % Reshape the slice to size 200x50
%     slice1 = reshape(slice1, [200, 50]);
%     % slice2 = reshape(slice2, [200, 50]);
% 
%     % Find the maximum value along the first dimension 
%     max_values_g1(:, i) = max(slice1(:));
%     % max_values_g2(:, i) = max(slice2(:)); 
% 
%     % Find the minimum value along the first dimension
%     min_values_g1(:, i) = min(slice1(:));
%     % min_values_g2(:, i) = min(slice2(:));
% 
%     fprintf('max/min of G1: %.2f - %.2f\n', max_values_g1, min_values_g1);
% 
%     max_min_index_G1(:, i) = [i; max(slice1(:)); min(slice1(:))];
%     % max_min_index_G2(:, i) = [i; max(slice2(:)); min(slice2(:))];
% 
% end
% 
% for i = 1:size(reshaped_array2, 3)
%     % Get the slice along the first and second dimensions
%     % slice1 = reshaped_array1(:,:,i);
%     slice2 = reshaped_array2(:,:,i);
% 
%     % Reshape the slice to size 200x50
%     % slice1 = reshape(slice1, [200, 50]);
%     slice2 = reshape(slice2, [200, 50]);
% 
%     % Find the maximum value along the first dimension 
%     % max_values_g1(:, i) = max(slice1(:));
%     max_values_g2(:, i) = max(slice2(:)); 
% 
%     % Find the minimum value along the first dimension
%     % min_values_g1(:, i) = min(slice1(:));
%     min_values_g2(:, i) = min(slice2(:));
% 
%     % fprintf('max/min of G2: %.2f - %.2f\n', max_values_g2, min_values_g2);
% 
%     % max_min_index_G1(:, i) = [i; max(slice1(:)); min(slice1(:))];
%     max_min_index_G2(:, i) = [i; max(slice2(:)); min(slice2(:))];
% 
% end
% 
% % max_min_index_G1(:) = [i; max_values_g1; min_values_g1];
% 
% % Display the size of max and min value arrays
% % fprintf('Size of max/min values G1: %.2f %.2f\n', size(max_values_g1), size(min_values_g2));
% % fprintf('Size of max values G2/G2: %.4f %.4f\n', size(max_values_g1), size(min_values_g2));
% % disp('Size of max values array:');
% % disp(size(max_values));
% % disp('Size of min values array:');
% % disp(size(min_values));
% 
% 
% max_min_index_G1 = max_min_index_G1.';
% max_min_index_G2 = max_min_index_G2.';
% % fprintf('Max Min Index of G1: %f \n', max_min_index_G1);
% 
% % writematrix(max_min_index_G1,'max_min_index_G1_121_01.xlsx')
% % writematrix(max_min_index_G2,'max_min_index_G2_121_01.xlsx')
 
toc

fprintf('Done!\n');

