clear
clc
        
gamma = 1.2; % gamma = 0.5:0.1:1.5
log_omega = -1; % log_omega = 0:-1:-4
omega = power(10,log_omega);        

corr_g1 = permute(cell2mat(struct2cell(load('/Users/ismaila/Documents/C-Codes/SCI_GraphTheory/sci_data/SCI/fc/corr_hc.mat'))), [2 3 1]);
corr_g2 = permute(cell2mat(struct2cell(load('/Users/ismaila/Documents/C-Codes/SCI_GraphTheory/sci_data/SCI/fc/corr_sci.mat'))), [2 3 1]);

% corr_g1 = permute(cell2mat(struct2cell(load('/Users/ismaila/Documents/C-Codes/SCI_GraphTheory/sci_data/SCI/fc/corr_sci_c.mat'))), [2 3 1]);
% corr_g2 = permute(cell2mat(struct2cell(load('/Users/ismaila/Documents/C-Codes/SCI_GraphTheory/sci_data/SCI/fc/corr_sci_t.mat'))), [2 3 1]);

% model settings 
A_g1 = squeeze(num2cell(corr_g1,[1 2])); % adjacency/connectivity matrix
A_g2 = squeeze(num2cell(corr_g2,[1 2]));         
N = length(A_g1{1}); % number of nodes (same for both groups)
T_g1 = size(corr_g1,3); % number of layers (subjects)
T_g2 = size(corr_g2,3);

% multi-layer modularity (Q) and partitions
% Group 1
[B_g1, twom_g1] = multicat(A_g1,gamma,omega);
% [B_g1, twom_g1] = multicatdir_f(A_g1,gamma,omega);
PP_g1 = @(S_g1)postprocess_categorical_multilayer(S_g1,T_g1);
[S_g1,Q_g1] = iterated_genlouvain(B_g1,10000,0,1,'moverandw',[], PP_g1); % 4th entry(randord): 0[move] or 1[moverand] | 5th: move, moverand, or moverandw
Q_g1 = Q_g1/twom_g1;
S_g1 = reshape(S_g1,N,T_g1);
C_g1 = mode(S_g1,2); % consensus
K_g1 = max(S_g1,[],'all'); % number of communities
% Group 2
[B_g2,twom_g2] = multicat(A_g2,gamma,omega);
PP_g2 = @(S_g2)postprocess_categorical_multilayer(S_g2,T_g2);
[S_g2,Q_g2] = iterated_genlouvain(B_g2,10000,0,1,'moverandw',[], PP_g2); % 4th entry(randord): 0[move] or 1[moverand] | 5th: move, moverand, or moverandw
Q_g2 = Q_g2/twom_g2;
S_g2 = reshape(S_g2, N, T_g2);
C_g2 = mode(S_g2,2); % consensus
K_g2 = max(S_g2,[],'all'); % number of communities

cd '/Users/ismaila/Documents/C-Codes/SCI_GraphTheory/sci_data/SCI/modularity_var/';

filename = sprintf('S_hc_%.1f,%.1f.mat', gamma, log_omega); save(filename,'S_g1');
filename = sprintf('S_sci_%.1f,%.1f.mat', gamma, log_omega); save(filename,'S_g2');

% filename = sprintf('S_sci_c_%.1f,%.1f.mat', gamma, log_omega); save(filename,'S_g1');
% filename = sprintf('S_sci_t_%.1f,%.1f.mat', gamma, log_omega); save(filename,'S_g2');

