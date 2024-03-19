[CS_g1, CQ_g1, X_new_g1, qpc_g1] = consensus_iterative_bassett(S_g1);
[CS_g2, CQ_g2, X_new_g2, qpc_g2] = consensus_iterative_bassett(S_g2);

% cd '/Users/ismaila/Documents/C-Codes/SCI_GraphTheory/sci_data/SCI/modularity_var/';

filename = sprintf('CS_hc_%.1f,%.1f.mat', gamma, log_omega); save(filename,'CS_g1');
filename = sprintf('CS_sci_%.1f,%.1f.mat', gamma, log_omega); save(filename,'CS_g2');