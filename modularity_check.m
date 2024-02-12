clear
clc

corr_g1 = permute(cell2mat(struct2cell(load('/Users/ismaila/Documents/C-Codes/SCI_GraphTheory/sci_data/SCI/fc/corr_hc.mat'))), [2 3 1]);
corr_g2 = permute(cell2mat(struct2cell(load('/Users/ismaila/Documents/C-Codes/SCI_GraphTheory/sci_data/SCI/fc/corr_sci.mat'))), [2 3 1]);

tic
n = 50;
m_index = zeros(6, 80);
count=1;
for x = 0.0:0.1:1.5

    for y = 0:-1:-4

        m_g1 = zeros(1, n);
        m_g2 = zeros(1, n);

        parfor i = 1:n
            % Use temporary variables for x and y inside parfor
            temp_x = x;
            temp_y = y;
            
            omega = power(10, temp_y);
            A_g1 = squeeze(num2cell(corr_g1, [1 2]));
            A_g2 = squeeze(num2cell(corr_g2, [1 2]));
            N = length(A_g1{1});
            T_g1 = size(corr_g1, 3);
            T_g2 = size(corr_g2, 3);

            [B_g1, twom_g1] = multicat(A_g1, temp_x, omega);
            PP_g1 = @(S_g1)postprocess_categorical_multilayer(S_g1, T_g1);
            [S_g1, Q_g1] = iterated_genlouvain(B_g1, 10000, 0, 1, 'moverandw', [], PP_g1);
            Q_g1 = Q_g1 / twom_g1;

            S_g1 = reshape(S_g1, N, T_g1);
            C_g1 = mode(S_g1, 2);
            K_g1 = max(S_g1, [], 'all');

            [B_g2, twom_g2] = multicat(A_g2, temp_x, omega);
            PP_g2 = @(S_g2)postprocess_categorical_multilayer(S_g2, T_g2);
            [S_g2, Q_g2] = iterated_genlouvain(B_g2, 10000, 0, 1, 'moverandw', [], PP_g2);
            Q_g2 = Q_g2 / twom_g2;

            S_g2 = reshape(S_g2, N, T_g2);
            C_g2 = mode(S_g2, 2);
            K_g2 = max(S_g2, [], 'all');

            m_g1(i) = Q_g1;
            m_g2(i) = Q_g2;
            
        end

        m_g1_avg = mean(m_g1);
        m_g1_std = std(m_g1);
        m_g2_avg = mean(m_g2);
        m_g2_std = std(m_g2);

        fprintf('%d) Gamma:%.2f Omega:%.2f Modularity:: HC:%.4f ± %.4f & SCI:%.4f ± %.4f\n',count, x, y, m_g1_avg, m_g1_std, m_g2_avg, m_g2_std);

        m_g1_avg = str2double(sprintf('%.4f', m_g1_avg));
        m_g1_std = str2double(sprintf('%.4f', m_g1_std));
        m_g2_avg = str2double(sprintf('%.4f', m_g2_avg));
        m_g2_std = str2double(sprintf('%.4f', m_g2_std));
        count = count+1;
        m_index(:, count) = [x; y; m_g1_avg; m_g1_std; m_g2_avg; m_g2_std];
        % (counter_x - 1) * 5 + counter_y (x-1)*5+y
    end
end

toc

m_index_all = m_index.';
fprintf('Modularity of hc & sci is %.4f\n', m_index_all);

% m_g1_a = mean(m_g1)
% m_g1_a = mode(m_g1)

% Mod_all = [m_g1; m_g1]
writematrix(m_index_all,'m_index_all.csv')

fprintf('Done!\n');
