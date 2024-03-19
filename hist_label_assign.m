
%%%%%%%%%%%%%%%%%%%%% plot 1 region from 1 participant over 50 iterations
% S_g1_temp (200x76x50)
clc

% Slice the array to get the first 500 slices
S_g1_temp_500 = S_g1_temp(:,:,501:1000);
S_g1_temp_200 = S_g1_temp(:,:,201:400);
S_g1_temp_100 = S_g1_temp(:,:,501:600);
S_g1_temp_50 = S_g1_temp(:,:,671:720);
S_g1_temp_25 = S_g1_temp(:,:,401:425);
disp(size(S_g1_temp));
% S_g1_temp

% Define the indexes 5 30 80 120 180
brainregion_index = 120; % Choose the brain region index you want to plot
participant_index = 5; % Choose the participant index you want to plot

% Extract the slice from S_g1_temp
participant_index = squeeze(S_g1_temp_200(brainregion_index, participant_index, :));

% Reshape the slice data into a 200x1X1000 array (already correct, no need to reshape further)
reshaped_data = participant_index; % participant_index.' 

disp(size(reshaped_data));

% Save the reshaped data to a CSV file 
% writematrix(reshaped_data,'hist_data.xlsx')

% % Plot the histogram
% 
% histogram(reshaped_data(:), 'Normalization', 'count');
% % histogram(reshaped_data(:));
% xlabel('Label Distribution'); %value
% ylabel('Frequency'); 
% legend('Labels')
% title('Histogram of modularity label assignment Data');

% Plot the histogram
subplot(2, 1, 1);
histogram(reshaped_data(:));
xlabel('Regions');
ylabel('Frequency');
title(['Histogram of label assignment of ', num2str(brainregion_index), 'th brain region']);

% Plot the line graph
subplot(2, 1, 2);
plot(1:numel(reshaped_data(:)), reshaped_data(:));
xlabel('Runs');
ylabel('Labels');
ylim([0.5 6.5]);
title(['Line Graph label assignment of ', num2str(brainregion_index), 'th brain region']);
