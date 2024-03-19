% Define the size of the array
nodeCount = 200;
communityCount = 76;
totalIterations = 100;

% Assuming you have the communityAssignments array already available

% Initialize variables S_g1_temp(:,:,1)
sameCommunityCount = 0;

for k = 1:totalIterations
% Iterate over each node pair (i, j)
    for i = 1:nodeCount
        for j = i+1:communityCount % Avoid duplicate pairs
        % Check if nodes i and j are assigned to the same community in each iteration
            if S_g1_temp(i, :, k) == S_g1_temp(j, :, k) 
            % if S_g1_temp(5, 5, k) == S_g1_temp(5, 5, k+1) 
                fprintf('%d == %d ?\n', S_g1_temp(i, :, k), S_g1_temp(j, :, k) );
                sameCommunityCount = sameCommunityCount + 1;
            end
        end
    end
end

% Calculate probability
totalPairs = 7 * (7 - 1) / 2; % Total number of unique node pairs communitycount*(cc-1)/2
probability = sameCommunityCount / (totalPairs * totalIterations);

% Display result
fprintf('Probability of node i and j being in the same community: %.4f\n', probability);
