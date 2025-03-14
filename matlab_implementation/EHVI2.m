function ehvi = EHVI2(means, sigmas, goal, ref, pareto)
    % EHVI calculates the Expected Hypervolume Improvement for a set of test points.
    % Inputs:
    %   - means: GP mean estimation of objectives of the test points.
    %   - sigmas: Uncertainty of GP mean estimations.
    %   - goal: Row vector defining which objectives to be minimized or maximized.
    %   - ref: Hypervolume reference for calculations.
    %   - pareto: Current true Pareto front obtained so far.
    % Output:
    %   - ehvi: Expected Hypervolume Improvement for each test point.

    N_obj = size(means, 2);
    n_points = size(means, 1);

    % Turn the problem into minimizing for all objectives
    for i = 1:size(goal, 2)
        if goal(i) == 1
            means(:, i) = -1 * means(:, i);
            pareto(:, i) = -1 * pareto(:, i);
        end
    end

    % Sorting the non_dominated points considering the first objective
    [~, d] = sort(pareto(:, 1));
    pareto = pareto(d, :);
    ind = zeros(n_points, 1);
    ehvi = zeros(n_points, 1);

    % EHVI calculation for test points
    for i = 1:n_points
        if ind(i) == 1
            ehvi(i) = 0;
        else
            hvi = 0;
            box = 1;
            % EHVI over the box from infinity to the ref point
            for j = 1:N_obj
                s = (ref(j) - means(i, j)) / sigmas(i, j);
                box = box * ((ref(j) - means(i, j)) * normcdf(s) + sigmas(i, j) * normpdf(s));
            end
            % Calculate how much adding a test point can improve the hypervolume
            hvi = recursiveIteration(means(i, :), sigmas(i, :), ref, pareto);
            ehvi(i) = box - hvi;
        end
    end
end

function improvement = recursiveIteration(means, sigmas, ref, pareto)
    N_obj = size(pareto, 2);
    improvement = 0;
    hvi_temp = 1;
    while size(pareto, 1) > 1
        s_up = (ref - means) ./ sigmas;
        s_low = (pareto(1, :) - means) ./ sigmas;
        up = ((ref - means) .* normcdf(s_up)) + (sigmas .* normpdf(s_up));
        low = ((pareto(1, :) - means) .* normcdf(s_low)) + (sigmas .* normpdf(s_low));
        hvi_temp = hvi_temp .* prod(up - low, 2);
        pareto = max([pareto(1, :); pareto(2:end, :)]);
        pareto = Pareto_finder(pareto, zeros(1, N_obj));
    end
    improvement = hvi_temp;
end
