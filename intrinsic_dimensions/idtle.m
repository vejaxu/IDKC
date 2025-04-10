function id = idtle(X,dists,epsilon)
%% TLE estimator: excluding central measurements, including reflections (recommended default)
% X - matrix of nearest neighbors (k x d), sorted by distance
% dists - nearest-neighbor distances (1 x k), sorted
% epsilon - threshold for dropping measurements in order to avoid numerical issues
    if nargin < 2
        error('Two parameters required: X - matrix of nearest neighbors (k x d), sorted by distance; dists - nearest-neighbor distances (1 x k), sorted');
    end
    if nargin == 2
        % if epsilon is too large, too many measurements can be dropped, yielding NaN or very large IDs
        % if epsilon is too small, numerical issues can produce imaginary values
        epsilon = 0.00001; % default value expected to work in the vast majority of cases
    end
    r = dists(end); % distance to k-th neighbor
    %% Boundary case 1: If r = 0, this is fatal, since the neighborhood would be degenerate
    if r == 0
        error('All k-NN distances are zero!');
    end
    %% Main computation
    k = length(dists);
    V = squareform(pdist(X));
    Di = repmat(dists',1,k);
    Dj = Di';
    Z2 = 2*Di.^2 + 2*Dj.^2 - V.^2;
    S = r * (((Di.^2 + V.^2 - Dj.^2).^2 + 4*V.^2 .* (r^2 - Di.^2)).^0.5 - (Di.^2 + V.^2 - Dj.^2)) ./ (2*(r^2 - Di.^2));
    T = r * (((Di.^2 + Z2   - Dj.^2).^2 + 4*Z2   .* (r^2 - Di.^2)).^0.5 - (Di.^2 + Z2   - Dj.^2)) ./ (2*(r^2 - Di.^2));
    Dr = dists == r; % handle case of repeating k-NN distances
    S(Dr,:) = r * V(Dr,:).^2 ./ (r^2 + V(Dr,:).^2 - Dj(Dr,:).^2);
    T(Dr,:) = r * Z2(Dr,:)   ./ (r^2 + Z2(Dr,:)   - Dj(Dr,:).^2);
    %% Boundary case 2: If u_i = 0, then for all 1 <= j <= k the measurements s_ij and t_ij reduce to u_j
    Di0 = Di == 0; 
    S(Di0) = Dj(Di0);
    T(Di0) = Dj(Di0);
    %% Boundary case 3: If u_j = 0, then for all 1 <= j <= k the measurements s_ij and t_ij reduce to (r v_ij)/(r + v_ij)
    Dj0 = Dj == 0; 
    S(Dj0) = r * V(Dj0) ./ (r + V(Dj0));
    T(Dj0) = r * V(Dj0) ./ (r + V(Dj0));
    %% Boundary case 4: If v_ij = 0, then the measurement s_ij is zero and must be dropped. The measurement t_ij should be dropped as well
    V0 = V == 0;
    V0(logical(eye(k))) = 0;
    S(V0) = r; % by setting to r, s_ij will not contribute to the sum s1s
    T(V0) = r; % by setting to r, t_ij will not contribute to the sum s1t
    nV0 = sum(V0(:)); % will subtract twice this number during ID computation below
    %% Drop S & T measurements below epsilon: If s_ij is thrown out, then for the sake of balance, t_ij should be thrown out as well (and vice versa)
    STeps = S < epsilon | T < epsilon | isnan(S) | isnan(T); % also drop NaN measurements, "legitemately" obtained as 0/0
    STeps(logical(eye(k))) = 0;
    nSTeps = sum(STeps(:));
    S(STeps) = r;
    S = log(S/r);
    T(STeps) = r;
    T = log(T/r);
    S(logical(eye(k))) = 0; % delete diagonal elements
    T(logical(eye(k))) = 0;
    %% Sum over the whole matrices
    s1s = sum(S(:));
    s1t = sum(T(:));
    %% Compute ID, subtracting numbers of dropped measurements
    s1sum = s1s+s1t;
    if s1sum > -epsilon % Boundary case 5: (almost) all kNN distances are equal
        id = 0;
    else
        id = -2*(k*(k-1)-nSTeps-nV0) / s1sum;
    end
end
