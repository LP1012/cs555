
theta = (pi/N)*[N:-1:0]';
xb    = cos(theta); xb = L*(xb+1)/2;    %% CHEBYSHEV distribution

% xb    = L*[0:N]'/N;                     %% Uniform

h  = diff(xb);
x  = xb(2:end-1);


% spacings to left/right of each interior node j=2..N-1
hL   = xb(2:N)    - xb(1:N-1);  % h_- (length n)
hR   = xb(3:N+1)  - xb(2:N);    % h_+ (length n)
hbar = 0.5*(hL + hR);           % avg h_j

% main and off-diagonals of S
main  = -(1./hL + 1./hR);
upper =  1./hR(1:end-1);      upper = [0; upper];
lower =  1./hL(2:end);        lower = [lower; 0];

% assemble sparse S and diagonal M
n = length(main);
S = spdiags([lower, main, upper], [-1 0 1], n, n);
M = spdiags(1./hbar, 0, n, n);

A = -M*S;

