
%
% HEAT EQUATION, u_t = nu \nabla^2 u, USING m-POINT FINITE DIFFERENCE AND ADI
%

hdr

% N is expected to be set externally (e.g., from tst_h2d.m)
m=N-1;

Lx=1; Ly=1;
T = 1.2; nstep = ceil(T/dt); dt=T/nstep;
nu = .05;

dx=Lx/N; xb = dx*[0:N]'; x=xb(2:end-1);
e = ones(m,1); A = spdiags([-e 2*e -e],-1:1, m,m);
Ax = nu*A/(dx*dx);
Ix = speye(m);


dy=Ly/N; yb = dy*[0:N]'; y=yb(2:end-1);
e = ones(m,1); A = spdiags([-e 2*e -e],-1:1, m,m);
Ay = nu*A/(dy*dy);
Iy = speye(m);

[X,Y]=ndgrid(x,y);
[Xb,Yb]=ndgrid(xb,yb);

%
%  Set up ADI operators (Peaceman-Rachford scheme)
%
%  Half-step 1: (I + dt/2 * Ax) * U* = (I - dt/2 * Ay) * U^n
%  Half-step 2: (I + dt/2 * Ay) * U^(n+1) = (I - dt/2 * Ax) * U*
%

HLx = Ix + (dt/2)*Ax;   % Left operator for x-direction (implicit)
HRx = Ix - (dt/2)*Ax;   % Right operator for x-direction (explicit)
HLy = Iy + (dt/2)*Ay;   % Left operator for y-direction (implicit)
HRy = Iy - (dt/2)*Ay;   % Right operator for y-direction (explicit)

% Factor the tridiagonal matrices for efficient solves
warning('off', 'all');
[Lxf, Uxf] = lu(HLx);
[Lyf, Uyf] = lu(HLy);
warning('on', 'all');

%
%  Set up Exact Solution + RHS
%

kx = 1; ky = 3;

U0 = sin(kx*pi*X/Lx).*sin(ky*pi*Y/Ly);
U  = U0;

lamxy = -nu*( (kx*pi/Lx)^2 + (ky*pi/Ly)^2 );
thx   = pi*kx*dx/Lx;
thy   = pi*ky*dy/Ly;
lamx  = 2*(1-cos(thx))/(dx*dx);
lamy  = 2*(1-cos(thy))/(dy*dy);
lamxy = -nu*(lamx+lamy);              % Judge accuracy by discrete lambdas

% ADI time-stepping loop
% U is an m x m matrix where rows correspond to x and columns to y
for istep=1:nstep; time=istep*dt;
  
  % Half-step 1: implicit in x, explicit in y
  % Solve (I + dt/2 * Ax) * Ustar = (I - dt/2 * Ay) * U^n
  % Apply HRy along columns (y-direction), then solve HLx along rows (x-direction)
  Rhs = U * HRy';                     % Apply (I - dt/2*Ay) to each row (y-direction)
  Ustar = Uxf \ (Lxf \ Rhs);          % Solve (I + dt/2*Ax) for each column (x-direction)
  
  % Half-step 2: implicit in y, explicit in x
  % Solve (I + dt/2 * Ay) * U^(n+1) = (I - dt/2 * Ax) * Ustar
  % Apply HRx along rows (x-direction), then solve HLy along columns (y-direction)
  Rhs = HRx * Ustar;                  % Apply (I - dt/2*Ax) to each column (x-direction)
  U = (Uyf \ (Lyf \ Rhs'))';          % Solve (I + dt/2*Ay) for each row (y-direction)
  
end;

Uex = exp(lamxy*time)*U0;
Err = Uex-U;
eo  = e2;
e2  = norm(Err,"fro") / norm(Uex,"fro");
ratio = eo/e2;

format shorte;
disp([N dt nstep time e2 ratio])

% semilogy(t2k,e2k,'r-',lw,2,t2k,e2k,'k.',ms,11); drawnow; hold on;
