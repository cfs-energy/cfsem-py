clear; clc;

%% Parameters (example values)
a      = 1;      % meters
b      = 2;      % meters
c      = 3;      % meters (assumed > 0)
sigma1 = 0.1;
mu0    = 4*pi*1e-7; % permeability
dy    = 1e-9;     % small lateral shift

%% Define source triangle vertices:
O = [0, 0, 0];
A = [a, 0, 0];
B = [a, b, 0];
C = [0, 0, c]; 
D = [0, dy, c];
E = [0, dy, 0];
F = [a, dy, 0]; % E projected on AB

%% 1. Triangle EAB (the extra region when shifting)
% Edge OB: from O to B.
V = (B - O)/norm(B-O);  % = [a, b, 0]
% Compute the projection parameter 
t = dot(E - O, V);
% Projection point P on OB:
P = O + t*V;

% Triangle EFB
a1 = norm(F-E);
b1 = norm(F-B);

% triangle AFE
a2 = norm(F-E);
b2 = norm(F-A);

% triangle AEO
a3 = norm(E-O);
b3 = norm(A-O);

% triangle OPE
a4 = norm(E-P);   
b4 = norm(P-O);   

% triangle PBE
a5 = norm(E-P);
b5 = norm(P-B);


%% 4. Define function F for potential of a right triangle.
% the sigma 0 equation
F0 = @(aa,bb,cc) (1/(4*pi*mu0)) * ( (aa/2)*log((sqrt(aa^2+bb^2+cc^2)+bb)/(sqrt(aa^2+bb^2+cc^2)-bb)) - abs(cc)*atan((aa*bb)/(sqrt(aa^2+cc^2)^2+abs(cc)*sqrt(aa^2+bb^2+cc^2)) ) );

% the sigma1 equation
F1 = @(aa,bb,cc) (1/(4*pi*mu0)) * ( ((aa^2+cc^2)/4)*log((sqrt(aa^2+bb^2+cc^2)+bb)/(sqrt(aa^2+bb^2+cc^2)-bb)) - (bb*cc^2/(2*sqrt(aa^2+bb^2)))*log((sqrt(aa^2+bb^2)+sqrt(aa^2+bb^2+cc^2))/abs(cc)) );

% the sigma2 equation
F2 = @(aa,bb,cc) (1/(4*pi*mu0)) * ( (aa/2)*(sqrt(aa^2+bb^2+cc^2)-sqrt(aa^2+cc^2)) + (aa*cc^2/(2*sqrt(aa^2+bb^2)))*log((sqrt(aa^2+bb^2)+sqrt(aa^2+bb^2+cc^2))/abs(cc)) - (cc^2/2)*log((sqrt(aa^2+cc^2)+aa)/abs(cc)) );


%% 5. Form the net potential change due to the shift.
% The shifted potential for the sigma1 term is approximated by
%   phi_shift_sigma1 = F(a1,b1,c) + F(a2,b2,c) - F(a3,b3,c).
sigmaEP = sigma1 * dot((P-E)/norm(P-E), (A-O)/norm(A-O));
sigmaOB = sigma1 * dot((O-B)/norm(O-B), (A-O)/norm(A-O));
phi1 = sigma1 * F1(a,b,c);
phi2 = sigma1 * (F1(a1,b1,c) + F1(a2,b2,c) + F2(a3,b3,c)) - (sigmaEP * (F1(a4,b4,c) + F1(a5,b5,c)) + sigmaOB*(F2(a4,b4,c) - F2(a5,b5,c)));

%% 6. Compute the derivative d(phi)/dx (using the finite-difference limit)
Hy_sigma1_finite_difference = -(phi2 - phi1) / dy;

% precompute terms
Dabc = sqrt(a^2 + b^2 + c^2);
Dab = sqrt(a^2 + b^2);
Dac = sqrt(a^2 + c^2);

%% 7. Then the sigma1 contribution to Hx is
fprintf('Hy_sigma1 from puzzling triangles: %12.14g\n', Hy_sigma1_finite_difference);

%% 8. For comparison: Direct numerical integration of the sigma1 part.
% For sigma1, the integrand for φ is: (sigma1*x) / sqrt(x^2+y^2+c^2).
% Its contribution to Hx is then:
integrand_Hy_sigma1  = @(x,y) ( -y .* ( sigma1*x  ) ) ./ ((x.^2+y.^2+c^2).^(3/2));

Hy_num  = (1/(4*pi*mu0)) * integral2(integrand_Hy_sigma1, 0, a, @(x) 0, @(x) (b/a)*x, 'RelTol',1e-14,'AbsTol',1e-14);

fprintf('Hy_sigma1 from direct numerical integration: %12.14g\n', Hy_num);

fprintf('difference :  %12.14g\n', Hy_sigma1_finite_difference - Hy_num);
