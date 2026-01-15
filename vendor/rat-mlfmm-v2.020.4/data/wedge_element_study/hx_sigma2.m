clear; clc;

%% Parameters (example values)
a      = 0.01;      % meters
b      = 0.02;      % meters
c      = 0.005;      % meters (assumed > 0)
sigma2 = 0.1;
mu0    = 4*pi*1e-7; % permeability
dx    = 1e-9;     % small lateral shift

%% Define source triangle vertices:
O = [0, 0, 0];
A = [a, 0, 0];
B = [a, b, 0];
C = [0, 0, c]; 
D = [dx, 0, c];
E = [dx, 0, 0];

%% 1. Triangle EAB (the extra region when shifting)
a1 = a - dx; 
b1 = b;      

%% 2. Determine projection of E onto edge OB.
% Edge OB: from O to B.
V = (B - O)/norm(B-O);  % = [a, b, 0]
% Compute the projection parameter t = ((E-O) dot V) / (V dot V)
t = dot(E - O, V);
% Projection point P on OB:
P = O + t*V;

%% 3. Triangles from projection:
% For triangle EPO
a2 = norm(E - P);   
b2 = norm(P - O);   

% For triangle EBP
a3 = norm(E - P);            
b3 = norm(P - B);   

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
sigmaEP = sigma2 * dot((P-E)/norm(P-E), [0;1;0]);
sigmaOB = sigma2 * dot((O-B)/norm(O-B), [0;1;0]);
phi1 = sigma2*F2(a,b,c);
phi2 = sigma2*F2(a1,b1,c) + sigmaEP*F1(a2,b2,c) + sigmaEP*F1(a3,b3,c) + sigmaOB*F2(a2,b2,c) - sigmaOB*F2(a3,b3,c);


%% 6. Compute the derivative d(phi)/dx (using the finite-difference limit)
Hx_sigma2_finite_difference = -(phi2 - phi1) / dx;

% precompute terms
Dabc = sqrt(a^2 + b^2 + c^2);
Dab = sqrt(a^2 + b^2);
Dac = sqrt(a^2 + c^2);

%% 7. Then the sigma1 contribution to Hx is
fprintf('Hx_sigma1 from puzzling triangles: %12.10g\n', Hx_sigma2_finite_difference);

%% 8. For comparison: Direct numerical integration of the sigma1 part.
% For sigma1, the integrand for φ is: (sigma1*x) / sqrt(x^2+y^2+c^2).
% Its contribution to Hx is then:
integrand_Hx_sigma2 = @(x,y) (-x .* (sigma2*y)) ./ ((x.^2+y.^2+c^2).^(3/2));
Hx_num = (1/(4*pi*mu0)) * integral2(integrand_Hx_sigma2, 0, a, @(x) 0, @(x) (b/a)*x, 'RelTol',1e-14,'AbsTol',1e-14);

fprintf('Hx_sigma1 from direct numerical integration: %12.16g\n', Hx_num);

fprintf('difference :  %12.16g\n', Hx_sigma2_finite_difference - Hx_num);


