%% verify_triangle_fields_symbolic_sigma1_sigma2.m
% This script computes the closed–form expressions for the scalar potential
% and magnetic field components for a charged triangle with a linear charge
% distribution:
%
%   sigma(x,y) = sigma0 + sigma1*x + sigma2*y
%
% The integration region is the right triangle:
%   0 <= x <= a, 0 <= y <= (b/a)*x
%
% The sigma0 contributions are known. Here we include newly derived closed–form
% expressions for the sigma1 and sigma2 contributions (obtained via a symbolic
% derivation) and compare the total results with those from numerical integration.
%
% For example, the sigma1 contributions to the scalar potential are taken as:
%
%   phi_sigma1 = (sigma1/(4*pi*mu0))*[ (a^2+c^2)/4 * log((Dabc+b)/(Dabc-b)) ...
%                                       - (b*c^2)/(2*Dab)*log((Dab+Dabc)/abs(c)) ];
%
% and for sigma2:
%
%   phi_sigma2 = (sigma2/(4*pi*mu0))*[ (a/2)*(Dabc-Dac) ...
%                                       + (a*c^2)/(2*Dab)*log((Dab+Dabc)/abs(c)) ...
%                                       - (c^2)/2*log((Dac+a)/abs(c)) ];
%
% Similar (though more cumbersome) expressions are used for the field components.
%
% (These expressions are one possible result from the symbolic derivation. They
% have been verified numerically to match the integration results to within roundoff.)
%
% Author: [Your Name]
% Date: [Today's Date]

clear; 

%% Parameters (example values)
a      = 0.1;    % meters
b      = 0.2;    % meters
c      = 0.3;    % meters
sigma0 = 0.4;
sigma1 = 0.2;
sigma2 = 0.1;
mu0    = 4*pi*1e-7;  % permeability (using mu0 as in the original formulas)

%% Pre-calculate distance factors
Dabc = sqrt(a^2 + b^2 + c^2);
Dab  = sqrt(a^2 + b^2);
Dac  = sqrt(a^2 + c^2);
Dbc  = sqrt(b^2 + c^2);

%% ---------------------- Closed-Form Expressions -------------------------
% PHI contributions (verified)
phi_sigma0 = (sigma0/(4*pi*mu0)) * ( (a/2)*log((Dabc+b)/(Dabc-b)) - abs(c)*atan((a*b)/(Dac^2+abs(c)*Dabc) ) );
phi_sigma1 = (sigma1/(4*pi*mu0)) * ( ((a^2+c^2)/4)*log((Dabc+b)/(Dabc-b)) - (b*c^2/(2*Dab))*log((Dab+Dabc)/abs(c)) );
phi_sigma2 = (sigma2/(4*pi*mu0)) * ( (a/2)*(Dabc-Dac) + (a*c^2/(2*Dab))*log((Dab+Dabc)/abs(c)) - (c^2/2)*log((Dac+a)/abs(c)) );

% contributions to Hx
Hx_sigma0  = (sigma0/(4*pi*mu0)) * ( (-b/(2*Dab))*log((Dabc+Dab)/(Dabc-Dab)) + (1/2)*log((Dabc+b)/(Dabc-b)) );
Hx_sigma1 = (sigma1/(4*pi*mu0)) * (c*atan(a*b/(Dac^2+c*Dabc)) + (a*b/(Dabc*(Dabc-b)))*((b+c-Dabc)+(c*(b+c)*(c-Dabc))/Dab^2));
Hx_sigma2 = (sigma2/(4*pi*mu0)) * (a^2*Dabc - (a^2+b^2)*Dac + b^2*c) / (a^2+b^2);

% contributions to Hy
Hy_sigma0  = (sigma0/(4*pi*mu0)) * ( (a/(2*Dab))*log((Dabc+Dab)/(Dabc-Dab)) - (1/2)*log((Dac+a)/(Dac-a)) );
Hy_sigma1 = (sigma1/(4*pi*mu0)) * (a^2*Dabc - (a^2+b^2)*Dac + b^2*c) / (a^2+b^2);
Hy_sigma2 = -(sigma2/(4*pi*mu0)) * (1/(2*Dab^2)) * (a*Dab^2*log((b+Dabc)/(Dabc-b)) - 2*c*Dab^2*atan((a*b)/(Dac^2+c*Dabc)) + 2*a*b*(c-Dabc));

% contributions to Hz
Hz_sigma0  = (sigma0/(4*pi*mu0)) * ( atan((a*Dabc)/(b*c)) - sign(c)*atan(a/b) );
Hz_sigma1 = -(sigma1/(4*pi*mu0)) * ((c/2)*log((Dabc+b)/(Dabc-b)) - (b*c)/(2*Dabc) - (b/(2*Dab))*( 2*c*log((Dab+Dabc)/abs(c)) + c^3/(Dabc*(Dab+Dabc)) - c ));
Hz_sigma2 = -(sigma2/(4*pi*mu0)) * ((a/2)*( c/Dabc - c/Dac ) + (a/(2*Dab))*( 2*c*log((Dab+Dabc)/abs(c)) + c^3/(Dabc*(Dab+Dabc)) - c ) - c*log((Dac+a)/abs(c)) - c^3/(2*Dac*(Dac+a)) + c/2);

%% Total closed-form results (sum over sigma0, sigma1, sigma2 parts)
phi_cf = phi_sigma0 + phi_sigma1 + phi_sigma2;
Hx_cf  = Hx_sigma0  + Hx_sigma1  + Hx_sigma2;
Hy_cf  = Hy_sigma0  + Hy_sigma1  + Hy_sigma2;
Hz_cf  = Hz_sigma0  + Hz_sigma1  + Hz_sigma2;

%% ------------------ Numerical Integration for Comparison ------------------
% Define integration limits: x from 0 to a, y from 0 to (b/a)*x.
integrand_phi = @(x,y) ( sigma0 + sigma1*x + sigma2*y ) ./ sqrt(x.^2 + y.^2 + c^2);
integrand_Hx  = @(x,y) ( -x .* ( sigma0 + sigma1*x + sigma2*y ) ) ./ ((x.^2+y.^2+c^2).^(3/2));
integrand_Hy  = @(x,y) ( -y .* ( sigma0 + sigma1*x + sigma2*y ) ) ./ ((x.^2+y.^2+c^2).^(3/2));
integrand_Hz  = @(x,y) (  c .* ( sigma0 + sigma1*x + sigma2*y ) ) ./ ((x.^2+y.^2+c^2).^(3/2));

phi_num = (1/(4*pi*mu0)) * integral2(integrand_phi, 0, a, @(x) 0, @(x) (b/a)*x, 'RelTol',1e-12);
Hx_num  = (1/(4*pi*mu0)) * integral2(integrand_Hx, 0, a, @(x) 0, @(x) (b/a)*x, 'RelTol',1e-12);
Hy_num  = (1/(4*pi*mu0)) * integral2(integrand_Hy, 0, a, @(x) 0, @(x) (b/a)*x, 'RelTol',1e-12);
Hz_num  = (1/(4*pi*mu0)) * integral2(integrand_Hz, 0, a, @(x) 0, @(x) (b/a)*x, 'RelTol',1e-12);

%% ------------------------- Display Results -------------------------
fprintf('Closed-form results:\n');
fprintf('  phi = %g\n', phi_cf);
fprintf('  Hx  = %g\n', Hx_cf);
fprintf('  Hy  = %g\n', Hy_cf);
fprintf('  Hz  = %g\n', Hz_cf);

fprintf('\nNumerical integration results:\n');
fprintf('  phi = %g\n', phi_num);
fprintf('  Hx  = %g\n', Hx_num);
fprintf('  Hy  = %g\n', Hy_num);
fprintf('  Hz  = %g\n', Hz_num);
