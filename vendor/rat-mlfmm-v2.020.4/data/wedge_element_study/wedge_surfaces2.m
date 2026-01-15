clear; clc;

%% Parameters (example values)
a      = 1.0;    % meters
b      = 2.0;    % meters
c      = 3.0;    % meters (assumed > 0)
sigma0 = 0.0;
sigma1 = 0.1;
sigma2 = 0.0;
mu0    = 4*pi*1e-7;  % permeability

%% Define sigma function (linear charge)
sigma = @(x,y) sigma0 + sigma1*x + sigma2*y;

%% Original closed-form potential for observation at (0,0,c)
phi_cf = (sigma0/(4*pi*mu0)) * ( (a/2)*log((sqrt(a^2+b^2+c^2)+b)/(sqrt(a^2+b^2+c^2)-b)) - c*atan((a*b)/(sqrt(a^2+c^2)^2+c*sqrt(a^2+b^2+c^2)) ) );
% (phi_cf as given in the literature is valid only for the unshifted case)

%% Define a function to compute the potential for a shifted observation point
% The integration region is the triangle: x in [0,a], y in [0,(b/a)*x].
% The observation point is at (x0,y0,c).
phi_shift = @(x0,y0) (1/(4*pi*mu0)) * integral2(...
    @(x,y) sigma(x,y) ./ sqrt((x - x0).^2 + (y - y0).^2 + c^2), ...
    0, a, @(x) 0, @(x) (b/a)*x, 'RelTol',1e-8);

%% Compute the potential at the unshifted point (should equal phi_cf numerically)
phi0 = phi_shift(0,0);

%% Use finite differences to compute derivatives with respect to x0 and y0
delta = 1e-6;
% Hx = - d(phi)/dx0 at (0,0,c)
Hx_diff = -(phi_shift(delta,0) - phi0)/delta;
% Hy = - d(phi)/dy0 at (0,0,c)
Hy_diff = -(phi_shift(0,delta) - phi0)/delta;

%% Now compute the direct numerical integration of the H-field components
integrand_phi = @(x,y) sigma(x,y) ./ sqrt(x.^2 + y.^2 + c^2);
integrand_Hx  = @(x,y) -x .* sigma(x,y) ./ ((x.^2+y.^2+c^2).^(3/2));
integrand_Hy  = @(x,y) -y .* sigma(x,y) ./ ((x.^2+y.^2+c^2).^(3/2));
integrand_Hz  = @(x,y) c  .* sigma(x,y) ./ ((x.^2+y.^2+c^2).^(3/2));

phi_num = (1/(4*pi*mu0)) * integral2(integrand_phi, 0, a, @(x) 0, @(x) (b/a)*x, 'RelTol',1e-8);
Hx_num  = (1/(4*pi*mu0)) * integral2(integrand_Hx, 0, a, @(x) 0, @(x) (b/a)*x, 'RelTol',1e-8);
Hy_num  = (1/(4*pi*mu0)) * integral2(integrand_Hy, 0, a, @(x) 0, @(x) (b/a)*x, 'RelTol',1e-8);
Hz_num  = (1/(4*pi*mu0)) * integral2(integrand_Hz, 0, a, @(x) 0, @(x) (b/a)*x, 'RelTol',1e-8);

%% Display results
fprintf('Potential (phi):\n');
fprintf('  Closed-form (unshifted): %g\n', phi_cf);
fprintf('  Numerical (phi_shift(0,0)): %g\n', phi0);
fprintf('  Numerical integration: %g\n', phi_num);

fprintf('\nHx (from differentiation of phi): %g\n', Hx_diff);
fprintf('Hx (direct numerical integration): %g\n', Hx_num);

fprintf('\nHy (from differentiation of phi): %g\n', Hy_diff);
fprintf('Hy (direct numerical integration): %g\n', Hy_num);

fprintf('\nHz (from differentiation of phi w.r.t. c):\n');
fprintf('Hz (direct numerical integration): %g\n', Hz_num);
