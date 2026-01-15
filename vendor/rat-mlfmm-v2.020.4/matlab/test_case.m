% Copyright 2022 Jeroen van Nugteren

% Permission is hereby granted, free of charge, to any person obtaining a copy of this software
% and associated documentation files (the "Software"), to deal in the Software without
% restriction, including without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
% Software is furnished to do so, subject to the following conditions:

% The above copyright notice and this permission notice shall be included in all copies or
% substantial portions of the Software.

% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
% FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
% OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
% WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
% CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

%% settings
% add the build directory which should
% contain the mexmlfmm.mex file
addpath('../build/Release/lib');

% settings related to accuracy
allow_use_gpu = false; % note that this is for both direct and mlfmm
num_refine = 200; % tree refinement target
num_expansions = 7; % number of expansions used for the multipoles and localpoles
large_interaction_list = true; % use larger neighbour list to increase accuracy

% sources
van_lanen = true; % use line elements instead of point elements
loop_radius = 0.05; % [m]
num_elements = 10000; % [#]
current = 1000; % [A]

% targets
xmin = -0.1; % [m]
xmax = 0.1; % [m]
num_x = 200; % [#]
zmin = -0.07; % [m]
zmax = 0.07; % [m]
num_z = 150; % [#]


%% create sources and targets
% create a loop current in the xy plane
theta = linspace(0,2*pi,num_elements+1);
Rn = [loop_radius*sin(theta);loop_radius*cos(theta);zeros(1,num_elements+1)];
dRs = diff(Rn,1,2);
Rs = (Rn(:,2:end) + Rn(:,1:end-1))/2;
Is = ones(1,num_elements)*current; 
epss = ones(1,num_elements)*1e-3;

% create a target xz plane
xa = linspace(xmin,xmax,num_x);
za = linspace(zmin,zmax,num_z);
[x,z] = meshgrid(xa,za);
y = zeros(num_x,num_z);
Rt = [x(:)';y(:)';z(:)'];


%% run MLFMM
% setup mlfmm
[flag] = mexmlfmm('setup',num_refine,num_expansions,large_interaction_list,allow_use_gpu);

% show output in terminal
% [flag] = mexmlfmm('enable_logging');

% set the sources to the mlfmm
[flag] = mexmlfmm('set_sources',Rs,dRs,Is,epss,van_lanen);

% set the targets to the mlfmm
[flag] = mexmlfmm('set_targets',Rt);

% setup the tree structure
timer = tic;
[flag] = mexmlfmm('setup_tree');
tree_time = toc(timer);

% after setting up the tree it is allowed to update the currents
% between successive calls of the calculate function
% [flag] = mexmlfmm('update_currents',Is);
% [flag] = mexmlfmm('update_direction',dR);

% run direct calculation
timer = tic;
disp("running direct ...");
[flag] = mexmlfmm('calculate_direct');
direct_time = toc(timer);

% get the field
[flag,At1,Bt1] = mexmlfmm('get_field');

% run multipole method
timer = tic;
disp("running mlfmm ...");
[flag] = mexmlfmm('calculate_mlfmm');
fmm_time = toc(timer);

% get the field
[flag,At2,Bt2] = mexmlfmm('get_field');

% cleanup
[flag] = mexmlfmm('cleanup');


%% post process output
% reshape output
Ax1 = reshape(At1(1,:),num_z,num_x);
Ay1 = reshape(At1(2,:),num_z,num_x);
Az1 = reshape(At1(3,:),num_z,num_x);
Am1 = sqrt(Ax1.*Ax1 + Ay1.*Ay1 + Az1.*Az1);

% reshape output
Bx1 = reshape(Bt1(1,:),num_z,num_x);
By1 = reshape(Bt1(2,:),num_z,num_x);
Bz1 = reshape(Bt1(3,:),num_z,num_x);
Bm1 = sqrt(Bx1.*Bx1 + By1.*By1 + Bz1.*Bz1);

% reshape output
Ax2 = reshape(At2(1,:),num_z,num_x);
Ay2 = reshape(At2(2,:),num_z,num_x);
Az2 = reshape(At2(3,:),num_z,num_x);
Am2 = sqrt(Ax2.*Ax2 + Ay2.*Ay2 + Az2.*Az2);

% reshape output
Bx2 = reshape(Bt2(1,:),num_z,num_x);
By2 = reshape(Bt2(2,:),num_z,num_x);
Bz2 = reshape(Bt2(3,:),num_z,num_x);
Bm2 = sqrt(Bx2.*Bx2 + By2.*By2 + Bz2.*Bz2);


%% analyse result
% use larger font size on high dpi screen
if exist('OCTAVE_VERSION', 'builtin')~=0
    set(0, "defaulttextfontsize", 24)  % title
    set(0, "defaultaxesfontsize", 16)  % axes labels
    set(0, "defaultlinelinewidth", 2)
end

% show timings
disp(['direct time         : ',num2str(direct_time),' [s]']);
disp(['rat-mlfmm tree time : ',num2str(tree_time),' [s]']);
disp(['rat-mlfmm fmm time  : ',num2str(fmm_time),' [s]']);

% create a figure
f = figure('Position',[100,100,1600,400]);

ax1 = subplot(2,3,1,'Parent',f);
imagesc(xa,za,Bm1,'Parent',ax1);
cb1 = colorbar(ax1);
title(ax1,['direct (',num2str(direct_time),' [s])']);
ylabel(cb1,'flux density [T]');
xlabel(ax1,'x [m]');
ylabel(ax1,'z [m]');

ax2 = subplot(2,3,2,'Parent',f);
imagesc(xa,za,Bm2,'Parent',ax2);
cb2 = colorbar(ax2);
title(ax2,['rat-mlfmm (',num2str(fmm_time),' [s])']);
ylabel(cb2,'flux density [T]');
xlabel(ax2,'x [m]');
ylabel(ax2,'z [m]');

ax3 = subplot(2,3,3,'Parent',f);
imagesc(xa,za,abs(Bm1 - Bm2),'Parent',ax3);
cb3 = colorbar(ax3);
title(ax3,'difference');
ylabel(cb3,'flux density [T]');
xlabel(ax3,'x [m]');
ylabel(ax3,'z [m]');

ax4 = subplot(2,3,4,'Parent',f);
imagesc(xa,za,Am1,'Parent',ax4);
cb4 = colorbar(ax4);
title(ax4,['direct (',num2str(direct_time),' [s])']);
ylabel(cb4,'vector potential [Vs/m]');
xlabel(ax4,'x [m]');
ylabel(ax4,'z [m]');

ax5 = subplot(2,3,5,'Parent',f);
imagesc(xa,za,Am2,'Parent',ax5);
cb5 = colorbar(ax5);
title(ax5,['rat-mlfmm (',num2str(fmm_time),' [s])']);
ylabel(cb5,'vector potential [Vs/m]');
xlabel(ax5,'x [m]');
ylabel(ax5,'z [m]');

ax6 = subplot(2,3,6,'Parent',f);
imagesc(xa,za,abs(Am1 - Am2),'Parent',ax6);
cb6 = colorbar(ax6);
title(ax6,'difference');
ylabel(cb6,'vector potential [Vs/m]');
xlabel(ax6,'x [m]');
ylabel(ax6,'z [m]');
