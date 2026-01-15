// Copyright 2025 Jeroen van Nugteren

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
// OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>

#include "common/extra.hh"
#include "magnetictargets.hh"
#include "currentsurface.hh"
#include "mlfmm.hh"
#include "settings.hh"

// main
int main(){
	// settings
	arma::uword num_exp = 5;
	arma::uword num_refine = 240;
	double J = 400e3;

	// make volume sources
	ShCurrentSurfacePr mysources = CurrentSurface::create();
	//mysources->setup_cylinder_shell(0.03,0.03,6,60);
	mysources->setup_moebius_strip(2,0.03, 0.02,12,80);

	// arma::Mat<double> X2 = mysources->get_target_coords().t();
	// X2.save("plot_this.txt",arma::csv_ascii);

	// set current using longitudinal vector
	arma::Mat<double> L = mysources->get_node_long_vector();
	mysources->set_current_density_nodes(L*J);

	// create target level
	ShMagneticTargetsPr mytargets = MagneticTargets::create();
	mytargets->set_xz_plane(0.1, 0.05, 0, 0, 0, 100, 50);

	// input settings
	ShSettingsPr settings = Settings::create();
	settings->set_num_exp(num_exp);
	settings->set_num_refine(num_refine);

	// setup and run MLFMM
	ShMlfmmPr myfmm = Mlfmm::create(settings);
	myfmm->set_sources(mysources);
	myfmm->set_targets(mysources);
	myfmm->setup(); myfmm->calculate();	// report

	// get results
	arma::Mat<double> Bfmm = mysources->get_field("B",3);
	//Bfmm.save("/Users/jvn/Dropbox/Development/plot_this.txt",arma::csv_ascii);

	// output
	arma::Mat<double> X = arma::join_horiz(mysources->get_target_coords().t(),Bfmm.t());
	arma::Mat<arma::uword> sn = mysources->get_elements().t();
	//X.save("/Users/jvn/Dropbox/Development/plot_this.txt",arma::csv_ascii);
	X.save("plot_this.txt",arma::csv_ascii);
	sn.save("elements.txt",arma::csv_ascii);

	// return
	return 0;
}