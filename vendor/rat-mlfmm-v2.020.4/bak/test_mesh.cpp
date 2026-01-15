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
#include "currentmesh.hh"
#include "mlfmm.hh"
#include "mtsurface.hh"
#include "settings.hh"

// main
int main(){
	// settings
	arma::uword num_exp = 5;
	arma::uword num_refine = 240;
	arma::uword num_gauss = 3;
	double J = 400e6;

	// use armadillo timer
	arma::wall_clock timer;

	// make volume sources
	ShCurrentMeshPr mymesh = CurrentMesh::create();
	mymesh->setup_cylinder(0.04,0.05,0.01,6,6,100);
	
	// calculate and set current density
	// arma::Mat<double> Rn = mymesh->get_node_coords();
	// arma::Mat<double> L(3,Rn.n_cols,arma::fill::zeros);
	// L.row(0) = -Rn.row(1); L.row(1) = Rn.row(0); 
	// L.each_row()/=Extra::vec_norm(L);
	// mymesh->set_current_density_nodes(L*J);

	arma::Mat<double> Ln = mymesh->get_node_long_vector();
	mymesh->set_current_density_nodes(Ln*J);
	mymesh->apply_rotation(arma::datum::pi/4,arma::datum::pi/5,arma::datum::pi/3);

	// get elements for checking
	// arma::Mat<arma::uword> S = mymesh->get_surface_elements();
	// S = S.t(); Rn = Rn.t();
	// S.save("/Users/jvn/Dropbox/Development/elements.txt",arma::csv_ascii);
	// Rn.save("/Users/jvn/Dropbox/Development/nodes.txt",arma::csv_ascii);
	
	// add mesh
	mymesh->set_num_gauss(num_gauss);
	//mymesh->set_num_dist(0);
	// target points
	// arma::Mat<double> Rt(3,1000,arma::fill::zeros);
	// Rt.row(0) = arma::linspace<arma::Row<double> >(-0.2,0.2,Rt.n_cols);

	// arma::uword N = 400;
	// arma::Mat<double> x(N,N);
	// x.each_row() = arma::linspace<arma::Row<double> >(-0.1,0.1,N);
	// arma::Mat<double> y(N,N);
	// y.each_col() = arma::linspace<arma::Col<double> >(-0.1,0.1,N);
	// arma::Mat<double> z = arma::Mat<double>(N,N,arma::fill::ones)*0.0;
	// arma::Mat<double> Rt(3,N*N);
	// Rt.row(0) = arma::reshape(x,1,N*N);
	// Rt.row(1) = arma::reshape(y,1,N*N);
	// Rt.row(2) = arma::reshape(z,1,N*N);

	// arma::Mat<double> Rt = mysources->get_node_coords();

	// // create target level
	// MagneticTargets* mytargets = new MagneticTargets;
	// mytargets->set_coords(Rt);
	ShMTSurfacePr mysurf = mymesh->create_surface();

	// create settings
	ShSettingsPr settings = Settings::create();
	settings->set_num_exp(num_exp);
	settings->set_num_refine(num_refine);

	// setup and run MLFMM
	ShMlfmmPr myfmm = Mlfmm::create(settings);
	myfmm->set_sources(mymesh);
	myfmm->set_targets(mysurf);
	myfmm->setup(); myfmm->calculate();	// report

	// get results
	arma::Mat<double> Bfmm = mysurf->get_field("B",3);
	arma::Mat<double> X = arma::join_horiz(mysurf->get_target_coords().t(),Bfmm.t());
	arma::Mat<arma::uword> sn = mysurf->get_elements().t();
	//X.save("/Users/jvn/Dropbox/Development/plot_this.txt",arma::csv_ascii);
	X.save("plot_this.txt",arma::csv_ascii);
	//sn.save("/Users/jvn/Dropbox/Development/elements.txt",arma::csv_ascii);
	sn.save("elements.txt",arma::csv_ascii);

	// return
	return 0;
}