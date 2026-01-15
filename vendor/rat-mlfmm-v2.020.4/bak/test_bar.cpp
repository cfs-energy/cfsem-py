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
// #include "ilist.hh"
// #include "grid.hh"
// #include "nodelevel.hh"
#include "mgnmesh.hh"
#include "mlfmm.hh"

// main
int main(){
	// settings
	arma::uword num_exp = 5;
	arma::uword num_refine = 240;
	//arma::uword num_gauss = 3;
	double M = 1e6;

	// make volume sources
	//VolumeSources* mysources = new VolumeSources;
	//CoilMesh_f* mysources = new CoilMesh_f;
	ShMgnMeshPr mysources = MgnMesh::create();
	mysources->setup_cylinder(0.01, 0.02, 0.01, 6, 6, 120);
	//mysources->setup_cube(0.01,0.01,0.04,5,5,4*5);

	// axial magnetisation
	// arma::Mat<double> Mn(3,mysources->get_num_nodes(),arma::fill::zeros);
	// Mn.row(2).fill(M);
	// mysources->set_magnetisation_nodes(Mn);
	
	// radial magnetisation
	arma::Mat<double> Rn = mysources->get_node_coords();
	arma::Mat<double> Mn = M*(Rn.each_row()/Extra::vec_norm(Rn));
	mysources->set_magnetisation_nodes(Mn);	
	
	// create target coordinates
	ShMagneticTargetsPr mytargets = MagneticTargets::create();
	//mytargets->set_xy_plane(0.06, 0.05, 0, 0, 0, 120, 100);
	mytargets->set_xz_plane(0.1, 0.05, 0, 0, 0, 200, 100);
	//arma::Mat<double> Rt = mysources->get_node_coords();
	//arma::Mat<double> Rt = mysources->get_centroids();
	//mytargets->set_coords(Rt);

	// create settings
	ShSettingsPr settings = Settings::create();
	settings->set_num_exp(num_exp);
	settings->set_num_refine(num_refine);

	// creaet MLFMM
	ShMlfmmPr myfmm = Mlfmm::create(settings,mysources,mytargets);

	// run MLFMM
	myfmm->setup(); myfmm->calculate();

	// get results
	arma::Mat<double> Bfmm = mytargets->get_field("B",3);

	// // surface field fix
	// arma::Mat<arma::uword> n = mysources->get_vol_elements();
	// arma::Mat<arma::uword> sn = mysources->get_surf_elements();
	// arma::Mat<double> Ne = mysources->get_surf_face_normal();

	arma::Mat<arma::uword> e = mysources->get_surf_edges().t();
	arma::Mat<double> Xn = mysources->get_node_coords().t();
	//std::cout<<e.t()<<std::endl;
	// e.save("/Users/jvn/Dropbox/Development/elements.txt",arma::csv_ascii);
	// Xn.save("/Users/jvn/Dropbox/Development/coords.txt",arma::csv_ascii);

	// output
	arma::Mat<double> X = arma::join_horiz(mytargets->get_target_coords().t(),Bfmm.t());
	//arma::Mat<arma::uword> sn = mysources->get_surface_elements().t();
	X.save("plot_this.txt",arma::csv_ascii);
	//X.save("plot_this.txt",arma::csv_ascii);
	//sn.save("/Users/jvn/Dropbox/Development/elements.txt",arma::csv_ascii);

	// return
	return 0;
}