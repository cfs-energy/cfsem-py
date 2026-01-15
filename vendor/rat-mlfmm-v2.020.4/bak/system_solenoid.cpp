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
#include "soleno.hh"
#include "mlfmm.hh"
#include "settings.hh"

// main
int main(){
	// settings
	arma::uword Nt = 1000;
	arma::uword num_exp = 7;
	arma::uword num_refine = 240;
	arma::uword num_gauss = 3;
	double num_dist = 1.5;

	// geometry
	arma::Row<double> Rin = {0.1};
	arma::Row<double> Rout = {0.11};
	arma::Row<double> height = {0.1};
	arma::Row<double> J = {400e6};
	arma::Row<arma::uword> num_rad = {5};
	arma::Row<arma::uword> num_height = {20};
	arma::Row<arma::uword> num_azym = {90};
	arma::Row<arma::uword> nlayer = {5};

	// create list of sources
	arma::field<ShSourcesPr> mysourcelist(Rin.n_elem);

	// walk over coils
	for(arma::uword i=0;i<Rin.n_elem;i++){
		// make volume sources
		ShCurrentMeshPr mysource = CurrentMesh::create();
		mysource->setup_cylinder(Rin(i),Rout(i),height(i),
			num_rad(i),num_height(i),num_azym(i));

		// finalise setup by setting up volumes
		mysource->calculate_element_volume();
		
		// calculate and set current density
		arma::Mat<double> Rn = mysource->get_node_coords();
		arma::Mat<double> L(3,Rn.n_cols,arma::fill::zeros);
		L.row(0) = -Rn.row(1); L.row(1) = Rn.row(0); 
		L.each_row()/=Extra::vec_norm(L);
		mysource->set_current_density_nodes(L*J(i));

		// set other properties
		mysource->set_num_gauss(num_gauss);
		mysource->set_num_dist(num_dist);

		// add to sourcelist
		mysourcelist(i) = mysource;
	}

	// targets on line in radial direction
	ShMagneticTargetsPr myradialtargets = MagneticTargets::create();
	arma::Mat<double> Rtr(3,Nt,arma::fill::zeros);
	Rtr.row(0) = arma::linspace<arma::Row<double> >(-0.2,0.2,Nt);
	myradialtargets->set_coords(Rtr);

	// targets on line axially but not in center
	ShMagneticTargetsPr myaxialtargets = MagneticTargets::create();
	arma::Mat<double> Rta(3,Nt,arma::fill::zeros);
	Rta.row(0).fill(0.05);
	Rta.row(2) = arma::linspace<arma::Row<double> >(-0.2,0.2,Nt);
	myaxialtargets->set_coords(Rta);

	// create settings
	ShSettingsPr settings = Settings::create();
	settings->set_num_exp(num_exp);
	settings->set_num_refine(num_refine);

	// setup and run MLFMM
	ShMlfmmPr myfmm = Mlfmm::create(settings);
	myfmm->add_targets(myradialtargets);
	myfmm->add_targets(myaxialtargets);
	myfmm->add_sources(mysourcelist);
	myfmm->setup(); myfmm->calculate();	// report

	// get results for radial targets
	arma::Mat<double> Bfmmr = myradialtargets->get_field("B",3);
	arma::Row<double> Brfmmr = Bfmmr.row(0);
	arma::Row<double> Bzfmmr = Bfmmr.row(2);

	// get results for axial targets
	arma::Mat<double> Bfmma = myaxialtargets->get_field("B",3);
	arma::Row<double> Brfmma = Bfmma.row(0); 
	arma::Row<double> Bzfmma = Bfmma.row(2);

	// calculate total current in coil
	arma::Row<double> I = J%(Rout-Rin)%height;

	// run soleno for radial targets
	arma::Mat<double> Brsolr, Bzsolr;
	arma::Mat<double> Rrr = Rtr.row(0);
	arma::Mat<double> Rzr = Rtr.row(2);
	Soleno::calc_B(Brsolr, Bzsolr, Rrr, Rzr, Rin, Rout, -height/2, height/2, I, nlayer);

	// check difference in axial field
	double pctdiffBzr = arma::max(arma::abs(Bzsolr-Bzfmmr)/arma::max(arma::abs(Bzsolr)));
	std::printf("difference in Bz, %5.2f pct\n",pctdiffBzr);

	// run soleno for axial targets
	arma::Mat<double> Brsola, Bzsola;
	arma::Mat<double> Rra = Rta.row(0);
	arma::Mat<double> Rza = Rta.row(2);
	Soleno::calc_B(Brsola, Bzsola, Rra, Rza, Rin, Rout, -height/2, height/2, I, nlayer);

	// check difference in radial and axial field
	double pctdiffBra = arma::max(arma::abs(Brsola-Brfmma)/arma::max(arma::abs(Brsola)));
	double pctdiffBza = arma::max(arma::abs(Brsola-Brfmma)/arma::max(arma::abs(Brsola)));
	std::printf("difference in Br, %5.2f pct\n",pctdiffBra);
	std::printf("difference in Bz, %5.2f pct\n",pctdiffBza);

	// debugging
	// CoilMesh* mysource = dynamic_cast<CoilMesh*>(mysourcelist(0));
	// arma::Mat<double> Rn = mysource->get_volume().t();
	// Rn.save("plot_this.txt",arma::csv_ascii);

	// arma::Mat<double> X(Nt,8);
	// X.col(0) = Brsolr.t();
	// X.col(1) = Brfmmr.t();
	// X.col(2) = Bzsolr.t();
	// X.col(3) = Bzfmmr.t();
	// X.col(4) = Brsola.t();
	// X.col(5) = Brfmma.t();
	// X.col(6) = Bzsola.t();
	// X.col(7) = Bzfmma.t();
	// X.save("plot_this.txt",arma::csv_ascii);

	assert(pctdiffBzr<1.0);
	assert(pctdiffBra<1.0);
	assert(pctdiffBza<1.0);

	// return
	return 0;
}