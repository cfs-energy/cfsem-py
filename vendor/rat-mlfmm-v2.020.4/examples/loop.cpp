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

// general headers
#include <armadillo>

// specific headers
#include "currentsources.hh"
#include "mgntargets.hh"
#include "mlfmm.hh"

// main function
int main(){
	// settings
	const rat::fltp radius = 0.05; // loop radius in [m]
	const arma::uword num_sources = 500; // number of source elements in loop
	const arma::uword num_targets = 400; // number of target points along axis
	const rat::fltp current = 400; // loop current in [A]
	const rat::fltp zmin = -0.05; // target axis start-coordinate in [m]
	const rat::fltp zmax = 0.05; // target axis end-coordinate in [m]

	// create coordinates on a circle in cylindrical coordinates
	const arma::Row<rat::fltp> rho = arma::Row<rat::fltp>(num_sources+1,arma::fill::ones)*radius;
	const arma::Row<rat::fltp> theta = arma::linspace<arma::Row<rat::fltp> >(0,2*arma::Datum<rat::fltp>::pi,num_sources+1);

	// convert to carthesian and store in a 3 by N matrix with rows x, y and z
	const arma::Mat<rat::fltp> Rn = arma::join_vert(
		rho%arma::cos(theta), rho%arma::sin(theta), 
		arma::Row<rat::fltp>(num_sources+1,arma::fill::zeros));

	// calculate coordinates and direction vectors of line elements. 
	// This is achieved by taking the average and difference between 
	// two consecutive points, respectively. These are then also
	// stored in 3 by N matrices with rows x,y,z and dx,dy,dz, respectively
	const arma::Mat<rat::fltp> Rs = (Rn.tail_cols(num_sources) + Rn.head_cols(num_sources))/2;
	const arma::Mat<rat::fltp> dRs = arma::diff(Rn,1,1);

 	// create corresponding currents and softening factors
 	const arma::Row<rat::fltp> Is = arma::Row<rat::fltp>(num_sources,arma::fill::ones)*current;
	const arma::Row<rat::fltp> epss = 0.7*rat::cmn::Extra::vec_norm(dRs);

	// create target points along axis
	const arma::Mat<rat::fltp> Rt = arma::join_vert(
		arma::Mat<rat::fltp>(2,num_targets,arma::fill::zeros),
		arma::linspace<arma::Row<rat::fltp> >(zmin,zmax,num_targets));

	// setup source and target objects
	rat::fmm::ShCurrentSourcesPr src = rat::fmm::CurrentSources::create(Rs,dRs,Is,epss);
	rat::fmm::ShMgnTargetsPr tar = rat::fmm::MgnTargets::create(Rt); tar->set_field_type('H',3);

	// run MLFMM on sources and targets
	rat::fmm::ShMlfmmPr mlfmm = rat::fmm::Mlfmm::create(src,tar);
	mlfmm->setup();	mlfmm->calculate();

	// get resulting field in 3 by N matrix with rows Bx, By and Bz
	const arma::Mat<rat::fltp> B = tar->get_field('B');

	// return normally
	return 0;
}