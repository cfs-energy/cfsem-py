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
#include <cassert>

#include "common/extra.hh"
#include "currentsources.hh"
#include "targets.hh"
#include "mlfmm.hh"

// main
int main(){
	// settings
	arma::uword Ns = 1000;
	arma::uword Nt = Ns;
	arma::uword num_exp = 7;
	arma::uword num_refine = 50;
	double eps_r = 1.0;
	double constant = (1e7/(4*arma::datum::pi*eps_r));

	// tell user what this thing does
	std::printf("\nScript for testing one dimensional sources.\n\n");

	// set random seed
	arma::arma_rng::set_seed(1001);

	// create quasi random current source coordinates
	arma::Mat<double> Rs = 
		Extra::random_coordinates(0, 0.1, 0.1, 1, Ns); // x,y,z,size,N
	arma::Mat<double> Q = arma::Mat<double>(10,Ns,arma::fill::randu)-0.5;
		
	// create source currents
	ShCurrentSourcesPr mycharges = CurrentSources::create();
	mycharges->set_coords(Rs);
	mycharges->set_currents(Q);
	mycharges->set_num_dim(Q.n_rows);
	
	// create target coordinates
	arma::Mat<double> Rt = 
		Extra::random_coordinates(0, 0, 0, 1, Nt); // x,y,z,size,N
		
	// create target level
	ShTargetsPr mytargets = Targets::create();
	mytargets->set_field_type("A",Q.n_rows);
	mytargets->set_coords(Rt);

	// setup mlfmm
	ShMlfmmPr myfmm = Mlfmm::create(mycharges,mytargets);
	
	// input settings
	ShSettingsPr settings = myfmm->settings();
	settings->set_num_exp(num_exp);
	settings->set_num_refine(num_refine);
	

	// compare with direct
	myfmm->calculate_direct();

	// get results
	arma::Mat<double> Vdir = constant*mytargets->get_field("A",3);

	// run MLFMM
	myfmm->setup(); myfmm->calculate();

	// get results
	arma::Mat<double> Vfmm = constant*mytargets->get_field("A",3);

	// display
	std::printf("difference: %6.4f pct\n",
		100*arma::max(arma::max((Vfmm-Vdir)/arma::max(arma::max(Vdir)))));

	return 0;
}