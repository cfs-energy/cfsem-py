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
#include <iostream>
#include <cmath>
#include <complex>

// specific headers
#include "rat/common/error.hh"
#include "rat/common/extra.hh"
#include "spharm.hh"

// test for spherical harmonics

// main
int main(){
	// number of expansions
	const int num_exp = 3; // do not change
	const arma::uword num_coords = 500;

	// coordinate to check
	// rho (no influence)
	// theta (between 0 and Pi)
	// phi (between -2Pi and 2Pi)
	arma::arma_rng::set_seed(1001);
	arma::Row<rat::fltp> rho = 0.1*arma::Row<rat::fltp>(num_coords,arma::fill::ones);
	arma::Row<rat::fltp> theta = arma::Datum<rat::fltp>::pi*arma::Row<rat::fltp>(num_coords,arma::fill::randu);
	arma::Row<rat::fltp> phi = 4*arma::Datum<rat::fltp>::pi*arma::Row<rat::fltp>(num_coords,arma::fill::randu)-2*arma::Datum<rat::fltp>::pi;
		
	// allocate prefac
	arma::Row<rat::fltp> prefac(num_coords);
	arma::Row<std::complex<rat::fltp> > Yrow(num_coords); 

	// make coordinates
	arma::Mat<rat::fltp> sphcoord(3,num_coords);
	sphcoord.row(0) = rho;
	sphcoord.row(1) = theta; 
	sphcoord.row(2) = phi; 

	// make spherical harmonics
	rat::fmm::Spharm myspharm(num_exp, sphcoord);

	// display harmonic
	myspharm.display(rat::cmn::Log::create());


	// get harmonic from spharm
	arma::Mat<std::complex<rat::fltp> > Yspharm = myspharm.get_all();

	// cross check with analytic 
	// harmonics are Schmidt semi-normalized and include (-1)^m term
	// http://mathworld.wolfram.com/SphericalHarmonic.html
	// Euler formula
	// exp(ix) = cos(x) + isin(x)
	arma::Mat<std::complex<rat::fltp> > Y(rat::fmm::Extra::polesize(num_exp),num_coords,arma::fill::zeros);

	// n=0, m=0
	Y.row(rat::fmm::Extra::nm2fidx(0,0)).fill(1.0f);

	// n=1, m=-1,1
	prefac = -(1.0f/2)*std::sqrt(2)*arma::sin(theta);
	Yrow.set_real(prefac%arma::cos(-phi));
	Yrow.set_imag(prefac%arma::sin(-phi));
	Y.row(rat::fmm::Extra::nm2fidx(1,-1)) = Yrow;
	Yrow.set_real(prefac%arma::cos(phi));
	Yrow.set_imag(prefac%arma::sin(phi));
	Y.row(rat::fmm::Extra::nm2fidx(1,+1)) = Yrow;

	// n=1, m=0
	Yrow.set_real(arma::cos(theta));
	Yrow.set_imag(arma::Row<rat::fltp>(num_coords,arma::fill::zeros));
	Y.row(rat::fmm::Extra::nm2fidx(1,0)) = Yrow;

	// n=2, m=-2,2
	prefac = (1.0f/4)*std::sqrt(6)*arma::pow(arma::sin(theta),2);
	Yrow.set_real(prefac%arma::cos(-2*phi));
	Yrow.set_imag(prefac%arma::sin(-2*phi));
	Y.row(rat::fmm::Extra::nm2fidx(2,-2)) = Yrow;
	Yrow.set_real(prefac%arma::cos(2*phi));
	Yrow.set_imag(prefac%arma::sin(2*phi));
	Y.row(rat::fmm::Extra::nm2fidx(2,+2)) = Yrow;

	// n=2, m=-1,1
	prefac = -(1.0f/2)*std::sqrt(6)*arma::sin(theta)%arma::cos(theta);
	Yrow.set_real(prefac%arma::cos(-phi));
	Yrow.set_imag(prefac%arma::sin(-phi));
	Y.row(rat::fmm::Extra::nm2fidx(2,-1)) = Yrow;
	Yrow.set_real(prefac%arma::cos(phi));
	Yrow.set_imag(prefac%arma::sin(phi));
	Y.row(rat::fmm::Extra::nm2fidx(2,+1)) = Yrow;

	// n=2, m=0
	Yrow.set_real(0.5f*(3*arma::cos(theta)%arma::cos(theta)-1));
	Yrow.set_imag(arma::Row<rat::fltp>(num_coords,arma::fill::zeros));
	Y.row(rat::fmm::Extra::nm2fidx(2,0)) = Yrow;

	// n=3, m=-3,3
	prefac = -(1.0f/8)*std::sqrt(20)*arma::pow(arma::sin(theta),3);
	Yrow.set_real(prefac%arma::cos(-3*phi));
	Yrow.set_imag(prefac%arma::sin(-3*phi));
	Y.row(rat::fmm::Extra::nm2fidx(3,-3)) = Yrow;
	Yrow.set_real(prefac%arma::cos(3*phi));
	Yrow.set_imag(prefac%arma::sin(3*phi));
	Y.row(rat::fmm::Extra::nm2fidx(3,+3)) = Yrow;

	// n=3, m=-2,2
	prefac = (1.0f/4)*std::sqrt(30)*arma::pow(arma::sin(theta),2)%arma::cos(theta);
	Yrow.set_real(prefac%arma::cos(-2*phi));
	Yrow.set_imag(prefac%arma::sin(-2*phi));
	Y.row(rat::fmm::Extra::nm2fidx(3,-2)) = Yrow;
	Yrow.set_real(prefac%arma::cos(2*phi));
	Yrow.set_imag(prefac%arma::sin(2*phi));
	Y.row(rat::fmm::Extra::nm2fidx(3,+2)) = Yrow;

	// n=3, m=-1,1
	prefac = -(1.0f/8)*std::sqrt(12)*arma::sin(theta)%(5*arma::pow(arma::cos(theta),2)-1);
	Yrow.set_real(prefac%arma::cos(-phi));
	Yrow.set_imag(prefac%arma::sin(-phi));
	Y.row(rat::fmm::Extra::nm2fidx(3,-1)) = Yrow;
	Yrow.set_real(prefac%arma::cos(phi));
	Yrow.set_imag(prefac%arma::sin(phi));
	Y.row(rat::fmm::Extra::nm2fidx(3,+1)) = Yrow;

	// n=3, m=0
	Yrow.set_real((1.0f/4)*std::sqrt(4)*(5*arma::pow(arma::cos(theta),3)-3*arma::cos(theta)));
	Yrow.set_imag(arma::Row<rat::fltp>(num_coords,arma::fill::zeros));
	Y.row(rat::fmm::Extra::nm2fidx(3,0)) = Yrow;

	// cross-check
	rat::fltp norm = arma::max(arma::sqrt(arma::sum(arma::real(Y)%arma::real(Y) + arma::imag(Y)%arma::imag(Y))));
	arma::Mat<rat::fltp> pct_diff_real = 100*arma::abs(arma::real(Y)-arma::real(Yspharm))/norm;
	arma::Mat<rat::fltp> pct_diff_imag = 100*arma::abs(arma::imag(Y)-arma::imag(Yspharm))/norm;

	// check difference
	//assert(arma::all(arma::all(pct_diff_real<1e-6)));
	//assert(arma::all(arma::all(pct_diff_imag<1e-6)));

	// std::cout<<pct_diff_real<<std::endl;
	// std::cout<<pct_diff_imag<<std::endl;

	// check difference in real part
	if(!arma::all(arma::all(pct_diff_real<1e-2))){
		rat_throw_line("real part of spherical harmonics calculation deviates from theoretical value");
	}
		
	// check differnce in complex part
	if(!arma::all(arma::all(pct_diff_imag<1e-2))){
		rat_throw_line("complex part of spherical harmonics calculation deviates from theoretical value");
	}

	// return
	return 0;
}