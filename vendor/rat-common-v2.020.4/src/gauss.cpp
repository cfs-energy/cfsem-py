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

// include header file
#include "gauss.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// default constructor
	Gauss::Gauss(){
		// setup points
		calculate();
	}

	// constructor
	Gauss::Gauss(const arma::sword num_gauss, const fltp tol){
		// check input
		assert(num_gauss!=0);

		// set number of gauss points
		num_gauss_ = num_gauss;
		set_tol(tol);

		// setup points
		calculate();
	}

	// set tolerance
	void Gauss::set_tol(const fltp tol){
		tol_ = tol;
	}

	// set number of gauss points
	void Gauss::set_num_gauss(const arma::sword num_gauss){
		num_gauss_ = num_gauss;
	}

	// get tolerance
	fltp Gauss::get_tol()const{
		return tol_;
	}

	// get number of gauss 
	arma::sword Gauss::get_num_gauss()const{
		return num_gauss_;
	}

	// Generates the abscissa and weights for a Gauss-Legendre quadrature.
	// Reference:  Numerical Recipes in Fortran 77, Cornell press.
	void Gauss::calculate(){
		// can not be zero
		assert(num_gauss_!=0);

		// get number of gauss points
		arma::uword ng = static_cast<arma::uword>(std::abs(num_gauss_));

		// gauss point distribution
		if(num_gauss_>0){
			// allocate output
			xg_.zeros(ng); wg_.zeros(ng);

			// calculate
			fltp m = (ng+RAT_CONST(1.0))/2;
			for(arma::uword ii=0;ii<m;ii++){
				// initial estimate
				fltp z = std::cos(arma::Datum<fltp>::pi*(fltp(ii)+RAT_CONST(1.0)-RAT_CONST(0.25))/(fltp(ng)+RAT_CONST(0.5)));
				fltp z1 = z+1;
				fltp pp = 0; // not used
				while(std::abs(z-z1)>tol_){
					fltp p1 = 1, p2 = 0;
					for(arma::uword jj=1;jj<=ng;jj++){
						fltp p3 = p2;
						p2 = p1;

						// The Legendre polynomial.
						p1 = ((2*jj-1)*z*p2-(jj-1)*p3)/jj;
					}

					// The L.P. derivative.
					pp = ng*(z*p1-p2)/(z*z-1);
					z1 = z;
					z = z1-p1/pp;
				}

				// Build up the abscissas.
				xg_(ii) = -z;
				xg_(ng-1-ii) = z;

				// Build up the weights.
				wg_(ii) = 2/((1-z*z)*(pp*pp));
				wg_(ng-1-ii) = wg_(ii);
			}
		}

		// regular distribution
		if(num_gauss_<0){
			xg_ = arma::linspace<arma::Row<fltp> >(-RAT_CONST(1.0) + RAT_CONST(1.0)/ng, RAT_CONST(1.0) - RAT_CONST(1.0)/ng,ng);
			wg_ = arma::Row<fltp>(ng,arma::fill::ones)*RAT_CONST(2.0)/static_cast<fltp>(ng);
		}
	}

	// function for getting absissas
	const arma::Row<fltp>& Gauss::get_abscissae() const{
		return xg_;
	}

	// function for getting absissas
	const arma::Row<fltp>& Gauss::get_weights() const{
		return wg_;
	}

}}