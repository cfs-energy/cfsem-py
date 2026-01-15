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

#ifndef GAUSS_HH
#define GAUSS_HH

#include <armadillo> 
#include <complex>
#include <cmath>
#include <cassert>

// that are used throughout the code
class Gauss{
	private:
		
		// settings
		double tol_ = 1e-7;
		arma::sword num_gauss_ = 3;

		// abscissas
		arma::Row<double> xg_;
		
		// weights
		arma::Row<double> wg_;

	// methods
	public:
		// constructor
		Gauss();
		Gauss(arma::sword num_gauss);

		// calculation
		void calculate();
		arma::Row<double> get_abscissae() const;
		arma::Row<double> get_weights() const;

		// static arma::Mat<arma::uword> setup_grid(const arma::uword N0, const arma::uword N1, const arma::uword N2);
		// static arma::Mat<double> setup_gauss_grid(const arma::uword N);
		// static arma::Mat<double> setup_regular_grid(const arma::uword N);	
};
 
#endif