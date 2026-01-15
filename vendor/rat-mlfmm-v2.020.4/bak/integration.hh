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

#ifndef INTEGRATION_HH
#define INTEGRATION_HH

#include <armadillo> 
#include <cassert>
#include <cmath> // contains M_PI
#include <algorithm>

#include "common/extra.hh"
#include "common/gauss.hh"
#include "hexahedron.hh"
#include "quadrilateral.hh"
#include "savart.hh"
#include "triangle.hh"

// contains numerical and analytical integrals for magnetisation element
class Integration{
	// methods
	public:
		// field calculations
		static arma::Mat<double> calc_analytic(const arma::Mat<double> &Rt, const arma::Mat<double> &Rn);
		static arma::Mat<double> calc_numeric_surface(const arma::Mat<double> &Rt, const arma::Mat<double> &Rnh, const arma::sword num_gauss, const bool is_self_field);
		static arma::Mat<double> calc_numeric_volume(const arma::Mat<double> &Rt,const arma::Mat<double> &Rnh, const double Ve, const arma::sword num_gauss);
		static arma::Mat<double> calc_savart(const arma::Mat<double> &Rt, const arma::Mat<double> &Rnh, const double Ve, const bool is_self_field);
		static arma::Mat<double> calc_analytic_tet(const arma::Mat<double> &Rt,const arma::Mat<double> &Rnh);
};

#endif