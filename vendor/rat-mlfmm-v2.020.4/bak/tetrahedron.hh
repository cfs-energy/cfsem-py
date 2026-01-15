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

#ifndef TETRAHEDRON_HH
#define TETRAHEDRON_HH

#include <armadillo> 
#include <cmath>
#include <cassert>

#include "common/extra.hh"
#include "common/parfor.hh"

// that are used throughout the code
class Tetrahedron{
	// methods
	public:
		static arma::Mat<arma::uword>::fixed<4,3> get_faces();
		static arma::Row<double> calc_volume(const arma::Mat<double> &Rn, const arma::Mat<arma::uword> &n);
		static arma::Row<double> special_determinant(const arma::Mat<double> &R1, const arma::Mat<double> &R2, const arma::Mat<double> &R3, const arma::Mat<double> &R4);
		static arma::Row<arma::uword> is_inside(const arma::Mat<double> &R, const arma::Mat<double> &Rn, const arma::Mat<arma::uword> &n);

		// shape function
		static arma::Mat<arma::sword>::fixed<4,4> get_corner_nodes();
		static arma::Mat<double> quad2cart(const arma::Mat<double> &Rn, const arma::Mat<double> &Rq);
};
 
#endif