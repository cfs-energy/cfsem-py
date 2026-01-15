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

#ifndef HEXAHEDRON_HH
#define HEXAHEDRON_HH

#include <armadillo> 
#include <complex>
#include <cmath>
#include <cassert>

#include "common/extra.hh"
#include "common/parfor.hh"
#include "tetrahedron.hh"
#include "quadrilateral.hh"

// library of math functions related to volume elements

// definition of the hexahedrons
// nodes:
// o-->xi 
// | 0---1     4---5
// nu|   |     |   |
//   3---2     7---6
//   mu=-1     mu=1:

// sides:
// Side Node-1 Node-2 Node-3 Node-4
// Side-1 (S1) 0 1 5 4 in xi, mu
// Side-2 (S2) 1 2 6 5 in nu, mu
// Side-3 (S3) 2 3 7 6 in xi, mu
// Side-4 (S4) 3 0 4 7 in nu, mu
// Side-5 (S5) 0 3 2 1 in xi, nu
// Side-6 (S6) 4 5 6 7 in xi, nu

// that are used throughout the code
class Hexahedron{
	// methods
	public:
		// definition matrices of hexahedron
		static arma::Mat<arma::sword>::fixed<8,3> get_corner_nodes();
		static arma::Mat<arma::uword>::fixed<6,4> get_faces();
		static arma::Mat<arma::uword>::fixed<12,2> get_edges();
		static arma::Mat<arma::sword>::fixed<6,3> get_facenormal();
		static arma::Mat<arma::uword>::fixed<6,3> get_facedim();
		static arma::Col<arma::sword>::fixed<6> get_facedir();

		// hexahedron conversion matrices
		static arma::Mat<arma::uword>::fixed<5,4> tetrahedron_conversion_matrix();
		static arma::Mat<arma::uword>::fixed<12,4> tetrahedron_conversion_matrix_special();
		
		// shape function and its derivatives
		static arma::Mat<double> shape_function(const arma::Mat<double> &Rq);
		static arma::Mat<double> shape_function_derivative(const arma::Mat<double> &Rq);
		static arma::Mat<double> shape_function_derivative_cart(const arma::Mat<double> &Rn, const arma::Mat<double> &Rq);
		
		// quadrilateral coordinates
		static arma::Mat<double> quad2cart(const arma::Mat<double> &Rn, const arma::Mat<double> &Rq);
		static arma::Mat<double> cart2quad(const arma::Mat<double> &Rn, const arma::Mat<double> &Rc, const double tol);

		// derivatives of quadrilateral coordinates
		static arma::Mat<double> quad2cart_derivative(const arma::Mat<double> &Rn, const arma::Mat<double> &Rq, const arma::Mat<double> &phi);
		static arma::Mat<double> quad2cart_curl(const arma::Mat<double> &Rn, const arma::Mat<double> &Rq, const arma::Mat<double> &V);

		// functions for checking inside or outside
		static arma::Row<arma::uword> is_inside(const arma::Mat<double> &R, const arma::Mat<double> &Rn, const arma::Mat<arma::uword> &n);

		// volume calculation
		static arma::Row<double> calc_volume(const arma::Mat<double> &Rn, const arma::Mat<arma::uword> &n);

		// check if faces are correct
		static arma::Row<arma::uword> is_clockwise(const arma::Mat<double> &R, const arma::Mat<arma::uword> &n);

		// grid for avoiding singularities
		static void setup_source_grid(arma::Mat<double> &Rqgrd, arma::Row<double> &wgrd, const arma::Col<double>::fixed<3> &Rqs, const arma::Row<double> &xg, const arma::Row<double> &wg);
};
 
#endif