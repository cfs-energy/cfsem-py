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

#ifndef HB_CURVE_HH
#define HB_CURVE_HH

#include <armadillo> 
#include <cassert>
#include <algorithm>

#include "common/extra.hh"

// shared pointer definition
typedef std::shared_ptr<class HBCurve> ShHBCurvePr;
typedef arma::field<ShHBCurvePr> ShHBCurvePrList;

// hb curve class template
class HBCurve{
	// properties
	private:
		// hb curve row(0) = H, row(1) = B
		bool mono_interp_ = false;
		
		// HB data
		arma::Col<double> Hinterp_;
		arma::Col<double> Binterp_;

		// iron filling fraction
		double ff_ = 1.0;

		// finite difference
		double dH_ = 0.1;

	// methods
	public:
		// constructor
		HBCurve();

		// factory
		static ShHBCurvePr create();

		// set functions
		void set_roxie();
		void set_comsol();
		void set_opera();
		void set_extern(const arma::Row<double> &Hinterp, const arma::Row<double> &Binterp, const double ff);

		// calculation functions
		arma::Row<double> calc_magnetic_flux_density(const arma::Row<double> &Hmag) const;
		arma::Row<double> calc_magnetisation(const arma::Row<double> &Hmag) const;
		arma::Row<double> calc_susceptibility(const arma::Row<double> &Hmag) const;
		arma::Row<double> calc_susceptibility_diff(const arma::Row<double> &Hmag) const;

};

#endif