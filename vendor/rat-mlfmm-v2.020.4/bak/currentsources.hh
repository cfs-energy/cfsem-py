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

#ifndef CURRENT_SOURCES_HH
#define CURRENT_SOURCES_HH

#include <armadillo> 
#include <cassert>
#include <cmath> // contains M_PI
#include <algorithm>
#include <memory>

#include "common/extra.hh"
#include "pointsources.hh"
#include "targets.hh"
#include "savart.hh"
#include "stmat.hh"

// shared pointer definition
typedef std::shared_ptr<class CurrentSources> ShCurrentSourcesPr;

// collection of line current sources
class CurrentSources: public PointSources{
	// properties
	protected:
		// effective current density in [Am]
		arma::Mat<double> Ieff_;

		// source to multipole matrices
		StMat_So2Mp_J M_J_;
		arma::Mat<double> dR_;

	// methods
	public:
		// constructor
		CurrentSources();

		// factory
		static ShCurrentSourcesPr create();

		// setup points 
		void set_points(const arma::Mat<double> &R, const arma::Mat<double> &Ieff);
		void set_points(arma::field<ShCurrentSourcesPr> &srcs);

		// set coordinates and currents of elements
		void set_currents(const arma::Mat<double> &Ieff);
		arma::Mat<double> get_currents() const;

		// direct field calculation for both A and B
		void calc_direct(ShTargetsPr &tar) const;
		void calc_direct(ShTargetsPr &tar, const arma::Row<arma::uword> &tidx, const arma::Row<arma::uword> &sidx) const;

		// source to multipole
		void setup_source_to_multipole(const arma::Mat<double> &dR, const arma::uword num_exp);
		arma::Mat<std::complex<double> > source_to_multipole(const arma::Row<arma::uword> &indices, const arma::uword num_exp) const;
};

#endif
