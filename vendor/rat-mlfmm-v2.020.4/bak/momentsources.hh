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

#ifndef MOMENT_SOURCES_HH
#define MOMENT_SOURCES_HH

#include <armadillo> 
#include <complex>
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
typedef std::shared_ptr<class MomentSources> ShMomentSourcesPr;

// collection of line sources
// is derived from the sources class
class MomentSources: public PointSources{
	// properties
	private:
		// effective current density in [Am]
		arma::Mat<double> Meff_;

		// source to multipole matrices
		StMat_So2Mp_M M_M_;
		arma::Mat<double> dR_;

	// methods
	public:
		// constructor
		MomentSources();

		// factory
		static ShMomentSourcesPr create();

		// set coordinates and currents of elements
		void set_moments(const arma::Mat<double> &Meff);

		// direct field calculation for both A and B
		void calc_direct(ShTargetsPr &tar) const;
		void calc_direct(ShTargetsPr &tar, const arma::Row<arma::uword> &tidx, const arma::Row<arma::uword> &sidx) const;

		// source to multipole
		void setup_source_to_multipole(const arma::Mat<double> &dR, arma::uword num_exp);
		arma::Mat<std::complex<double> > source_to_multipole(const arma::Row<arma::uword> &indices, arma::uword num_exp) const;

		// display information
		void display() const;
};

#endif
