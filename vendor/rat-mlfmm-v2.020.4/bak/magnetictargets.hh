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

#ifndef MAGNETIC_TARGET_POINTS_HH
#define MAGNETIC_TARGET_POINTS_HH

#include <armadillo> 
#include <complex>
#include <cassert>
#include <cmath> // contains M_PI
#include <algorithm>
#include <mutex>
#include <memory>

#include "common/extra.hh"
#include "stmat.hh"
#include "mtargets.hh"

// shared pointer definition
typedef std::shared_ptr<class MagneticTargets> ShMagneticTargetsPr;

// interaction between target points and mlfmm for
// magnetic field and/or vector potential calculation
class MagneticTargets: public MTargets{
	// properties
	protected:
		// coordinates
		arma::Mat<double> Rt_;

		// number of target points
		arma::uword num_targets_;

		// multipole operation matrices
		StMat_Lp2Ta_A M_A_; // for vector potential
		StMat_Lp2Ta_H M_H_; // for magnetic field
		arma::Mat<double> dR_;

	// methods
	public:
		// constructor
		MagneticTargets();

		// factory
		static ShMagneticTargetsPr create();

		// setting of coordinates
		void set_coords(const arma::Mat<double> &Rt);

		// required methods for all targets
		arma::Mat<double> get_target_coords() const;
		arma::Mat<double> get_target_coords(const arma::Row<arma::uword> &indices) const;
		arma::uword num_targets() const;

		// setup coordinates for plane
		void set_xy_plane(double ellx, double elly, double xoff, double yoff, double zoff, arma::uword nx, arma::uword ny);
		void set_xz_plane(double ellx, double ellz, double xoff, double yoff, double zoff, arma::uword nx, arma::uword nz);
};

#endif