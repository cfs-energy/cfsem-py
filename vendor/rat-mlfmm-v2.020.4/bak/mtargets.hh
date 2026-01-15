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

#ifndef MAGNETIC_TARGETS_HH
#define MAGNETIC_TARGETS_HH

#include <armadillo> 
#include <complex>
#include <cassert>

#include "common/extra.hh"
#include "stmat.hh"
#include "targets.hh"

// shared pointer definition
typedef std::shared_ptr<class MTargets> ShMTargetsPr;

// interaction between target points and mlfmm for
// magnetic field and/or vector potential calculation
class MTargets: public Targets{
	// properties
	protected:
		// localpole to target matrices
		StMat_Lp2Ta_A M_A_; // for vector potential
		StMat_Lp2Ta_H M_H_; // for magnetic field
		arma::Mat<double> dRl2t_;

	// methods
	public:
		// constructor
		MTargets();

		// localpole to target
		virtual void setup_localpole_to_target(const arma::Mat<double> &dR, const arma::uword num_exp);
		virtual void localpole_to_target(const arma::Mat<std::complex<double> > &Lp, const arma::Row<arma::uword> &indices, const arma::uword num_exp);

		// getting of calculated field
		virtual arma::Mat<double> get_field(const std::string &type, arma::uword num_dim) const;
};

#endif