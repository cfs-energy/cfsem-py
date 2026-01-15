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

// include guard
#ifndef FMM_MAGNETIC_TARGETS_HH
#define FMM_MAGNETIC_TARGETS_HH

// general headers
#include <armadillo> 
#include <cassert>
#include <algorithm>
#include <memory>

// parallel processing headers
#include <mutex>
#include <atomic>

// common headers
#include "rat/common/extra.hh"

// mlfmm headers
#include "targetpoints.hh"
#include "fmmmat.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// shared pointer definition
	typedef std::shared_ptr<class LpTargets> ShLpTargetsPr;

	// target points for mlfmm calculation
	// base class partially virtual
	class LpTargets: public TargetPoints{
		// properties
		protected:
			// multipole operation matrices
			arma::Mat<fltp> dR_;

		// methods
		public:
			// constructor
			LpTargets();
			explicit LpTargets(const arma::Mat<fltp> &Rt);
			
			// factory
			static ShLpTargetsPr create();
			static ShLpTargetsPr create(const arma::Mat<fltp> &Rt);

			// localpole to target
			void setup_localpole_to_target(const arma::Mat<fltp> &dR, const ShSettingsPr &stngs) override;
			void localpole_to_target(const arma::Mat<std::complex<fltp> > &Lp, const arma::Row<arma::uword> &first_target, const arma::Row<arma::uword> &last_target, const arma::uword num_dim, const ShSettingsPr &stngs) override;
	};

}}

#endif

