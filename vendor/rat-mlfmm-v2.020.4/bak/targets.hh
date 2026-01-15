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
#ifndef FMM_TARGETS_HH
#define FMM_TARGETS_HH

// general headers
#include <armadillo> 
#include <cassert>
#include <algorithm>
#include <memory>

#include <mutex>
#include <atomic>

// common headers
#include "rat/common/defines.hh"
#include "rat/common/extra.hh"

// mlfmm headers
#include "stmat.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// shared pointer definition
	typedef std::shared_ptr<class Targets> ShTargetsPr;
	typedef arma::field<ShTargetsPr> ShTargetsPrList;

	// target points for mlfmm calculation
	// base class partially virtual
	class Targets{
		// methods
		public: 	
			// virtual destructor (obligatory)
			virtual ~Targets(){};
			
			// all source types should have these methods
			// in order to commmunicate with MLFMM
			virtual arma::Mat<double> get_target_coords() const = 0;
			virtual arma::Mat<double> get_target_coords(const arma::Row<arma::uword> &indices) const = 0;

			// sort sources
			virtual void sort(const arma::Row<arma::uword> &sort_idx) = 0;
			virtual void unsort(const arma::Row<arma::uword> &sort_idx) = 0;

			// localpole to target virtual functions
			virtual void setup_localpole_to_target(const arma::Mat<double> &dR, const arma::uword num_exp) = 0;
			virtual void localpole_to_target(const arma::Mat<std::complex<double> > &Lp, const arma::Row<arma::uword> &first_target, const arma::Row<arma::uword> &last_target, const arma::uword num_dim, const arma::uword num_exp) = 0;

			// get number of dimensions of specific type
			virtual arma::uword get_target_num_dim(const std::string &field_type) const = 0;

			// getting basic information
			virtual arma::uword num_targets() const = 0;

			// setting of calculated field
			virtual void set_field(const std::string &type, const arma::Mat<double> &Mset) = 0;
			virtual void add_field(const std::string &type, const arma::Mat<double> &Madd, const bool with_lock) = 0;
			virtual void set_field(const std::string &type, const arma::Row<arma::uword> &indices, const arma::Mat<double> &Mset) = 0;
			virtual void add_field(const std::string &type, const arma::Row<arma::uword> &indices, const arma::Mat<double> &Madd, const bool with_lock) = 0;
			
			// allocation function
			virtual void allocate() = 0;

			// getting basic information
			virtual bool has(const std::string &type) const = 0;

	};

}}

#endif

