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

#ifndef SOFT_CURRENT_SOURCES_HH
#define SOFT_CURRENT_SOURCES_HH

#include <armadillo> 
#include <cassert>
#include <cmath> // contains M_PI
#include <algorithm>
#include <memory>

#include "common/extra.hh"
#include "targets.hh"
#include "savart.hh"
#include "currentsources.hh"

// shared pointer definition
class SoftCurrentSources;
typedef std::shared_ptr<SoftCurrentSources> ShSoftCurrentSourcesPr;

// collection of line current sources
class SoftCurrentSources: public CurrentSources{
	// properties
	private:
		arma::Row<double> eps_;
		
	// methods
	public:
		// constructor
		SoftCurrentSources();
		
		// factory
		static ShSoftCurrentSourcesPr create();

		// setting of the softening factor
		void set_softening(const arma::Row<double> &eps);

		// direct field calculation for both A and B
		void calc_direct(ShTargetsPr &tar) const;
		void calc_direct(ShTargetsPr &tar, const arma::Row<arma::uword> &tidx, const arma::Row<arma::uword> &sidx) const;
};

#endif
