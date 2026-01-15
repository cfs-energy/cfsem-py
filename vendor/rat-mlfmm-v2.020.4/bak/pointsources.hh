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

#ifndef POINT_SOURCES_HH
#define POINT_SOURCES_HH

#include <armadillo> 
#include <cassert>
#include <cmath> // contains M_PI
#include <algorithm>
#include <memory>

#include "common/extra.hh"
#include "sources.hh"
#include "targets.hh"
#include "savart.hh"
#include "stmat.hh"

// shared pointer definition
typedef std::shared_ptr<class PointSources> ShPointSourcesPr;

// collection of line sources
// is derived from the sources class (still mostly virtual)
class PointSources: public Sources{
	// properties
	protected:
		// number of sources
		arma::uword num_sources_ = 0;

		// coordinates in [m]
		arma::Mat<double> Rs_;
	
	// methods
	public:
		// constructor
		PointSources();

		// set points by combining other pointsources
		void set_points(arma::field<ShPointSourcesPr> &srcs);

		// set coordinates and currents of elements
		void set_coords(const arma::Mat<double> &R);

		// getting source coordinates
		arma::Mat<double> get_source_coords() const;
		arma::Mat<double> get_source_coords(const arma::Mat<arma::uword> &indices) const;
		
		// count number of sources
		arma::uword num_sources() const;
};

#endif
