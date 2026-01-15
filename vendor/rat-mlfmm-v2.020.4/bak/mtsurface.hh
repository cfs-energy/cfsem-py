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

#ifndef MAGNETIC_TARGET_SURFACE_HH
#define MAGNETIC_TARGET_SURFACE_HH

#include <armadillo> 
#include <cassert>
#include <memory>

#include "surface.hh"
#include "mtargets.hh"

// shared pointer definition
typedef std::shared_ptr<class MTSurface> ShMTSurfacePr;
typedef arma::field<ShMTSurfacePr> ShMTSurfacePrList;

// hexahedron mesh with pre-set current density
class MTSurface: public Surface, public MTargets{
	// properties
	protected:
		// none

	// methods
	public:
		// constructor
		MTSurface();
		
		// factory
		static ShMTSurfacePr create();

		// required methods for all targets
		arma::Mat<double> get_target_coords() const;
		arma::Mat<double> get_target_coords(const arma::Row<arma::uword> &indices) const;
		arma::uword num_targets() const;

		// export to gmsh format
		virtual void export_gmsh(ShGmshFilePr gmsh);
};

#endif





