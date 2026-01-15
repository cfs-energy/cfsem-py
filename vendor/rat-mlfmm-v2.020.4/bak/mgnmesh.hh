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

#ifndef MGN_MESH_HH
#define MGN_MESH_HH

#include <armadillo> 
#include <cassert>
#include <cmath> // contains M_PI
#include <algorithm>
#include <memory>

#include "sources.hh"
#include "currentmesh.hh"
#include "currentsurface.hh"
#include "magnetictargets.hh"
#include "settings.hh"

// shared pointer definition
class MgnMesh;
typedef std::shared_ptr<MgnMesh> ShMgnMeshPr;

// hexahedron mesh with pre-set magnetisation
// wraps coilmesh and coilsurface to model bound currents
class MgnMesh: public Sources{
	// properties
	private:
		// counters
		arma::uword num_surf_ = 0;
		arma::uword num_vol_ = 0;
		
		// wrap source objects
		// volume currents
		ShCurrentMeshPr vol_;

		// surface currents
		ShCurrentSurfacePr surf_;

	// methods
	public:
		// constructor
		MgnMesh();
		
		// factory
		static ShMgnMeshPr create();

		// setting a hexahedronal mesh with volume elements
		void set_mesh(const arma::Mat<double> &Rn, const arma::Mat<arma::uword> &n);
		void calculate_element_volume();

		// method for setting up the surface 
		// mesh based on the volume mesh
		void surf_from_vol();

		// setting
		void set_magnetisation_nodes(const arma::Mat<double> &Mn);

		// coordinates
		arma::Mat<double> get_source_coords() const;
		arma::Mat<double> get_source_coords(const arma::Mat<arma::uword> &indices) const;
		
		// getting information
		arma::Mat<double> get_node_coords() const;
		arma::Mat<double> get_centroids() const;
		arma::Mat<arma::uword> get_vol_elements() const;
		arma::Mat<arma::uword> get_surf_elements() const;
		arma::Mat<arma::uword> get_surf_edges() const;
		arma::Mat<double> get_surf_face_normal() const;

		// source to multipole step
		void setup_source_to_multipole(const arma::Mat<double> &dR, const arma::uword num_exp);
		arma::Mat<std::complex<double> > source_to_multipole(const arma::Row<arma::uword> &indices, const arma::uword num_exp) const;

		// field calculation from specific sources
		void calc_direct(ShTargetsPr &tar) const;
		void calc_direct(ShTargetsPr &tar, const arma::Row<arma::uword> &tidx, const arma::Row<arma::uword> &sidx) const;

		// getting basic information
		arma::uword num_sources() const;
		arma::uword get_num_nodes() const;

		// several build-in basic shapes used for testing the code
		void setup_cylinder(const double Rin, const double Rout, const double height, const double nr, const double nz, const double nl);
		void setup_cube(const double dx, const double dy, const double dz, const arma::uword nx, const arma::uword ny, const arma::uword nz);
};	

#endif
