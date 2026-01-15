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

#ifndef MESH_HH
#define MESH_HH

#include <armadillo> 
#include <cassert>
#include <cmath> // contains M_PI
#include <algorithm>

#include "common/extra.hh"
#include "common/gauss.hh"
#include "hexahedron.hh"
#include "tetrahedron.hh"
// #include "sources.hh"
#include "surface.hh"
#include "settings.hh"
#include "gmshfile.hh"
// #include "targets.hh"

// shared pointer definition
typedef std::shared_ptr<class Mesh> ShMeshPr;
typedef arma::field<ShMeshPr> ShMeshPrList;

// hexahedron mesh with volume elements
// is derived from the sources class (still partially virtual)
class Mesh{
	// properties
	protected:
		// node locations
		arma::uword num_nodes_;
		arma::Mat<double> Rn_;

		// user identifier to keep track of nodes and elements
		// arma::Row<arma::uword> node_id_;
		// arma::Row<arma::uword> element_id_; 
		// arma::uword num_id_ = 0;

		// node orientation 
		// (note that description assumes rectangular cable)
		arma::Mat<double> Nn_; // perpendicular to wide face
		arma::Mat<double> Ln_; // along cable
		arma::Mat<double> Dn_; // parallel to wide face

		// element definition
		arma::uword num_elements_;
		arma::Mat<arma::uword> n_;
		
		// calculated element data
		arma::Mat<double> Re_; // element centroids
		arma::Row<double> element_radius_;

		// calculated element area or volume 
		// in [m^2] or [m^3] respectively
		arma::Row<double> Ve_;

	// methods
	public:
		// constructors
		Mesh();
		
		// setting a hexahedronal mesh with volume elements
		virtual void set_mesh(const arma::Mat<double> &Rn, const arma::Mat<arma::uword> &n);
		// void set_mesh(ShMeshPrList &meshes);

		// externally set node orientation vectors
		virtual void set_node_orientation(const arma::Mat<double> &Ln, const arma::Mat<double> &Nn, const arma::Mat<double> &Dn);

		// calculation
		void calculate_element_volume();
		// void setup_volume_grid();

		// getting
		arma::Row<double> get_volume() const;
		
		// virtual function replacements
		// getting for multipole formation
		// arma::Mat<double> get_source_coords() const;
		// arma::Mat<double> get_source_coords(const arma::Mat<arma::uword> &indices) const;
		// arma::uword num_sources() const;

		// required methods for all targets
		// arma::Mat<double> get_target_coords() const;
		// arma::Mat<double> get_target_coords(const arma::Row<arma::uword> &indices) const;
		// arma::uword num_targets() const;

		// counters
		arma::uword get_num_nodes() const;
		arma::uword get_num_elements() const;

		// getting node coordinates
		arma::Mat<double> get_node_coords() const;
		arma::Mat<arma::uword> get_elements() const;
		arma::Mat<arma::uword> get_edges() const;
		
		// getting node orientation vectors
		arma::Mat<double> get_node_long_vector() const;
		arma::Mat<double> get_node_normal_vector() const;
		arma::Mat<double> get_node_trans_vector() const;

		// // getting identifiers
		// arma::Row<arma::uword> get_node_id() const;
		// arma::Row<arma::uword> get_element_id() const;
		// arma::uword get_num_id() const;

		// coordinate transformations
		virtual void apply_translation(const arma::Col<double>::fixed<3> &dR);
		virtual void apply_translation(const double dx, const double dy, const double dz);
		virtual void apply_rotation(const double phi, const double theta, const double psi);
		void fix_clockwise();

		// several build-in basic shapes used for testing the code
		void setup_cylinder(const double Rin, const double Rout, const double height, const double nr, const double nz, const double nl);
		void setup_cube(const double dx, const double dy, const double dz, const arma::uword nx, const arma::uword ny, const arma::uword nz);

		// export to gmsh
		virtual void export_gmsh(ShGmshFilePr gmsh);

		// extract surface mesh
		arma::Mat<arma::uword> get_surface_elements() const;
};

#endif
