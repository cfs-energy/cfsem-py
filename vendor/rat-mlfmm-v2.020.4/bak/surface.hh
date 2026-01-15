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

#ifndef SURFACE_HH
#define SURFACE_HH

#include <armadillo> 
#include <cassert>
#include <cmath> // contains M_PI
#include <algorithm>

#include "common/extra.hh"
#include "common/gauss.hh"
#include "quadrilateral.hh"
#include "settings.hh"
#include "gmshfile.hh"

// shared pointer definition
typedef std::shared_ptr<class Surface> ShSurfacePr;

// hexahedron mesh with volume elements
// is derived from the sources class (still partially virtual)
class Surface{
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
		arma::Mat<double> element_radius_;

		// calculated element area in [m^2]
		arma::Row<double> Ae_;

		// element face normals (pointing outward)
		arma::Mat<double> Ne_;

	// methods
	public:
		// constructor
		Surface();

		// setting a hexagonal mesh with volume elements
		virtual void set_mesh(const arma::Mat<double> &Rn, const arma::Mat<arma::uword> &n);
		void set_mesh(arma::field<ShSurfacePr> &meshes);

		// mesh setup functions
		void calculate_element_areas();

		// getting node coordinates
		arma::Mat<double> get_node_coords() const;
		arma::Mat<arma::uword> get_elements() const;
		arma::Mat<arma::uword> get_edges() const;
		arma::Mat<double> get_face_normal() const;

		// getting node orientation
		arma::Mat<double> get_node_long_vector() const;
		arma::Mat<double> get_node_normal_vector() const;
		arma::Mat<double> get_node_trans_vector() const;

		// // getting identifiers
		// arma::Row<arma::uword> get_node_id() const;
		// arma::Row<arma::uword> get_element_id() const;
		// arma::uword get_num_id() const;
		
		// setting node orientation
		void set_node_orientation(const arma::Mat<double> &Ln, const arma::Mat<double> &Nn, const arma::Mat<double> &Dn);
		void set_face_normal(const arma::Mat<double> &Ne);

		// counters
		arma::uword get_num_nodes() const;
		arma::uword get_num_elements() const;

		// getting of areas
		arma::Row<double> get_area() const;

		// fix clockwise
		void fix_clockwise();

		// export to gmsh
		virtual void export_gmsh(ShGmshFilePr gmsh);

		// several build-in basic shapes used for testing the code
		void setup_xy_plane(double ellx, double elly, double xoff, double yoff, double zoff, arma::uword nx, arma::uword ny);
		void setup_xz_plane(double ellx, double ellz, double xoff, double yoff, double zoff, arma::uword nx, arma::uword nz);
		void setup_cylinder_shell(const double R, const double height, const double nz, const double nl);
		void setup_moebius_strip(const arma::uword nt, const double R, const double wstrip, const arma::uword nw, const arma::uword nl);
};

#endif
