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

#ifndef COIL_MESH_HH
#define COIL_MESH_HH

#include <armadillo> 
#include <cassert>
#include <cmath> // contains M_PI
#include <algorithm>
#include <iostream>
#include <fstream>
#include <memory>
#include <iomanip>

#include "mesh.hh"
#include "common/extra.hh"
#include "savart.hh"
#include "sources.hh"
#include "mtargets.hh"
#include "common/gauss.hh"
#include "hexahedron.hh"
#include "common/parfor.hh"
#include "stmat.hh"
#include "gmshfile.hh"

#include "currentsurface.hh"
#include "currentsources.hh"

// shared pointer definition
typedef std::shared_ptr<class CurrentMesh> ShCurrentMeshPr;
typedef arma::field<ShCurrentMeshPr> ShCurrentMeshPrList;

// hexahedron mesh with pre-set current density
class CurrentMesh: public Mesh, public Sources, public MTargets{
	// properties
	private:
		// keep track of whether a current was set
		bool nodal_currents_ = false;
		bool elemental_currents_ = false;

		// number of radii at which elements 
		// are no longer considered point sources
		double num_dist_ = 1.5;

		// number of gauss points used for volume approximation
		arma::sword num_gauss_ = 2;

		// calculated gauss point abscissae and weights
		arma::Row<double> xg_;
		arma::Row<double> wg_;

		// current density vector at nodes or elements in [A/m^2]
		arma::Mat<double> Jn_; // current is usually defined at nodes
		arma::Mat<double> Je_;

		// source to multipole matrices
		StMat_So2Mp_J M_J_;
		arma::Mat<double> dRs2m_;

		// // localpole to target matrices
		// StMat_Lp2Ta_A M_A_; // for vector potential
		// StMat_Lp2Ta_H M_H_; // for magnetic field
		// arma::Mat<double> dRl2t_;

	// source related methods
	public:
		// constructor
		CurrentMesh();

		// factory
		static ShCurrentMeshPr create();

		// set mesh
		// void set_mesh(const arma::Mat<double> &Rn, const arma::Mat<arma::uword> &n);

		// get field
		arma::Mat<double> get_field(const std::string &type, const arma::uword num_dim) const;

		// setting the current density
		// void allocate_current();
		void set_current_density_elements(const arma::Mat<double> &Je);
		void set_current_density_nodes(const arma::Mat<double> &Jn);
		void set_magnetisation_nodes(const arma::Mat<double> &Mn);

		// getting current density
		arma::Mat<double> get_current_density_nodes() const;
		arma::Mat<double> get_current_density_elements() const;
		arma::Mat<double> get_combined_current_density() const;

		// get type of current density set
		// check whether this mesh has nodal currents
		bool has_nodal_currents() const;
		bool has_elemental_currents() const;
		bool has_current() const;

		// calculation accuracy settings
		void set_num_gauss(const arma::sword num_gauss);
		void set_num_dist(const arma::uword num_dist);
		void setup_gauss_points();

		// virtual function replacements
		// getting for multipole formation
		arma::Mat<double> get_source_coords() const;
		arma::Mat<double> get_source_coords(const arma::Mat<arma::uword> &indices) const;
		arma::uword num_sources() const;

		// required methods for all targets
		arma::Mat<double> get_target_coords() const;
		arma::Mat<double> get_target_coords(const arma::Row<arma::uword> &indices) const;
		arma::uword num_targets() const;

		// source to multipole
		void setup_source_to_multipole(const arma::Mat<double> &dR, const arma::uword num_exp);
		arma::Mat<std::complex<double> > source_to_multipole(const arma::Row<arma::uword> &indices, const arma::uword num_exp) const;

		// override rotation function to include 
		// rotation of current density vector
		void apply_rotation(const double phi, const double theta, const double psi);

		// field calculation for both A and B
		void calc_direct(ShTargetsPr &tar) const;
		void calc_direct(ShTargetsPr &tar, const arma::Row<arma::uword> &tidx, const arma::Row<arma::uword> &sidx) const;
		void calc_field(ShTargetsPr &tar, const arma::Row<arma::uword> &tidx, const arma::Row<arma::uword> &sidx) const;

		// extract surface mesh
		ShMTSurfacePr create_surface() const;

		// extract point sources
		ShCurrentSourcesPr create_current_sources() const;

		// export to gmsh format
		virtual void export_gmsh(ShGmshFilePr gmsh);
		virtual void export_force_density(std::ofstream &fid) const;

		// integrate forces and torques
		arma::Col<double>::fixed<3> integrate_forces() const;
		arma::Col<double>::fixed<3> integrate_torque(const arma::Col<double>::fixed<3> &R) const;
};

#endif
