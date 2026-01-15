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

#ifndef NON_LINEAR_MESH_HH
#define NON_LINEAR_MESH_HH

#include <armadillo> 
#include <memory>

#include "mesh.hh"
#include "common/extra.hh"
#include "savart.hh"
#include "sources.hh"
#include "mtargets.hh"
#include "common/gauss.hh"
#include "hexahedron.hh"
#include "common/parfor.hh"
#include "stmat.hh"
#include "integration.hh"
#include "common/parfor.hh"
#include "sparse.hh"

// shared pointer definition
typedef std::shared_ptr<class NLMesh> ShNLMeshPr;
typedef arma::field<ShNLMeshPr> ShNLMeshPrList;

// special mesh for the non-linear solver
class NLMesh: public Mesh, public Sources, public MTargets{
	private:
		// target elements (instead of nodes)
		bool target_elements_ = false;

		// number of gauss points used for volume approximation
		bool enforce_symmetry_ = true; // add Mso2ta to Mso2ta.t() and divide by 2
		arma::sword second_num_gauss_ = 3; // put 1 for single integral (make adaptive?)
		arma::sword num_gauss_source_ = 2;
		arma::sword num_gauss_target_ = 2;

		// interpolation arrays for internal field calculation
		arma::Mat<double> Hs_; // calculated magnetic field (for interpolation)
		arma::Mat<double> Ms_; // calculated magnetic field (for interpolation)
		
		// elements
		arma::Mat<double> Me_; // magnetisation vector

		// source to multipole matrices
		StMat_So2Mp_M M_M_;
		arma::Mat<double> dR_;

	public:
		// constructor
		NLMesh();

		// factory
		static ShNLMeshPr create();

		// getting settings
		arma::uword get_num_gauss_source() const;
		arma::uword get_num_gauss_target() const;
		arma::uword get_num_gauss_second() const;

		// set target elements
		void set_target_elements(const bool target_elements);

		// virtual function replacements
		// getting for multipole formation
		arma::Mat<double> get_source_coords() const;
		arma::Mat<double> get_source_coords(const arma::Mat<arma::uword> &indices) const;
		arma::uword num_sources() const;

		// required methods for all targets
		arma::Mat<double> get_target_coords() const;
		arma::Mat<double> get_target_coords(const arma::Row<arma::uword> &indices) const;
		arma::uword num_targets() const;

		// set element magnetisation
		void set_magnetisation_elements(const arma::Mat<double> &Me);
		void set_solution(const arma::Mat<double> &Hs, const arma::Mat<double> &Ms);

		// field calculation for both A and B
		void calc_direct(ShTargetsPr &tar) const;
		void calc_direct(ShTargetsPr &tar, const arma::Row<arma::uword> &tidx, const arma::Row<arma::uword> &sidx) const;
		void calc_field(ShTargetsPr &tar, const arma::Row<arma::uword> &tidx, const arma::Row<arma::uword> &sidx) const;

		// source to multipole methods
		void setup_source_to_multipole(const arma::Mat<double> &dR, const arma::uword num_exp);
		arma::Mat<std::complex<double> > source_to_multipole(const arma::Row<arma::uword> &indices, const arma::uword num_exp) const;
		
		// override localpole to target to include gauss nodes
		void setup_localpole_to_target(const arma::Mat<double> &dR, const arma::uword num_exp);
		void localpole_to_target(const arma::Mat<std::complex<double> > &Lp, const arma::Row<arma::uword> &indices, const arma::uword num_exp);

		// matrix representations of multipole method steps
		arma::SpMat<std::complex<double> > source_to_multipole_matrix(const arma::uword num_exp_lim, const arma::field<arma::Row<arma::uword> > &multipole_index) const;
		arma::SpMat<std::complex<double> > localpole_to_target_matrix(const arma::uword num_exp_lim, const arma::field<arma::Row<arma::uword> > &localpole_index) const;
		void source_to_target_matrix(arma::Col<arma::uword> &mrow_c, arma::Col<arma::uword> &col, arma::Mat<double> &val, arma::Col<double> &vprec, const arma::field<arma::Row<arma::uword> > &s2t_list, const bool use_parallel) const;
};

#endif

