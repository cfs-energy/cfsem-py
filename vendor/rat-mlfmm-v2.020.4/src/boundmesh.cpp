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

// include header file
#include "boundmesh.hh"

// code specific to Rat
namespace rat{namespace fmm{
	
	// constructors
	BoundMesh::BoundMesh(){

	}

	// constructors
	BoundMesh::BoundMesh(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Mat<fltp> &Mn){
		set_mesh(Rn,n,Mn);
	}

	// factory
	ShBoundMeshPr BoundMesh::create(){
		return std::make_shared<BoundMesh>();
	}

	// factory
	ShBoundMeshPr BoundMesh::create(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Mat<fltp> &Mn){
		return std::make_shared<BoundMesh>(Rn,n,Mn);
	}

	// set number of gauss points
	void BoundMesh::set_num_gauss(
		const arma::sword num_gauss_volume, 
		const arma::sword num_gauss_surface){
		num_gauss_volume_ = num_gauss_volume;
		num_gauss_surface_ = num_gauss_surface;
	}

	// set mesh
	void BoundMesh::set_mesh(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Mat<fltp> &Mn){

		// create a current mesh with "bound current"
		current_mesh_ = CurrentMesh::create();
		current_mesh_->set_mesh(Rn,n);
		current_mesh_->set_magnetisation_nodes(Mn);

		// extract surface
		const arma::Mat<arma::uword> s = rat::cmn::Hexahedron::extract_surface(n);

		// find indexes of unique nodes
		const arma::Row<arma::uword> idx_node_surface = 
			arma::unique<arma::Row<arma::uword> >(arma::reshape(s,1,s.n_elem));

		// figure out which nodes to drop as 
		// they are no longer contained in the mesh
		arma::Row<arma::uword> node_indexing(Rn.n_cols,arma::fill::zeros);
		node_indexing.cols(idx_node_surface) = 
			arma::regspace<arma::Row<arma::uword> >(0,idx_node_surface.n_elem-1);

		// re-index connectivity
		const arma::Mat<arma::uword> ss = arma::reshape(
			node_indexing(arma::reshape(s,1,s.n_elem)),s.n_rows,s.n_cols);

		// create surface mesh
		current_surface_ = fmm::CurrentSurface::create();
		current_surface_->set_mesh(Rn.cols(idx_node_surface),ss);
		current_surface_->set_magnetisation_nodes(Mn.cols(idx_node_surface));

		// set number of gauss points
		current_mesh_->set_num_gauss(num_gauss_volume_);
		// current_surface_->set_num_gauss(num_gauss_surface_);

		// create interpolation for magnetisation
		interp_ = fmm::Interp::create();
		interp_->set_mesh(Rn,n);
		interp_->set_interpolation_values('M',Mn);

		// set to multi-sources
		add_sources(current_surface_);
		add_sources(current_mesh_);
		add_sources(interp_);
	}

}}

