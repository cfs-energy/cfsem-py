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

// include header
#include "mgnmesh.hh"

// constructor
MgnMesh::MgnMesh(){
	// number of dimensions
	set_src_num_dim(3);

	// create objects
	vol_ = CurrentMesh::create();
	surf_ = CurrentSurface::create();
}

// factory
ShMgnMeshPr MgnMesh::create(){
	//return ShMgnMeshPr(new MgnMesh);
	return std::make_shared<MgnMesh>();
}

// setting a hexahedron mesh with volume elements
void MgnMesh::set_mesh(
	const arma::Mat<double> &Rn, 
	const arma::Mat<arma::uword> &n){
	
	// forward to volume
	vol_->set_mesh(Rn,n);
	
	// count number of nodes
	num_vol_ = n.n_cols;

	// setup surface
	surf_from_vol();
}
		
// cylinder
void MgnMesh::setup_cylinder(const double Rin, const double Rout, const double height, const double nr, const double nz, const double nl){
	// forward to volume mesh
	vol_->setup_cylinder(Rin, Rout, height, nr, nz, nl);
	
	// get number of elements
	num_vol_ = vol_->get_num_elements();

	// setup surface
	surf_from_vol();
}

// cube
void MgnMesh::setup_cube(const double dx, const double dy, const double dz, const arma::uword nx, const arma::uword ny, const arma::uword nz){
	// forward to volume mesh
	vol_->setup_cube(dx, dy, dz, nx, ny, nz);

	// get number of elements
	num_vol_ = vol_->get_num_elements();

	// setup surface
	surf_from_vol();
}

// setup surface from volume
void MgnMesh::surf_from_vol(){
	// get nodes and surface elements from volume
	arma::Mat<double> Rn = vol_->get_node_coords();
	arma::Mat<arma::uword> sn = vol_->get_surface_elements();

	// set surface mesh
	surf_->set_mesh(Rn,sn);

	// count number of nodes
	num_surf_ = sn.n_cols;
}

// setting
void MgnMesh::set_magnetisation_nodes(
	const arma::Mat<double> &Mn){
	// forward to volume
	vol_->set_magnetisation_nodes(Mn);

	// forward to surface
	surf_->set_magnetisation_nodes(Mn);
}

// coordinates
arma::Mat<double> MgnMesh::get_node_coords() const{
	// return node coordinates stored in volume
	return vol_->get_node_coords();
}

// coordinates
arma::Mat<arma::uword> MgnMesh::get_vol_elements() const{
	// return node coordinates stored in volume
	return vol_->get_elements();
}

// coordinates
arma::Mat<arma::uword> MgnMesh::get_surf_elements() const{
	// return node coordinates stored in volume
	return surf_->get_elements();
}

// edges from surface
arma::Mat<arma::uword> MgnMesh::get_surf_edges() const{
	return surf_->get_edges();
}

	// finalise setup by setting up volumes
void MgnMesh::calculate_element_volume(){
	surf_->calculate_element_areas();
	vol_->calculate_element_volume();
}

// coordinates
arma::Mat<double> MgnMesh::get_source_coords() const{
	// stick coordinates from volume and surface together
	return arma::join_horiz(vol_->get_source_coords(),surf_->get_source_coords());
}

// get face normals from surface elements
arma::Mat<double> MgnMesh::get_surf_face_normal() const{
	return surf_->get_face_normal();
}

// coordinates
arma::Mat<double> MgnMesh::get_source_coords(
	const arma::Mat<arma::uword> &indices) const{

	// check input
	assert(arma::as_scalar(arma::all(indices<(num_vol_+num_surf_))));

	// find coordinates from volume
	arma::Col<arma::uword> idx_vol = arma::find(indices<num_vol_);
	arma::Col<arma::uword> idx_surf = arma::find(indices>=num_vol_);

	// allocate output
	arma::Mat<double> R(3,indices.n_elem);

	// fill output
	R.cols(idx_vol) = vol_->get_source_coords(indices.cols(idx_vol));
	R.cols(idx_surf) = surf_->get_source_coords(indices.cols(idx_surf)-num_vol_);

	// return coordinates
	return R;
}

// get element centroids
arma::Mat<double> MgnMesh::get_centroids() const{
	return vol_->get_source_coords();
}

// source to multipole step setup
void MgnMesh::setup_source_to_multipole(
	const arma::Mat<double> &dR,
	const arma::uword num_exp){
	
	// check size
	assert(dR.n_cols==num_vol_+num_surf_);

	// forward request
	vol_->setup_source_to_multipole(dR.cols(0,num_vol_-1), num_exp);
	surf_->setup_source_to_multipole(dR.cols(num_vol_,num_vol_+num_surf_-1),num_exp);
}


		
// source to multipole step
arma::Mat<std::complex<double> > MgnMesh::source_to_multipole(
	const arma::Row<arma::uword> &indices, const arma::uword num_exp) const{

	// check input
	assert(arma::all(indices<(num_vol_+num_surf_)));

	// find coordinates from volume
	arma::Col<arma::uword> idx_vol = arma::find(indices<num_vol_);
	arma::Col<arma::uword> idx_surf = arma::find(indices>=num_vol_);

	// allocate output multipole
	arma::Mat<std::complex<double> > Mp(Extra::polesize(num_exp),src_num_dim_,arma::fill::zeros);

	// forward calculation
	if(!idx_vol.is_empty())Mp += vol_->source_to_multipole(indices.cols(idx_vol), num_exp);
	if(!idx_surf.is_empty())Mp += surf_->source_to_multipole(indices.cols(idx_surf)-num_vol_, num_exp);

	// return multipole
	return Mp;
}

// field calculation from specific sources
void MgnMesh::calc_direct(ShTargetsPr &tar) const{
	// forward calculation
	vol_->calc_direct(tar);
	surf_->calc_direct(tar);
}

// field calculation from specific sources
void MgnMesh::calc_direct(ShTargetsPr &tar, 
	const arma::Row<arma::uword> &tidx, 
	const arma::Row<arma::uword> &sidx) const{

	// check input
	assert(arma::all(sidx<(num_vol_+num_surf_)));

	// find coordinates from volume
	arma::Col<arma::uword> idx_vol = arma::find(sidx<num_vol_);
	arma::Col<arma::uword> idx_surf = arma::find(sidx>=num_vol_);

	// forward calculation
	if(!idx_vol.is_empty())vol_->calc_direct(tar, tidx, sidx.cols(idx_vol));
	if(!idx_surf.is_empty())surf_->calc_direct(tar, tidx, sidx.cols(idx_surf)-num_vol_);
}

// getting basic information
arma::uword MgnMesh::num_sources() const{
	return num_vol_ + num_surf_;
}

// number of dimensions
arma::uword MgnMesh::get_num_nodes() const{
	return vol_->get_num_nodes();
}


		