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
#include "mesh.hh"

// default constructor
// default is hex
Mesh::Mesh(){
	
}

// set mesh
void Mesh::set_mesh(
	const arma::Mat<double> &Rn, 
	const arma::Mat<arma::uword> &n){

	// check input
	assert(Rn.n_rows==3); 
	assert(n.n_rows==8);
	if(!n.is_empty())
		assert(arma::as_scalar(arma::max(arma::max(n)))<Rn.n_cols);

	// set supplied values
	Rn_ = Rn; n_ = n;

	// get number of supplied nodes
	num_nodes_ = Rn_.n_cols;
	num_elements_ = n_.n_cols;
	
	// // set default id
	// node_id_.zeros(num_nodes_);
	// element_id_.zeros(num_elements_);
	// num_id_ = 1;
}

// // set mesh by merging other meshes
// void Mesh::set_mesh(ShMeshPrList &meshes){
// 	// get number of input meshes
// 	arma::uword num_meshes = meshes.n_elem;

// 	// allocate information arrays
// 	arma::Row<arma::uword> num_nodes_each(num_meshes);
// 	arma::Row<arma::uword> num_elements_each(num_meshes);

// 	// acquire data from each mesh
// 	for(arma::uword i=0;i<num_meshes;i++){
// 		// get numberof nodes
// 		num_nodes_each(i) = meshes(i)->get_num_nodes();
// 		num_elements_each(i) = meshes(i)->get_num_elements();
// 	}

// 	// get number of nodes
// 	num_nodes_ = arma::sum(num_nodes_each);
// 	num_elements_ = arma::sum(num_elements_each);

// 	// create node indexing array
// 	arma::Row<arma::uword> idx_nodes(num_meshes+1); idx_nodes(0) = 0;
// 	idx_nodes.cols(1,num_meshes) = arma::cumsum(num_nodes_each,1);

// 	// create element indexing array
// 	arma::Row<arma::uword> idx_elements(num_meshes+1); idx_elements(0) = 0;
// 	idx_elements.cols(1,num_meshes) = arma::cumsum(num_elements_each,1);

// 	// allocate node coordinates
// 	Rn_.set_size(3,num_nodes_); Ln_.set_size(3,num_nodes_);
// 	Dn_.set_size(3,num_nodes_); Nn_.set_size(3,num_nodes_);

// 	// allocate elements
// 	n_.set_size(8,num_elements_);

// 	// allocate identifiers
// 	node_id_.set_size(num_nodes_);
// 	element_id_.set_size(num_elements_);
// 	num_id_ = 0;

// 	// get coordinates
// 	for(arma::uword i=0;i<num_meshes;i++){
// 		// coordinates
// 		Rn_.cols(idx_nodes(i),idx_nodes(i+1)-1) = meshes(i)->get_node_coords();

// 		// orientation vectors
// 		Ln_.cols(idx_nodes(i),idx_nodes(i+1)-1) = meshes(i)->get_node_long_vector();
// 		Dn_.cols(idx_nodes(i),idx_nodes(i+1)-1) = meshes(i)->get_node_trans_vector();
// 		Nn_.cols(idx_nodes(i),idx_nodes(i+1)-1) = meshes(i)->get_node_normal_vector();

// 		// elements connectivity
// 		n_.cols(idx_elements(i),idx_elements(i+1)-1) = meshes(i)->get_elements() + idx_nodes(i);

// 		// set identifiers
// 		node_id_.cols(idx_nodes(i),idx_nodes(i+1)-1) = meshes(i)->get_node_id() + num_id_;
// 		element_id_.cols(idx_elements(i),idx_elements(i+1)-1) = meshes(i)->get_element_id() + num_id_;
// 		num_id_ += meshes(i)->num_id_;
// 	}
// }


// set orientation of the nodes in the mesh
void Mesh::set_node_orientation(
	const arma::Mat<double> &Ln, 
	const arma::Mat<double> &Nn, 
	const arma::Mat<double> &Dn){

	// check input
	assert(Ln.n_rows==3); assert(Ln.n_cols==num_nodes_);
	assert(Nn.n_rows==3); assert(Nn.n_cols==num_nodes_);
	assert(Dn.n_rows==3); assert(Dn.n_cols==num_nodes_);

	// set matrices to self
	Ln_ = Ln; Nn_ = Nn; Dn_ = Dn; 
}

// fix element orientation
void Mesh::fix_clockwise(){
	// find anti-clockwise hexahedra
	arma::Row<arma::uword> idx = arma::find(
		Hexahedron::is_clockwise(Rn_,n_)==0).t();

	// fix
	arma::Row<arma::uword>::fixed<4> face1 = arma::regspace<arma::Row<arma::uword> >(0,3);
	arma::Row<arma::uword>::fixed<4> face2 = arma::regspace<arma::Row<arma::uword> >(4,7);
	n_.submat(face1,idx) = arma::flipud(n_.submat(face1,idx));
	n_.submat(face2,idx) = arma::flipud(n_.submat(face2,idx));
}

// calculate hexahedron volumes
// by splitting it up into five tetrahedrons
// then adds their volumes
void Mesh::calculate_element_volume(){
	// check if it is really quad
	assert(n_.n_rows==8);

	// volume of elements
	Ve_ = Hexahedron::calc_volume(Rn_,n_);

	// allocate element centroids
	Re_.set_size(3,num_elements_);

	// walk over elements
	arma::Col<double>::fixed<3> Rq = {0,0,0};
	for(arma::uword i=0;i<num_elements_;i++){
		// calculate centroid using quadrilateral coordinates
		Re_.col(i) = Hexahedron::quad2cart(Rn_.cols(n_.col(i)),Rq);
	}

	// distance to center
	element_radius_.zeros(1,num_elements_);
	for(arma::uword i=0;i<n_.n_rows;i++){
		arma::Mat<double> Rr = (Re_ - Rn_.cols(n_.row(i)));
		element_radius_ = arma::max(element_radius_,
			arma::sqrt(arma::sum(Rr%Rr,0)));
	}
}

// get element volumes
arma::Row<double> Mesh::get_volume() const{
	// check if volumes were calculated
	assert(!Ve_.is_empty());

	// return element volumes
	return Ve_;
}

// // get element centroids
// arma::Mat<double> Mesh::get_source_coords() const{
// 	// check if coordinates were set
// 	assert(!Re_.is_empty());

// 	// return element centroids
// 	return Re_;
// }

// // get element centroids of specific elements
// arma::Mat<double> Mesh::get_source_coords(
// 	const arma::Mat<arma::uword> &indices) const{
// 	// check if coordinates were set
// 	assert(!Re_.is_empty());

// 	// return element centroids
// 	return Re_.cols(indices);
// }

// // get number of sources (each element has one source)
// arma::uword Mesh::num_sources() const{
// 	return num_elements_;
// }

// // get target point coordinates
// arma::Mat<double> Mesh::get_target_coords() const{
// 	// allocate
// 	arma::Mat<double> R;

// 	// select element or node coordinates
// 	if(target_elements_)R = Re_; else R = Rn_;

// 	// return coordinates
// 	return R;
// }

// // get target point coordinates
// arma::Mat<double> Mesh::get_target_coords(
// 	const arma::Row<arma::uword> &indices) const{
// 	// allocate
// 	arma::Mat<double> R;

// 	// select element or node coordinates
// 	if(target_elements_)R = Re_.cols(indices); else R = Rn_.cols(indices);

// 	// return coordinates
//  	return R;
// }

// // number of points stored
// arma::uword Mesh::num_targets() const{
// 	// allocate
// 	arma::uword nt;

// 	// select number of elements or nodes
// 	if(target_elements_)nt = num_elements_; else nt = num_nodes_;

// 	// return number of targets
// 	return nt;
// }


// get number of sources (each element has one source)
arma::uword Mesh::get_num_nodes() const{
	return num_nodes_;
}

// get number of sources (each element has one source)
arma::uword Mesh::get_num_elements() const{
	return num_elements_;
}

// get node coordinates
arma::Mat<double> Mesh::get_node_coords() const{
	// check if node coordinates were set
	assert(!Rn_.is_empty());

	// return node coordinates
	return Rn_;
}

// get element node indices
arma::Mat<arma::uword> Mesh::get_elements() const{
	// check if elements were set
	assert(!n_.is_empty());

	// return 
	return n_;
}


// getting node longitudinal orientation vectors
arma::Mat<double> Mesh::get_node_long_vector() const{
	return Ln_;
}

// getting node normal orientation vectors
arma::Mat<double> Mesh::get_node_normal_vector() const{
	return Nn_;
}

// getting node transverse orientation vectors
arma::Mat<double> Mesh::get_node_trans_vector() const{
	return Dn_;
}

// // getting node identifiers
// arma::Row<arma::uword> Mesh::get_node_id() const{
// 	return node_id_;
// }

// // getting element identifiers
// arma::Row<arma::uword> Mesh::get_element_id() const{
// 	return element_id_;
// }

// // getting element identifiers
// arma::uword Mesh::get_num_id() const{
// 	return num_id_;
// }

// extract surface from hexahedron mesh
// translated from matlab "sq_hexsurface.m"
arma::Mat<arma::uword> Mesh::get_surface_elements() const{
	// make sure it is a hexahedron mesh
	assert(n_.n_rows==8);

	// count number of references of each node
	// arma::Row<arma::uword> nref(num_nodes_,arma::fill::zeros);
	// nref.cols(arma::reshape(n_,1,num_nodes_*8)) += 1;

	// get list of faces for each element
	arma::Mat<arma::uword>::fixed<6,4> M = Hexahedron::get_faces();

	// create list of all faces present in mesh
	arma::Mat<arma::uword> S(4,num_elements_*6);
	for(arma::uword i=0;i<6;i++){
		S.cols(i*num_elements_,(i+1)*num_elements_-1) = n_.rows(M.row(i));
	}

	// sort each column in S
	arma::Mat<arma::uword> Ss(4,num_elements_*6);
	for(arma::uword i=0;i<num_elements_*6;i++){
		Ss.col(i) = arma::sort(S.col(i));
	}

	// sort S by rows
	for(arma::uword i=0;i<4;i++){
		arma::Col<arma::uword> idx = arma::stable_sort_index(Ss.row(3-i));
		Ss = Ss.cols(idx); S = S.cols(idx);
	}

	// find duplicates and mark
	arma::Row<arma::uword> duplicates = 
		arma::all(Ss.cols(0,num_elements_*6-2)==Ss.cols(1,num_elements_*6-1),0);

	// extend duplicate list to contain first and second
	arma::Row<arma::uword> extended(num_elements_*6,arma::fill::zeros);
	extended.head(num_elements_*6-1) += duplicates;
	extended.tail(num_elements_*6-1) += duplicates;

	// remove marked indices
	S = S.cols(arma::find(extended==0));

	// return
	return S;
}

// write to gmsh file
void Mesh::export_gmsh(ShGmshFilePr gmsh){	
	// write nodes
	gmsh->write_nodes(Rn_);

	// extract surface elements
	arma::Mat<arma::uword> ns = get_surface_elements();

	// write volume elements
	gmsh->write_elements(n_);

	// write surface elements
	// gmsh->write_elements(ns);
}



// OPERATIONS FOR SETTING UP BASIC MESH SHAPES
// ===========================================
// manually offset mesh position by vector dR
void Mesh::apply_translation(const arma::Col<double>::fixed<3> &dR){
	// check if node coordinates were set
	assert(!Rn_.is_empty());

	// apply translation
	Rn_.each_col() += dR;
}

// translate mesh by vector components
void Mesh::apply_translation(const double dx, const double dy, const double dz){
	// create vector
	arma::Col<double>::fixed<3> dR = {dx,dy,dz};

	// call overloaded method
	apply_translation(dR);
}

// manually rotate mesh arround origin
// note that this does not rotate the current density 
// when this class is used as a superclass for coil
void Mesh::apply_rotation(const double phi, const double theta, const double psi){
	// check if node coordinates were set
	assert(!Rn_.is_empty()); assert(!Nn_.is_empty());
	assert(!Ln_.is_empty()); assert(!Dn_.is_empty());

	// build rotation matrix
	arma::Mat<double>::fixed<3,3> A = Extra::create_rotation_matrix(phi,theta,psi);

	// rotation by matrix vector product
	Rn_ = A*Rn_; Nn_ = A*Nn_; Ln_ = A*Ln_; Dn_ = A*Dn_;
}

// basic build in mesh shape cylinder
// for testing purposes
// note that after calling this method
// it is still necessary to calculate element
// volumes to complete setup.
void Mesh::setup_cylinder(
	const double Rin, const double Rout, 
	const double height, const double nr, 
	const double nz, const double nl){

	// create azymuthal coordinates of nodes
	arma::Row<double> theta = arma::linspace<arma::Row<double> >(0,-(1.0-1.0/nl)*2*arma::datum::pi,nl);

	// create radial coordinates of nodes
	arma::Row<double> rho = arma::linspace<arma::Row<double> >(Rin,Rout,nr);

	// create axial cooridnates of nodes
	arma::Row<double> z = arma::linspace<arma::Row<double> >(-height/2,height/2,nz);

	// create matrix to hold node coordinates
	arma::Mat<double> xn(nl,nr*nz), yn(nl,nr*nz), zn(nl,nr*nz);

	// build node coordinates in two dimensions (first block of matrix)
	for(arma::uword i=0;i<nr;i++){
		// generate circles with different radii
		xn.col(i) = rho(i)*arma::cos(theta).t();
		yn.col(i) = rho(i)*arma::sin(theta).t();
		zn.col(i).fill(z(0));
	}

	// extrude to other axial planes
	for(arma::uword j=1;j<nz;j++){
		// copy coordinates from ground plane
		xn.cols(j*nr,(j+1)*nr-1) = xn.cols(0,nr-1);
		yn.cols(j*nr,(j+1)*nr-1) = yn.cols(0,nr-1);
		zn.cols(j*nr,(j+1)*nr-1).fill(z(j));
	}

	// number of nodes
	num_nodes_ = nr*nl*nz;
	
	// create node coordinates
	Rn_.set_size(3,num_nodes_);
	Rn_.row(0) = arma::reshape(xn,1,num_nodes_);
	Rn_.row(1) = arma::reshape(yn,1,num_nodes_);
	Rn_.row(2) = arma::reshape(zn,1,num_nodes_);

	// node orientation	
	Nn_ = Rn_.each_row()/Extra::vec_norm(Rn_);
	Dn_.zeros(3,num_nodes_); Dn_.row(2).fill(1.0);
	Ln_ = Extra::cross(Nn_,Dn_);

	// create matrix of node indices
	arma::Mat<arma::uword> node_idx = arma::regspace<arma::Mat<arma::uword> >(0,num_nodes_-1);
	node_idx.reshape(nl,nr*nz);

	// close mesh by setting last row to the first
	node_idx = arma::join_vert(node_idx,node_idx.row(0));

	// get definition of hexahedron element
	arma::Mat<arma::sword>::fixed<8,3> M = 
		arma::conv_to<arma::Mat<arma::sword> >::from(
		(Hexahedron::get_corner_nodes()+1)/2);

	// calculate number of elements
	num_elements_ = nl*(nr-1)*(nz-1);

	// allocate elements
	n_.set_size(8,num_elements_);

	// create elements between the nodes	
	for(arma::uword j=0;j<nz-1;j++){
		// walk over corner nodes
		for(arma::uword k=0;k<8;k++){
			// get matrix indexes
			arma::uword idx0 = M(k,0), idx1 = M(k,0)+nl+1-2;
			arma::uword idx2 = (j+M(k,2))*nr+M(k,1), idx3 = (j+M(k,2)+1)*nr+M(k,1)-2;
			arma::uword idx4 = j*nl*(nr-1), idx5 = (j+1)*nl*(nr-1)-1;

			// get node indexes for this corner
			n_.submat(arma::span(k,k),arma::span(idx4,idx5)) =
				arma::reshape(node_idx.submat(arma::span(idx0,idx1),
					arma::span(idx2,idx3)),1,nl*(nr-1));
		}
	}
}

// basic build in mesh shape cube
void Mesh::setup_cube(
	const double dx, const double dy, 
	const double dz, const arma::uword nx, 
	const arma::uword ny, const arma::uword nz){

	// create nodes
	arma::Mat<double> xn(nx,ny*nz);
	arma::Mat<double> yn(nx,ny*nz);
	arma::Mat<double> zn(nx,ny*nz);

	// fill y
	xn.each_col() = arma::linspace<arma::Col<double> >(-dx/2,dx/2,nx);
	arma::Row<double> za = arma::linspace<arma::Row<double> >(-dz/2,dz/2,nz);

	// walk over levels
	for(arma::uword i=0;i<nz;i++){
		arma::Mat<double> ysub(nx,ny);
		ysub.each_row() = arma::linspace<arma::Row<double> >(-dy/2,dy/2,ny);
		yn.cols(i*ny,(i+1)*ny-1) = ysub;
		zn.cols(i*ny,(i+1)*ny-1).fill(za(i));
	}

	// number of nodes
	num_nodes_ = nx*ny*nz;
	
	// create node coordinates
	Rn_.set_size(3,num_nodes_);
	Rn_.row(0) = arma::reshape(xn,1,num_nodes_);
	Rn_.row(1) = arma::reshape(yn,1,num_nodes_);
	Rn_.row(2) = arma::reshape(zn,1,num_nodes_);

	// node orientation	(arbitrary)
	Nn_.zeros(3,num_nodes_); Nn_.row(0).fill(1.0);
	Dn_.zeros(3,num_nodes_); Dn_.row(2).fill(1.0);
	Ln_.zeros(3,num_nodes_); Ln_.row(1).fill(1.0);

	// create matrix of node indices
	arma::Mat<arma::uword> node_idx = arma::regspace<arma::Mat<arma::uword> >(0,num_nodes_-1);
	node_idx.reshape(nx,ny*nz);

	// get definition of hexahedron element
	arma::Mat<arma::sword>::fixed<8,3> M = 
		arma::conv_to<arma::Mat<arma::sword> >::from(
		(Hexahedron::get_corner_nodes()+1)/2);

	// calculate number of elements
	num_elements_ = (nx-1)*(ny-1)*(nz-1);

	// allocate elements
	n_.set_size(8,num_elements_);

	// create elements between the nodes	
	for(arma::uword j=0;j<nz-1;j++){
		// walk over corner nodes
		for(arma::uword k=0;k<8;k++){
			// get matrix indexes
			arma::uword idx0 = M(k,0), idx1 = M(k,0)+nx-2;
			arma::uword idx2 = (j+M(k,2))*ny+M(k,1), idx3 = (j+M(k,2)+1)*ny+M(k,1)-2;
			arma::uword idx4 = j*(nx-1)*(ny-1), idx5 = (j+1)*(nx-1)*(ny-1)-1;

			// get node indexes for this corner
			n_.submat(arma::span(k,k),arma::span(idx4,idx5)) =
				arma::reshape(node_idx.submat(arma::span(idx0,idx1),
					arma::span(idx2,idx3)),1,(nx-1)*(ny-1));
		}
	}
}