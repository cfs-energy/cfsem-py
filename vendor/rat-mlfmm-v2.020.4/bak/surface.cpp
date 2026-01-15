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
#include "surface.hh"


// constructor
Surface::Surface(){
	
}

// setting a hexagonal mesh with volume elements
void Surface::set_mesh(
	const arma::Mat<double> &Rn, 
	const arma::Mat<arma::uword> &n){

	// check input
	assert(Rn.n_rows==3); 
	assert(n.n_rows==4);
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

	// get face normal vector
	// this is necessary to keep track of face oriention
	// during transformations
	arma::Mat<double> V0 = Rn_.cols(n_.row(1)) - Rn_.cols(n_.row(0));
	arma::Mat<double> V1 = Rn_.cols(n_.row(3)) - Rn_.cols(n_.row(0));
	arma::Mat<double> V01 = Extra::cross(V0,V1);
	Ne_ = V01.each_row()/Extra::vec_norm(V01);
}

// // set mesh by combining other surface meshes
// void Surface::set_mesh(arma::field<ShSurfacePr> &meshes){
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
// 	n_.set_size(4,num_elements_);
// 	Ne_.set_size(3,num_elements_);

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

// 		// element face normals
// 		Ne_.cols(idx_elements(i),idx_elements(i+1)-1) = meshes(i)->get_face_normal();

// 		// set identifiers
// 		node_id_.cols(idx_nodes(i),idx_nodes(i+1)-1) = meshes(i)->get_node_id() + num_id_;
// 		element_id_.cols(idx_elements(i),idx_elements(i+1)-1) = meshes(i)->get_element_id() + num_id_;
// 		num_id_ += meshes(i)->num_id_;
// 	}
// }

// fix element orientation
// makes elements clockwise with respect to their
// face normals
void Surface::fix_clockwise(){
	// select edges
	const arma::Mat<double> V1 = Rn_.cols(n_.row(1)) - Rn_.cols(n_.row(0));
	const arma::Mat<double> V2 = Rn_.cols(n_.row(2)) - Rn_.cols(n_.row(1));

	// cross product then use facenormal to determine orientation
	const arma::Row<double> d = Extra::dot(Ne_,Extra::cross(V1,V2));

	// return orientation
	const arma::Row<arma::uword> idx = arma::find(d>0).t();

	// fix
	n_.cols(idx) = arma::flipud(n_.cols(idx));
}

// calculate areas and centroids of the elements
void Surface::calculate_element_areas(){
	// check if it is really quad
	assert(n_.n_rows==4);

	// extract vectors that span the face from opposing nodes
	arma::Mat<double> V0 = Rn_.cols(n_.row(1)) - Rn_.cols(n_.row(0));
	arma::Mat<double> V1 = Rn_.cols(n_.row(3)) - Rn_.cols(n_.row(0));
	arma::Mat<double> V2 = Rn_.cols(n_.row(1)) - Rn_.cols(n_.row(2));
	arma::Mat<double> V3 = Rn_.cols(n_.row(3)) - Rn_.cols(n_.row(2));

	arma::Mat<double> V01 = Extra::cross(V0,V1);
	arma::Mat<double> V23 = Extra::cross(V2,V3);

	// get face normal vector
	// Ne_ = V01.each_row()/Extra::vec_norm(V01);

	// calculate face area using two triangles
	Ae_ = Extra::vec_norm(V01)/2 + Extra::vec_norm(V23)/2;

	// allocate centroids
	Re_.set_size(3,num_elements_);

	// walk over elements and calculate centroids
	arma::Col<double>::fixed<2> Rq = {0,0};
	for(arma::uword i=0;i<num_elements_;i++){
		// calculate centroid using quadrilateral coordinates
		Re_.col(i) = Quadrilateral::quad2cart(Rn_.cols(n_.col(i)),Rq);
	}

	// distance to center
	element_radius_.zeros(1,num_elements_);
	for(arma::uword i=0;i<n_.n_rows;i++){
		arma::Mat<double> Rr = (Re_ - Rn_.cols(n_.row(i)));
		element_radius_ = arma::max(element_radius_,
			arma::sqrt(arma::sum(Rr%Rr,0)));
	}

}

// get number of sources (each element has one source)
arma::uword Surface::get_num_nodes() const{
	return num_nodes_;
}

// get number of sources (each element has one source)
arma::uword Surface::get_num_elements() const{
	return num_elements_;
}

// get element volumes
arma::Row<double> Surface::get_area() const{
	// check if volumes were calculated
	assert(!Ae_.is_empty());

	// return element volumes
	return Ae_;
}


// get node coordinates
arma::Mat<double> Surface::get_node_coords() const{
	// check if node coordinates were set
	assert(!Rn_.is_empty());

	// return node coordinates
	return Rn_;
}

// get element node indices
arma::Mat<arma::uword> Surface::get_elements() const{
	// check if elements were set
	assert(!n_.is_empty());

	// return 
	return n_;
}

// getting node longitudinal orientation vectors
arma::Mat<double> Surface::get_node_long_vector() const{
	return Ln_;
}

// getting node normal orientation vectors
arma::Mat<double> Surface::get_node_normal_vector() const{
	return Nn_;
}

// getting node transverse orientation vectors
arma::Mat<double> Surface::get_node_trans_vector() const{
	return Dn_;
}

// get element face normal
arma::Mat<double> Surface::get_face_normal() const{
	return Ne_;
}

// externally set element face normals 
void Surface::set_face_normal(const arma::Mat<double> &Ne){
	Ne_ = Ne;
}

// // getting node identifiers
// arma::Row<arma::uword> Surface::get_node_id() const{
// 	return node_id_;
// }

// // getting element identifiers
// arma::Row<arma::uword> Surface::get_element_id() const{
// 	return element_id_;
// }

// // getting element identifiers
// arma::uword Surface::get_num_id() const{
// 	return num_id_;
// }

// set orientation of the nodes in the mesh
void Surface::set_node_orientation(
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

// get all surface mesh edge elements
arma::Mat<arma::uword> Surface::get_edges() const{
	// get edges from quadrilateral
	arma::Mat<arma::uword>::fixed<4,2> M = Quadrilateral::get_edges();

	// get edges from all quadrilaterals in mesh
	arma::Mat<arma::uword> e = arma::reshape(n_.rows(arma::reshape(M.t(),8,1)),2,M.n_rows*num_elements_);

	// filter unique edges
	// sort each column in e
	arma::Mat<arma::uword> ee(2,num_elements_*4);
	for(arma::uword i=0;i<num_elements_*4;i++){
		ee.col(i) = arma::sort(e.col(i));
	}

	// sort S by rows
	for(arma::uword i=0;i<2;i++){
		arma::Col<arma::uword> idx = arma::stable_sort_index(ee.row(1-i));
		ee = ee.cols(idx); e = e.cols(idx);
	}

	// find duplicates and mark
	arma::Row<arma::uword> duplicates = 
		arma::all(ee.cols(0,num_elements_*4-2)==ee.cols(1,num_elements_*4-1),0);

	// remove marked indices
	e = e.cols(arma::find(duplicates==0));

	// return e
	return e;
}




// basic build in mesh shape cylinder
// for testing purposes
// note that after calling this method
// it is still necessary to calculate element
// volumes to complete setup.
void Surface::setup_xy_plane(double ellx, double elly, 
	double xoff, double yoff, double zoff, 
	arma::uword nx, arma::uword ny){

	// allocate coordinates
	arma::Mat<double> x(ny,nx); arma::Mat<double> y(ny,nx); 
	arma::Mat<double> z(ny,nx,arma::fill::zeros);

	// set coordinates to matrix
	x.each_row() = arma::linspace<arma::Row<double> >(-ellx/2,ellx/2,nx) + xoff;
	y.each_col() = arma::linspace<arma::Col<double> >(-elly/2,elly/2,ny) + yoff;
	z += zoff;
	
	// store coordinates
	arma::Mat<double> Rn(3,nx*ny);
	Rn.row(0) = arma::reshape(x,1,nx*ny);
	Rn.row(1) = arma::reshape(y,1,nx*ny);
	Rn.row(2) = arma::reshape(z,1,nx*ny);

	// create matrix of node numbers
	arma::Mat<arma::uword> M = arma::reshape(
		arma::regspace<arma::Row<arma::uword> >(0,nx*ny-1),ny,nx);

	// number of elements
	arma::uword num_elements = (nx-1)*(ny-1);

	// allocate elements
	arma::Mat<arma::uword> n(4,num_elements);

	// create mesh
	for(arma::uword i=0;i<ny-1;i++){
		for(arma::uword j=0;j<nx-1;j++){
			n(0,i*(nx-1)+j) = M(i,j);
			n(1,i*(nx-1)+j) = M(i+1,j);
			n(2,i*(nx-1)+j) = M(i+1,j+1);
			n(3,i*(nx-1)+j) = M(i,j+1);
		}
	}

	// set mesh
	set_mesh(Rn,n);
}


// set plane coordinates
void Surface::setup_xz_plane(double ellx, double ellz, 
	double xoff, double yoff, double zoff, 
	arma::uword nx, arma::uword nz){

	// allocate coordinates
	arma::Mat<double> x(nz,nx); arma::Mat<double> z(nz,nx); 
	arma::Mat<double> y(nz,nx,arma::fill::zeros);

	// set coordinates to matrix
	x.each_row() = arma::linspace<arma::Row<double> >(-ellx/2,ellx/2,nx) + xoff;
	y += yoff;
	z.each_col() = arma::linspace<arma::Col<double> >(-ellz/2,ellz/2,nz) + zoff;
	
	// store coordinates
	arma::Mat<double> Rn(3,nx*nz);
	Rn.row(0) = arma::reshape(x,1,nx*nz);
	Rn.row(1) = arma::reshape(y,1,nx*nz);
	Rn.row(2) = arma::reshape(z,1,nx*nz);

	// create matrix of node numbers
	arma::Mat<arma::uword> M = arma::reshape(
		arma::regspace<arma::Row<arma::uword> >(0,nx*nz-1),nz,nx);

	// number of elements
	arma::uword num_elements = (nx-1)*(nz-1);

	// allocate elements
	arma::Mat<arma::uword> n(4,num_elements);

	// create mesh
	for(arma::uword i=0;i<nz-1;i++){
		for(arma::uword j=0;j<nx-1;j++){
			n(0,i*(nx-1)+j) = M(i,j);
			n(1,i*(nx-1)+j) = M(i+1,j);
			n(2,i*(nx-1)+j) = M(i+1,j+1);
			n(3,i*(nx-1)+j) = M(i,j+1);
		}
	}

	// set mesh
	set_mesh(Rn,n);
}

// basic build in mesh shape cylinder
// for testing purposes
// note that after calling this method
// it is still necessary to calculate element
// volumes to complete setup.
void Surface::setup_cylinder_shell(
	const double R, const double height, 
	const double nz, const double nl){

	// create azymuthal coordinates of nodes
	arma::Row<double> theta = arma::linspace<arma::Row<double> >(0,-(1.0-1.0/(nl+1))*2*arma::datum::pi,nl);

	// create axial cooridnates of nodes
	arma::Row<double> z = arma::linspace<arma::Row<double> >(-height/2,height/2,nz);

	// create matrix to hold node coordinates
	arma::Mat<double> xn(nl,nz), yn(nl,nz), zn(nl,nz);

	// build node coordinates in two dimensions (first block of matrix)
	for(arma::uword i=0;i<nz;i++){
		// generate circles with different radii
		xn.col(i) = R*arma::cos(theta).t();
		yn.col(i) = R*arma::sin(theta).t();
		zn.col(i).fill(z(i));
	}

	// number of nodes
	num_nodes_ = nl*nz;
	
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
	arma::Mat<arma::uword> node_idx = arma::regspace<arma::Mat<arma::uword> >(0,num_nodes_);
	node_idx.reshape(nl,nz);

	// close mesh by setting last row to the first
	node_idx = arma::join_vert(node_idx,node_idx.row(0));

	// get definition of quadrilateral element
	arma::Mat<arma::sword>::fixed<4,2> M = 
		arma::conv_to<arma::Mat<arma::sword> >::from(
		(Quadrilateral::get_corner_nodes()+1)/2);

	// calculate number of elements
	num_elements_ = nl*(nz-1);

	// allocate elements
	n_.set_size(4,num_elements_);

	// walk over corner nodes
	for(arma::uword k=0;k<4;k++){
		// get matrix indexes
		arma::uword idx0 = M(k,0), idx1 = M(k,0)+nl+1-2;
		arma::uword idx2 = M(k,1), idx3 = M(k,1)+nz-2;

		// get node indexes for this corner
		n_.row(k) = arma::reshape(
			node_idx.submat(arma::span(idx0,idx1),
			arma::span(idx2,idx3)),1,nl*(nz-1));
	}

	// check if volume is correct (within 0.1%)
	assert(((height*2*arma::datum::pi*R) - 
		arma::as_scalar(arma::sum(arma::sum(get_area()))))/
		(height*2*arma::datum::pi*R)<1e-3);
}


// moebius strip
// just for fun
void Surface::setup_moebius_strip(
	const arma::uword nt, const double R, const double wstrip, 
	const arma::uword nw, const arma::uword nl){

	// allocate coordinates
	arma::Mat<double> u(nl,nw);
	arma::Mat<double> v(nl,nw);

	// set coordinates
	u.each_col() = arma::linspace<arma::Col<double> >(0,(1.0-1.0/(nl+1))*2*arma::datum::pi,nl);
	v.each_row() = arma::linspace<arma::Row<double> >(-wstrip,wstrip,nw);

	// coordinate transformation
	arma::Mat<double> xn = (R+(v/2)%arma::cos(nt*u/2))%arma::cos(u);
	arma::Mat<double> yn = (R+(v/2)%arma::cos(nt*u/2))%arma::sin(u);
	arma::Mat<double> zn = (v/2)%arma::sin(nt*u/2);

	// longitudinal derivative
	arma::Mat<double> lxn = -0.25*nt*v%arma::cos(u)%arma::sin(u/2) - (R+0.5*v%arma::cos(nt*u/2))%arma::sin(u);
	arma::Mat<double> lyn = (R+0.5*v%arma::cos(nt*u/2))%arma::cos(u) - 0.25*nt*v%arma::sin(nt*u/2)%arma::sin(u);
	arma::Mat<double> lzn = 0.25*nt*v%arma::sin(u/2);

	// transverse derivative
	arma::Mat<double> dxn = 0.5*arma::cos(nt*u/2)%arma::cos(u);
	arma::Mat<double> dyn = 0.5*arma::cos(nt*u/2)%arma::sin(u);
	arma::Mat<double> dzn = 0.5*arma::sin(nt*u/2);

	// number of nodes
	num_nodes_ = nl*nw;
	
	// create node coordinates
	Rn_.set_size(3,num_nodes_);
	Rn_.row(0) = arma::reshape(xn,1,num_nodes_);
	Rn_.row(1) = arma::reshape(yn,1,num_nodes_);
	Rn_.row(2) = arma::reshape(zn,1,num_nodes_);

	// longitudinal vector
	Ln_.set_size(3,num_nodes_);
	Ln_.row(0) = arma::reshape(lxn,1,num_nodes_);
	Ln_.row(1) = arma::reshape(lyn,1,num_nodes_);
	Ln_.row(2) = arma::reshape(lzn,1,num_nodes_);
	Ln_.each_row()/=Extra::vec_norm(Ln_);

	// longitudinal vector
	Dn_.set_size(3,num_nodes_);
	Dn_.row(0) = arma::reshape(dxn,1,num_nodes_);
	Dn_.row(1) = arma::reshape(dyn,1,num_nodes_);
	Dn_.row(2) = arma::reshape(dzn,1,num_nodes_);
	Dn_.each_row()/=Extra::vec_norm(Dn_);

	// face normal vector
	Nn_ = Extra::cross(Ln_,Dn_);

	// create matrix of node indices
	arma::Mat<arma::uword> node_idx = arma::regspace<arma::Mat<arma::uword> >(0,num_nodes_);
	node_idx.reshape(nl,nw);

	// close mesh by setting last row to the first
	if(nt%2==1)node_idx = arma::join_vert(node_idx,arma::fliplr(node_idx.row(0)));
	else node_idx = arma::join_vert(node_idx,node_idx.row(0));

	// get definition of quadrilateral element
	arma::Mat<arma::sword>::fixed<4,2> M = 
		arma::conv_to<arma::Mat<arma::sword> >::from(
		(Quadrilateral::get_corner_nodes()+1)/2);

	// calculate number of elements
	num_elements_ = nl*(nw-1);

	// allocate elements
	n_.set_size(4,num_elements_);

	// walk over corner nodes
	for(arma::uword k=0;k<4;k++){
		// get matrix indexes
		arma::uword idx0 = M(k,0), idx1 = M(k,0)+nl+1-2;
		arma::uword idx2 = M(k,1), idx3 = M(k,1)+nw-2;

		// get node indexes for this corner
		n_.row(k) = arma::reshape(
			node_idx.submat(arma::span(idx0,idx1),
			arma::span(idx2,idx3)),1,nl*(nw-1));
	}
}


// write to gmsh file
void Surface::export_gmsh(ShGmshFilePr gmsh){	
	// write nodes
	gmsh->write_nodes(Rn_);

	// write volume elements
	gmsh->write_elements(n_);
}