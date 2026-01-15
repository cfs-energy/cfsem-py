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
#include "currentmesh.hh"

// constructor
CurrentMesh::CurrentMesh(){
	// number of dimensions of this source
	set_src_num_dim(3);

	// calculate gauss points
	setup_gauss_points();
}

// factory
ShCurrentMeshPr CurrentMesh::create(){
	//return ShCurrentMeshPr(new CurrentMesh);
	return std::make_shared<CurrentMesh>();
}

// set mesh
// void CurrentMesh::set_mesh(
// 	const arma::Mat<double> &Rn, 
// 	const arma::Mat<arma::uword> &n){
	
// 	// set mesh
// 	Mesh::set_mesh(Rn,n);
// }

// get element centroids
arma::Mat<double> CurrentMesh::get_source_coords() const{
	// check if coordinates were set
	assert(!Re_.is_empty());

	// return element centroids
	return Re_;
}

// get element centroids of specific elements
arma::Mat<double> CurrentMesh::get_source_coords(
	const arma::Mat<arma::uword> &indices) const{
	// check if coordinates were set
	assert(!Re_.is_empty());

	// return element centroids
	return Re_.cols(indices);
}

// get number of sources (each element has one source)
arma::uword CurrentMesh::num_sources() const{
	return num_elements_;
}

// get target point coordinates
arma::Mat<double> CurrentMesh::get_target_coords() const{
	// return coordinates
	return Rn_;
}

// get target point coordinates
arma::Mat<double> CurrentMesh::get_target_coords(
	const arma::Row<arma::uword> &indices) const{
	// return coordinates
 	return Rn_.cols(indices);
}

// number of points stored
arma::uword CurrentMesh::num_targets() const{
	// return number of targets
	return num_nodes_;
}

// set current density vector of each node
void CurrentMesh::set_current_density_nodes(
	const arma::Mat<double> &Jn){
	
	// check input
	assert(Jn.n_rows==3);
	assert(Jn.n_cols==num_nodes_);

	// set type
	nodal_currents_ = true;

	// set current density
	Jn_ = Jn;
}

// set current density vector of each element
void CurrentMesh::set_current_density_elements(
	const arma::Mat<double> &Je){

	// check input
	assert(Je.n_rows==3);
	assert(Je.n_cols==num_elements_);
	
	// set type
	elemental_currents_ = true;

	// set current density
	Je_ = Je;
}

// check whether this mesh has nodal currents
bool CurrentMesh::has_nodal_currents() const{
	return nodal_currents_;
}

// check whether this mesh has elemental currents
bool CurrentMesh::has_elemental_currents() const{
	return elemental_currents_;
}

// check whether this mesh has currents
bool CurrentMesh::has_current() const{
	return nodal_currents_ || elemental_currents_;
}

// getting current density vector at each node
arma::Mat<double> CurrentMesh::get_current_density_nodes() const{
	return Jn_;
}

// get the current density vector of each element
arma::Mat<double> CurrentMesh::get_current_density_elements() const{
	return Je_;
}

// get current density combined at elements
arma::Mat<double> CurrentMesh::get_combined_current_density() const{
	// allocate
	arma::Mat<double> Je(3,num_elements_,arma::fill::zeros);

	// get current density at elements
	if(nodal_currents_)
		for(arma::uword i=0;i<num_elements_;i++)
			Je.col(i) = arma::mean(Jn_.cols(n_.col(i)),1);
	if(elemental_currents_)
		Je += Je_;

	// return combined current density
	return Je;
}


// manually rotate mesh arround origin 
void CurrentMesh::apply_rotation(const double phi, const double theta, const double psi){
	// check if node coordinates were set
	assert(!Rn_.is_empty()); assert(!Nn_.is_empty()); 
	assert(!Ln_.is_empty()); assert(!Dn_.is_empty()); 
	assert(!Je_.is_empty()); assert(!Jn_.is_empty());

	// build rotation matrix
	arma::Mat<double>::fixed<3,3> A = Extra::create_rotation_matrix(phi,theta,psi);

	// rotation by matrix vector product
	Rn_ = A*Rn_; Nn_ = A*Nn_; Ln_ = A*Ln_; Dn_ = A*Dn_;

	// rotate current as well
	Je_ = A*Je_; Jn_ = A*Jn_;
}

// set current density from magnetisation at nodes
// this allows the modeling of bound currents
// note that this does not include the surface
// current density which needs to be modelled 
// separately
void CurrentMesh::set_magnetisation_nodes(
	const arma::Mat<double> &Mn){

	// allocate current
	Je_.set_size(3,num_elements_);

	// walk over elements
	arma::Col<double>::fixed<3> Rq = {0,0,0};
	for(arma::uword i=0;i<num_elements_;i++){
		// use derivatives to calculate the curl of the 
		// magnetisation at the center of the element.
		// this is the bound current
		Je_.col(i) = Hexahedron::quad2cart_curl(
			Rn_.cols(n_.col(i)),Rq,Mn.cols(n_.col(i)));
	}

	// set elemental currents to true
	elemental_currents_ = true;
}


// set number of gauss points
void CurrentMesh::set_num_gauss(const arma::sword num_gauss){
	// safety limit
	assert(num_gauss<100 && num_gauss_>-100);

	// set value
	num_gauss_ = num_gauss;

	// calculate gauss points
	setup_gauss_points();
}

// calculate guass points and abscissae
void CurrentMesh::setup_gauss_points(){
	// make gauss point calculator
	Gauss gp(num_gauss_);

	// extract abscissae and weights
	xg_ = gp.get_abscissae(); 
	wg_ = gp.get_weights();
}

// set number of element radii 
// within which gauss integration is used
void CurrentMesh::set_num_dist(const arma::uword num_dist){
	num_dist_ = num_dist;
}

// internal field calculation routine
void CurrentMesh::calc_field(
	ShTargetsPr &tar, const arma::Row<arma::uword> &tidx,
 	const arma::Row<arma::uword> &sidx) const{

	// ensure that gauss points were set
	assert(!xg_.is_empty());
	assert(!wg_.is_empty());
	
	// get target points
	const arma::Mat<double> Rt = tar->get_target_coords(tidx);

	// check calculaation type
	const bool has_vector_potential = tar->has("A");
	const bool has_magnetic_field = tar->has("H");

	// walk over all sources (parallel?)
	//for(arma::uword i=0;i<sidx.n_elem;i++){
	parfor(0,sidx.n_elem,false,[&](int i, int){
		// get my source
		arma::uword mysource = sidx(i);

		// current density in elements and at nodes
		arma::Col<double>::fixed<3> myJav(arma::fill::zeros);
		if(nodal_currents_)myJav += arma::mean(Jn_.cols(n_.col(mysource)),1);
		if(elemental_currents_)myJav += Je_.col(mysource);

		// calculate distance of 
		// all target points to this source
		const arma::Mat<double> Rrel = Rt.each_col() - Re_.col(mysource);
		const arma::Mat<double> dist = arma::sqrt(arma::sum(Rrel%Rrel,0));

		// find indexes of target points 
		// that are far away
		const arma::Row<arma::uword> indices_far = 
			arma::find(dist>num_dist_*element_radius_(mysource)).t();
		const arma::Row<arma::uword> indices_near = 
			arma::find(dist<=num_dist_*element_radius_(mysource)).t();

		// run normal calculation for far targets
		if(!indices_far.is_empty()){
			// for vector potential
			if(has_vector_potential){
				tar->add_field("A", tidx.cols(indices_far), Savart::calc_I2A(
					Re_.col(mysource),myJav*Ve_(mysource),
					Rt.cols(indices_far),false),true);
			}

			// for magnetic field
			if(has_magnetic_field){
				tar->add_field("H", tidx.cols(indices_far), Savart::calc_I2H(
					Re_.col(mysource),myJav*Ve_(mysource),
					Rt.cols(indices_far),false),true);
			}
		}

		// for target points that are close
		// use gauss points to do the integration
		if(!indices_near.is_empty()){
			// my nodes
			const arma::Mat<double>::fixed<3,8> myRn = Rn_.cols(n_.col(mysource));

			// get quadrilateral coordinates (iteratively)
			arma::Mat<double> Rqt = Hexahedron::cart2quad(
				myRn, Rt.cols(indices_near), 1e-6);

			// clamp such that quadrilateral coordinates 
			// are forced inside the element
			Rqt = arma::clamp(Rqt,-1.0,1.0);

			// allocate we store A and H locally to avoid
			// accessing the targets many times as this would
			// require a costly mutex lock
			arma::Mat<double> A, H;
			if(has_vector_potential)A.zeros(3,indices_near.n_elem);
			if(has_magnetic_field)H.zeros(3,indices_near.n_elem);

			// walk over target points
			for(arma::uword i=0;i<indices_near.n_cols;i++){
				// setup grid around singularity
				arma::Mat<double> Rqgrd; arma::Row<double> wgrd;
				Hexahedron::setup_source_grid(Rqgrd, wgrd, Rqt.col(i), xg_, wg_);
				assert(std::abs(arma::sum(wgrd)-1.0)<1e-6);
				
				// calculate points in carthesian coordinates
				const arma::Mat<double> Rc = Hexahedron::quad2cart(myRn,Rqgrd);
				
				// allocate effective current for gauss points
				arma::Mat<double> Ieff(3,Rc.n_cols,arma::fill::zeros);

				// calculate effective current from elemental currents
				if(elemental_currents_){
					Ieff.each_row() = Ve_(mysource)*wgrd;
					Ieff.each_col() %= Je_.col(mysource);
				}

				// calculate effective current from nodal currents
				if(nodal_currents_){
					Ieff += Ve_(mysource)*(Hexahedron::quad2cart(
						Jn_.cols(n_.col(mysource)),Rqgrd).each_row()%wgrd);
				}

				// calculate and sum field of all gauss points
				// for vector potential
				if(has_vector_potential){
					A.col(i) = Savart::calc_I2A(Rc,Ieff,Rt.col(indices_near(i)),false);
				}

				// for magnetic field
				if(has_magnetic_field){
					H.col(i) = Savart::calc_I2H(Rc,Ieff,Rt.col(indices_near(i)),false);
				}
			}

			// add field
			if(has_vector_potential)tar->add_field("A", tidx.cols(indices_near), A, true);
			if(has_magnetic_field)tar->add_field("H", tidx.cols(indices_near), H, true);

		}
	});
	//}
}

// calculation of vector potential for all sources
void CurrentMesh::calc_direct(
	ShTargetsPr &tar) const{
	
	// make index arrays
	arma::Row<arma::uword> sidx = 
		arma::regspace<arma::Row<arma::uword> >(0,num_elements_-1);
	arma::Row<arma::uword> tidx = 
		arma::regspace<arma::Row<arma::uword> >(0,tar->num_targets()-1);

	// forward calculation for now
	calc_field(tar,tidx,sidx);
}

// calculation of vector potential for specific sources
void CurrentMesh::calc_direct(
	ShTargetsPr &tar, 
	const arma::Row<arma::uword> &tidx, 
	const arma::Row<arma::uword> &sidx) const{
	
	// forward calculation for now
	calc_field(tar,tidx,sidx);
}

// create surface mesh
ShMTSurfacePr CurrentMesh::create_surface() const{
	// create surface object
	ShMTSurfacePr surf = MTSurface::create();

	// get surface connectivity
	arma::Mat<arma::uword> ns = get_surface_elements();

	// find indexes of unique nodes
	arma::Row<arma::uword> idx_node_keep = 
		arma::unique(arma::reshape(ns,1,ns.n_elem));
	
	// figure out which nodes to drop
	arma::Row<arma::uword> node_indexing(1,num_nodes_,arma::fill::zeros);
	node_indexing.cols(idx_node_keep) = 
		arma::regspace<arma::Row<arma::uword> >(0,idx_node_keep.n_elem-1);

	// re-index connectivity
	ns = arma::reshape(node_indexing(arma::reshape(ns,1,ns.n_elem)),4,ns.n_cols);

	// get node coordinates
	arma::Mat<double> Rns = Rn_.cols(idx_node_keep);

	// set nodes to new surface
	surf->set_mesh(Rns,ns);

	// return surface object
	return surf;
}


// create current sources from elements stored in mesh
// when not calculating inside coil this may be faster
ShCurrentSourcesPr CurrentMesh::create_current_sources() const{
	// check if element volumes calculated
	//calculate_element_volume();
	assert(!Ve_.is_empty());

	// calculate number of currents
	arma::uword num_gauss_element = num_gauss_*num_gauss_*num_gauss_;
	arma::uword num_currents = num_elements_*num_gauss_element;

	// allocate
	arma::Mat<double> R(3,num_currents);
	arma::Mat<double> Ieff(3,num_currents);

	// setup gauss grid
	arma::Mat<double> Rqgrd; arma::Row<double> wgrd;
	arma::Col<double>::fixed<3> Rqt = {1,1,1};
	Hexahedron::setup_source_grid(Rqgrd, wgrd, Rqt, xg_, wg_);
	assert(std::abs(arma::sum(wgrd)-1.0)<1e-6);
	assert(Rqgrd.n_cols==num_gauss_element);

	// calculate coordinates
	for(arma::uword i=0;i<num_elements_;i++){
		// generate coordinates
		R.cols(i*num_gauss_element,(i+1)*num_gauss_element-1) = 
			Hexahedron::quad2cart(Rn_.cols(n_.col(i)),Rqgrd);

		
		arma::Mat<double> Ieffe(3,num_gauss_element,arma::fill::zeros);

		// effective current from element
		if(elemental_currents_){
			Ieffe.each_row() = Ve_(i)*wgrd;
			Ieffe.each_col() %= Je_.col(i);
		}

		// effective current from nodes
		if(nodal_currents_){
			Ieff.cols(i*num_gauss_element,(i+1)*num_gauss_element-1) = 
				Ieffe + Ve_(i)*(Hexahedron::quad2cart(
				Jn_.cols(n_.col(i)),Rqgrd).each_row()%wgrd);
		}
	}

	// calculate effective currents
	ShCurrentSourcesPr sources = CurrentSources::create();
	sources->set_coords(R);
	sources->set_currents(Ieff);

	// return current sources
	return sources;	
}

// setup source to multipole matrices
void CurrentMesh::setup_source_to_multipole(
	const arma::Mat<double> &dR,
	const arma::uword num_exp){
	
	// memory efficient implementation (default)
	if(enable_memory_efficient_s2m_){
		dRs2m_ = dR;
	}

	// maximize speed over memory efficiency
	else{
		// set number of expansions and setup matrix
		M_J_.set_num_exp(num_exp);
		M_J_.calc_matrix(-dR);
	}	
}

// get multipole contribution of the sources with indices
// the contributions of the sources are already summed
arma::Mat<std::complex<double> > CurrentMesh::source_to_multipole(
	const arma::Row<arma::uword> &indices,
	const arma::uword num_exp) const{
		
	// allocate output multipole
	arma::Mat<std::complex<double> > Mp;
	
	// get current density at elements with indices
	// arma::Mat<double> Je;
	// if(Je_.is_empty()){
	// 	Je.set_size(3,indices.n_elem);
	// 	for(arma::uword i=0;i<indices.n_elem;i++)
	// 		Je.col(i) = arma::mean(Jn_.cols(n_.col(indices(i))),1);
	// }else{
	// 	Je = Je_.cols(indices);
	// }

	// get current density at elements with indices
	arma::Mat<double> Je(3,indices.n_elem,arma::fill::zeros);
	if(nodal_currents_)
		for(arma::uword i=0;i<indices.n_elem;i++)
			Je.col(i) = arma::mean(Jn_.cols(n_.col(indices(i))),1);
	if(elemental_currents_)
		Je += Je_.cols(indices);

	// memory efficient implementation (default)
	if(enable_memory_efficient_s2m_){
		// calculate contribution of currents and return calculated multipole
		StMat_So2Mp_J M_J;
		M_J.set_num_exp(num_exp);
		M_J.calc_matrix(-dRs2m_.cols(indices));
		Mp = M_J.apply(Je.each_row()%Ve_.cols(indices));
	}

	// faster less memory efficient implementation
	else{
		Mp = M_J_.apply(Je.each_row()%Ve_.cols(indices),indices);
	}

	// return multipole
	return Mp;
}

// get field from internal storage
arma::Mat<double> CurrentMesh::get_field(
	const std::string &type, const arma::uword num_dim) const{

	// make sure that type is only one character
	assert(type.length()==1);

	// allocate output
	arma::Mat<double> Mout;

	// force density (special case)
	if(type=="F"){
		assert(elemental_currents_==false); // safety until elemental currents fully removed
		Mout = Extra::cross(Jn_,MTargets::get_field("B",num_dim));
	}

	// normal cases
	else{
		Mout = MTargets::get_field(type,num_dim);
	}

	// add field to matrix
	return Mout; 
}

// write to gmsh file
void CurrentMesh::export_gmsh(ShGmshFilePr gmsh){
	// call parent class
	Mesh::export_gmsh(gmsh);

	// check if field calculated
	if(has_field()){
		// get magnetic flux densities
		arma::Mat<double> B = get_field("B",3);
		arma::Row<double> Bmag = arma::sqrt(arma::sum(B%B,0));

		// write magnetic flux densities
		gmsh->write_nodedata(B,"Mgn. Flux Density (B)");
		gmsh->write_nodedata(Bmag,"Mgn. Flux Density norm (|B|)");
			
		// get magnetic field
		arma::Mat<double> H = get_field("H",3);
		arma::Row<double> Hmag = arma::sqrt(arma::sum(H%H,0));

		// write magnetic field
		gmsh->write_nodedata(H,"Mgn. Field (H)");
		gmsh->write_nodedata(Hmag,"Mgn. Field norm (|H|)");	

		// get force density
		arma::Mat<double> F = get_field("F",3);
		arma::Row<double> Fmag = arma::sqrt(arma::sum(F%F,0));

		// write force density
		gmsh->write_nodedata(F,"Force Density (F)");
		gmsh->write_nodedata(Fmag,"Force Density norm(|F|)");

		// get force density
		arma::Row<double> Jmag = arma::sqrt(arma::sum(Jn_%Jn_,0));

		// write force density
		gmsh->write_nodedata(Jn_,"Current Density (J)");
		gmsh->write_nodedata(Jmag,"Current Density norm (|J|)");	
	}
}

// gmsh interface
// note that gmsh starts counting the nodes at one
void CurrentMesh::export_force_density(std::ofstream &fid) const{
	// get forces
	const arma::Mat<double> Fd = get_field("F",3);

	// set precision
	fid << std::fixed;
	fid << std::setprecision(7);

	// write to file
	fid << num_nodes_ << "\n";
	fid << "i, x, y, z, Fdx, Fdy, Fdz\n";
	for(arma::uword i=0;i<num_nodes_;i++){
		fid << i+1 << ", ";
		fid << Rn_(0,i) << ", " << Rn_(1,i) << ", " << Rn_(2,i) << ", ";
		fid << Fd(0,i) << ", " << Fd(1,i) << ", " << Fd(2,i) << "\n";
	}
}


// calculate integrated forces
arma::Col<double>::fixed<3> CurrentMesh::integrate_forces() const{
	// get force density at nodes
	const arma::Mat<double> Fdn = get_field("F",3);

	// calculate average force density at elements
	arma::Mat<double> Fde(3,num_elements_);
	for(arma::uword i=0;i<n_.n_cols;i++)
		Fde.col(i) = arma::mean(Fdn.cols(n_.col(i)),1);

	// check if volume calculated
	assert(!Ve_.is_empty());

	// calculate force
	arma::Mat<double> Fe = Fde.each_row()%Ve_;

	// sum forces
	arma::Col<double>::fixed<3> F = arma::sum(Fe,1);

	// return force on mesh
	return F;
}

// calculate torque around point
arma::Col<double>::fixed<3> CurrentMesh::integrate_torque(
	const arma::Col<double>::fixed<3> &R) const{
	
	// get force density at nodes
	const arma::Mat<double> Fdn = get_field("F",3);

	// calculate average force density at elements
	arma::Mat<double> Fde(3,num_elements_);
	for(arma::uword i=0;i<n_.n_cols;i++)
		Fde.col(i) = arma::mean(Fdn.cols(n_.col(i)),1);

	// check if volume calculated
	assert(!Ve_.is_empty());

	// calculate force
	arma::Mat<double> Fe = Fde.each_row()%Ve_;

	// calculate torque contribution
	arma::Mat<double> dRe = Re_.each_col()-R;
	arma::Mat<double> taue = Extra::cross(dRe, Fe);

	// sum tau
	arma::Col<double>::fixed<3> tau = arma::sum(taue,1);

	// return forces
	return tau;
}
