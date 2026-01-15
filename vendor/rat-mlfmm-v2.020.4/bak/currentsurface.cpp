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
#include "currentsurface.hh"

// constructor
CurrentSurface::CurrentSurface(){
	// number of dimensions of this source
	set_src_num_dim(3);

	// calculate gauss points
	setup_gauss_points();
}

// factory
ShCurrentSurfacePr CurrentSurface::create(){
	//return ShCurrentSurfacePr(new CurrentSurface);
	return std::make_shared<CurrentSurface>();
}

// // set mesh
// void CurrentSurface::set_mesh(
// 	const arma::Mat<double> &Rn, 
// 	const arma::Mat<arma::uword> &n){
	
// 	// set mesh
// 	Surface::set_mesh(Rn,n);
// }

// // set coil mesh from other coil meshes
// // this calls set mesh on mesh to avoid
// // code duplication
// void CurrentSurface::set_mesh(ShCurrentSurfacePrList &surface_meshes){
// 	// get number of meshes
// 	arma::uword num_meshes = surface_meshes.n_elem;

// 	// downcast mesh array
// 	arma::field<ShSurfacePr> meshes(num_meshes);
// 	for(arma::uword i=0;i<num_meshes;i++)
// 		meshes(i) = std::dynamic_pointer_cast<Surface>(surface_meshes(i));

// 	// forward construction to mesh class
// 	Surface::set_mesh(meshes);

// 	// allocate information arrays
// 	arma::Row<arma::uword> num_nodes_each(num_meshes);
// 	arma::Row<arma::uword> num_elements_each(num_meshes);

// 	// acquire data from each mesh
// 	for(arma::uword i=0;i<num_meshes;i++){
// 		// get numberof nodes
// 		num_nodes_each(i) = meshes(i)->get_num_nodes();
// 		num_elements_each(i) = meshes(i)->get_num_elements();
// 	}

// 	// create node indexing array
// 	arma::Row<arma::uword> idx_nodes(num_meshes+1); idx_nodes(0) = 0;
// 	idx_nodes.cols(1,num_meshes) = arma::cumsum(num_nodes_each,1);

// 	// create element indexing array
// 	arma::Row<arma::uword> idx_elements(num_meshes+1); idx_elements(0) = 0;
// 	idx_elements.cols(1,num_meshes) = arma::cumsum(num_elements_each,1);

// 	// allocate current
// 	Je_.set_size(3,num_elements_);
// 	Jn_.set_size(3,num_nodes_);

// 	// get currents
// 	for(arma::uword i=0;i<num_meshes;i++){
// 		Je_.cols(idx_elements(i),idx_elements(i+1)-1) = surface_meshes(i)->get_current_density_elements();
// 		Jn_.cols(idx_nodes(i),idx_nodes(i+1)-1) = surface_meshes(i)->get_current_density_nodes();
// 	}

// 	// current type
// 	for(arma::uword i=0;i<num_meshes;i++){
// 		nodal_currents_ = nodal_currents_ || surface_meshes(i)->has_nodal_currents();
// 		elemental_currents_ = elemental_currents_ || surface_meshes(i)->has_elemental_currents();
// 	}
	
// }


// get element centroids
arma::Mat<double> CurrentSurface::get_source_coords() const{
	// check if coordinates were set
	assert(!Re_.is_empty());

	// return element centroids
	return Re_;
}

// get element centroids of specific elements
arma::Mat<double> CurrentSurface::get_source_coords(
	const arma::Mat<arma::uword> &indices) const{
	// check if coordinates were set
	assert(!Re_.is_empty());

	// return element centroids
	return Re_.cols(indices);
}

// get number of sources (each element has one source)
arma::uword CurrentSurface::num_sources() const{
	return num_elements_;
}

// set current density vector of each node
// note that this is overridden by Je
void CurrentSurface::set_current_density_nodes(
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
void CurrentSurface::set_current_density_elements(
	const arma::Mat<double> &Je){
	
	// check input
	assert(Je.n_rows==3);
	assert(Je.n_cols==num_elements_);

	// check out of plane currents
	assert(arma::all(Extra::vec_norm(Extra::cross(Je,Ne_))/Extra::vec_norm(Je)<1e-6) || arma::all(arma::all(Je==0)));

	// set type
	elemental_currents_ = true;

	// set current density
	Je_ = Je;
}

// check whether this mesh has nodal currents
bool CurrentSurface::has_nodal_currents() const{
	return nodal_currents_;
}

// check whether this mesh has elemental currents
bool CurrentSurface::has_elemental_currents() const{
	return elemental_currents_;
}

// check whether this mesh has currents
bool CurrentSurface::has_current() const{
	return nodal_currents_ || elemental_currents_;
}

// set current density from magnetisation at nodes
// this allows the modeling of bound currents
// note that this does not include the volume
// current density which needs to be modelled 
// separately
void CurrentSurface::set_magnetisation_nodes(
	const arma::Mat<double> &Mn){

	// check input
	assert(Mn.n_cols==num_nodes_);

	// calculate magnetisation at center of element
	arma::Mat<double> Me(3,num_elements_);
	for(arma::uword i=0;i<num_elements_;i++){
		Me.col(i) = arma::mean(Mn.cols(n_.col(i)),1);
	}

	// calculate cross product with facenormal
	Je_ = Extra::cross(Me,Ne_);

	// set elemental currents to true
	elemental_currents_ = true;
}


// set number of gauss points
void CurrentSurface::set_num_gauss(const arma::sword num_gauss){
	// safety limit
	assert(num_gauss<100 && num_gauss_>-100);

	// set value
	num_gauss_ = num_gauss;

	// calculate gauss points
	setup_gauss_points();
}

// calculate guass points and abscissae
void CurrentSurface::setup_gauss_points(){
	// make gauss point calculator
	Gauss gp(num_gauss_);

	// extract abscissae and weights
	xg_ = gp.get_abscissae(); 
	wg_ = gp.get_weights();
	
	// equispaced points
	// xg_ = arma::linspace<arma::Row<double> >(-1.0+1.0/num_gauss_,1.0-1.0/num_gauss_,num_gauss_);
	// wg_ = 2*arma::Row<double>(num_gauss_,arma::fill::ones)/num_gauss_;
}

// set number of element radii 
// within which gauss integration is used
void CurrentSurface::set_num_dist(const arma::uword num_dist){
	num_dist_ = num_dist;
}

// getting current density vector at each node
arma::Mat<double> CurrentSurface::get_current_density_nodes() const{
	return Jn_;
}

// get the current density vector of each element
arma::Mat<double> CurrentSurface::get_current_density_elements() const{
	return Je_;
}

// create current sources from elements stored in mesh
// when not calculating inside coil this may be faster
ShCurrentSourcesPr CurrentSurface::create_current_sources() const{
	// check if element volumes calculated
	assert(!Ae_.is_empty());

	// calculate number of currents
	arma::uword num_gauss_element = num_gauss_*num_gauss_;
	arma::uword num_currents = num_elements_*num_gauss_element;

	// allocate
	arma::Mat<double> R(3,num_currents);
	arma::Mat<double> Ieff(3,num_currents);

	// setup gauss grid
	arma::Mat<double> Rqgrd; arma::Row<double> wgrd;
	arma::Col<double>::fixed<2> Rqt = {1,1};
	Quadrilateral::setup_source_grid(Rqgrd, wgrd, Rqt, xg_, wg_);
	assert(std::abs(arma::sum(wgrd)-1.0)<1e-6);
	assert(Rqgrd.n_cols==num_gauss_element);

	// calculate coordinates
	for(arma::uword i=0;i<num_elements_;i++){
		// generate coordinates
		R.cols(i*num_gauss_element,(i+1)*num_gauss_element-1) = 
			Quadrilateral::quad2cart(Rn_.cols(n_.col(i)),Rqgrd);

		
		arma::Mat<double> Ieffe(3,num_gauss_element,arma::fill::zeros);

		// effective current from element
		if(elemental_currents_){
			Ieffe.each_row() = Ae_(i)*wgrd;
			Ieffe.each_col() %= Je_.col(i);
		}

		// calculate effective current for grid points
		if(nodal_currents_){
			Ieff.cols(i*num_gauss_element,(i+1)*num_gauss_element-1) = 
				Ieffe + Ae_(i)*(Quadrilateral::quad2cart(
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


// internal field calculation routine
void CurrentSurface::calc_field(
	ShTargetsPr &tar, const arma::Row<arma::uword> &tidx,
 	const arma::Row<arma::uword> &sidx) const{

	// ensure that gauss points were set
	assert(!xg_.is_empty());
	assert(!wg_.is_empty());
	
	// get target points
	arma::Mat<double> Rt = tar->get_target_coords(tidx);
	
	// walk over all sources (parallel?)
	//for(arma::uword i=0;i<sidx.n_elem;i++){
	parfor(0,sidx.n_elem,true,[&](int i, int){
		// get my source
		arma::uword mysource = sidx(i);

		// current density in element
		// arma::Col<double>::fixed<3> myJav;
		// if(Je_.is_empty()){
		// 	assert(!Jn_.is_empty());
		// 	myJav = arma::mean(Jn_.cols(n_.col(mysource)),1);
		// }else{
		// 	myJav = Je_.col(mysource);
		// }

		// current density in element
		// arma::Col<double>::fixed<3> myJav = 
		// 	arma::mean(Jn_.cols(n_.col(mysource)),1) + Je_.col(mysource);

		// current density in elements and at nodes
		arma::Col<double>::fixed<3> myJav(arma::fill::zeros);
		if(nodal_currents_)myJav += arma::mean(Jn_.cols(n_.col(mysource)),1);
		if(elemental_currents_)myJav += Je_.col(mysource);

		// calculate distance of 
		// all target points to this source
		arma::Mat<double> dist = arma::sqrt(
			arma::sum(arma::pow(Rt.each_col() - 
			Re_.col(mysource),2),0));

		// find indexes of target points 
		// that are far away
		arma::Row<arma::uword> indices_far = 
			arma::find(dist>num_dist_*element_radius_(mysource)).t();
		arma::Row<arma::uword> indices_near = 
			arma::find(dist<=num_dist_*element_radius_(mysource)).t();

		// run normal calculation for far targets
		if(!indices_far.is_empty()){
			// for vector potential
			if(tar->has("A")){
				tar->add_field("A", tidx.cols(indices_far), Savart::calc_I2A(
					Re_.col(mysource),myJav*Ae_(mysource),
					Rt.cols(indices_far),false), true);
			}

			// for magnetic field
			if(tar->has("H")){
				tar->add_field("H", tidx.cols(indices_far), Savart::calc_I2H(
					Re_.col(mysource),myJav*Ae_(mysource),
					Rt.cols(indices_far),false), true);
			}
		}

		// for target points that are close
		// use gauss points to do the integration
		if(!indices_near.is_empty()){
			// my nodes
			arma::Mat<double>::fixed<3,4> myRn = Rn_.cols(n_.col(mysource));

			// get quadrilateral coordinates (iteratively)
			arma::Mat<double> Rqt = Quadrilateral::cart2quad(
				myRn, Rt.cols(indices_near), 1e-6);

			// check if all Rqt<1.0001 & Rqt>-1.0001
			// calculate distance d = cross(N,R)
			// d<1e-9 & d>-typical_size -> just inside  B = +mu0*cross(Je,N)/2
			// d>1e-9 & d<+typical_size -> just outside B = -mu0*cross(Je,N)/2
			// remaining indices do
			// if not inside use pre-made grid

			// allocate we store A and H locally to avoid
			// accessing the targets many times as this would
			// require a costly mutex lock
			arma::Mat<double> A, H;
			if(tar->has("A"))A.set_size(3,indices_near.n_elem);
			if(tar->has("H"))H.set_size(3,indices_near.n_elem);

			// walk over target points
			for(arma::uword i=0;i<indices_near.n_cols;i++){
				// distance from plane
				double dist = arma::as_scalar(Extra::dot(Ne_.col(
					mysource),Re_.col(mysource) - Rt.col(indices_near(i))));

				// in plane
				if(arma::all(Rqt.col(i)>-1.0000001 && Rqt.col(i)<1.0000001) && dist<1e-6 && dist>-1e-6){
					if(tar->has("H")){
						H.col(i).fill(0); //Extra::cross(Ne_.col(mysource),Je_.col(mysource))/8.0;
					}
				}

				// further away
				else{
					// setup grid around singularity	
					// clamp such that quadrilateral 
					// coordinates are forced inside the face element
					arma::Mat<double> Rqgrd; arma::Row<double> wgrd;
					Quadrilateral::setup_source_grid(Rqgrd, wgrd, 
						arma::clamp(Rqt.col(i),-1.0,1.0), xg_, wg_);
					assert(std::abs(arma::sum(wgrd)-1.0)<1e-6);

					// calculate points in carthesian coordinates
					arma::Mat<double> Rc = Quadrilateral::quad2cart(myRn,Rqgrd);
					
					// calculate effective current for grid points
					// arma::Mat<double> Ieff;
					// if(Je_.is_empty()){
					// 	Ieff = Ae_(mysource)*(Quadrilateral::quad2cart(
					// 		Jn_.cols(n_.col(mysource)),Rqgrd).each_row()%wgrd);
					// }else{
					// 	Ieff.set_size(3,Rc.n_cols);
					// 	Ieff.each_row() = Ae_(mysource)*wgrd;
					// 	Ieff.each_col() %= Je_.col(mysource);
					// }

					// allocate effective current for gauss points
					arma::Mat<double> Ieff(3,Rc.n_cols,arma::fill::zeros);

					// calculate effective current from elemental currents
					if(elemental_currents_){
						Ieff.each_row() = Ae_(mysource)*wgrd;
						Ieff.each_col() %= Je_.col(mysource);
					}

					// calculate effective current from nodal currents
					if(nodal_currents_){
						Ieff += Ae_(mysource)*(Quadrilateral::quad2cart(
							Jn_.cols(n_.col(mysource)),Rqgrd).each_row()%wgrd);
					}

					// calculate and sum field of all gauss points
					// for vector potential
					if(tar->has("A")){
						A.col(i) = Savart::calc_I2A(Rc,Ieff,Rt.col(indices_near(i)),false);
					}

					// for magnetic field (magnetisation makes this a hypersingular integral)
					if(tar->has("H")){
						H.col(i) = Savart::calc_I2H(Rc,Ieff,Rt.col(indices_near(i)),false);
					}
				}
			}

			// add field
			if(tar->has("A"))tar->add_field("A", tidx.cols(indices_near), A, true);
			if(tar->has("H"))tar->add_field("H", tidx.cols(indices_near), H, true);
		}
	});
	//}
}

// calculation of vector potential for all sources
void CurrentSurface::calc_direct(
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
void CurrentSurface::calc_direct(
	ShTargetsPr &tar, 
	const arma::Row<arma::uword> &tidx, 
	const arma::Row<arma::uword> &sidx) const{
	
	// forward calculation for now
	calc_field(tar,tidx,sidx);
}

// setup source to multipole matrices
void CurrentSurface::setup_source_to_multipole(
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
arma::Mat<std::complex<double> > CurrentSurface::source_to_multipole(
	const arma::Row<arma::uword> &indices,
	const arma::uword num_exp) const{
		
	// check currents
	assert(Je_.n_rows==src_num_dim_);

	// allocate output multipole
	arma::Mat<std::complex<double> > Mp;
	
	// get current density at elements with indices
	// arma::Mat<double> Je;
	// if(Je_.is_empty()){
	// 	Je.set_size(3,indices.n_elem);
	// 	for(arma::uword i=0;i<indices.n_elem;i++){
	// 		Je.col(i) = arma::mean(Jn_.cols(n_.col(indices(i))),1);
	// 	}
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
		Mp = M_J.apply(Je.each_row()%Ae_.cols(indices));
	}

	// faster less memory efficient implementation
	else{
		Mp = M_J_.apply(Je.each_row()%Ae_.cols(indices),indices);
	}

	// check multipole size
	assert(Mp.n_cols==src_num_dim_);

	// return multipole
	return Mp;
}

