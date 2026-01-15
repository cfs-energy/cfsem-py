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
#include "currentmesh.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// default constructor
	// default is hex
	CurrentMesh::CurrentMesh(){
		
	}

	// constructor from group of meshes
	CurrentMesh::CurrentMesh(const ShCurrentMeshPrList &meshes){
		set_mesh(meshes);
	}

	// constructor with mesh input
	CurrentMesh::CurrentMesh(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n){
		set_mesh(Rn,n);
	}


	// factory
	ShCurrentMeshPr CurrentMesh::create(){
		return std::make_shared<CurrentMesh>();
	}

	// factory
	ShCurrentMeshPr CurrentMesh::create(const ShCurrentMeshPrList &meshes){
		return std::make_shared<CurrentMesh>(meshes);
	}

	// factory
	ShCurrentMeshPr CurrentMesh::create(const arma::Mat<fltp> &Rn, const arma::Mat<arma::uword> &n){
		return std::make_shared<CurrentMesh>(Rn,n);
	}

	// constructor
	void CurrentMesh::set_mesh(const ShCurrentMeshPrList &meshes){
		// number of input meshes
		const arma::uword num_meshes = meshes.n_elem;

		// allocate mesh
		arma::field<arma::Mat<fltp> > Rnfld(1,num_meshes);
		arma::field<arma::Mat<arma::uword> > nfld(1,num_meshes);
		arma::field<arma::Mat<fltp> > Jefld(1,num_meshes);

		// gather
		arma::uword node_shift = 0;
		for(arma::uword i=0;i<num_meshes;i++){
			// copy data
			Rnfld(i) = meshes(i)->Rn_;
			nfld(i) = meshes(i)->n_ + node_shift;
			Jefld(i) = meshes(i)->Je_;

			// account for node shifting
			node_shift += Rnfld(i).n_cols;
		}

		// combine and store in self
		Rn_ = cmn::Extra::field2mat(Rnfld);
		n_ = cmn::Extra::field2mat(nfld);
		Je_ = cmn::Extra::field2mat(Jefld);

		// set counters
		num_nodes_ = Rn_.n_cols;
		num_elements_ = n_.n_cols;

		// calculate volume and centroids
		calculate_element_volume();
	}

	// set mesh
	void CurrentMesh::set_mesh(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n){

		// check user input
		if(Rn.empty())rat_throw_line("source node coordinates not set");
		if(Rn.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(n.n_rows!=8)rat_throw_line("element matrix must have eight rows");

		// set supplied values
		Rn_ = Rn; n_ = n;

		// get number of supplied nodes
		num_nodes_ = Rn_.n_cols;
		num_elements_ = n_.n_cols;

		// calculate volume and centroids
		calculate_element_volume();
	}

	// set mesh
	void CurrentMesh::set_mesh(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Mat<fltp> &Je){

		// check user input
		if(Rn.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(n.n_rows!=8)rat_throw_line("element matrix must have eight rows");
		if(Je.n_rows!=3)rat_throw_line("current density matrix must have three rows");
		if(Je.n_cols!=n.n_cols)rat_throw_line("current density and element matrix must have same number of columns");
		if(n.max()>=Rn.n_cols && n.n_cols>0)rat_throw_line("element matrix contains index outside coordinate matrix");

		// set supplied values
		Rn_ = Rn; n_ = n;

		// get number of supplied nodes
		num_nodes_ = Rn_.n_cols;
		num_elements_ = n_.n_cols;

		// set current density
		Je_ = Je;

		// calculate volume and centroids
		calculate_element_volume();
	}

	// set current density vector of each element
	void CurrentMesh::set_current_density_elements(
		const arma::Mat<fltp> &Je){

		// check input
		if(Je.n_rows!=3)rat_throw_line("current density matrix must have three rows");
		if(Je.n_cols!=num_elements_)rat_throw_line("current density matrix number of columns must equal number of elements");

		// set current density
		Je_ = Je;
	}

	// set current density from magnetisation at nodes
	// this allows the modeling of bound currents
	// note that this does not include the surface
	// current density which needs to be modelled 
	// separately
	void CurrentMesh::set_magnetisation_nodes(
		const arma::Mat<fltp> &Mn){
		
		// check input
		if(Mn.n_rows!=3)rat_throw_line("magnetisation matrix must have three rows");
		if(Mn.n_cols!=num_nodes_)rat_throw_line("magnetisation matrix number of columns must equal number of nodes");
		
		// allocate 
		Je_.set_size(3,num_elements_);

		// walk over elements
		const arma::Col<fltp>::fixed<3> Rq = {0,0,0};
		for(arma::uword i=0;i<num_elements_;i++){
			// use derivatives to calculate the curl of the 
			// magnetisation at the center of the element.
			// this is the bound current
			Je_.col(i) = cmn::Hexahedron::quad2cart_curl(
				Rn_.cols(n_.col(i)),Rq,Mn.cols(n_.col(i)));
		}
	}

	// get elmenet size
	fltp CurrentMesh::element_size() const{
		assert(n_.n_rows==8);
		return arma::max(cmn::Extra::vec_norm(Rn_.cols(n_.row(6)) - Rn_.cols(n_.row(0))));
	}

	// set number of gauss points
	void CurrentMesh::set_num_gauss(const arma::sword num_gauss){
		if(num_gauss==0)rat_throw_line("number of gauss points can not be zero");
		num_gauss_ = num_gauss;
	}

	// set distance scaling
	void CurrentMesh::set_num_dist(const fltp num_dist){
		if(num_dist<=0)rat_throw_line("nuber of distances must be larger than zero");
		num_dist_ = num_dist;
	}

	// calculate hexahedron volumes
	// by splitting it up into five tetrahedrons
	// then adds their volumes
	void CurrentMesh::calculate_element_volume(){
		// check if it is really hexahedron
		assert(n_.n_rows==8);

		// volume of elements
		Ve_ = cmn::Hexahedron::calc_volume(Rn_,n_);

		// allocate element centroids
		Re_.set_size(3,num_elements_);

		// walk over elements
		arma::Col<fltp>::fixed<3> Rq = {0,0,0};
		for(arma::uword i=0;i<num_elements_;i++){
			// calculate centroid using quadrilateral coordinates
			Re_.col(i) = cmn::Hexahedron::quad2cart(Rn_.cols(n_.col(i)),Rq);
		}

		// distance to center
		element_radius_.zeros(1,num_elements_);
		for(arma::uword i=0;i<n_.n_rows;i++){
			arma::Mat<fltp> Rr = (Re_ - Rn_.cols(n_.row(i)));
			element_radius_ = arma::max(element_radius_,
				arma::sqrt(arma::sum(Rr%Rr,0)));
		}
	}

	// get element volumes
	arma::Row<fltp> CurrentMesh::get_volume() const{
		// check if volumes were calculated
		assert(!Ve_.is_empty());

		// return element volumes
		return Ve_;
	}

	// get number of sources (each element has one source)
	arma::uword CurrentMesh::get_num_nodes() const{
		assert(num_nodes_>=1);
		return num_nodes_;
	}

	// get number of sources (each element has one source)
	arma::uword CurrentMesh::get_num_elements() const{
		assert(num_elements_>=1);
		return num_elements_;
	}

	// get node coordinates
	arma::Mat<fltp> CurrentMesh::get_node_coords() const{
		// check if node coordinates were set
		if(Rn_.is_empty())rat_throw_line("coordinates matrix not set");

		// return node coordinates
		return Rn_;
	}

	// get element node indices
	arma::Mat<arma::uword> CurrentMesh::get_elements() const{
		// check if elements were set
		if(n_.is_empty())rat_throw_line("element matrix not set");

		// return 
		return n_;
	}

	// get element centroids
	arma::Mat<fltp> CurrentMesh::get_source_coords() const{
		// check if coordinates were set
		if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix not calculated");

		// return element centroids
		return Re_;
	}

	// // get element centroids of specific elements
	// arma::Mat<fltp> CurrentMesh::get_source_coords(
	// 	const arma::Row<arma::uword> &indices) const{
	// 	// check if coordinates were set
	// 	if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix not calculated");

	// 	// return element centroids
	// 	return Re_.cols(indices);
	// }

	// count number of sources stored
	arma::uword CurrentMesh::num_sources() const{
		// return number of elements
		return num_elements_;
	}

	// get number of dimensions
	arma::uword CurrentMesh::get_num_dim() const{
		return num_dim_;
	}

	// extract surface from hexahedron mesh
	// translated from matlab "sq_hexsurface.m"
	arma::Mat<arma::uword> CurrentMesh::get_surface_elements() const{
		// make sure it is a hexahedron mesh
		assert(n_.n_rows==8);

		// count number of references of each node
		// arma::Row<arma::uword> nref(num_nodes_,arma::fill::zeros);
		// nref.cols(arma::reshape(n_,1,num_nodes_*8)) += 1;

		// get list of faces for each element
		arma::Mat<arma::uword>::fixed<6,4> M = cmn::Hexahedron::get_faces();

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

	// sorting function
	void CurrentMesh::sort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		if(n_.is_empty())rat_throw_line("element node index matrix not set");
		if(Je_.is_empty())rat_throw_line("current density matrix not set");
		if(Re_.is_empty())rat_throw_line("element centroid matrix not calculated");
		if(Ve_.is_empty())rat_throw_line("element volume vector not calculated");
		if(element_radius_.is_empty())rat_throw_line("element radius vector not calculated");
		
		// check if sort array right length
		assert(n_.n_cols == sort_idx.n_elem);

		// sort sources
		n_ = n_.cols(sort_idx);
		Je_ = Je_.cols(sort_idx);
		Re_ = Re_.cols(sort_idx);
		Ve_ = Ve_.cols(sort_idx);
		element_radius_ = element_radius_.cols(sort_idx);
	}

	// unsorting function
	void CurrentMesh::unsort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		if(n_.is_empty())rat_throw_line("element node index matrix not set");
		if(Je_.is_empty())rat_throw_line("current density matrix not set");
		if(Re_.is_empty())rat_throw_line("element centroid matrix not calculated");
		if(Ve_.is_empty())rat_throw_line("element volume vector not calculated");
		if(element_radius_.is_empty())rat_throw_line("element radius vector not calculated");
		
		// check if sort array right length
		assert(n_.n_cols == sort_idx.n_elem);

		// sort sources
		n_.cols(sort_idx) = n_;
		Je_.cols(sort_idx) = Je_;
		Re_.cols(sort_idx) = Re_;
		Ve_.cols(sort_idx) = Ve_;
		element_radius_.cols(sort_idx) = element_radius_;
	}


	// setup source to multipole matrices
	void CurrentMesh::setup_source_to_multipole(
		const arma::Mat<fltp> &dR,
		const ShSettingsPr &stngs){
			
		// get number of expansions
		const int num_exp = stngs->get_num_exp();

		// memory efficient implementation (default)
		if(stngs->get_memory_efficient_s2m()){
			dR_ = dR;
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
	void CurrentMesh::source_to_multipole(
		arma::Mat<std::complex<fltp> > &Mp,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source,
		const ShSettingsPr &stngs) const{

		// get number of expansions
		const int num_exp = stngs->get_num_exp();

		// check input
		assert(first_source.n_elem==last_source.n_elem);
		assert(!Mp.is_empty());

		// calculate elemental currents
		// arma::Mat<fltp> Je(3,num_elements_,arma::fill::zeros);
		// for(arma::uword i=0;i<num_elements_;i++)
		// 	Je.col(i) = arma::mean(Jn_.cols(n_.col(i)),1);

		// calculate effective current
		const arma::Mat<fltp> Ieff = Je_.each_row()%Ve_;

		// memory efficient implementation (default)
		if(stngs->get_memory_efficient_s2m()){		
			// walk over source nodes
			cmn::parfor(0,first_source.n_elem,stngs->get_parallel_s2m(),[&](arma::uword i, int){
				// calculate contribution of currents and return calculated multipole
				StMat_So2Mp_J M_J;
				M_J.set_num_exp(num_exp);
				M_J.calc_matrix(-dR_.cols(first_source(i),last_source(i)));

				// apply matrix
				Mp.cols(i*num_dim_, (i+1)*num_dim_-1) = M_J.apply(
					Ieff.cols(first_source(i),last_source(i)));
			});
		}

		// faster less memory efficient implementation
		else{
			// walk over source nodes
			cmn::parfor(0,first_source.n_elem,stngs->get_parallel_s2m(),[&](arma::uword i, int){
				// add child source contribution to this multipole
				Mp.cols(i*num_dim_, (i+1)*num_dim_-1) = M_J_.apply(
					Ieff.cols(first_source(i),last_source(i)),
					first_source(i),last_source(i));
			});
		}
	}

	// source to target kernel
	void CurrentMesh::source_to_target(
		const ShTargetsPr &tar, const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list,
		const arma::Row<arma::uword> &first_source, const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, const arma::Row<arma::uword> &last_target,
		const ShSettingsPr &stngs,
		const cmn::ShLogPr &lg) const{
		
		// calculate elemental currents
		//arma::Mat<fltp> Je(3,num_elements_,arma::fill::zeros);
		//for(arma::uword i=0;i<num_elements_;i++)
		//	Je.col(i) = arma::mean(Jn_.cols(n_.col(i)),1);

		// calculate effective current
		//const arma::Mat<fltp> Ieff = Je.each_row()%Ve_;

		// get targets
		// const arma::Mat<fltp> Rt = tar->get_target_coords();

		// generate gauss points
		// make gauss point calculator
		cmn::Gauss gp(num_gauss_);

		// extract abscissae and weights
		const arma::Row<fltp> xg = gp.get_abscissae(); 
		const arma::Row<fltp> wg = gp.get_weights();

		// forward calculation of vector potential to extra
		if(tar->has('A')){
			// allocate
			arma::Mat<fltp> A(num_dim_,tar->num_targets(),arma::fill::zeros);

			// walk over source nodes
			cmn::parfor(0,target_list.n_elem,stngs->get_parallel_s2t(),[&](arma::uword i, int){
			//for(arma::uword i=0;i<target_list.n_elem;i++){
				// check cancel flag
				if(lg->is_cancelled())return;

				// get target node
				const arma::uword target_idx = target_list(i);

				// get location of the targets
				const arma::uword ft = first_target(target_idx);
				const arma::uword lt = last_target(target_idx);
				assert(ft<=lt); assert(lt<tar->num_targets());

				// my target positions
				const arma::Mat<fltp> myRt = tar->get_target_coords(ft,lt);

				// walk over source nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// get target node
					const arma::uword source_idx = source_list(i)(j);

					// get my source elements
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<num_elements_);

					// walk over sources in list
					for(arma::uword n=0;n<(ls-fs+1);n++){
						// get source index
						const arma::uword mysource = fs+n;

						// calculate distance of 
						// all target points to this source
						const arma::Row<fltp> dist = cmn::Extra::vec_norm(myRt.each_col() - Re_.col(mysource));

						// find indexes of target points 
						// that are far away
						const arma::Row<arma::uword> indices_far = 
							arma::find(dist>num_dist_*element_radius_(mysource)).t();
						const arma::Row<arma::uword> indices_near = 
							arma::find(dist<=num_dist_*element_radius_(mysource)).t();

						// run normal calculation for far targets
						if(!indices_far.is_empty()){
							A.cols(ft+indices_far) += Savart::calc_I2A(Re_.col(mysource),
								Je_.col(mysource)*Ve_(mysource), myRt.cols(indices_far),false);
						}

						// for target points that are close
						// use gauss points to do the integration
						if(!indices_near.is_empty()){
							// my nodes
							const arma::Mat<fltp>::fixed<3,8> myRn = Rn_.cols(n_.col(mysource));

							// get quadrilateral coordinates (iteratively)
							arma::Mat<fltp> Rqt = cmn::Hexahedron::cart2quad(
								myRn, myRt.cols(indices_near), 1e-4);

							// clamp such that quadrilateral coordinates 
							// are forced inside the element
							Rqt = arma::clamp(Rqt,-RAT_CONST(1.0),RAT_CONST(1.0));

							// walk over targets in list
							for(arma::uword k=0;k<indices_near.n_elem;k++){
								// get target index
								const arma::uword mytarget = ft+indices_near(k);

								// setup grid around singularity
								arma::Mat<fltp> Rqgrd; arma::Row<fltp> wgrd;
								cmn::Hexahedron::setup_source_grid(Rqgrd, wgrd, Rqt.col(k), xg, wg);
								assert(std::abs(arma::sum(wgrd)-RAT_CONST(1.0))<1e-4);
								
								// calculate points in carthesian coordinates
								const arma::Mat<fltp> Rc = cmn::Hexahedron::quad2cart(myRn,Rqgrd);
								
								// allocate effective current for gauss points
								arma::Mat<fltp> Ieffc(3,Rc.n_cols);
								Ieffc.each_row() = wgrd; Ieffc.each_col() %= Ve_(mysource)*Je_.col(mysource);
								// const arma::Mat<fltp> Ieffc = Ve_(mysource)*(cmn::Hexahedron::quad2cart(
								// 	Jn_.cols(n_.col(mysource)),Rqgrd).each_row()%wgrd);

								// calculate and sum field of all gauss points
								// for vector potential
								// A.col(mytarget) += Savart::calc_I2A(
								// 	Rc,Ieffc,Rt.col(mytarget),false);
								A.col(mytarget) += Savart::calc_I2A(
									Rc,Ieffc,myRt.col(indices_near(k)),false);
							}
						}

					}
				}
			});
			//}

			// check cancel flag
			if(lg->is_cancelled())return;
			

			//std::cout<<H.t()<<std::endl;

			// set field to targets
			tar->add_field('A',A,true);
		}

		// forward calculation of vector potential to extra
		if(tar->has('H') || tar->has('B')){
			// allocate
			arma::Mat<fltp> H(num_dim_,tar->num_targets(),arma::fill::zeros);

			// walk over source nodes
			cmn::parfor(0,target_list.n_elem,stngs->get_parallel_s2t(),[&](arma::uword i, int){
			//for(arma::uword i=0;i<target_list.n_elem;i++){
				// check cancel flag
				if(lg->is_cancelled())return;

				// get target node
				const arma::uword target_idx = target_list(i);

				// get location of the targets
				const arma::uword ft = first_target(target_idx);
				const arma::uword lt = last_target(target_idx);
				assert(ft<=lt); assert(lt<tar->num_targets());

				// my target positions
				const arma::Mat<fltp> myRt = tar->get_target_coords(ft,lt);

				// walk over source nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// get target node
					const arma::uword source_idx = source_list(i)(j);

					// get my source elements
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<num_elements_);

					// walk over sources in list
					for(arma::uword n=0;n<(ls-fs+1);n++){
						// get source index
						const arma::uword mysource = fs+n;

						// calculate distance of 
						// all target points to this source
						const arma::Row<fltp> dist = cmn::Extra::vec_norm(myRt.each_col() - Re_.col(mysource));

						// find indexes of target points 
						// that are far away
						const arma::Row<arma::uword> indices_far = 
							arma::find(dist>num_dist_*element_radius_(mysource)).t();
						const arma::Row<arma::uword> indices_near = 
							arma::find(dist<=num_dist_*element_radius_(mysource)).t();

						// run normal calculation for far targets
						if(!indices_far.is_empty()){
							H.cols(ft+indices_far) += Savart::calc_I2H(Re_.col(mysource),
								Je_.col(mysource)*Ve_(mysource), myRt.cols(indices_far),false);
						}

						// for target points that are close
						// use gauss points to do the integration
						if(!indices_near.is_empty()){
							// my nodes
							const arma::Mat<fltp>::fixed<3,8> myRn = Rn_.cols(n_.col(mysource));

							// get quadrilateral coordinates (iteratively)
							arma::Mat<fltp> Rqt = cmn::Hexahedron::cart2quad(
								myRn, myRt.cols(indices_near), 1e-4);

							// clamp such that quadrilateral coordinates 
							// are forced inside the element
							Rqt = arma::clamp(Rqt,-RAT_CONST(1.0),RAT_CONST(1.0));

							// walk over targets in list
							for(arma::uword k=0;k<indices_near.n_elem;k++){
								// get target index
								const arma::uword mytarget = ft+indices_near(k);

								// setup grid around singularity
								arma::Mat<fltp> Rqgrd; arma::Row<fltp> wgrd;
								cmn::Hexahedron::setup_source_grid(Rqgrd, wgrd, Rqt.col(k), xg, wg);
								assert(std::abs(arma::sum(wgrd)-RAT_CONST(1.0))<1e-4);
								
								// calculate points in carthesian coordinates
								const arma::Mat<fltp> Rc = cmn::Hexahedron::quad2cart(myRn,Rqgrd);
								
								// allocate effective current for gauss points
								arma::Mat<fltp> Ieffc(3,Rc.n_cols);
								Ieffc.each_row() = wgrd; Ieffc.each_col() %= Ve_(mysource)*Je_.col(mysource);
								// const arma::Mat<fltp> Ieffc = Ve_(mysource)*(cmn::Hexahedron::quad2cart(
								// 	Jn_.cols(n_.col(mysource)),Rqgrd).each_row()%wgrd);

								// calculate and sum field of all gauss points
								// for vector potential
								H.col(mytarget) += Savart::calc_I2H(Rc,Ieffc,myRt.col(indices_near(k)),false);
							}
						}

					}
				}
			});
			//}

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			if(tar->has('H'))tar->add_field('H',H,true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,true);
		}
	}

	// direct calculation
	void CurrentMesh::calc_direct(const ShTargetsPr &tar, const ShSettingsPr &stngs, const cmn::ShLogPr &lg) const{
		
		// generate gauss points
		// make gauss point calculator
		cmn::Gauss gp(num_gauss_);

		// extract abscissae and weights
		const arma::Row<fltp> xg = gp.get_abscissae(); 
		const arma::Row<fltp> wg = gp.get_weights();

		// forward calculation of vector potential to extra
		if(tar->has('A')){
			// get targets
			const arma::Mat<fltp>& Rt = tar->get_target_coords();

			// allocate
			arma::Mat<fltp> A(num_dim_,tar->num_targets(),arma::fill::zeros);

			// walk over sources in list
			for(arma::uword n=0;n<num_elements_;n++){
				// calculate distance of 
				// all target points to this source
				const arma::Mat<fltp> dist = cmn::Extra::vec_norm(Rt.each_col() - Re_.col(n));

				// find indexes of target points 
				// that are far away
				const arma::Row<arma::uword> indices_far = 
					arma::find(dist>num_dist_*element_radius_(n)).t();
				const arma::Row<arma::uword> indices_near = 
					arma::find(dist<=num_dist_*element_radius_(n)).t();

				// run normal calculation for far targets
				if(!indices_far.is_empty()){
					A.cols(indices_far) += Savart::calc_I2A(Re_.col(n),
						Je_.col(n)*Ve_(n), Rt.cols(indices_far),stngs->get_parallel_s2t());
				}

				// for target points that are close
				// use gauss points to do the integration
				if(!indices_near.is_empty()){
					// my nodes
					const arma::Mat<fltp>::fixed<3,8> myRn = Rn_.cols(n_.col(n));

					// get quadrilateral coordinates (iteratively)
					arma::Mat<fltp> Rqt = cmn::Hexahedron::cart2quad(
						myRn, Rt.cols(indices_near), 1e-4);

					// clamp such that quadrilateral coordinates 
					// are forced inside the element
					Rqt = arma::clamp(Rqt,-1.0,1.0);

					// walk over targets in list
					//for(arma::uword k=0;k<indices_near.n_elem;k++){
					cmn::parfor(0,indices_near.n_elem,stngs->get_parallel_s2t(),[&](arma::uword k, int){
						// check cancel flag
						if(lg->is_cancelled())return;

						// get my target
						const arma::uword mytarget = indices_near(k);

						// setup grid around singularity
						arma::Mat<fltp> Rqgrd; arma::Row<fltp> wgrd;
						cmn::Hexahedron::setup_source_grid(Rqgrd, wgrd, Rqt.col(k), xg, wg);
						assert(std::abs(arma::sum(wgrd)-RAT_CONST(1.0))<RAT_CONST(1e-6));
						
						// calculate points in carthesian coordinates
						const arma::Mat<fltp> Rc = cmn::Hexahedron::quad2cart(myRn,Rqgrd);
						
						// allocate effective current for gauss points
						arma::Mat<fltp> Ieffc(3,Rc.n_cols);
						Ieffc.each_row() = wgrd; Ieffc.each_col() %= Ve_(n)*Je_.col(n);
						// const arma::Mat<fltp> Ieffc = Ve_(n)*(cmn::Hexahedron::quad2cart(
						// 	Jn_.cols(n_.col(n)),Rqgrd).each_row()%wgrd);

						// calculate and sum field of all gauss points
						// for vector potential
						A.col(mytarget) += Savart::calc_I2A(Rc,Ieffc,Rt.col(mytarget),false);
					//}
					});
				}
			}

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			tar->add_field('A',A,true);
		}

		// forward calculation of vector potential to extra
		if(tar->has('H') || tar->has('B')){
			// get targets
			const arma::Mat<fltp>& Rt = tar->get_target_coords();

			// allocate
			arma::Mat<fltp> H(num_dim_,tar->num_targets(),arma::fill::zeros);

			// walk over sources in list
			for(arma::uword n=0;n<num_elements_;n++){
				// calculate distance of 
				// all target points to this source
				const arma::Mat<fltp> dist = cmn::Extra::vec_norm(Rt.each_col() - Re_.col(n));

				// find indexes of target points 
				// that are far away
				const arma::Row<arma::uword> indices_far = 
					arma::find(dist>num_dist_*element_radius_(n)).t();
				const arma::Row<arma::uword> indices_near = 
					arma::find(dist<=num_dist_*element_radius_(n)).t();

				// run normal calculation for far targets
				if(!indices_far.is_empty()){
					H.cols(indices_far) += Savart::calc_I2H(Re_.col(n),
						Je_.col(n)*Ve_(n), Rt.cols(indices_far),stngs->get_parallel_s2t());
				}

				// for target points that are close
				// use gauss points to do the integration
				if(!indices_near.is_empty()){
					// my nodes
					const arma::Mat<fltp>::fixed<3,8> myRn = Rn_.cols(n_.col(n));

					// get quadrilateral coordinates (iteratively)
					arma::Mat<fltp> Rqt = cmn::Hexahedron::cart2quad(
						myRn, Rt.cols(indices_near), RAT_CONST(1e-4));

					// clamp such that quadrilateral coordinates 
					// are forced inside the element
					Rqt = arma::clamp(Rqt,-RAT_CONST(1.0),RAT_CONST(1.0));

					// walk over targets in list
					//for(arma::uword k=0;k<indices_near.n_elem;k++){
					cmn::parfor(0,indices_near.n_elem,stngs->get_parallel_s2t(),[&](arma::uword k, int){
						// check cancel flag
						if(lg->is_cancelled())return;

						// get my target
						const arma::uword mytarget = indices_near(k);

						// setup grid around singularity
						arma::Mat<fltp> Rqgrd; arma::Row<fltp> wgrd;
						cmn::Hexahedron::setup_source_grid(Rqgrd, wgrd, Rqt.col(k), xg, wg);
						assert(std::abs(arma::sum(wgrd)-RAT_CONST(1.0))<RAT_CONST(1e-6));
						
						// calculate points in carthesian coordinates
						const arma::Mat<fltp> Rc = cmn::Hexahedron::quad2cart(myRn,Rqgrd);
						
						// allocate effective current for gauss points
						arma::Mat<fltp> Ieffc(3,Rc.n_cols);
						Ieffc.each_row() = wgrd; Ieffc.each_col() %= Ve_(n)*Je_.col(n);
						// const arma::Mat<fltp> Ieffc = Ve_(n)*(cmn::Hexahedron::quad2cart(
						// 	Jn_.cols(n_.col(n)),Rqgrd).each_row()%wgrd);

						// calculate and sum field of all gauss points
						// for vector potential
						H.col(mytarget) += Savart::calc_I2H(Rc,Ieffc,Rt.col(mytarget),false);
					//}
					});
				}
			}

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			if(tar->has('H'))tar->add_field('H',H,true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,true);
		}

	}

	// basic build in mesh shape cylinder
	// for testing purposes
	// note that after calling this method
	// it is still necessary to calculate element
	// volumes to complete setup.
	void CurrentMesh::setup_solenoid(
		const fltp Rin, const fltp Rout, 
		const fltp height, const arma::uword nr, 
		const arma::uword nz, const arma::uword nl, 
		const fltp J){

		// check user input
		if(Rout<=Rin)rat_throw_line("outer radius must be larger than inner radius");
		if(height<=0)rat_throw_line("height must be larger than zero");
		if(nr<=1)rat_throw_line("number of radial coordinates must be larger than one");
		if(nz<=1)rat_throw_line("number of axial coordinates must be larger than one");
		if(nl<=1)rat_throw_line("number of azymuthal coordinates must be larger than one");

		// create azymuthal coordinates of nodes
		arma::Row<fltp> theta = arma::linspace<arma::Row<fltp> >(0,-(RAT_CONST(1.0)-RAT_CONST(1.0)/nl)*2*arma::Datum<fltp>::pi,nl);

		// create radial coordinates of nodes
		arma::Row<fltp> rho = arma::linspace<arma::Row<fltp> >(Rin,Rout,nr);

		// create axial cooridnates of nodes
		arma::Row<fltp> z = arma::linspace<arma::Row<fltp> >(-height/2,height/2,nz);

		// create matrix to hold node coordinates
		arma::Mat<fltp> xn(nl,nr*nz), yn(nl,nr*nz), zn(nl,nr*nz);

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

		
		// create matrix of node indices
		arma::Mat<arma::uword> node_idx = arma::regspace<arma::Mat<arma::uword> >(0,num_nodes_-1);
		node_idx.reshape(nl,nr*nz);

		// close mesh by setting last row to the first
		node_idx = arma::join_vert(node_idx,node_idx.row(0));

		// get definition of hexahedron element
		arma::Mat<arma::sword>::fixed<8,3> M = 
			arma::conv_to<arma::Mat<arma::sword> >::from(
			(cmn::Hexahedron::get_corner_nodes()+1)/2);

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

		// calculate volume and element centroids
		calculate_element_volume();

		// element orientation	
		arma::Mat<fltp> Ne = arma::join_vert(Re_.rows(0,1),arma::Row<fltp>(num_elements_,arma::fill::zeros));
		Ne = Ne.each_row()/cmn::Extra::vec_norm(Ne);
		arma::Mat<fltp> De(3,num_elements_,arma::fill::zeros); De.row(2).fill(RAT_CONST(1.0));
		arma::Mat<fltp> Le = cmn::Extra::cross(Ne,De);

		// set currents
		Je_ = -J*Le; 

		// check current density size
		assert(Je_.n_cols==num_elements_);
		assert(Je_.n_rows==3);
	}

}}