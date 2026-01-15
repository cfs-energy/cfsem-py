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
#include "mgnchargesurface.hh"

// rat-mlfmm headers
#include "chargedpolyhedron.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// default constructor
	// default is hex
	MgnChargeSurface::MgnChargeSurface(){
		set_field_type('H',3);
	}

	// default constructor
	// default is hex
	MgnChargeSurface::MgnChargeSurface(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Row<fltp> &sigma) : MgnChargeSurface(){
		set_mesh(Rn,n,sigma);
	}

	// factory
	ShMgnChargeSurfacePr MgnChargeSurface::create(){
		return std::make_shared<MgnChargeSurface>();
	}

	// factory
	ShMgnChargeSurfacePr MgnChargeSurface::create(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Row<fltp> &sigma){
		return std::make_shared<MgnChargeSurface>(Rn,n,sigma);
	}

	// setting a hexagonal mesh with volume elements
	void MgnChargeSurface::set_mesh(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Row<fltp> &sigma){

		// check input
		if(Rn.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(n.n_rows!=3 && n.n_rows!=4)rat_throw_line("element node index matrix must have three or four rows");
		if(n.max()>=Rn.n_cols)rat_throw_line("element matrix contains index outside coordinate matrix");

		// set supplied values
		Rn_ = Rn; n_ = n; sigma_ = sigma; 

		// get number of supplied nodes
		num_nodes_ = Rn_.n_cols;
		num_elements_ = n_.n_cols;	

		// calculate areas
		calculate_element_areas();

		// calculate field at element centroids
		Rt_ = Re_; num_targets_ = Rt_.n_cols;
	}

	// calculate areas and centroids of the elements
	void MgnChargeSurface::calculate_element_areas(){
		// check if it is really quad
		if(n_.n_rows!=3 && n_.n_rows!=4)rat_throw_line("element node index matrix must have three or four rows");

		// calculate face normals
		Ne_ = n_.n_rows==3 ? 
			cmn::Triangle::calc_face_normals(Rn_,n_) :
			cmn::Quadrilateral::calc_face_normals(Rn_,n_);

		// calculate face area using two triangles
		Ae_ = n_.n_rows==3 ? 
			cmn::Triangle::calc_area(Rn_,n_) :
			cmn::Quadrilateral::calc_area(Rn_,n_);

		// allocate centroids
		Re_.set_size(3,num_elements_);

		// walk over elements and calculate centroids
		arma::Col<fltp>::fixed<2> Rq = {0,0};
		for(arma::uword i=0;i<num_elements_;i++){
			// calculate centroid using quadrilateral coordinates
			Re_.col(i) = n_.n_rows==3 ? 
				cmn::Triangle::quad2cart(Rn_.cols(n_.col(i)),Rq) : 
				cmn::Quadrilateral::quad2cart(Rn_.cols(n_.col(i)),Rq);
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
	arma::Row<fltp> MgnChargeSurface::get_area() const{
		// check if volumes were calculated
		if(Ae_.is_empty())rat_throw_line("element area vector was not calculated");

		// return element volumes
		return Ae_;
	}

	// get number of sources (each element has one source)
	arma::uword MgnChargeSurface::get_num_nodes() const{
		assert(num_nodes_>0);
		return num_nodes_;
	}

	// get number of sources (each element has one source)
	arma::uword MgnChargeSurface::get_num_elements() const{
		assert(num_elements_>0);
		return num_elements_;
	}

	// get node coordinates
	const arma::Mat<fltp>& MgnChargeSurface::get_node_coords() const{
		// check if node coordinates were set
		if(Rn_.is_empty())rat_throw_line("node coordinate matrix not set");

		// return node coordinates
		return Rn_;
	}

	// get element node indices
	const arma::Mat<arma::uword>& MgnChargeSurface::get_elements() const{
		// check if elements were set
		if(n_.is_empty())rat_throw_line("element node index matrix not set");

		// return 
		return n_;
	}

	// get element centroids
	arma::Mat<fltp> MgnChargeSurface::get_source_coords() const{
		// check if coordinates were set
		if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix not calculated");

		// return element centroids
		return Re_;
	}

	// // get element centroids of specific elements
	// arma::Mat<fltp> MgnChargeSurface::get_source_coords(
	// 	const arma::Row<arma::uword> &indices) const{
	// 	// check if coordinates were set
	// 	if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix not calculated");

	// 	// return element centroids
	// 	return Re_.cols(indices);
	// }

	// count number of sources stored
	arma::uword MgnChargeSurface::num_sources() const{
		// return number of elements
		assert(num_elements_>0);
		return num_elements_;
	}

	// get number of dimensions
	arma::uword MgnChargeSurface::get_num_dim() const{
		assert(num_dim_>0);
		return num_dim_;
	}

	// get elmenet size
	fltp MgnChargeSurface::element_size() const{
		return 2*arma::max(element_radius_);
	}

	// sorting function
	void MgnChargeSurface::sort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		if(n_.is_empty())rat_throw_line("element matrix was not set");
		if(sigma_.is_empty())rat_throw_line("current density matrix was not set");
		if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix was not set");
		if(Ae_.is_empty())rat_throw_line("element area vector was not set");
		if(Ne_.is_empty())rat_throw_line("face normal matrix was not set");
		if(element_radius_.is_empty())rat_throw_line("element radius vector was not set");

		// check if sort array right length
		assert(n_.n_cols == sort_idx.n_elem);

		// sort sources
		n_ = n_.cols(sort_idx);
		sigma_ = sigma_.cols(sort_idx);
		Re_ = Re_.cols(sort_idx);
		Ae_ = Ae_.cols(sort_idx);
		Ne_ = Ne_.cols(sort_idx);
		element_radius_ = element_radius_.cols(sort_idx);
	}

	// unsorting function
	void MgnChargeSurface::unsort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		if(n_.is_empty())rat_throw_line("element matrix was not set");
		if(sigma_.is_empty())rat_throw_line("current density matrix was not set");
		if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix was not set");
		if(Ae_.is_empty())rat_throw_line("element area vector was not set");
		if(Ne_.is_empty())rat_throw_line("face normal matrix was not set");
		if(element_radius_.is_empty())rat_throw_line("element radius vector was not set");

		// check if sort array right length
		assert(n_.n_cols == sort_idx.n_elem);

		// sort sources
		n_.cols(sort_idx) = n_;
		sigma_.cols(sort_idx) = sigma_;
		Re_.cols(sort_idx) = Re_;
		Ae_.cols(sort_idx) = Ae_;
		Ne_.cols(sort_idx) = Ne_;
		element_radius_.cols(sort_idx) = element_radius_;
	}


	// setup source to multipole matrices
	void MgnChargeSurface::setup_source_to_multipole(
		const arma::Mat<fltp> &dR,
		const ShSettingsPr &stngs){
		
		// memory efficient implementation (default)
		if(stngs->get_memory_efficient_s2m()){
			dR_ = dR;
		}

		// maximize speed over memory efficiency
		else{
			// get number of expansions
			const int num_exp = stngs->get_num_exp();

			// set number of expansions and setup matrix
			M_J_.set_num_exp(num_exp);
			M_J_.calc_matrix(-dR);
		}
	}

	// get multipole contribution of the sources with indices
	// the contributions of the sources are already summed
	void MgnChargeSurface::source_to_multipole(
		arma::Mat<std::complex<fltp> > &Mp,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source,
		const ShSettingsPr &stngs) const{
		
		// check input
		assert(first_source.n_elem==last_source.n_elem);
		assert(!Mp.is_empty());

		// calculate elemental currents
		// arma::Mat<fltp> Je(3,num_elements_,arma::fill::zeros);
		// for(arma::uword i=0;i<num_elements_;i++)
		// 	Je.col(i) = arma::mean(Jn_.cols(n_.col(i)),1);

		// get number of expansions
		const int num_exp = stngs->get_num_exp();

		// calculate effective current
		const arma::Mat<fltp> charge = sigma_%Ae_;

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
					charge.cols(first_source(i),last_source(i)));
			});
		}

		// faster less memory efficient implementation
		else{
			// walk over source nodes
			cmn::parfor(0,first_source.n_elem,stngs->get_parallel_s2m(),[&](arma::uword i, int){
				// add child source contribution to this multipole
				Mp.cols(i*num_dim_, (i+1)*num_dim_-1) = M_J_.apply(
					charge.cols(first_source(i),last_source(i)),
					first_source(i),last_source(i));
			});
		}
	}

	// source to target kernel
	void MgnChargeSurface::source_to_target(
		const ShTargetsPr &tar, const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list,
		const arma::Row<arma::uword> &first_source, const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, const arma::Row<arma::uword> &last_target, 
		const ShSettingsPr &stngs,
		const cmn::ShLogPr &lg) const{

		// forward calculation of vector potential to extra
		if(tar->has('H') || tar->has('B')){
			// allocate
			arma::Mat<fltp> H(3,tar->num_targets(),arma::fill::zeros);

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
					// check cancel flag
					if(lg->is_cancelled())return;

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
						const arma::Row<fltp> dist = cmn::Extra::vec_norm(
							myRt.each_col() - Re_.col(mysource));

						// find indexes of target points 
						// that are far away
						const arma::Row<arma::uword> indices_far = 
							arma::find(dist>num_dist_*element_radius_(mysource)).t();
						const arma::Row<arma::uword> indices_near = 
							arma::find(dist<=num_dist_*element_radius_(mysource)).t();

						// run normal calculation for far targets
						if(!indices_far.is_empty()){
							const arma::Mat<fltp> dR = myRt.cols(indices_far).eval().each_col() - Re_.col(mysource);
							H.cols(ft+indices_far) += sigma_(mysource)*Ae_(mysource)*(dR.each_row()/arma::pow(cmn::Extra::vec_norm(dR),3))/(4*arma::Datum<fltp>::pi);
						}

						// for target points that are close
						// use gauss points to do the integration
						if(!indices_near.is_empty()){
							// use charged polygon analytical equations
							H.cols(ft+indices_near) += 
								sigma_(mysource)*ChargedPolyhedron::calc_magnetic_field(
								Rn_.cols(n_.col(mysource)), myRt.cols(indices_near))/(4*arma::Datum<fltp>::pi);
						}

					}
				}
			});

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			if(tar->has('H'))tar->add_field('H',H,true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,true);
		}
	}

	// direct calculation
	void MgnChargeSurface::calc_direct(
		const ShTargetsPr &tar, const ShSettingsPr &/*stngs*/, const cmn::ShLogPr &lg) const{
		// get targets
		const arma::Mat<fltp> Rt = tar->get_target_coords();

		// forward calculation of vector potential to extra
		if(tar->has('H') || tar->has('B')){
			// allocate
			arma::Mat<fltp> H(3,tar->num_targets(),arma::fill::zeros);

			// walk over sources in list
			for(arma::uword n=0;n<num_elements_;n++){
				// check cancel flag
				if(lg->is_cancelled())return;

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
					const arma::Mat<fltp> dR = Rt.cols(indices_far).eval().each_col() - Re_.col(n);
					H.cols(indices_far) += sigma_(n)*Ae_(n)*(dR.each_row()/arma::pow(cmn::Extra::vec_norm(dR),3))/(4*arma::Datum<fltp>::pi);
				}

				// for target points that are close
				// use gauss points to do the integration
				if(!indices_near.is_empty()){
					H.cols(indices_near) += sigma_(n)*ChargedPolyhedron::calc_magnetic_field(
						Rn_.cols(n_.col(n)), Rt.cols(indices_near))/(4*arma::Datum<fltp>::pi);
				}
			}

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			if(tar->has('H'))tar->add_field('H',H,true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,true);
		}
	}

	// subdivision into smaller elements
	ShSourcesPr MgnChargeSurface::subdivide(const fltp grid_size){
		// shape depenent parameters
		const arma::uword num_sub_nodes = n_.n_rows==3 ? 6llu : 9llu;
		const arma::uword num_sub_elements = 4llu;
		const arma::Mat<arma::uword> E = n_.n_rows==3 ? 
			cmn::Triangle::get_edges().eval() : 
			cmn::Quadrilateral::get_edges().eval();

		// element subdivision table
		const arma::Mat<arma::uword> N = n_.n_rows==3 ? 
			arma::reshape(arma::Mat<arma::uword>{0,3,5, 3,1,4, 3,4,5, 5,4,2},3,4) : 
			arma::reshape(arma::Mat<arma::uword>{0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,6,3},4,4);

		// find elements to subdivide
		const arma::Col<arma::uword> idx_sub = arma::find(2*element_radius_>grid_size);
		const arma::Col<arma::uword> idx_keep = arma::find(2*element_radius_<=grid_size);

		// allocate new nodes and elements
		arma::Mat<fltp> Rnnew(3,idx_sub.n_elem*num_sub_nodes);
		arma::Mat<arma::uword> nnew(n_.n_rows,idx_sub.n_elem*num_sub_elements);
		arma::Row<fltp> sigmanew(idx_sub.n_elem*num_sub_elements);

		// walk over elements
		for(arma::uword i=0;i<idx_sub.n_elem;i++){
			// get nodes for this elements
			const arma::Mat<fltp> Rn = Rn_.cols(n_.col(idx_sub(i)));

			// create new nodes on edges
			arma::Mat<fltp> Re(3,E.n_rows);
			for(arma::uword j=0;j<E.n_rows;j++)
				Re.col(j) = arma::mean(Rn.cols(E.row(j)),1);
			
			// create new node in middle
			const arma::Col<fltp>::fixed<3> Rm = arma::mean(Rn,1);

			// indexing
			const arma::uword idx1 = i*num_sub_nodes;
			const arma::uword idx2 = (i+1)*num_sub_nodes - 1;

			// combine and store nodes
			Rnnew.cols(idx1,idx2) = n_.n_rows==3 ? 
				arma::join_horiz(Rn,Re) : arma::join_horiz(Rn,Re,Rm);

			// create elements
			const arma::uword idx3 = i*num_sub_elements;
			const arma::uword idx4 = (i+1)*num_sub_elements - 1;
			nnew.cols(idx3,idx4) = N + i*num_sub_nodes;

			// currents
			if(!sigmanew.empty())
				sigmanew.cols(idx3,idx4) = arma::repmat(sigma_.col(idx_sub(i)),1,num_sub_elements);
		}

		// create new charges
		const ShMgnChargeSurfacePr new_charges = MgnChargeSurface::create(
			arma::join_horiz(Rnnew,Rn_),
			arma::join_horiz(nnew,n_.cols(idx_keep)+Rnnew.n_cols),
			arma::join_horiz(sigmanew,sigma_.cols(idx_keep)));
		new_charges->cleanup_mesh();
		return new_charges;
	}

	// cleanup mesh
	void MgnChargeSurface::cleanup_mesh(){
		// calculate nodes that are still connected
		arma::Col<arma::uword> node_valence(Rn_.n_cols,arma::fill::zeros);
		for(arma::uword i=0;i<n_.n_cols;i++)node_valence.rows(n_.col(i))+=1;

		// create reindexing array
		const arma::Col<arma::uword> reindex1 = arma::join_vert(
			arma::Col<arma::uword>{0}, arma::cumsum(node_valence>0));

		// shed rows
		Rn_.shed_cols(arma::find(node_valence==0));

		// merge co-inciding nodes
		const arma::Row<arma::uword> reindex2 = cmn::Extra::combine_nodes(Rn_);

		// update elements
		n_ = arma::reshape(reindex2(reindex1(arma::vectorise(n_))),n_.n_rows,n_.n_cols);

		// perform setup again
		num_nodes_ = Rn_.n_cols;
		num_elements_ = n_.n_cols;

		// calculate volume and centroids
		calculate_element_areas();

		// calculate field at element centroids
		Rt_ = Re_; num_targets_ = Rt_.n_cols;
	}

	// calculate force
	arma::Mat<fltp> MgnChargeSurface::calc_force()const{
		return arma::Datum<fltp>::mu_0*(get_field('H').each_row()%(Ae_%sigma_));
	}

}}

