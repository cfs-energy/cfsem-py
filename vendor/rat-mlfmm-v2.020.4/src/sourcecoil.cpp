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
#include "sourcecoil.hh"

// include fmm headers
#include "brick8node.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// default constructor
	// default is hex
	SourceCoil::SourceCoil(){
		
	}

	// constructor with mesh input
	SourceCoil::SourceCoil(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Row<fltp> &Ie){
		set_mesh(Rn,n,Ie);
	}

	// factory
	ShSourceCoilPr SourceCoil::create(){
		return std::make_shared<SourceCoil>();
	}

	// factory
	ShSourceCoilPr SourceCoil::create(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Row<fltp> &Ie){
		return std::make_shared<SourceCoil>(Rn,n,Ie);
	}

	// set mesh
	void SourceCoil::set_mesh(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Row<fltp> &Ie){

		// check user input
		if(Rn.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(n.n_rows!=8)rat_throw_line("element matrix must have eight rows");
		if(Ie.n_cols!=n.n_cols)rat_throw_line("current density and element matrix must have same number of columns");
		if(n.max()>=Rn.n_cols && n.n_cols>0)rat_throw_line("element matrix contains index outside coordinate matrix");

		// set supplied values
		Rn_ = Rn; n_ = n;

		// set current density
		Ie_ = Ie;

		// calculate volume and centroids
		calculate_element_volume();
	}

	// setters
	void SourceCoil::set_num_gauss(const arma::sword num_gauss){
		num_gauss_ = num_gauss;
	}

	void SourceCoil::set_num_planes(const arma::uword num_planes){
		num_planes_ = num_planes;
	}

	// get element size
	fltp SourceCoil::element_size() const{
		assert(n_.n_rows==8);
		return arma::max(cmn::Extra::vec_norm(Rn_.cols(n_.row(6)) - Rn_.cols(n_.row(0))));
	}

	// calculate hexahedron volumes
	// by splitting it up into five tetrahedrons
	// then adds their volumes
	void SourceCoil::calculate_element_volume(){
		// check if it is really hexahedron
		assert(n_.n_rows==8);
		assert(!Ie_.empty());

		// volume of elements
		Ve_ = cmn::Hexahedron::calc_volume(Rn_,n_);

		// calculate effective current
		Re_.set_size(3,n_.n_cols);
		for(arma::uword i=0;i<n_.n_cols;i++)
			Re_.col(i) = arma::mean(Rn_.cols(n_.col(i)),1);

		// distance to center
		element_radius_.set_size(n_.n_cols);
		for(arma::uword i=0;i<n_.n_cols;i++)
			element_radius_(i) = arma::max(cmn::Extra::vec_norm(Rn_.cols(n_.col(i)).eval().each_col() - Re_.col(i)));
	}

	// get element volumes
	const arma::Row<fltp>& SourceCoil::get_volume() const{
		// check if volumes were calculated
		assert(!Ve_.is_empty());

		// return element volumes
		return Ve_;
	}

	// get number of sources (each element has one source)
	arma::uword SourceCoil::get_num_nodes() const{
		return Rn_.n_cols;
	}

	// get number of sources (each element has one source)
	arma::uword SourceCoil::get_num_elements() const{
		return n_.n_cols;
	}

	// get node coordinates
	const arma::Mat<fltp>& SourceCoil::get_node_coords() const{
		return Rn_;
	}

	// get element node indices
	const arma::Mat<arma::uword>& SourceCoil::get_elements() const{
		return n_;
	}

	// get element centroids
	arma::Mat<fltp> SourceCoil::get_source_coords() const{
		// check if coordinates were set
		if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix not calculated");

		// return element centroids
		return Re_;
	}

	// count number of sources stored
	arma::uword SourceCoil::num_sources() const{
		// return number of elements
		return n_.n_cols;
	}

	// get number of dimensions
	arma::uword SourceCoil::get_num_dim() const{
		return num_dim_;
	}

	// sorting function
	void SourceCoil::sort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		if(n_.is_empty())rat_throw_line("element node index matrix not set");
		if(Ie_.is_empty())rat_throw_line("current density matrix not set");
		if(Re_.is_empty())rat_throw_line("element centroid matrix not calculated");
		if(Ve_.is_empty())rat_throw_line("element volume vector not calculated");
		if(element_radius_.is_empty())rat_throw_line("element radius vector not calculated");
		
		// check if sort array right length
		assert(n_.n_cols == sort_idx.n_elem);

		// sort sources
		n_ = n_.cols(sort_idx);
		Ie_ = Ie_.cols(sort_idx);
		Re_ = Re_.cols(sort_idx);
		Ve_ = Ve_.cols(sort_idx);
		element_radius_ = element_radius_.cols(sort_idx);
	}

	// unsorting function
	void SourceCoil::unsort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		if(n_.is_empty())rat_throw_line("element node index matrix not set");
		if(Ie_.is_empty())rat_throw_line("current density matrix not set");
		if(Re_.is_empty())rat_throw_line("element centroid matrix not calculated");
		if(Ve_.is_empty())rat_throw_line("element volume vector not calculated");
		if(element_radius_.is_empty())rat_throw_line("element radius vector not calculated");
		
		// check if sort array right length
		assert(n_.n_cols == sort_idx.n_elem);

		// sort sources
		n_.cols(sort_idx) = n_;
		Ie_.cols(sort_idx) = Ie_;
		Re_.cols(sort_idx) = Re_;
		Ve_.cols(sort_idx) = Ve_;
		element_radius_.cols(sort_idx) = element_radius_;
	}


	// setup source to multipole matrices
	void SourceCoil::setup_source_to_multipole(
		const arma::Mat<fltp> &dR,
		const ShSettingsPr &/*stngs*/){
			
		// get number of expansions
		// const int num_exp = stngs->get_num_exp();

		// memory efficient implementation (default)
		dR_ = dR;
	}

	// get multipole contribution of the sources with indices
	// the contributions of the sources are already summed
	void SourceCoil::source_to_multipole(
		arma::Mat<std::complex<fltp> > &Mp,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source,
		const ShSettingsPr &stngs) const{

		// get number of expansions
		const int num_exp = stngs->get_num_exp();

		// check input
		assert(first_source.n_elem==last_source.n_elem);
		assert(!Mp.is_empty());

		// create gauss points
		const arma::Mat<rat::fltp> gp = cmn::Hexahedron::create_gauss_points(num_gauss_);
		const arma::Mat<rat::fltp> Rqg = gp.rows(0,2);
		const arma::Row<rat::fltp> wg = gp.row(3);

		// derivative of shape function at gauss points
		const arma::Mat<fltp> dN = cmn::Hexahedron::shape_function_derivative(Rqg);

		// vector shape function for nedelec/raviart thomas elements
		const arma::Mat<fltp> Vq = cmn::Hexahedron::raviart_thomas_hdiv_shape_function(Rqg);

		// only get faces 5 and 6
		const arma::Mat<fltp> Vq56 = Vq.rows(12,17);

		// walk over source nodes
		cmn::parfor(0,first_source.n_elem,stngs->get_parallel_s2m(),[&](arma::uword i, int){
			// create a local copy for the multipole
			arma::Mat<std::complex<fltp> > Mp_local(Mp.n_rows,num_dim_,arma::fill::zeros);

			// walk over sources that belong to this multipole
			for(arma::uword j=first_source(i);j<=last_source(i);j++){
				// get planes
				const arma::Mat<fltp>::fixed<3,8> Rn = Rn_.cols(n_.col(j));

				// integration points in cartesian coordinates
				const arma::Mat<fltp> Rcg = cmn::Hexahedron::quad2cart(Rn,Rqg);

				// jacobian matrices at integration points stored column wise
				const arma::Mat<fltp> J = cmn::Hexahedron::shape_function_jacobian(Rn,dN);

				// determinant of jacobian matrices
				const arma::Row<fltp> Jdet = cmn::Hexahedron::jacobian2determinant(J);

				// piola transform to get carthesian vectors
				// note that the faces in RAT are slightly different from defelement website
				const arma::Mat<fltp> Vc = cmn::Hexahedron::quad2cart_contravariant_piola(Vq56,J,Jdet);
				const arma::Mat<fltp> Vc0123 = Vc.rows(0,2); // face 5
				const arma::Mat<fltp> Vc4567 = Vc.rows(3,5); // face 6

				// calculate current at integration points
				const arma::Mat<fltp> Ieff = (Vc0123 - Vc4567).eval().each_row()%(Ie_(j)*wg%Jdet);

				// calculate offset
				const arma::Mat<fltp> Roffset = Rcg.each_col() - Re_.col(j);

				// create the source to multipole matrix for these elements
				StMat_So2Mp_J M_J;
				M_J.set_num_exp(num_exp);
				M_J.calc_matrix(-dR_.col(j) + Roffset.each_col()); // this + or -?

				// apply matrix and add to multipole
				Mp_local += M_J.apply(Ieff);
			}

			// set multipole
			Mp.cols(i*num_dim_, (i+1)*num_dim_-1) = Mp_local;
		});
	}

	// source to target kernel
	void SourceCoil::source_to_target(
		const ShTargetsPr &tar, const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list,
		const arma::Row<arma::uword> &first_source, const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, const arma::Row<arma::uword> &last_target,
		const ShSettingsPr &stngs,const cmn::ShLogPr &lg) const{


		// gauss points
		const arma::Mat<fltp> gp = cmn::Hexahedron::create_gauss_points(num_gauss_);
		const arma::Mat<fltp> Rqg = gp.rows(0,2); 
		const arma::Row<fltp> wg = gp.row(3);

		// get distance factor
		const fltp near_far_factor = near_far_factor_(std::min(near_far_factor_.n_elem-1, static_cast<arma::uword>(std::abs(num_gauss_))));

		// derivative of shape function at gauss points
		const arma::Mat<fltp> dN = cmn::Hexahedron::shape_function_derivative(Rqg);

		// vector shape function for nedelec/raviart thomas elements
		const arma::Mat<fltp> Vq = cmn::Hexahedron::raviart_thomas_hdiv_shape_function(Rqg);

		// only get faces 5 and 6, current is in mu direction
		const arma::Mat<fltp> Vq56 = Vq.rows(12,17); // [12-15] and [16-17]

		// allocate
		arma::Mat<fltp> Rs(3,n_.n_cols*Rqg.n_cols);
		arma::Mat<fltp> Ieff(3,n_.n_cols*Rqg.n_cols);

		// walk over source nodes
		cmn::parfor(0,n_.n_cols,true,[&](arma::uword i, arma::uword /*cpu*/) {
		// for(arma::uword i=0;i<n.n_cols;i++){
			// check cancel flag
			if(lg->is_cancelled())return;

			// get element nodes
			const arma::Mat<fltp>::fixed<3,8> Rn = Rn_.cols(n_.col(i));

			// integration points in cartesian coordinates
			Rs.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1) = cmn::Hexahedron::quad2cart(Rn,Rqg);

			// jacobian matrices at integration points stored column wise
			const arma::Mat<fltp> J = cmn::Hexahedron::shape_function_jacobian(Rn,dN);
			if(!J.is_finite())rat_throw_line("jacobian matrix contains non-finite values, i.e. malformed mesh");

			// determinant of jacobian matrices
			const arma::Row<fltp> Jdet = cmn::Hexahedron::jacobian2determinant(J);
			if(arma::any(Jdet<0))rat_throw_line("jacobian determinant less than zero, i.e. inverted elements detected");

			// piola transform to get carthesian vectors
			// note that the faces in RAT are slightly different from defelement website
			const arma::Mat<fltp> Vc = cmn::Hexahedron::quad2cart_contravariant_piola(Vq56,J,Jdet);
			const arma::Mat<fltp> Vc0123 = Vc.rows(0,2); // face 5
			const arma::Mat<fltp> Vc4567 = Vc.rows(3,5); // face 6

			// calculate current at integration points
			Ieff.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1) = (Vc0123 - Vc4567).eval().each_row()%(Ie_(i)*wg%Jdet);
		});

		// check cancel flag
		if(lg->is_cancelled())return;

		// forward calculation of vector potential to extra
		if(tar->has('A')){
			// allocate
			arma::Mat<fltp> A(num_dim_,tar->num_targets(),arma::fill::zeros);

			// walk over target nodes
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
				arma::Mat<fltp> local_A(3,lt-ft+1,arma::fill::zeros);

				// walk over source nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// check cancel flag
					if(lg->is_cancelled())return;

					// get target node
					const arma::uword source_idx = source_list(i)(j);

					// get my source elements
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<n_.n_cols);

					// walk over sources in list
					for(arma::uword mysource=fs;mysource<=ls;mysource++){
						// get source
						const arma::Mat<fltp> myRn = Rn_.cols(n_.col(mysource));

						// find target points far away
						const arma::Row<fltp> rhot = cmn::Extra::vec_norm(myRt.each_col() - Re_.col(mysource));
						const fltp rhon_max = near_far_factor*element_radius_(mysource);

						// find targets closeby and far away
						const arma::Col<arma::uword> idx_far = arma::find(rhot>rhon_max);
						const arma::Col<arma::uword> idx_near = arma::find(rhot<=rhon_max);

						// nearby targets
						if(!idx_near.empty()){
							local_A.cols(idx_near) += Brick8Node::calc_vector_potential(
								myRn,myRt.cols(idx_near),Ie_(mysource),num_planes_);
						}

						// far away targets
						if(!idx_far.empty()){
							local_A.cols(idx_far) += Savart::calc_I2A(
								Rs.cols(mysource*Rqg.n_cols,(mysource+1)*Rqg.n_cols-1),
								Ieff.cols(mysource*Rqg.n_cols,(mysource+1)*Rqg.n_cols-1),
								myRt.cols(idx_far),false);
						}
					}
				}

				// set to global
				A.cols(ft,lt) = local_A;
			});

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			tar->add_field('A',A,true);
		}

		// forward calculation of vector potential to extra
		if(tar->has('H') || tar->has('B')){
			// allocate
			arma::Mat<fltp> H(num_dim_,tar->num_targets(),arma::fill::zeros);

			// walk over target nodes
			cmn::parfor(0,target_list.n_elem,stngs->get_parallel_s2t(),[&](arma::uword i, int){
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
				arma::Mat<fltp> local_H(3,lt-ft+1,arma::fill::zeros);

				// walk over source nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// check cancel flag
					if(lg->is_cancelled())return;

					// get target node
					const arma::uword source_idx = source_list(i)(j);

					// get my source elements
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<n_.n_cols);

					// walk over sources in list
					for(arma::uword mysource=fs;mysource<=ls;mysource++){
						// get source
						const arma::Mat<fltp> myRn = Rn_.cols(n_.col(mysource));

						// find target points far away
						const arma::Row<fltp> rhot = cmn::Extra::vec_norm(myRt.each_col() - Re_.col(mysource));
						const fltp rhon_max = near_far_factor*element_radius_(mysource);

						// find targets closeby and far away
						const arma::Col<arma::uword> idx_near = arma::find(rhot<=rhon_max);
						const arma::Col<arma::uword> idx_far = arma::find(rhot>rhon_max);

						// nearby targets
						if(!idx_near.empty()){
							local_H.cols(idx_near) += Brick8Node::calc_magnetic_field_singularity(
								myRn,myRt.cols(idx_near),Ie_(mysource),num_planes_);
						}

						// far away targets
						if(!idx_far.empty()){
							local_H.cols(idx_far) += Savart::calc_I2H(
								Rs.cols(mysource*Rqg.n_cols,(mysource+1)*Rqg.n_cols-1),
								Ieff.cols(mysource*Rqg.n_cols,(mysource+1)*Rqg.n_cols-1),
								myRt.cols(idx_far),false);
						}
					}
				}

				// store to global
				H.cols(ft,lt) = local_H;
			});

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			if(tar->has('H'))tar->add_field('H',H,true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,true);
		}
	}

	// direct calculation
	void SourceCoil::calc_direct(const ShTargetsPr &tar, const ShSettingsPr &stngs, const cmn::ShLogPr &lg) const{
		// gauss points
		const arma::Mat<fltp> gp = cmn::Hexahedron::create_gauss_points(num_gauss_);
		const arma::Mat<fltp> Rqg = gp.rows(0,2); 
		const arma::Row<fltp> wg = gp.row(3);

		// get distance factor
		const fltp near_far_factor = near_far_factor_(std::min(near_far_factor_.n_elem-1, static_cast<arma::uword>(std::abs(num_gauss_))));

		// derivative of shape function at gauss points
		const arma::Mat<fltp> dN = cmn::Hexahedron::shape_function_derivative(Rqg);

		// vector shape function for nedelec/raviart thomas elements
		const arma::Mat<fltp> Vq = cmn::Hexahedron::raviart_thomas_hdiv_shape_function(Rqg);

		// only get faces 5 and 6, current is in mu direction
		const arma::Mat<fltp> Vq56 = Vq.rows(12,17); // [12-15] and [16-17]

		// allocate
		arma::Mat<fltp> Rs(3,n_.n_cols*Rqg.n_cols);
		arma::Mat<fltp> Ieff(3,n_.n_cols*Rqg.n_cols);

		// walk over source nodes
		cmn::parfor(0,n_.n_cols,true,[&](arma::uword i, arma::uword /*cpu*/) {
		// for(arma::uword i=0;i<n.n_cols;i++){
			// check cancel flag
			if(lg->is_cancelled())return;

			// get element nodes
			const arma::Mat<fltp>::fixed<3,8> Rn = Rn_.cols(n_.col(i));

			// integration points in cartesian coordinates
			Rs.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1) = cmn::Hexahedron::quad2cart(Rn,Rqg);

			// jacobian matrices at integration points stored column wise
			const arma::Mat<fltp> J = cmn::Hexahedron::shape_function_jacobian(Rn,dN);
			if(!J.is_finite())rat_throw_line("jacobian matrix contains non-finite values, i.e. malformed mesh");

			// determinant of jacobian matrices
			const arma::Row<fltp> Jdet = cmn::Hexahedron::jacobian2determinant(J);
			if(arma::any(Jdet<0))rat_throw_line("jacobian determinant less than zero, i.e. inverted elements detected");

			// piola transform to get carthesian vectors
			// note that the faces in RAT are slightly different from defelement website
			const arma::Mat<fltp> Vc = cmn::Hexahedron::quad2cart_contravariant_piola(Vq56,J,Jdet);
			const arma::Mat<fltp> Vc0123 = Vc.rows(0,2); // face 5
			const arma::Mat<fltp> Vc4567 = Vc.rows(3,5); // face 6

			// calculate current at integration points
			Ieff.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1) = (Vc0123 - Vc4567).eval().each_row()%(Ie_(i)*wg%Jdet);
		});

		// check cancel flag
		if(lg->is_cancelled())return;

		// forward calculation of vector potential to extra
		if(tar->has('A')){
			// get targets
			const arma::Mat<fltp>& Rt = tar->get_target_coords();

			// allocate
			arma::Mat<fltp> A(num_dim_,tar->num_targets(),arma::fill::zeros);

			// walk over all targets
			cmn::parfor(0,Rt.n_cols,stngs->get_parallel_s2t(),[&](arma::uword k, int){
				// check cancel flag
				if(lg->is_cancelled())return;

				// allocate field
				arma::Col<fltp>::fixed<3> At(arma::fill::zeros);

				// walk over sources in list
				for(arma::uword i=0;i<n_.n_cols;i++){
					// get source
					const arma::Mat<fltp> myRn = Rn_.cols(n_.col(i));

					// find target points far away
					const fltp rhot = arma::as_scalar(cmn::Extra::vec_norm(Rt.col(k) - Re_.col(i)));
					const fltp rhon_max = near_far_factor*element_radius_(i);

					// nearby targets
					if(rhot<=rhon_max)At += Brick8Node::calc_vector_potential_singularity(myRn,Rt.col(k),Ie_(i),num_planes_);

					// far away targets
					else At += Savart::calc_I2A(Rs.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1), Ieff.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1), Rt.col(k),false);
				}

				// store
				A.col(k) = At;
			});

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			tar->add_field('A',A,true);
		}

		// check cancel flag
		if(lg->is_cancelled())return;

		// forward calculation of vector potential to extra
		if(tar->has('H') || tar->has('B')){
			// get targets
			const arma::Mat<fltp>& Rt = tar->get_target_coords();

			// allocate
			arma::Mat<fltp> H(num_dim_,tar->num_targets(),arma::fill::zeros);

			// walk over all targets
			cmn::parfor(0,Rt.n_cols,stngs->get_parallel_s2t(),[&](arma::uword k, int){
				// check cancel flag
				if(lg->is_cancelled())return;

				// allocate field
				arma::Col<fltp>::fixed<3> Ht(arma::fill::zeros);

				// walk over sources in list
				for(arma::uword i=0;i<n_.n_cols;i++){
					// get source
					const arma::Mat<fltp> myRn = Rn_.cols(n_.col(i));

					// find target points far away
					const fltp rhot = arma::as_scalar(cmn::Extra::vec_norm(Rt.col(k) - Re_.col(i)));
					const fltp rhon_max = near_far_factor*element_radius_(i);

					// nearby targets
					if(rhot<=rhon_max)Ht += Brick8Node::calc_magnetic_field_singularity(myRn,Rt.col(k),Ie_(i),num_planes_);

					// far away targets
					else Ht += Savart::calc_I2H(Rs.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1), Ieff.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1), Rt.col(k),false);
				}

				// store
				H.col(k) = Ht;
			});

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			if(tar->has('H'))tar->add_field('H',H,true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,true);
		}

	}


	// subdivide a hexahedron
	void SourceCoil::subdivide_core(
		arma::Mat<fltp> &Rn, 
		arma::Mat<arma::uword> &n, 
		const arma::Col<arma::uword>::fixed<3>& num_div){

		// check input
		assert(Rn.n_cols==8);
		assert(Rn.n_rows==3);

		// quadrilateral coordinates of the new nodes
		const arma::Row<fltp> nu_node = arma::linspace<arma::Row<fltp> >(-1.0,1.0,num_div(0)+1);
		const arma::Row<fltp> mu_node = arma::linspace<arma::Row<fltp> >(-1.0,1.0,num_div(1)+1);
		const arma::Row<fltp> xi_node = arma::linspace<arma::Row<fltp> >(-1.0,1.0,num_div(2)+1);

		// create coordinates
		const arma::uword num_nodes = (num_div(0)+1)*(num_div(1)+1)*(num_div(2)+1);
		arma::uword node_cnt = 0llu;
		arma::Mat<fltp> Rnlocal(3,num_nodes);
		for(arma::uword k=0;k<num_div(0)+1;k++){
			for(arma::uword n=0;n<num_div(1)+1;n++){
				for(arma::uword m=0;m<num_div(2)+1;m++){
					const arma::Col<fltp>::fixed<3> Rq{nu_node(k),mu_node(n),xi_node(m)};
					Rnlocal.col(node_cnt++) = cmn::Hexahedron::quad2cart(Rn,Rq);
				}
			}
		}
		assert(node_cnt==num_nodes);

		// create element
		const arma::uword num_elements = num_div(0)*num_div(1)*num_div(2);
		arma::uword element_cnt = 0llu;
		arma::Mat<arma::uword> nlocal(8,num_elements);
		for(arma::uword k=0;k<num_div(0);k++){
			for(arma::uword n=0;n<num_div(1);n++){
				for(arma::uword m=0;m<num_div(2);m++){
					nlocal.col(element_cnt++) = 
						arma::Col<arma::uword>::fixed<8>{
						(k+0)*(num_div(1)+1)*(num_div(2)+1) + (n+0)*(num_div(2)+1) + (m+0), 
						(k+1)*(num_div(1)+1)*(num_div(2)+1) + (n+0)*(num_div(2)+1) + (m+0), 
						(k+1)*(num_div(1)+1)*(num_div(2)+1) + (n+1)*(num_div(2)+1) + (m+0), 
						(k+0)*(num_div(1)+1)*(num_div(2)+1) + (n+1)*(num_div(2)+1) + (m+0), 
						(k+0)*(num_div(1)+1)*(num_div(2)+1) + (n+0)*(num_div(2)+1) + (m+1), 
						(k+1)*(num_div(1)+1)*(num_div(2)+1) + (n+0)*(num_div(2)+1) + (m+1), 
						(k+1)*(num_div(1)+1)*(num_div(2)+1) + (n+1)*(num_div(2)+1) + (m+1), 
						(k+0)*(num_div(1)+1)*(num_div(2)+1) + (n+1)*(num_div(2)+1) + (m+1)};
				}
			}
		}
		assert(element_cnt==num_elements);

		// assign to output
		Rn = Rnlocal; n = nlocal;
	}



	// subdivide
	ShSourcesPr SourceCoil::subdivide(const fltp grid_size){
		// check directions
		const arma::Mat<arma::uword>::fixed<6,4> F = cmn::Hexahedron::get_faces();

		// allocate element size
		arma::Mat<fltp> de(3,n_.n_cols);

		// walk over elements
		for(arma::uword i=0;i<n_.n_cols;i++){
			// get face coordinates
			const arma::Mat<fltp>::fixed<3,8> Rn = Rn_.cols(n_.col(i));
			arma::Mat<fltp>::fixed<3,6> Rfc;
			for(arma::uword j=0;j<F.n_rows;j++)
				Rfc.col(j) = arma::mean(Rn.cols(F.row(j)),1);

			// calculate element size in each direction
			de(0,i) = arma::as_scalar(cmn::Extra::vec_norm(Rfc.col(3) - Rfc.col(1))); // from 1265 to 3047 along nu
			de(1,i) = arma::as_scalar(cmn::Extra::vec_norm(Rfc.col(2) - Rfc.col(0))); // from 0154 to 2376 along mu
			de(2,i) = arma::as_scalar(cmn::Extra::vec_norm(Rfc.col(5) - Rfc.col(4))); // from 0123 to 4567 along xi
		}

		// find elements that are too large and distinguish between different cases
		const arma::Row<arma::uword> b012 = arma::all(de>grid_size,0);
		const arma::Row<arma::uword> b01 = arma::all(de.rows(arma::Col<arma::uword>{0,1})>grid_size,0);
		const arma::Row<arma::uword> b02 = arma::all(de.rows(arma::Col<arma::uword>{0,2})>grid_size,0);
		const arma::Row<arma::uword> b12 = arma::all(de.rows(arma::Col<arma::uword>{1,2})>grid_size,0);
		const arma::Row<arma::uword> b0 = arma::all(de.row(0)>grid_size,0);
		const arma::Row<arma::uword> b1 = arma::all(de.row(1)>grid_size,0);
		const arma::Row<arma::uword> b2 = arma::all(de.row(2)>grid_size,0);
		const arma::Col<arma::uword> idx012 = arma::find(b012);
		const arma::Col<arma::uword> idx01 = arma::find(b01 && (b012==0));
		const arma::Col<arma::uword> idx02 = arma::find(b02 && (b012==0));
		const arma::Col<arma::uword> idx12 = arma::find(b12 && (b012==0));
		const arma::Col<arma::uword> idx0 = arma::find(b0 && (b012==0) && (b01==0) && (b02==0));
		const arma::Col<arma::uword> idx1 = arma::find(b1 && (b012==0) && (b01==0) && (b12==0));
		const arma::Col<arma::uword> idx2 = arma::find(b2 && (b012==0) && (b02==0) && (b12==0));
		const arma::Col<arma::uword> idx_keep = arma::find(arma::all(de<=grid_size,0));

		// count number of new elements
		const arma::uword num_elem = 8*idx012.n_elem + 4*(idx01.n_elem + idx02.n_elem + idx12.n_elem) + 2*(idx0.n_elem + idx1.n_elem + idx2.n_elem) + idx_keep.n_elem;
		const arma::uword num_nodes = 27*idx012.n_elem + 18*(idx01.n_elem + idx02.n_elem + idx12.n_elem) + 12*(idx0.n_elem + idx1.n_elem + idx2.n_elem) + 8*idx_keep.n_elem;
		
		// allocate new elements
		arma::Mat<fltp> Rn(3,num_nodes); 
		arma::Mat<arma::uword> n(8,num_elem); 
		arma::Row<fltp> I(num_elem);
		arma::uword node_cnt = 0, element_cnt = 0;

		// subdivisions
		for(arma::uword i=0;i<idx012.n_elem;i++){
			const arma::uword myidx = idx012(i);
			arma::Mat<fltp> Rnlocal = Rn_.cols(n_.col(myidx));
			arma::Mat<arma::uword> nlocal;
			const arma::Col<arma::uword> nsub{2,2,2};
			subdivide_core(Rnlocal,nlocal,nsub);
			Rn.cols(node_cnt,node_cnt+27-1) = Rnlocal;
			n.cols(element_cnt,element_cnt+8-1) = nlocal + node_cnt;
			I.cols(element_cnt,element_cnt+8-1).fill(Ie_(myidx)/(nsub(0)*nsub(1)));
			node_cnt+=27; element_cnt+=8;
		}

		for(arma::uword i=0;i<idx01.n_elem;i++){
			const arma::uword myidx = idx01(i);
			arma::Mat<fltp> Rnlocal = Rn_.cols(n_.col(myidx));
			arma::Mat<arma::uword> nlocal;
			const arma::Col<arma::uword> nsub{2,2,1};
			subdivide_core(Rnlocal,nlocal,nsub);
			Rn.cols(node_cnt,node_cnt+18-1) = Rnlocal;
			n.cols(element_cnt,element_cnt+4-1) = nlocal + node_cnt;
			I.cols(element_cnt,element_cnt+4-1).fill(Ie_(myidx)/(nsub(0)*nsub(1)));
			node_cnt+=18; element_cnt+=4;
		}

		for(arma::uword i=0;i<idx02.n_elem;i++){
			const arma::uword myidx = idx02(i);
			arma::Mat<fltp> Rnlocal = Rn_.cols(n_.col(myidx));
			arma::Mat<arma::uword> nlocal;
			const arma::Col<arma::uword> nsub{2,1,2};
			subdivide_core(Rnlocal,nlocal,nsub);
			Rn.cols(node_cnt,node_cnt+18-1) = Rnlocal;
			n.cols(element_cnt,element_cnt+4-1) = nlocal + node_cnt;
			I.cols(element_cnt,element_cnt+4-1).fill(Ie_(myidx)/(nsub(0)*nsub(1)));
			node_cnt+=18; element_cnt+=4;
		}

		for(arma::uword i=0;i<idx12.n_elem;i++){
			const arma::uword myidx = idx12(i);
			arma::Mat<fltp> Rnlocal = Rn_.cols(n_.col(myidx));
			arma::Mat<arma::uword> nlocal;
			const arma::Col<arma::uword> nsub{1,2,2};
			subdivide_core(Rnlocal,nlocal,nsub);
			Rn.cols(node_cnt,node_cnt+18-1) = Rnlocal;
			n.cols(element_cnt,element_cnt+4-1) = nlocal + node_cnt;
			I.cols(element_cnt,element_cnt+4-1).fill(Ie_(myidx)/(nsub(0)*nsub(1)));
			node_cnt+=18; element_cnt+=4;
		}

		for(arma::uword i=0;i<idx0.n_elem;i++){
			const arma::uword myidx = idx0(i);
			arma::Mat<fltp> Rnlocal = Rn_.cols(n_.col(myidx));
			arma::Mat<arma::uword> nlocal;
			const arma::Col<arma::uword> nsub{2,1,1};
			subdivide_core(Rnlocal,nlocal,nsub);
			Rn.cols(node_cnt,node_cnt+12-1) = Rnlocal;
			n.cols(element_cnt,element_cnt+2-1) = nlocal + node_cnt;
			I.cols(element_cnt,element_cnt+2-1).fill(Ie_(myidx)/(nsub(0)*nsub(1)));
			node_cnt+=12; element_cnt+=2;
		}

		for(arma::uword i=0;i<idx1.n_elem;i++){
			const arma::uword myidx = idx1(i);
			arma::Mat<fltp> Rnlocal = Rn_.cols(n_.col(myidx));
			arma::Mat<arma::uword> nlocal;
			const arma::Col<arma::uword> nsub{1,2,1};
			subdivide_core(Rnlocal,nlocal,nsub);
			Rn.cols(node_cnt,node_cnt+12-1) = Rnlocal;
			n.cols(element_cnt,element_cnt+2-1) = nlocal + node_cnt;
			I.cols(element_cnt,element_cnt+2-1).fill(Ie_(myidx)/(nsub(0)*nsub(1)));
			node_cnt+=12; element_cnt+=2;
		}

		for(arma::uword i=0;i<idx2.n_elem;i++){
			const arma::uword myidx = idx2(i);
			arma::Mat<fltp> Rnlocal = Rn_.cols(n_.col(myidx));
			arma::Mat<arma::uword> nlocal;
			const arma::Col<arma::uword> nsub{1,1,2};
			subdivide_core(Rnlocal,nlocal,nsub);
			Rn.cols(node_cnt,node_cnt+12-1) = Rnlocal;
			n.cols(element_cnt,element_cnt+2-1) = nlocal + node_cnt;
			I.cols(element_cnt,element_cnt+2-1).fill(Ie_(myidx)/(nsub(0)*nsub(1)));
			node_cnt+=12; element_cnt+=2;
		}

		for(arma::uword i=0;i<idx_keep.n_elem;i++){
			const arma::uword myidx = idx_keep(i);
			arma::Mat<fltp> Rnlocal = Rn_.cols(n_.col(myidx));
			arma::Mat<arma::uword> nlocal;
			const arma::Col<arma::uword> nsub{1,1,1};
			subdivide_core(Rnlocal,nlocal,nsub);
			Rn.cols(node_cnt,node_cnt+8-1) = Rnlocal;
			n.cols(element_cnt,element_cnt+1-1) = nlocal + node_cnt;
			I.cols(element_cnt,element_cnt+1-1).fill(Ie_(myidx)/(nsub(0)*nsub(1)));
			node_cnt+=8; element_cnt+=1;
		}

		// sanity check
		assert(node_cnt==num_nodes);
		assert(element_cnt==num_elem);

		// cleanup
		const arma::Row<arma::uword> sort_idx = cmn::Extra::combine_nodes(Rn);
		n = arma::reshape(sort_idx.cols(arma::vectorise(n)),8,num_elem);

		// create another source coil object
		const ShSourceCoilPr src = SourceCoil::create(Rn,n,I); 

		// transfer settings! TODO

		return src;
	}

}}