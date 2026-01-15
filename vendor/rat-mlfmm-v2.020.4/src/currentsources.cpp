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
#include "currentsources.hh"

// code specific to Rat
namespace rat{namespace fmm{
	
	// constructor
	CurrentSources::CurrentSources(){

	}

	// constructor with input
	CurrentSources::CurrentSources(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &dRs, 
		const arma::Row<fltp> &Is,
		const arma::Row<fltp> &epss){
		
		// check matrix size
		if(Rs.n_cols!=dRs.n_cols)rat_throw_line("coordinate and direction have different number of columns");
		if(Rs.n_cols!=Is.n_cols)rat_throw_line("coordinate and current have different number of columns");
		if(Rs.n_cols!=epss.n_cols)rat_throw_line("coordinate and softening have different number of columns");

		// set sources 
		set_coords(Rs); set_direction(dRs);
		set_currents(Is); set_softening(epss);
	}

	// constructor
	CurrentSources::CurrentSources(const ShCurrentSourcesPrList &csl){
		// use merge function
		set_sources(csl);
	}

	// factory
	ShCurrentSourcesPr CurrentSources::create(){
		return std::make_shared<CurrentSources>();
	}

	// factory with input
	ShCurrentSourcesPr CurrentSources::create(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &dRs, 
		const arma::Row<fltp> &Is,
		const arma::Row<fltp> &epss){

		//return ShIListPr(new IList);
		return std::make_shared<CurrentSources>(Rs,dRs,Is,epss);
	}

	// factory combining current sources
	ShCurrentSourcesPr CurrentSources::create(
		const ShCurrentSourcesPrList &csl){
		return std::make_shared<CurrentSources>(csl);	
	}

	// get number of dimensions
	arma::uword CurrentSources::get_num_dim() const{
		return num_dim_;
	}

	// merge function
	void CurrentSources::set_sources(const ShCurrentSourcesPrList &csl){
		// check input
		if(csl.is_empty())rat_throw_line("sources list is empty");

		// allocate
		arma::field<arma::Mat<fltp> > Rsfld(1,csl.n_elem);
		arma::field<arma::Mat<fltp> > dRsfld(1,csl.n_elem);
		arma::field<arma::Mat<fltp> > Isfld(1,csl.n_elem);
		arma::field<arma::Mat<fltp> > epssfld(1,csl.n_elem);

		// copy properties
		for(arma::uword i=0;i<csl.n_elem;i++){
			Rsfld(i) = csl(i)->get_source_coords();
			dRsfld(i) = csl(i)->get_source_direction();
			Isfld(i) = csl(i)->get_source_currents();
			epssfld(i) = csl(i)->get_source_softening();
		}

		// combine and store
		set_coords(cmn::Extra::field2mat(Rsfld)); set_direction(cmn::Extra::field2mat(dRsfld));
		set_currents(cmn::Extra::field2mat(Isfld)); set_softening(cmn::Extra::field2mat(epssfld));

		// van lanen
		set_van_Lanen(csl(0)->get_van_Lanen());
	}

	// set coordinate vectors
	void CurrentSources::set_coords(const arma::Mat<fltp> &Rs){
		// check user input
		if(Rs.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(!Rs.is_finite())rat_throw_line("coordinate matrix must be finite");
		
		// coordinates can only be set once
		//if(!Rs_.is_empty())rat_throw_line("coordinates can only be set once");

		// set coordinate vectors
		Rs_ = Rs;

		// set number of sources
		num_sources_ = Rs_.n_cols;
	}

	// set direction vector
	void CurrentSources::set_direction(const arma::Mat<fltp> &dRs){
		// check user input
		if(dRs.n_rows!=3)rat_throw_line("direction matrix must have three rows");
		if(!dRs.is_finite())rat_throw_line("direction matrix must be finite");
		
		// set direction vectors
		dRs_ = dRs;
	}

	// get direction vector
	arma::Mat<fltp> CurrentSources::get_direction() const{
		return dRs_;
	}

	// get typical element size (to limit grid)
	fltp CurrentSources::element_size() const{
		if(use_van_Lanen_){
			return arma::max(cmn::Extra::vec_norm(dRs_));
		}else{
			return RAT_CONST(0.0);
		}
	}

	// subdivide
	ShSourcesPr CurrentSources::subdivide(const fltp grid_size){
		if(!use_van_Lanen_)return NULL;
		const arma::Col<arma::uword> idx_div = arma::find(cmn::Extra::vec_norm(dRs_)>grid_size);
		const arma::Col<arma::uword> idx_keep = arma::find(cmn::Extra::vec_norm(dRs_)<=grid_size);
		const arma::Mat<fltp> Rnew = arma::join_horiz(Rs_.cols(idx_div)-dRs_.cols(idx_div)/4,Rs_.cols(idx_div)+dRs_.cols(idx_div)/4,Rs_.cols(idx_keep));
		const arma::Mat<fltp> dRnew = arma::join_horiz(dRs_.cols(idx_div)/2,dRs_.cols(idx_div)/2,dRs_.cols(idx_keep));
		const arma::Row<fltp> Inew = arma::join_horiz(Is_.cols(idx_div),Is_.cols(idx_div),Is_.cols(idx_keep));
		const arma::Row<fltp> epsnew = arma::join_horiz(epss_.cols(idx_div),epss_.cols(idx_div),epss_.cols(idx_keep));
		return CurrentSources::create(Rnew,dRnew,Inew,epsnew);
	}

	// set currents
	void CurrentSources::set_currents(const arma::Row<fltp> &Is){
		// check user input
		if(!Is.is_finite())rat_throw_line("current vector must be finite");

		// set currents
		Is_ = Is;
	}

	// setting the softening factor
	void CurrentSources::set_softening(const arma::Row<fltp> &epss){
		// check user input
		if(!epss.is_finite())rat_throw_line("softening vector must be finite");

		// set softening
		epss_ = epss;
	}

	// sorting function
	void CurrentSources::sort_sources(const arma::Row<arma::uword> &sort_idx){
		// check user input
		if(Rs_.is_empty())rat_throw_line("coordinate matrix was not set");
		if(dRs_.is_empty())rat_throw_line("direction matrix was not set");
		if(Is_.is_empty())rat_throw_line("current vector was not set");
		if(epss_.is_empty())rat_throw_line("softening vector was not set");

		// check if sort array right length
		assert(Rs_.n_cols == sort_idx.n_elem);

		// sort sources
		Rs_ = Rs_.cols(sort_idx);
		dRs_ = dRs_.cols(sort_idx);
		Is_ = Is_.cols(sort_idx);
		epss_ = epss_.cols(sort_idx);
	}

	// unsorting function
	void CurrentSources::unsort_sources(const arma::Row<arma::uword> &sort_idx){
		// check user input
		if(Rs_.is_empty())rat_throw_line("coordinate matrix was not set");
		if(dRs_.is_empty())rat_throw_line("direction matrix was not set");
		if(Is_.is_empty())rat_throw_line("current vector was not set");
		if(epss_.is_empty())rat_throw_line("softening vector was not set");

		// check if sort array right length
		assert(Rs_.n_cols == sort_idx.n_elem);

		// sort sources
		Rs_.cols(sort_idx) = Rs_;
		dRs_.cols(sort_idx) = dRs_;
		Is_.cols(sort_idx) = Is_;
		epss_.cols(sort_idx) = epss_;
	}

	// count number of sources stored
	arma::uword CurrentSources::num_sources() const{
		// return number of elements
		return num_sources_;
	}

	// method for getting all coordinates
	arma::Mat<fltp> CurrentSources::get_source_coords() const{
		// check if source coordinates set
		if(Rs_.n_rows==0)rat_throw_line("coordinate matrix was not set");
		if(Rs_.n_rows!=3)rat_throw_line("coordinate matrix has wrong dimensions");

		// return coordinates
		return Rs_;
	}

	// method for getting all coordinates
	const arma::Mat<fltp>& CurrentSources::get_source_direction() const{
		// check if source coordinates set
		if(dRs_.n_rows==0)rat_throw_line("direction matrix was not set");
		if(dRs_.n_rows!=3)rat_throw_line("direction matrix has wrong dimensions");

		// return coordinates
		return dRs_;
	}

	// method for getting all coordinates
	const arma::Row<fltp>& CurrentSources::get_source_currents() const{
		// check if source coordinates set
		if(Is_.n_rows==0)rat_throw_line("direction matrix was not set");

		// return coordinates
		return Is_;
	}

	// method for getting all coordinates
	const arma::Row<fltp>& CurrentSources::get_source_softening() const{
		// check if source coordinates set
		if(epss_.n_rows==0)rat_throw_line("direction matrix was not set");
		
		// return coordinates
		return epss_;
	}


	// // method for getting coordinates with specific indices
	// arma::Mat<fltp> CurrentSources::get_source_coords(
	// 	const arma::Row<arma::uword> &indices) const{
	// 	// check if source coordinates set
	// 	if(Rs_.is_empty())rat_throw_line("coordinate matrix was not set");

	// 	// return coordinates
	// 	return Rs_.cols(indices);
	// }

	// set van Lanen kernel
	void CurrentSources::set_van_Lanen(const bool use_van_Lanen){
		use_van_Lanen_ = use_van_Lanen;
	}

	// get van Lanen
	bool CurrentSources::get_van_Lanen() const{
		return use_van_Lanen_;
	}

	// setup source to multipole matrices
	void CurrentSources::setup_source_to_multipole(
		const arma::Mat<fltp> &dR, 
		const ShSettingsPr &stngs){

		// setup CUDA
		#ifdef ENABLE_CUDA_KERNELS
		if(stngs->get_so2ta_enable_gpu() || stngs->get_so2mp_enable_gpu())GpuKernels::show_device_info(stngs->get_gpu_devices());
		if(stngs->get_so2mp_enable_gpu()){
			dR_ = dR; return;
		}
		#endif

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
	void CurrentSources::source_to_multipole(
		arma::Mat<std::complex<fltp> > &Mp,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source,
		const ShSettingsPr &stngs) const{

		// check input
		assert(first_source.n_elem==last_source.n_elem);
		assert(!Mp.is_empty());

		// check input source indexing
		assert(first_source(0)==0);
		assert(arma::all(arma::abs(
			first_source.tail_cols(first_source.n_elem-1)-
			last_source.head_cols(last_source.n_elem-1)-1)==0));
		assert(last_source(last_source.n_elem-1)==num_sources_-1);

		// get number of expansions
		const int num_exp = stngs->get_num_exp();

		// intercept GPU kernel
		#ifdef ENABLE_CUDA_KERNELS
		if(stngs->get_so2mp_enable_gpu()){
			source_to_multipole_gpu(Mp,first_source,last_source,stngs);
			return;
		}
		#endif

		// calculate effective current
		const arma::Mat<fltp> Ieff = dRs_.each_row()%Is_;

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

	// direct calculation of vector potential or magnetic 
	// field for all sources at all target points
	void CurrentSources::calc_direct(
		const ShTargetsPr &tar, 
		const ShSettingsPr &stngs, 
		const cmn::ShLogPr &lg) const{

		// intercept GPU kernel
		#ifdef ENABLE_CUDA_KERNELS
		if(stngs->get_so2ta_enable_gpu()){
			calc_direct_gpu(tar, stngs);
			return;
		}
		#endif

		// get target coordinates
		const arma::Mat<fltp> Rt = tar->get_target_coords();

		// forward calculation of vector potential to extra
		if(tar->has('A')){
			// allocate field
			arma::Mat<fltp> A;

			// run normal savart kernel
			if(!use_van_Lanen_){
				A = Savart::calc_I2A_s(Rs_, dRs_.each_row()%Is_, epss_, Rt, stngs->get_parallel_s2t());
			}

			// run van Lanen kernel
			else{
				A = Savart::calc_I2A_vl_dir(Rs_, dRs_, Is_, epss_, Rt, stngs->get_parallel_s2t(), lg);
			}

			// set
			tar->add_field('A',A,false);
		}

		// forward calculation of magnetic field to extra
		if(tar->has('H') || tar->has('B')){
			// allocate field
			arma::Mat<fltp> H;

			// run special savart kernel
			if(!use_van_Lanen_){
				H = Savart::calc_I2H_s(Rs_, dRs_.each_row()%Is_, epss_, Rt, stngs->get_parallel_s2t());
			}

			// run normal savart kernel
			else{
				H = Savart::calc_I2H_vl_dir(Rs_, dRs_, Is_, epss_, Rt, stngs->get_parallel_s2t(), lg);
			}

			// set
			if(tar->has('H'))tar->add_field('H',H,false);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,false);
		}
	}


	// source to target kernel
	void CurrentSources::source_to_target(
		const ShTargetsPr &tar, const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target,
		const ShSettingsPr &stngs, 
		const cmn::ShLogPr &lg) const{

		// check input source indexing
		// assert(first_source(0)==0);
		// assert(arma::all(arma::abs(
		// 	first_source.tail_cols(first_source.n_elem-1)-
		// 	last_source.head_cols(last_source.n_elem-1)-1)==0));
		// assert(last_source(last_source.n_elem-1)==num_sources_-1);

		// check input target indexing
		// assert(first_target(0)==0);
		// assert(arma::all(arma::abs(
		// 	first_target.tail_cols(first_target.n_elem-1)-
		// 	last_target.head_cols(last_target.n_elem-1)-1)==0));
		// assert(last_target(last_target.n_elem-1)==tar->num_targets()-1);

		// intercept GPU kernel
		#ifdef ENABLE_CUDA_KERNELS
		if(stngs->get_so2ta_enable_gpu()){
			source_to_target_gpu(tar,
				target_list,source_list,
				first_source,last_source,
				first_target,last_target,
				stngs);
			return;
		}
		#endif

		// check sizes
		assert(target_list.n_elem==source_list.n_elem);

		// get targets
		// const arma::Mat<fltp>& Rt = tar->get_target_coords();

		// calculate effective current
		const arma::Mat<fltp> Ieff = dRs_.each_row()%Is_;

		// forward calculation of vector potential to extra
		if(tar->has('A')){

			// allocate
			arma::Mat<fltp> A(num_dim_,tar->num_targets(),arma::fill::zeros);
				
			// walk over source nodes
			//for(arma::uword i=0;i<target_list.n_elem;i++){
			cmn::parfor(0,target_list.n_elem,stngs->get_parallel_s2t(),[&](arma::uword i, int){
				// check cancel flag
				if(lg->is_cancelled())return;

				// get target node
				const arma::uword target_idx = target_list(i);

				// get location of the targets
				const arma::uword ft = first_target(target_idx);
				const arma::uword lt = last_target(target_idx);
				assert(ft<=lt); assert(lt<tar->num_targets());
				
				// get target coordinates
				const arma::Mat<fltp> myRt = tar->get_target_coords(ft,lt);

				// check if source list has sources
				assert(source_list(i).n_elem>0);

				// walk over target nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// get target node
					const arma::uword source_idx = source_list(i)(j);
					
					// get my source
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<num_sources_);

					// run normal savart kernel
					if(!use_van_Lanen_){
						A.cols(ft,lt) += Savart::calc_I2A_s(
							Rs_.cols(fs,ls), Ieff.cols(fs,ls), 
							epss_.cols(fs,ls), myRt, false);
					}

					// run savart van Lanen kernel
					else{
						A.cols(ft,lt) += Savart::calc_I2A_vl(
							Rs_.cols(fs,ls), dRs_.cols(fs,ls), Is_.cols(fs,ls), 
							epss_.cols(fs,ls), myRt, false);
					}
				}
			});
			//}

			// check cancel flag
			if(lg->is_cancelled())return;

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

				// get target coordinates
				const arma::Mat<fltp> myRt = tar->get_target_coords(ft,lt);

				// walk over target nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// get target node
					const arma::uword source_idx = source_list(i)(j);

					// get my source
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<num_sources_);

					// run normal savart kernel
					if(!use_van_Lanen_){
						// run savart kernel
						H.cols(ft,lt) += Savart::calc_I2H_s(
							Rs_.cols(fs,ls), Ieff.cols(fs,ls), 
							epss_.cols(fs,ls), myRt, false);
					}

					// run savart van Lanen kernel
					else{
						H.cols(ft,lt) += Savart::calc_I2H_vl(
							Rs_.cols(fs,ls), dRs_.cols(fs,ls), Is_.cols(fs,ls), 
							epss_.cols(fs,ls), myRt, false);
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

	// translate sources
	void CurrentSources::translate(const fltp dx, const fltp dy, const fltp dz){
		const arma::Col<fltp>::fixed<3> &dR = {dx,dy,dz};
		Rs_.each_col() += dR;
	}

	// setup solenoidal elements 
	// for comparing results with SOLENO
	// the center of the solenoid is placed at the origin
	void CurrentSources::setup_solenoid(
		const fltp Rin, const fltp Rout, 
		const fltp height, const arma::uword nr, 
		const arma::uword nz, const arma::uword nl,
		const fltp J){

		// calculate element size
		//const fltp dtheta = 2*arma::Datum<fltp>::pi/(nl+1);
		const fltp drho = (Rout-Rin)/nr;
		const fltp dz = height/nz;

		// create azymuthal coordinates of elements
		arma::Row<fltp> theta = arma::linspace<arma::Row<fltp> >(0,2*arma::Datum<fltp>::pi,nl+1);

		// create radial coordinates of elements
		arma::Row<fltp> rho = arma::linspace<arma::Row<fltp> >(Rin+drho/2,Rout-drho/2,nr);

		// create axial cooridnates of elements
		arma::Row<fltp> z = arma::linspace<arma::Row<fltp> >(-height/2+dz/2,height/2-dz/2,nz);

		// create matrix to hold node coordinates
		arma::Mat<fltp> xcis(nl+1,nr*nz), ycis(nl+1,nr*nz), zcis(nl+1,nr*nz);

		// build node coordinates in two dimensions (first block of matrix)
		for(arma::uword i=0;i<nr;i++){
			// generate circles with different radii
			xcis.col(i) = rho(i)*arma::cos(theta).t();
			ycis.col(i) = rho(i)*arma::sin(theta).t();
			zcis.col(i).fill(z(0));
		}

		// extrude to other axial planes
		for(arma::uword j=1;j<nz;j++){
			// copy coordinates from ground plane
			xcis.cols(j*nr,(j+1)*nr-1) = xcis.cols(0,nr-1);
			ycis.cols(j*nr,(j+1)*nr-1) = ycis.cols(0,nr-1);
			zcis.cols(j*nr,(j+1)*nr-1).fill(z(j));
		}

		// number of nodes
		num_sources_ = nr*nl*nz;

		// calculate coords
		arma::Mat<fltp> xe = (xcis.rows(1,nl) + xcis.rows(0,nl-1))/2;
		arma::Mat<fltp> ye = (ycis.rows(1,nl) + ycis.rows(0,nl-1))/2;
		arma::Mat<fltp> ze = (zcis.rows(1,nl) + zcis.rows(0,nl-1))/2;

		// calculate element direction
		arma::Mat<fltp> dxe = arma::diff(xcis,1,0);
		arma::Mat<fltp> dye = arma::diff(ycis,1,0);
		arma::Mat<fltp> dze = arma::diff(zcis,1,0);
		
		// reshape coordinates
		Rs_.set_size(3,num_sources_);
		Rs_.row(0) = arma::reshape(xe,1,num_sources_);
		Rs_.row(1) = arma::reshape(ye,1,num_sources_);
		Rs_.row(2) = arma::reshape(ze,1,num_sources_);

		// reshape vectors
		dRs_.set_size(3,num_sources_);
		dRs_.row(0) = arma::reshape(dxe,1,num_sources_);
		dRs_.row(1) = arma::reshape(dye,1,num_sources_);
		dRs_.row(2) = arma::reshape(dze,1,num_sources_);

		// set currents
		Is_.set_size(num_sources_);
		Is_.fill(J*height*(Rout-Rin)/(nr*nz));

		// softening
		epss_.set_size(num_sources_); 
		epss_.fill(std::sqrt((drho*dz)/arma::Datum<fltp>::pi));
	}



	// GPU specific methods
	#ifdef ENABLE_CUDA_KERNELS

	// direct calculation of vector potential or magnetic 
	// field for all sources at all target points
	void CurrentSources::calc_direct_gpu(const ShTargetsPr &tar, const ShSettingsPr &stngs) const{

		// get targets
		const arma::Mat<fltp>& Rt = tar->get_target_coords();
		const arma::uword num_targets = tar->num_targets();
		
		// check if cuda has different precision from rat
		arma::Mat<cufltp> Rs_single(4,num_sources_); 
		arma::Mat<cufltp> Ieff_single(4,num_sources_); 
		arma::Mat<cufltp> Rt_single(4,num_targets); 
		#ifdef RAT_CUDA_CONVERSION_NEEDED
			// convert to single precision fltps
			// also store in 4XN matrix as to accomodate CUDA fltp4
			Rs_single.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(Rs_); 
			Rs_single.row(3) = arma::conv_to<arma::Row<cufltp> >::from(epss_); 	
			Ieff_single.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(dRs_);
			Ieff_single.row(3) = arma::conv_to<arma::Row<cufltp> >::from(Is_); 
			Rt_single.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(Rt);
		#else
			// keep original precision only store in
			// 4XN matrix as to accomodate CUDA fltp4
			Rs_single.rows(0,2) = Rs_; Rs_single.row(3) = epss_; 
			Ieff_single.rows(0,2) = dRs_; Ieff_single.row(3) = Is_; 
			Rt_single.rows(0,2) = Rt;
		#endif

		// check sizes
		assert(Rt_single.n_cols==num_targets); assert(Rt_single.n_rows==4);
		assert(Rs_single.n_cols==num_sources_); assert(Rs_single.n_rows==4);
		assert(Ieff_single.n_cols==num_sources_); assert(Ieff_single.n_rows==4);

		// check which fields to calculate
		const bool calc_A = tar->has('A');
		const bool calc_H = tar->has('H') || tar->has('B');

		// allocate magnetic field
		arma::Mat<cufltp> A_single, H_single; 
		if(calc_A)A_single.set_size(4,num_targets);
		if(calc_H)H_single.set_size(4,num_targets);

		// get gpu devices
		std::set<int> gpu_devices = stngs->get_gpu_devices();
		if(gpu_devices.empty())rat_throw_line("No GPU devices set for calculation");

		// ofload to GPU kernel
		const bool is_managed = false;
		GpuKernels::direct_kernel(
			A_single.memptr(), H_single.memptr(), Rt_single.memptr(), 
			static_cast<long unsigned int>(num_targets), 
			Rs_single.memptr(), Ieff_single.memptr(), 
			static_cast<long unsigned int>(num_sources_), calc_A, calc_H, use_van_Lanen_, 
			gpu_devices, is_managed);
		
		// check if cuda has different precision from rat
		#ifdef RAT_CUDA_CONVERSION_NEEDED
			// get and store vector potential
			if(tar->has('A'))tar->add_field('A',arma::conv_to<arma::Mat<fltp> >::from(A_single.rows(0,2)),true);
				
			// get and store magnetic field
			const arma::Mat<fltp> H = arma::conv_to<arma::Mat<fltp> >::from(H_single.rows(0,2));
			if(tar->has('H'))tar->add_field('H',H,true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,true);
		#else
			// get and store vector potential
			if(tar->has('A'))tar->add_field('A',A_single.rows(0,2),true);
				
			// get and store magnetic field
			if(tar->has('H'))tar->add_field('H',H_single.rows(0,2),true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H_single.rows(0,2),true);
		#endif
	}	

	// source to target GPU kernel
	void CurrentSources::source_to_target_gpu(
		const ShTargetsPr &tar, const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target,
		const ShSettingsPr &stngs) const{

		// do not run gpu if there is no source-target interactions
		if(source_list.is_empty() || target_list.is_empty())return;
		// arma::wall_clock timer; timer.tic();

		// get counters
		const arma::uword num_target_nodes = target_list.n_elem;
		const arma::uword num_source_nodes = first_source.n_elem;
		const arma::uword num_target_list = target_list.n_elem;

		// create indexes for source and target lists
		arma::Col<arma::uword> source_list_idx(num_target_list + 1);
		source_list_idx(0) = 0;
		for(arma::uword i=0;i<num_target_list;i++)
			source_list_idx(i+1) = source_list(i).n_elem;

		// accumulate sizes to create indexes
 		source_list_idx = arma::cumsum(source_list_idx);

		// nnumber of source nodes
 		const arma::uword num_source_list = source_list_idx(num_target_list);

		// combine source list
 		arma::Col<arma::uword> source_list_combined(num_source_list);

 		// fill source list
		for(arma::uword i=0;i<num_target_list;i++){
			source_list_combined.rows(source_list_idx(i),source_list_idx(i+1)-1) = source_list(i); 
		}

		// get only active targets
		const arma::Row<arma::uword> ft = first_target.cols(target_list);
		const arma::Row<arma::uword> lt = last_target.cols(target_list);
		const arma::Row<arma::uword> nt = lt-ft+1; 
		arma::Row<arma::uword> ti(num_target_nodes+1); 
		ti(0) = 0; ti.cols(1,num_target_nodes) = arma::cumsum(nt);
		const arma::uword ntt = ti.back();

		// get target coordinates of active targets
		arma::Mat<fltp> Rtt(3,ntt);
		for(arma::uword i=0;i<num_target_nodes;i++)
			Rtt.cols(ti(i),ti(i+1)-1) = tar->get_target_coords(ft(i),lt(i));

		// convert indices to 32 bit integers
		const arma::Row<long unsigned int> target_list_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(
			arma::regspace<arma::Row<arma::uword> >(0,num_target_nodes-1));
		const arma::Row<long unsigned int> source_list_idx_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(source_list_idx);
		const arma::Row<long unsigned int> source_list_combined_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(source_list_combined.t());
		const arma::Row<long unsigned int> first_source_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(first_source);
		const arma::Row<long unsigned int> last_source_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(last_source);
		const arma::Row<long unsigned int> first_target_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(ti.head_cols(num_target_nodes));
		const arma::Row<long unsigned int> last_target_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(ti.tail_cols(num_target_nodes)-1);

		// check if cuda has different precision from rat
		arma::Mat<cufltp> Rs4(4,num_sources_); 
		arma::Mat<cufltp> Ieff4(4,num_sources_); 
		arma::Mat<cufltp> Rtt4(4,ntt); 
		#ifdef RAT_CUDA_CONVERSION_NEEDED
			// convert to fltp4
			Rs4.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(Rs_); 
			Rs4.row(3) = arma::conv_to<arma::Row<cufltp> >::from(epss_); 	
			Ieff4.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(dRs_);
			Ieff4.row(3) = arma::conv_to<arma::Row<cufltp> >::from(Is_); 
			Rtt4.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(Rtt);

			// Rs4.rows(0,2) = Extra::parallel_convert<cufltp>(Rs_);
			// Rs4.row(3) = Extra::parallel_convert<cufltp>(epss_); 	
			// Ieff4.rows(0,2) = Extra::parallel_convert<cufltp>(dRs_);
			// Ieff4.row(3) = Extra::parallel_convert<cufltp>(Is_); 
			// Rtt4.rows(0,2) = Extra::parallel_convert<cufltp>(Rtt);
		#else
			// convert to fltp4
			Rs4.rows(0,2) = Rs_; Rs4.row(3) = epss_; 
			Ieff4.rows(0,2) = dRs_;	Ieff4.row(3) = Is_; 
			Rtt4.rows(0,2) = Rtt;
		#endif

		// check which fields to calculate
		const bool calc_A = tar->has('A');
		const bool calc_H = tar->has('H') || tar->has('B');

		// allocate magnetic field
		arma::Mat<cufltp> A, H; 
		if(calc_A)A.zeros(4,ntt);
		if(calc_H)H.zeros(4,ntt);

		// check sizes
		assert(Rtt4.n_cols==ntt); assert(Rtt4.n_rows==4);
		assert(Rs4.n_cols==num_sources_); assert(Rs4.n_rows==4);
		assert(Ieff4.n_cols==num_sources_); assert(Ieff4.n_rows==4);
		assert(first_target_single.n_elem==num_target_nodes); 
		assert(last_target_single.n_elem==num_target_nodes);
		assert(first_source_single.n_elem==num_source_nodes); 
		assert(last_source_single.n_elem==num_source_nodes);
		assert(target_list_single.n_elem==num_target_list); 
		assert(source_list_idx_single.n_elem==num_target_list+1);
		assert(source_list_combined_single.n_elem==num_source_list);

		// get gpu devices
		std::set<int> gpu_devices = stngs->get_gpu_devices();
		if(gpu_devices.empty())rat_throw_line("No GPU devices set for calculation");

		// ofload to GPU kernel
		GpuKernels::so2ta_kernel(
			A.memptr(), H.memptr(), Rtt4.memptr(), static_cast<long unsigned int>(ntt), 
			Rs4.memptr(), Ieff4.memptr(), static_cast<long unsigned int>(num_sources_),
			first_target_single.memptr(), last_target_single.memptr(), static_cast<long unsigned int>(num_target_nodes),
			first_source_single.memptr(), last_source_single.memptr(), static_cast<long unsigned int>(num_source_nodes),
			target_list_single.memptr(), source_list_idx_single.memptr(), static_cast<long unsigned int>(num_target_list),
			source_list_combined_single.memptr(), static_cast<long unsigned int>(num_source_list), calc_A, calc_H, use_van_Lanen_, 
			gpu_devices);

		// check if cuda has different precision from rat
		#ifdef RAT_CUDA_CONVERSION_NEEDED
			// get and store vector potential
			if(tar->has('A'))tar->add_field('A',arma::conv_to<arma::Mat<fltp> >::from(A.rows(0,2)),ft,lt,ti,true);
		
			// get and store magnetic field
			if(tar->has('H'))tar->add_field('H',arma::conv_to<arma::Mat<fltp> >::from(H.rows(0,2)),ft,lt,ti,true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*arma::conv_to<arma::Mat<fltp> >::from(H.rows(0,2)),ft,lt,ti,true);
		#else
			// get and store vector potential
			if(tar->has('A'))tar->add_field('A',A.rows(0,2),ft,lt,ti,true);
		
			// get and store magnetic field
			if(tar->has('H'))tar->add_field('H',H.rows(0,2),ft,lt,ti,true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H.rows(0,2),ft,lt,ti,true);
		#endif		
	}


	// source to target GPU kernel
	// old kernel is generally slower because it needs
	// to copy all targets to the GPU and back instead
	// of only the useful targets
	void CurrentSources::source_to_target_gpu_alt(
		const ShTargetsPr &tar, const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target,
		const ShSettingsPr &stngs) const{

		// do not run gpu if there is no source-target interactions
		if(source_list.is_empty() || target_list.is_empty())return;

		// get targets
		const arma::Mat<fltp> Rt = tar->get_target_coords(); // targets
		const arma::uword num_targets = Rt.n_cols;
		
		// get counters
		const arma::uword num_target_nodes = first_target.n_elem;
		const arma::uword num_source_nodes = first_source.n_elem;
		const arma::uword num_target_list = target_list.n_elem;
		assert(source_list.n_elem==num_target_list);
		assert(last_source.n_elem==num_source_nodes);
		assert(last_target.n_elem==num_target_nodes);

		// create indexes for source and target lists
		arma::Col<arma::uword> source_list_idx(num_target_list + 1);
		source_list_idx(0) = 0;
		for(arma::uword i=0;i<num_target_list;i++)
			source_list_idx(i+1) = source_list(i).n_elem;

		// accumulate sizes to create indexes
		source_list_idx = arma::cumsum(source_list_idx);

		// nnumber of source nodes
		const arma::uword num_source_list = source_list_idx(num_target_list);

		// combine source list
		arma::Col<arma::uword> source_list_combined(num_source_list);

		// fill source list
		for(arma::uword i=0;i<num_target_list;i++){
			source_list_combined.rows(source_list_idx(i),source_list_idx(i+1)-1) = source_list(i); 
		}

		// check if cuda has different precision from rat
		arma::Mat<cufltp> Rs_single(4,num_sources_); 
		arma::Mat<cufltp> Ieff_single(4,num_sources_); 
		arma::Mat<cufltp> Rt_single(4,num_targets); 
		#ifdef RAT_CUDA_CONVERSION_NEEDED
			// convert to fltp4
			Rs_single.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(Rs_); 
			Rs_single.row(3) = arma::conv_to<arma::Mat<cufltp> >::from(epss_); 	
			Ieff_single.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(dRs_);
			Ieff_single.row(3) = arma::conv_to<arma::Mat<cufltp> >::from(Is_); 
			Rt_single.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(Rt);
		#else
			// convert to fltp4
			Rs_single.rows(0,2) = Rs_; Rs_single.row(3) = epss_; 
			Ieff_single.rows(0,2) = dRs_; Ieff_single.row(3) = Is_; 
			Rt_single.rows(0,2) = Rt;
		#endif

		// convert indices to 32 bit integers
		const arma::Row<long unsigned int> target_list_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(target_list);
		const arma::Row<long unsigned int> source_list_idx_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(source_list_idx);
		const arma::Row<long unsigned int> source_list_combined_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(source_list_combined.t());
		const arma::Row<long unsigned int> first_source_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(first_source);
		const arma::Row<long unsigned int> last_source_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(last_source);
		const arma::Row<long unsigned int> first_target_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(first_target);
		const arma::Row<long unsigned int> last_target_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(last_target);

		// check sizes
		assert(Rt_single.n_cols==num_targets); assert(Rt_single.n_rows==4);
		assert(Rs_single.n_cols==num_sources_); assert(Rs_single.n_rows==4);
		assert(Ieff_single.n_cols==num_sources_); assert(Ieff_single.n_rows==4);
		assert(first_target_single.n_elem==num_target_nodes); 
		assert(last_target_single.n_elem==num_target_nodes);
		assert(first_source_single.n_elem==num_source_nodes); 
		assert(last_source_single.n_elem==num_source_nodes);
		assert(target_list_single.n_elem==num_target_list); 
		assert(source_list_idx_single.n_elem==num_target_list+1);
		assert(source_list_combined_single.n_elem==num_source_list);

		// check which fields to calculate
		const bool calc_A = tar->has('A');
		const bool calc_H = tar->has('H') || tar->has('B');

		// allocate magnetic field
		arma::Mat<cufltp> A_single, H_single; 
		if(calc_A)A_single.zeros(4,num_targets);
		if(calc_H)H_single.zeros(4,num_targets);

		// get gpu devices
		std::set<int> gpu_devices = stngs->get_gpu_devices();
		if(gpu_devices.empty())rat_throw_line("No GPU devices set for calculation");

		// ofload to GPU kernel
		GpuKernels::so2ta_kernel(
			A_single.memptr(), H_single.memptr(), Rt_single.memptr(), static_cast<long unsigned int>(num_targets), 
			Rs_single.memptr(), Ieff_single.memptr(), static_cast<long unsigned int>(num_sources_),
			first_target_single.memptr(), last_target_single.memptr(), static_cast<long unsigned int>(num_target_nodes),
			first_source_single.memptr(), last_source_single.memptr(), static_cast<long unsigned int>(num_source_nodes),
			target_list_single.memptr(), source_list_idx_single.memptr(), static_cast<long unsigned int>(num_target_list),
			source_list_combined_single.memptr(), static_cast<long unsigned int>(num_source_list), calc_A, calc_H, use_van_Lanen_, 
			gpu_devices);

		// check if cuda has different precision from rat
		#ifdef RAT_CUDA_CONVERSION_NEEDED
			// get and store vector potential
			if(tar->has('A'))tar->add_field('A',arma::conv_to<arma::Mat<fltp> >::from(A_single.rows(0,2)),true);
				
			// get and store magnetic field
			if(tar->has('H'))tar->add_field('H',arma::conv_to<arma::Mat<fltp> >::from(H_single.rows(0,2)),true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*arma::conv_to<arma::Mat<fltp> >::from(H_single.rows(0,2)),true);
		#else
			// get and store vector potential
			if(tar->has('A'))tar->add_field('A',A_single.rows(0,2),true);
				
			// get and store magnetic field
			if(tar->has('H'))tar->add_field('H',H_single.rows(0,2),true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H_single.rows(0,2),true);
		#endif

	}

	// source to target GPU kernel
	void CurrentSources::source_to_multipole_gpu(
		arma::Mat<std::complex<fltp> > &Mp,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source,
		const ShSettingsPr &stngs) const{

		// get number of expansions
		const int num_exp = stngs->get_num_exp();

		// convert to singles
		// const arma::Row<long unsigned int> first_source_single = 
		// 	arma::conv_to<arma::Row<long unsigned int> >::from(first_source);
		// const arma::Row<long unsigned int> last_source_single = 
		// 	arma::conv_to<arma::Row<long unsigned int> >::from(last_source);

		// convert to single precision fltps
		// also store in 4XN matrix as to accomodate CUDA fltp4
		// arma::wall_clock timer2; timer2.tic();
		arma::Mat<cufltp> Rn_single(4,num_sources_);
		arma::Mat<cufltp> Ieff_single(4,num_sources_);
		#ifdef RAT_CUDA_CONVERSION_NEEDED
			Rn_single.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(dR_); 
			Rn_single.row(3) = arma::conv_to<arma::Row<cufltp> >::from(epss_);
			Ieff_single.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(dRs_); 
			Ieff_single.row(3) = arma::conv_to<arma::Row<cufltp> >::from(Is_);
		#else
			Rn_single.rows(0,2) = dR_; Rn_single.row(3) = epss_;
			Ieff_single.rows(0,2) = dRs_; Ieff_single.row(3) = Is_;
		#endif		

		// get counters
		const arma::uword num_sources = Rs_.n_cols;
		const arma::uword num_nodes = first_source.n_elem;

		// allocate output multipoles
		arma::Mat<std::complex<cufltp> > Mp_half(Extra::hfpolesize(num_exp), Mp.n_cols);

		// ofload to GPU kernel
		GpuKernels::so2mp_kernel(
			Mp_half.memptr(), num_exp,
			Rn_single.memptr(), Ieff_single.memptr(), num_sources,
			first_source.memptr(), last_source.memptr(), num_nodes,
			stngs->get_gpu_devices());

		// convert back and create m<0 half from complex conjugate of m>0 half
		for(int n=0;n<=num_exp;n++){
			for(int m=0;m<=n;m++){
				// check if cuda has different precision from rat
				#ifdef RAT_CUDA_CONVERSION_NEEDED
					Mp.row(Extra::nm2fidx(n,m)) = 
						arma::conv_to<arma::Row<std::complex<fltp> > >::from(
						Mp_half.row(Extra::hfnm2fidx(n,m)));		
				#else
					Mp.row(Extra::nm2fidx(n,m)) = Mp_half.row(Extra::hfnm2fidx(n,m));
				#endif
				if(m!=0)Mp.row(Extra::nm2fidx(n,-m)) = arma::conj(Mp.row(Extra::nm2fidx(n,m)));
			}
		}

		// free memory
		// GpuKernels::free_pinned(Mp_single_ptr);
	}


	// end of CUDA specific code
	#endif

}}
