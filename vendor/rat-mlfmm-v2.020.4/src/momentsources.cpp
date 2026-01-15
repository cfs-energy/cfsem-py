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
#include "momentsources.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// constructor
	MomentSources::MomentSources(){

	}

	// constructor with input
	MomentSources::MomentSources(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &Mm){
		
		// check input
		assert(Rs.n_rows==3); assert(Mm.n_rows==3);
		assert(Rs.n_cols==Mm.n_cols); 

		// set sources 
		set_coords(Rs); set_moments(Mm); 
	}
			
	// factory
	ShMomentSourcesPr MomentSources::create(){
		//return ShIListPr(new IList);
		return std::make_shared<MomentSources>();
	}

	// factory with input
	ShMomentSourcesPr MomentSources::create(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &Mm){

		//return ShIListPr(new IList);
		return std::make_shared<MomentSources>(Rs,Mm);
	}

	// constructor
	arma::uword MomentSources::get_num_dim() const{
		return num_dim_;
	}

	// set coordinate vectors
	void MomentSources::set_coords(const arma::Mat<fltp> &Rs){
		// check input
		if(Rs.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(!Rs.is_finite())rat_throw_line("coordinate matrix must be finite");
		
		// coordinates can only be set once
		if(!Rs_.is_empty())rat_throw_line("coordinates can only be set once");
		
		// set coordinate vectors
		Rs_ = Rs;

		// set number of sources
		num_sources_ = Rs_.n_cols;
	}

	// set direction vector
	void MomentSources::set_moments(const arma::Mat<fltp> &Mm){
		// check input
		if(Mm.n_rows!=3)rat_throw_line("magnetic moment matrix must have three rows");
		
		assert(Mm.is_finite());

		// set direction vectors
		Mm_ = Mm;
	}

	// sorting function
	void MomentSources::sort_sources(
		const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		assert(!Rs_.is_empty());
		assert(!Mm_.is_empty());

		// check if sort array right length
		assert(Rs_.n_cols == sort_idx.n_elem);

		// sort sources
		Rs_ = Rs_.cols(sort_idx);
		Mm_ = Mm_.cols(sort_idx);
	}

	// unsorting function
	void MomentSources::unsort_sources(
		const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		assert(!Rs_.is_empty());
		assert(!Mm_.is_empty());

		// check if sort array right length
		assert(Rs_.n_cols == sort_idx.n_elem);

		// sort sources
		Rs_.cols(sort_idx) = Rs_;
		Mm_.cols(sort_idx) = Mm_;
	}


	// count number of sources stored
	arma::uword MomentSources::num_sources() const{
		// return number of elements
		return num_sources_;
	}

	// method for getting all coordinates
	arma::Mat<fltp> MomentSources::get_source_coords() const{
		// return coordinates
		return Rs_;
	}

	// // method for getting coordinates with specific indices
	// arma::Mat<fltp> MomentSources::get_source_coords(
	// 	const arma::Row<arma::uword> &indices) const{

	// 	// return coordinates
	// 	return Rs_.cols(indices);
	// }

	// setup source to multipole matrices
	void MomentSources::setup_source_to_multipole(
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
			M_M_.set_num_exp(num_exp);
			M_M_.calc_matrix(-dR);
		}	
	}

	// get multipole contribution of the sources with indices
	// the contributions of the sources are already summed
	void MomentSources::source_to_multipole(
		arma::Mat<std::complex<fltp> > &Mp,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source,
		const ShSettingsPr &stngs) const{

		// check input
		assert(first_source.n_elem==last_source.n_elem);
		assert(!Mp.is_empty());

		// memory efficient implementation (default)
		if(stngs->get_memory_efficient_s2m()){		
			// get number of expansions
			const int num_exp = stngs->get_num_exp();
			
			// walk over source nodes
			cmn::parfor(0,first_source.n_elem,stngs->get_parallel_s2m(),[&](arma::uword i, int){
				// calculate contribution of currents and return calculated multipole
				StMat_So2Mp_M M_M;
				M_M.set_num_exp(num_exp);
				M_M.set_use_parallel(false);
				M_M.calc_matrix(-dR_.cols(first_source(i),last_source(i)));

				// apply matrix
				Mp.cols(i*num_dim_, (i+1)*num_dim_-1) = M_M.apply(
					Mm_.cols(first_source(i),last_source(i)));
			});
		}

		// faster less memory efficient implementation
		else{
			// walk over source nodes
			cmn::parfor(0,first_source.n_elem,stngs->get_parallel_s2m(),[&](arma::uword i, int){
				// add child source contribution to this multipole
				Mp.cols(i*num_dim_, (i+1)*num_dim_-1) = M_M_.apply(
					Mm_.cols(first_source(i),last_source(i)),
					first_source(i),last_source(i));
			});
		}
	}

	// direct calculation of vector potential or magnetic 
	// field for all sources at all target points
	void MomentSources::calc_direct(
		const ShTargetsPr &tar, 
		const ShSettingsPr &/*stngs*/, 
		const rat::cmn::ShLogPr& /*lg*/) const{
		
		// get target coordinates
		const arma::Mat<fltp> Rt = tar->get_target_coords();

		// forward calculation of vector potential to extra
		if(tar->has('A')){
			// run normal savart kernel
			const arma::Mat<fltp> A = Savart::calc_M2A(Rs_, Mm_, Rt, true);

			// set
			tar->add_field('A',A,false);
		}

		// forward calculation of magnetic field to extra
		if(tar->has('H') || tar->has('B')){
			// calculate
			const arma::Mat<fltp> H = Savart::calc_M2H(Rs_, Mm_, Rt, true);

			// set
			if(tar->has('H'))tar->add_field('H',H,false);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,false);
		}
	}


	// source to target kernel
	void MomentSources::source_to_target(
		const ShTargetsPr &tar, const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list,
		const arma::Row<arma::uword> &first_source, const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, const arma::Row<arma::uword> &last_target, 
		const ShSettingsPr &stngs, 
		const cmn::ShLogPr &lg) const{

		// get targets
		// const arma::Mat<fltp> Rt = tar->get_target_coords();

		// forward calculation of vector potential to extra
		if(tar->has('A')){
			// allocate
			arma::Mat<fltp> A(num_dim_,tar->num_targets(),arma::fill::zeros);

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

				// get target points
				const arma::Mat<fltp> Rt = tar->get_target_coords(ft,lt);

				// walk over source nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// get target node
					const arma::uword source_idx = source_list(i)(j);

					// get my source
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<num_sources_);

					// run normal savart kernel
					A.cols(ft,lt) += Savart::calc_M2A(Rs_.cols(fs,ls), 
						Mm_.cols(fs,ls), Rt, false);
				}
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

				// get target points
				const arma::Mat<fltp> Rt = tar->get_target_coords(ft,lt);

				// walk over target nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// get target node
					const arma::uword source_idx = source_list(i)(j);

					// get my source
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<num_sources_);

					// run savart kernel
					H.cols(ft,lt) += Savart::calc_M2H(Rs_.cols(fs,ls), 
						Mm_.cols(fs,ls), Rt, false);
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

	// setup a magnetised ring
	// for testing purposes
	void MomentSources::setup_ring_magnet(
		const fltp Rin, const fltp Rout, 
		const fltp height, const arma::uword nr, 
		const arma::uword nz, const arma::uword nl,
		const fltp Max, const fltp Mrad){

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

		// allocate volumes
		arma::Mat<fltp> ve(nl,nr*nz);

		// calculate element volumes
		for(arma::uword i=0;i<nr;i++){
			for(arma::uword j=0;j<nz;j++){
				fltp ri = rho(i)-drho/2; fltp ro = rho(i)+drho/2;
				ve.col(i*nz + j).fill(dz*arma::Datum<fltp>::pi*(ro*ro-ri*ri)/nl);
			}
		}

		// reshape coordinates
		Rs_.set_size(3,num_sources_);
		Rs_.row(0) = arma::reshape(xe,1,num_sources_);
		Rs_.row(1) = arma::reshape(ye,1,num_sources_);
		Rs_.row(2) = arma::reshape(ze,1,num_sources_);
		arma::Row<fltp> V = arma::reshape(ve,1,num_sources_);

		// axial magnetic moment
		Mm_.zeros(3,num_sources_);
		Mm_.row(2) += V*Max;

		// radial magnetic moment
		Mm_.row(0) += Mrad*V%Rs_.row(0)/arma::sqrt(Rs_.row(0)%Rs_.row(0) + Rs_.row(1)%Rs_.row(1));
		Mm_.row(1) += Mrad*V%Rs_.row(1)/arma::sqrt(Rs_.row(0)%Rs_.row(0) + Rs_.row(1)%Rs_.row(1));
	}

}}