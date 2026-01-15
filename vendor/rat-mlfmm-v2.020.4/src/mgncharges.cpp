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
#include "mgncharges.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// constructor
	MgnCharges::MgnCharges(){
		set_field_type('H',3);
	}

	// constructor with input
	MgnCharges::MgnCharges(
		const arma::Mat<fltp> &Rs, 
		const arma::Row<fltp> &sigma_s,
		const arma::Row<fltp> &epss) : MgnCharges(){
		
		// check input
		assert(Rs.n_rows==3);
		assert(Rs.n_cols==sigma_s.n_cols); 

		// set sources 
		set_coords(Rs); set_magnetic_charge(sigma_s); set_softening(epss);
	}

	// factory
	ShMgnChargesPr MgnCharges::create(){
		//return ShIListPr(new IList);
		return std::make_shared<MgnCharges>();
	}

	// factory with input
	ShMgnChargesPr MgnCharges::create(
		const arma::Mat<fltp> &Rs, 
		const arma::Row<fltp> &sigma_s,
		const arma::Row<fltp> &epss){
		return std::make_shared<MgnCharges>(Rs,sigma_s,epss);
	}

	// get number of dimensions
	arma::uword MgnCharges::get_num_dim() const{
		return num_dim_;
	}

	// set coordinate vectors
	void MgnCharges::set_coords(const arma::Mat<fltp> &R){
		// check input
		if(R.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(!R.is_finite())rat_throw_line("coordinate matrix must be finite");
		
		// coordinates can only be set once
		if(!Rs_.is_empty())rat_throw_line("coordinates can only be set once");
		if(!Rt_.is_empty())rat_throw_line("coordinates can only be set once");

		// set coordinate vectors
		Rs_ = R; Rt_ = R;

		// set number of sources
		num_sources_ = Rs_.n_cols; num_targets_ = Rt_.n_cols;
	}

	// set number of solar masses
	void MgnCharges::set_magnetic_charge(const arma::Row<fltp> &sigma_s){
		assert(sigma_s.is_finite());

		sigma_s_ = sigma_s;
	}

	// set softening factor
	void MgnCharges::set_softening(const arma::Row<fltp> &epss){
		assert(epss.is_finite());
		epss_ = epss;
	}

	// sorting function
	void MgnCharges::sort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		assert(!Rs_.is_empty());
		assert(!sigma_s_.is_empty());

		// check if sort array right length
		assert(Rs_.n_cols == sort_idx.n_elem);

		// sort sources
		Rs_ = Rs_.cols(sort_idx);
		sigma_s_ = sigma_s_.cols(sort_idx);
		epss_ = epss_.cols(sort_idx);
	}

	// unsorting function
	void MgnCharges::unsort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		assert(!Rs_.is_empty());
		assert(!sigma_s_.is_empty());

		// check if sort array right length
		assert(Rs_.n_cols == sort_idx.n_elem);

		// sort sources
		Rs_.cols(sort_idx) = Rs_;
		sigma_s_.cols(sort_idx) = sigma_s_;
		epss_.cols(sort_idx) = epss_;
	}

	// count number of sources stored
	arma::uword MgnCharges::num_sources() const{
		// return number of elements
		return num_sources_;
	}

	// method for getting all coordinates
	arma::Mat<fltp> MgnCharges::get_source_coords() const{
		// return coordinates
		return Rs_;
	}

	// // method for getting coordinates with specific indices
	// arma::Mat<fltp> MgnCharges::get_source_coords(
	// 	const arma::Row<arma::uword> &indices) const{

	// 	// return coordinates
	// 	return Rs_.cols(indices);
	// }

	// method for getting all coordinates
	const arma::Mat<fltp>& MgnCharges::get_charge() const{
		// return coordinates
		return sigma_s_;
	}

	// setup source to multipole matrices
	void MgnCharges::setup_source_to_multipole(
		const arma::Mat<fltp> &dRs, 
		const fmm::ShSettingsPr &stngs){

		// get number of expansions
		const int num_exp = stngs->get_num_exp();

		// memory efficient implementation (default)
		if(stngs->get_memory_efficient_s2m()){
			dRs_ = dRs;
		}

		// maximize speed over memory efficiency
		else{
			// set number of expansions and setup matrix
			M_J_.set_num_exp(num_exp);
			M_J_.calc_matrix(-dRs);
		}
	}

	// get multipole contribution of the sources with indices
	// the contributions of the sources are already summed
	void MgnCharges::source_to_multipole(
		arma::Mat<std::complex<fltp> > &Mp,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source,
		const fmm::ShSettingsPr &stngs) const{

		// check input
		assert(first_source.n_elem==last_source.n_elem);
		assert(!Mp.is_empty());

		// get number of expansions
		const int num_exp = stngs->get_num_exp();

		// memory efficient implementation (default)
		if(stngs->get_memory_efficient_s2m()){	
			// walk over source nodes
			cmn::parfor(0,first_source.n_elem,stngs->get_parallel_s2m(),[&](arma::uword i, int){
				// calculate contribution of currents and return calculated multipole
				fmm::StMat_So2Mp_J M_J;
				M_J.set_num_exp(num_exp);
				M_J.calc_matrix(-dRs_.cols(first_source(i),last_source(i)));

				// apply matrix
				Mp.cols(i*num_dim_, (i+1)*num_dim_-1) = M_J.apply(
					sigma_s_.cols(first_source(i),last_source(i)));
			});
		}

		// faster less memory efficient implementation
		else{
			// walk over source nodes
			cmn::parfor(0,first_source.n_elem,stngs->get_parallel_s2m(),[&](arma::uword i, int){
				// add child source contribution to this multipole
				Mp.cols(i*num_dim_, (i+1)*num_dim_-1) = M_J_.apply(
					sigma_s_.cols(first_source(i),last_source(i)),
					first_source(i),last_source(i));
			});
		}

		// check multipoles
		assert(Mp.is_finite());
	}

	// direct calculation of vector potential or magnetic 
	// field for all sources at all target points
	void MgnCharges::calc_direct(
		const fmm::ShTargetsPr &tar, 
		const fmm::ShSettingsPr &stngs,
		const cmn::ShLogPr &lg) const{
		
		// get target coordinates
		const arma::Mat<fltp> Rt = tar->get_target_coords();

		// forward calculation of vector potential to extra
		if(tar->has('S')){
			// allocate output
			arma::Row<fltp> S(Rt.n_cols);

			// third power of softening
			const arma::Row<fltp> eps3 = epss_%epss_%epss_;

			// walk over targets
			cmn::parfor(0,Rt.n_cols,stngs->get_parallel_s2t(),[&](arma::uword i, int){
				// check cancel flag
				if(lg->is_cancelled())return;

				// relative position
				const arma::Mat<fltp> dR = Rt.col(i) - Rs_.each_col();

				// distance
				const arma::Row<fltp> rho = cmn::Extra::vec_norm(dR);

				// scale charge
				const arma::Row<fltp> rho3 = rho%rho%rho;
				const arma::Row<fltp> sigma_eff_s = sigma_s_%arma::clamp(rho3/eps3,RAT_CONST(0.0),RAT_CONST(1.0));

				// check distance
				const arma::Col<arma::uword> id = arma::find(rho>1e-9);

				// calculate contribution of each source to V
				S(i) = arma::accu(sigma_eff_s(id)/rho(id));
			});

			// check cancel flag
			if(lg->is_cancelled())return;

			// set
			tar->add_field('S',S,false);
		}

		// forward calculation of vector potential to extra
		if(tar->has('H') || tar->has('B')){
			// allocate output
			arma::Mat<fltp> H(3,Rt.n_cols);

			// softening
			const arma::Row<fltp> eps3 = epss_%epss_%epss_;

			// walk over targets
			cmn::parfor(0,Rt.n_cols,stngs->get_parallel_s2t(),[&](arma::uword i, int){
				// check cancel flag
				if(lg->is_cancelled())return;

				// relative position
				const arma::Mat<fltp> dR = Rt.col(i) - Rs_.each_col();

				// distance
				const arma::Row<fltp> rho = cmn::Extra::vec_norm(dR);

				// third power of distance
				const arma::Row<fltp> rho3 = rho%rho%rho;

				// scale charge
				const arma::Row<fltp> sigma_eff_s = sigma_s_%arma::clamp(rho3/eps3,RAT_CONST(0.0),RAT_CONST(1.0));

				// check distance
				const arma::Col<arma::uword> id = arma::find(rho>1e-9);

				// add to H
				// H.col(i) = -arma::sum(dR.cols(id).eval().each_row()%(sigma_eff_s.cols(id)/rho3.cols(id)),1);
				H.col(i) = arma::sum(dR.cols(id).eval().each_row()%(sigma_eff_s.cols(id)/rho3.cols(id)),1);
			});

			// check cancel flag
			if(lg->is_cancelled())return;

			// divide by 4 pi
			H/=4*arma::Datum<fltp>::pi;

			// set
			if(tar->has('H'))tar->add_field('H',H,false);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,false);
		}

	}


	// source to target kernel
	void MgnCharges::source_to_target(
		const fmm::ShTargetsPr &tar, 
		const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target, 
		const fmm::ShSettingsPr &stngs,
		const cmn::ShLogPr &lg) const{

		// get targets
		const arma::Mat<fltp> Rt = tar->get_target_coords();

		// calculation of magnetic potential
		if(tar->has('S')){
			// allocate
			arma::Row<fltp> S(tar->num_targets(),arma::fill::zeros);

			// walk over target nodes
			cmn::parfor(0,target_list.n_elem,stngs->get_parallel_s2t(),[&](arma::uword i, int){
				// check cancel flag
				if(lg->is_cancelled())return;

				// get target node
				const arma::uword tni = target_list(i);

				// get location of the targets
				const arma::uword ft = first_target(tni);
				const arma::uword lt = last_target(tni);
				assert(ft<=lt); assert(lt<tar->num_targets());

				// walk over source nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// check cancel flag
					if(lg->is_cancelled())return;

					// get source node
					const arma::uword sni = source_list(i)(j);

					// get my source
					const arma::uword fs = first_source(sni);
					const arma::uword ls = last_source(sni);
					assert(fs<=ls); assert(ls<num_sources_);

					// get sources
					const arma::Mat<fltp> Rs = Rs_.cols(fs,ls);
					const arma::Row<fltp> sigma_s = sigma_s_.cols(fs,ls);
					const arma::Row<fltp> epss_s = epss_.cols(fs,ls);

					// third power of eps
					const arma::Row<fltp> eps3 = epss_s%epss_s%epss_s;

					// walk over target elements
					for(arma::uword k=ft;k<=lt;k++){
						// relative position
						const arma::Mat<fltp> dR = Rt.col(k) - Rs.each_col();

						// distance
						const arma::Row<fltp> rho = cmn::Extra::vec_norm(dR);

						// scale charge
						const arma::Row<fltp> rho3 = rho%rho%rho;
						const arma::Row<fltp> sigma_eff_s = sigma_s%arma::clamp(rho3/eps3,RAT_CONST(0.0),RAT_CONST(1.0));

						// check distance
						const arma::Col<arma::uword> id = arma::find(rho>1e-9);

						// add to S
						S(k) += arma::accu(sigma_eff_s(id)/rho(id));
					}
				}
			});

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			tar->add_field('S',S,true);
		}

		// calculation of magnetic potential
		if(tar->has('H') || tar->has('B')){
			// allocate output
			arma::Mat<fltp> H(3,Rt.n_cols,arma::fill::zeros);

			// walk over target nodes
			cmn::parfor(0,target_list.n_elem,stngs->get_parallel_s2t(),[&](arma::uword i, int){
				// check cancel flag
				if(lg->is_cancelled())return;

				// get target node
				const arma::uword tni = target_list(i);

				// get location of the targets
				const arma::uword ft = first_target(tni);
				const arma::uword lt = last_target(tni);
				assert(ft<=lt); assert(lt<tar->num_targets());

				// walk over source nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// check cancel flag
					if(lg->is_cancelled())return;

					// get source node
					const arma::uword sni = source_list(i)(j);

					// get my source
					const arma::uword fs = first_source(sni);
					const arma::uword ls = last_source(sni);
					assert(fs<=ls); assert(ls<num_sources_);

					// get sources
					const arma::Mat<fltp> Rs = Rs_.cols(fs,ls);
					const arma::Row<fltp> sigma_s = sigma_s_.cols(fs,ls);
					const arma::Row<fltp> epss_s = epss_.cols(fs,ls);

					// third power of eps
					const arma::Row<fltp> eps3 = epss_s%epss_s%epss_s;

					// walk over target elements
					for(arma::uword k=ft;k<=lt;k++){
						// relative position
						const arma::Mat<fltp> dR = Rt.col(k) - Rs.each_col();

						// distance
						const arma::Row<fltp> rho = cmn::Extra::vec_norm(dR);

						// third power of distance
						const arma::Row<fltp> rho3 = rho%rho%rho;

						// scale charge
						const arma::Row<fltp> sigma_eff_s = sigma_s%arma::clamp(rho3/eps3,RAT_CONST(0.0),RAT_CONST(1.0));

						// check distance
						const arma::Col<arma::uword> id = arma::find(rho>1e-9);

						// add to S
						// H.col(k) += -arma::sum(dR.cols(id).eval().each_row()%(sigma_eff_s.cols(id)/rho3.cols(id)),1);
						H.col(k) += arma::sum(dR.cols(id).eval().each_row()%(sigma_eff_s.cols(id)/rho3.cols(id)),1);
					}
				}
			});

			// check cancel flag
			if(lg->is_cancelled())return;

			// divide by 4 pi
			H/=4*arma::Datum<fltp>::pi;

			// set field to targets
			if(tar->has('H'))tar->add_field('H',H,true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,true);
		}
	}


	// calculate force on each charge
	arma::Mat<fltp> MgnCharges::calc_force()const{
		return arma::Datum<fltp>::mu_0*(get_field('H').each_row()%sigma_s_);
	}

}}