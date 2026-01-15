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
#include "stars.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// gravitational constant for solar dynamics
	// https://en.wikipedia.org/wiki/Gravitational_constant
	// In astrophysics, it is convenient to measure distances 
	// in parsecs (pc), velocities in kilometers per second 
	// (km/s) and masses in solar units M⊙. In these units, 
	// the gravitational constant is: 
	const fltp G = 4.302e-3; // pc M^-1 (km/s)^2

	// constructor
	Stars::Stars(){
		// set field type
		add_field_type('V',1);
		add_field_type('F',3);
	}

	// constructor with input
	Stars::Stars(
		const arma::Mat<fltp> &Rs, 
		const arma::Row<fltp> &Ms,
		const arma::Row<fltp> &epss){
		
		// check input
		assert(Rs.n_rows==3);
		assert(Rs.n_cols==Ms.n_cols); 
		assert(Rs.n_cols==epss.n_cols);

		// set sources 
		set_coords(Rs); set_mass(Ms); set_softening(epss);

		// set field type
		add_field_type('V',1);
		add_field_type('F',3);
	}
			
	// factory
	ShStarsPr Stars::create(){
		//return ShIListPr(new IList);
		return std::make_shared<Stars>();
	}

	// factory with input
	ShStarsPr Stars::create(
		const arma::Mat<fltp> &Rs, 
		const arma::Row<fltp> &Ms,
		const arma::Row<fltp> &epss){

		//return ShIListPr(new IList);
		return std::make_shared<Stars>(Rs,Ms,epss);
	}

	// get number of dimensions
	arma::uword Stars::get_num_dim() const{
		return num_dim_;
	}

	// set coordinate vectors
	void Stars::set_coords(const arma::Mat<fltp> &Rs){
		// check input
		if(Rs.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(!Rs.is_finite())rat_throw_line("coordinate matrix must be finite");
		
		// coordinates can only be set once
		if(!Rs_.is_empty())rat_throw_line("coordinates can only be set once");
		
		// set coordinate vectors
		Rs_ = Rs;

		// set number of sources
		num_sources_ = Rs_.n_cols;

		// also set target coords
		set_target_coords(Rs);
	}

	// set number of solar masses
	void Stars::set_mass(const arma::Row<fltp> &Ms){
		assert(Ms.is_finite());

		Ms_ = Ms;
	}

	// setting the softening factor
	void Stars::set_softening(const arma::Row<fltp> &epss){
		// check input
		assert(epss.is_finite());

		// set softening
		epss_ = epss;
	}

	// sorting function
	void Stars::sort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		assert(!Rs_.is_empty());
		assert(!Ms_.is_empty());
		assert(!epss_.is_empty());

		// check if sort array right length
		assert(Rs_.n_cols == sort_idx.n_elem);

		// sort sources
		Rs_ = Rs_.cols(sort_idx);
		Ms_ = Ms_.cols(sort_idx);
		epss_ = epss_.cols(sort_idx);
	}

	// unsorting function
	void Stars::unsort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		assert(!Rs_.is_empty());
		assert(!Ms_.is_empty());
		assert(!epss_.is_empty());

		// check if sort array right length
		assert(Rs_.n_cols == sort_idx.n_elem);

		// sort sources
		Rs_.cols(sort_idx) = Rs_;
		Ms_.cols(sort_idx) = Ms_;
		epss_.cols(sort_idx) = epss_;
	}

	// count number of sources stored
	arma::uword Stars::num_sources() const{
		// return number of elements
		return num_sources_;
	}

	// method for getting all coordinates
	arma::Mat<fltp> Stars::get_source_coords() const{
		// return coordinates
		return Rs_;
	}

	// // method for getting coordinates with specific indices
	// arma::Mat<fltp> Stars::get_source_coords(
	// 	const arma::Row<arma::uword> &indices) const{

	// 	// return coordinates
	// 	return Rs_.cols(indices);
	// }

	// method for getting all coordinates
	arma::Mat<fltp> Stars::get_mass() const{
		// return coordinates
		return Ms_;
	}

	// setup source to multipole matrices
	void Stars::setup_source_to_multipole(
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
	void Stars::source_to_multipole(
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
				StMat_So2Mp_J M_J;
				M_J.set_num_exp(num_exp);
				M_J.calc_matrix(-dR_.cols(first_source(i),last_source(i)));

				// apply matrix
				Mp.cols(i*num_dim_, (i+1)*num_dim_-1) = M_J.apply(
					Ms_.cols(first_source(i),last_source(i)));
			});
		}

		// faster less memory efficient implementation
		else{
			// walk over source nodes
			cmn::parfor(0,first_source.n_elem,stngs->get_parallel_s2m(),[&](arma::uword i, int){
				// add child source contribution to this multipole
				Mp.cols(i*num_dim_, (i+1)*num_dim_-1) = M_J_.apply(
					Ms_.cols(first_source(i),last_source(i)),
					first_source(i),last_source(i));
			});
		}
	}

	// direct calculation of vector potential or magnetic 
	// field for all sources at all target points
	void Stars::calc_direct(const ShTargetsPr &tar, const ShSettingsPr &/*stngs*/, const cmn::ShLogPr &/*lg*/) const{
		// get target coordinates
		const arma::Mat<fltp> Rt = tar->get_target_coords();

		// forward calculation of vector potential to extra
		if(tar->has('V')){
			// calculate
			arma::Row<fltp> V = calc_M2V_s(Rs_, Ms_, epss_, Rt, true);

			// set
			tar->add_field('V',V,false);
		}

		// forward calculation of vector potential to extra
		if(tar->has('F')){
			// calculate
			arma::Mat<fltp> F = calc_M2F_s(Rs_, Ms_, epss_, Rt, true);

			// set
			tar->add_field('F',F,false);
		}


	}


	// source to target kernel
	void Stars::source_to_target(
		const ShTargetsPr &tar, const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list,
		const arma::Row<arma::uword> &first_source, const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, const arma::Row<arma::uword> &last_target, 
		const ShSettingsPr &stngs, const cmn::ShLogPr &lg) const{

		// get targets
		// const arma::Mat<fltp> Rt = tar->get_target_coords();

		// calculation of gravitational potential
		if(tar->has('V')){
			// allocate
			arma::Row<fltp> V(tar->num_targets(),arma::fill::zeros);

			// walk over source nodes
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

				// walk over target nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// get source node
					const arma::uword source_idx = source_list(i)(j);

					// get my source
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<num_sources_);

					// run kernel
					V.cols(ft,lt) += calc_M2V_s(Rs_.cols(fs,ls), Ms_.cols(fs,ls), 
						epss_.cols(fs,ls), Rt, false);
				}
			});
			
			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			tar->add_field('V',V,true);
		}

		// calculation of gravitational potential
		if(tar->has('F')){
			// allocate
			arma::Mat<fltp> F(3,tar->num_targets(),arma::fill::zeros);

			// walk over source nodes
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
				
				// walk over target nodes
				for(arma::uword j=0;j<source_list(target_idx).n_elem;j++){
					// get source node
					const arma::uword source_idx = source_list(i)(j);

					// get my source
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<num_sources_);

					// run kernel
					F.cols(ft,lt) += calc_M2F_s(Rs_.cols(fs,ls), Ms_.cols(fs,ls), 
						epss_.cols(fs,ls), Rt, false);
				}
			});

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			tar->add_field('F',F,true);
		}
	}

	// localpole to target setup function
	void Stars::setup_localpole_to_target(
		const arma::Mat<fltp> &dR, 
		const arma::uword /*num_dim*/,
		const ShSettingsPr &stngs){

		// check input
		assert(dR.is_finite());

		// memory efficient implementation (default)
		if(stngs->get_memory_efficient_l2t()){
			dRl2p_ = dR;
		}

		// maximize computation speed
		else{
			// get number of expansions
			const int num_exp = stngs->get_num_exp();

			// gravitational potential
			if(has('V')){
				// calculate matrix for all target points
				M_A_.set_num_exp(num_exp);
				M_A_.calc_matrix(-dR);
			}

			// forces
			if(has('F')){
				// calculate matrix for all target points
				M_F_.set_num_exp(num_exp);
				M_F_.calc_matrix(-dR);
			}
		}
	}

	// localpole to target function
	void Stars::localpole_to_target(
		const arma::Mat<std::complex<fltp> > &Lp, 
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target, 
		const arma::uword num_dim, 
		const ShSettingsPr &stngs){

		// memory efficient implementation (default)
		// note that this method is called many times in parallel
		// do not re-use the class property of M_A and M_H
		if(stngs->get_memory_efficient_l2t()){
			// check if dR was set
			assert(!dRl2p_.is_empty());

			// get number of expansions
			const int num_exp = stngs->get_num_exp();

			// gravitational potential
			if(has('V')){
				// create temporary storage for vector potential
				arma::Row<fltp> V(num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate matrix for these target points
					StMat_Lp2Ta M_A;
					M_A.set_num_exp(num_exp);
					M_A.calc_matrix(-dRl2p_.cols(first_target(i),last_target(i)));

					// calculate and add vector potential
					V.cols(first_target(i),last_target(i)) -= 
						G*M_A.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1));
				});

				// add to self
				add_field('V',V,true);
			}

			// gravitational potential
			if(has('F')){
				// create temporary storage for vector potential
				arma::Mat<fltp> F(3,num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate matrix for these target points
					StMat_Lp2Ta_Grad M_F;
					M_F.set_num_exp(num_exp);
					M_F.calc_matrix(-dRl2p_.cols(first_target(i),last_target(i)));

					// calculate and add vector potential
					F.cols(first_target(i),last_target(i)) -= 
						G*M_F.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1));
				});

				// add to self
				add_field('F',F,true);
			}
		}

		// maximize computation speed using pre-calculated matrix
		else{
			// gravitational potential
			if(has('V')){
				// check if localpole to target matrix was set
				assert(!M_A_.is_empty());

				// create temporary storage for vector potential
				arma::Row<fltp> V(num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate and add vector potential
					V.cols(first_target(i),last_target(i)) -= 
						G*M_A_.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1),
						first_target(i),last_target(i));
				});

				// add to self
				add_field('V',V,true);
			}

			// gravitational potential
			if(has('F')){
				// check if localpole to target matrix was set
				assert(!M_F_.is_empty());

				// create temporary storage for vector potential
				arma::Mat<fltp> F(3,num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate and add vector potential
					F.cols(first_target(i),last_target(i)) -= 
						G*M_F_.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1),
						first_target(i),last_target(i));
				});

				// add to self
				add_field('F',F,true);
			}
		}

	}

	// calculation of gravity from point sources
	// softness applied to avoid infinite forces
	arma::Row<fltp> Stars::calc_M2V_s(
		const arma::Mat<fltp> &Rs, 
		const arma::Row<fltp> &Ms, 
		const arma::Row<fltp> &epss,
		const arma::Mat<fltp> &Rt, 
		const bool use_parallel){
		
		// check input
		assert(Rs.n_rows==3); //assert(Ieff.n_rows==3);  
		assert(Rt.n_rows==3); assert(Rs.n_cols==Ms.n_cols);

		// allocate output
		arma::Row<fltp> V(Rt.n_cols);

		// power of eps
		const arma::Row<fltp> eps3 = epss%epss%epss;	

		// walk over targets
		//for(int i=0;i<(int)Rt.n_cols;i++){
		cmn::parfor(0,Rt.n_cols,use_parallel,[&](arma::uword i, int) { // second int is CPU number
			// relative position
			const arma::Mat<fltp> dR = Rt.col(i) - Rs.each_col();

			// distance
			const arma::Row<fltp> rho = arma::sqrt(arma::sum(dR%dR,0));

			// powers of distance
			const arma::Row<fltp> rho3 = rho%rho%rho;
			
			// scale current
			const arma::Mat<fltp> Ms_s = Ms%arma::clamp(rho3/eps3,0.0,1.0);

			// calculate contribution of each source to V
			arma::Row<fltp> Vcon = Ms_s/rho;

			// fix self field
			Vcon.cols(arma::find(rho<1e-6)).fill(0.0);

			// calculate field
			V(i) = -G*arma::sum(Vcon);
		});

		// return vector potential at each target point
		return V;
	}

	// calculation of gravity from point sources
	// softness applied to avoid infinite forces
	arma::Mat<fltp> Stars::calc_M2F_s(
		const arma::Mat<fltp> &Rs, 
		const arma::Row<fltp> &Ms, 
		const arma::Row<fltp> &epss,
		const arma::Mat<fltp> &Rt,
		const bool use_parallel){
		
		// check input
		assert(Rs.n_rows==3); //assert(Ieff.n_rows==3);  
		assert(Rt.n_rows==3); assert(Rs.n_cols==Ms.n_cols);

		// allocate output
		arma::Mat<fltp> F(3,Rt.n_cols);

		// power of eps
		const arma::Row<fltp> eps3 = epss%epss%epss;	

		// walk over targets
		//for(int i=0;i<(int)Rt.n_cols;i++){
		cmn::parfor(0,Rt.n_cols,use_parallel,[&](arma::uword i, int) { // second int is CPU number
			// relative position
			const arma::Mat<fltp> dR = Rt.col(i) - Rs.each_col();

			// distance
			const arma::Row<fltp> rho = cmn::Extra::vec_norm(dR);

			// powers of distance
			const arma::Row<fltp> rho3 = rho%rho%rho;
			
			// scale current
			const arma::Mat<fltp> Ms_s = Ms%arma::clamp(rho3/eps3,0.0,1.0);

			// calculate contribution of each source to V
			arma::Mat<fltp> Fcon = dR.each_row()%(Ms_s/rho3);

			// fix self field
			Fcon.cols(arma::find(rho<1e-6)).fill(0.0);

			// calculate field
			F.col(i) = -G*arma::sum(Fcon,1);
		});

		// return vector potential at each target point
		return F;
	}

}}