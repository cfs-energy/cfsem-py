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
#include "mgntargets.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// default constructor
	MgnTargets::MgnTargets(){
		// set field type
		add_field_type('A',3);
		add_field_type('B',3);
	}	

	// default constructor
	MgnTargets::MgnTargets(const arma::Mat<fltp> &Rt) : MgnTargets(){
		// set coordinates using appropriate method
		set_target_coords(Rt);
	}	

	// factory
	ShMgnTargetsPr MgnTargets::create(){
		return std::make_shared<MgnTargets>();
	}

	// factory
	ShMgnTargetsPr MgnTargets::create(const arma::Mat<fltp> &Rt){
		return std::make_shared<MgnTargets>(Rt);
	}

	// setup localpole to target matrix
	void MgnTargets::setup_localpole_to_target(
		const arma::Mat<fltp> &dRt,
		const arma::uword num_dim,
		const ShSettingsPr &stngs){

		// check input
		assert(dRt.is_finite());

		// memory efficient implementation (default)
		if(stngs->get_memory_efficient_l2t()){
			dRt_ = dRt;
		}

		// maximize computation speed
		else{
			// get number of expansions
			const int num_exp = stngs->get_num_exp();

			// for scalar potential 
			if(num_dim==1){
				// matrix for magnetic field
				if(has('H') || has('B')){
					// calculate matrix for all target points
					V_H_.set_num_exp(num_exp);
					V_H_.calc_matrix(-dRt);
				}
			}

			// for vector potential 
			else{
				// matrix for vector potential
				if(has('A')){
					// calculate matrix for all target points
					M_A_.set_num_exp(num_exp);
					M_A_.calc_matrix(-dRt);
				}

				// matrix for magnetic field
				if(has('H') || has('B')){
					// calculate matrix for all target points
					M_H_.set_num_exp(num_exp);
					M_H_.calc_matrix(-dRt);
				}
			}
		}
	}


	// localpole to target for scalar potential
	void MgnTargets::localpole_to_target_scalar_potential(
		const arma::Mat<std::complex<fltp> > &Lp,
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target,
		const arma::uword num_dim, 
		const ShSettingsPr &stngs){

		// check multipoles
		assert(Lp.is_finite());
		
		// memory efficient implementation
		if(stngs->get_memory_efficient_l2t()){
			// number of expansions
			const int num_exp = stngs->get_num_exp();

			// magnetic scalar potential
			if(has('S')){
				// create temporary storage for vector potential
				arma::Row<fltp> S(num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate matrix for these target points
					fmm::StMat_Lp2Ta M_A;
					M_A.set_num_exp(num_exp);
					M_A.calc_matrix(-dRt_.cols(first_target(i),last_target(i)));

					// calculate and add vector potential
					S.cols(first_target(i),last_target(i)) += M_A.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1));
				});

				// check field
				assert(S.is_finite());
		
				// add to self
				add_field('S',S,true);
			}

			// magnetic scalar potential
			if(has('H') || has('B')){
				// create temporary storage for vector potential
				arma::Mat<fltp> H(3,num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate matrix for these target points
					StMat_Lp2Ta_Grad V_H;
					V_H.set_num_exp(num_exp);
					V_H.calc_matrix(-dRt_.cols(first_target(i),last_target(i)));

					// calculate and add vector potential
					// H.cols(first_target(i),last_target(i)) += -V_H.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1));
					H.cols(first_target(i),last_target(i)) += V_H.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1));
				});

				// divide by 4 pi
				H/=4*arma::Datum<fltp>::pi;

				// check field
				assert(H.is_finite());

				// add to self
				if(has('H'))add_field('H',H,true);
				if(has('B'))add_field('B',arma::Datum<fltp>::mu_0*H,true);
			}
		}

		// normal implementation
		else{
			// magnetic scalar potential
			if(has('S')){
				// check if localpole to target matrix was set
				assert(!M_A_.is_empty());

				// create temporary storage for vector potential
				arma::Row<fltp> S(num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					S.cols(first_target(i),last_target(i)) += M_A_.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1), first_target(i), last_target(i));
				});

				// add to self
				add_field('S',S,true);
			}

			// magnetic field
			if(has('H') || has('B')){
				// check if localpole to target matrix was set
				assert(!M_H_.is_empty());

				// create temporary storage for vector potential
				arma::Mat<fltp> H(3,num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// H.cols(first_target(i),last_target(i)) += -V_H_.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1), first_target(i), last_target(i));
					H.cols(first_target(i),last_target(i)) += V_H_.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1), first_target(i), last_target(i));
				});

				// divide by 4 pi
				H/=4*arma::Datum<fltp>::pi;

				// add to self
				if(has('H'))add_field('H',H,true);
				if(has('B'))add_field('B',arma::Datum<fltp>::mu_0*H,true);
			}
		}
	}

	// localpole to target for vector potential
	void MgnTargets::localpole_to_target_vector_potential(
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
			assert(!dRt_.is_empty());

			// number of expansions
			const int num_exp = stngs->get_num_exp();

			// vector potential
			if(has('A')){
				// create temporary storage for vector potential
				arma::Mat<fltp> A(num_dim,num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate matrix for these target points
					StMat_Lp2Ta M_A;
					M_A.set_num_exp(num_exp);
					M_A.calc_matrix(-dRt_.cols(first_target(i),last_target(i)));

					// calculate and add vector potential
					A.cols(first_target(i),last_target(i)) += 
						RAT_CONST(1e-7)*M_A.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1));
				});

				// add to self
				add_field('A',A,true);
			}

			// magnetic field
			if(has('H') || has('B')){
				// create temporary storage for magnetic field
				arma::Mat<fltp> H(num_dim,num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate matrix for these target points
					StMat_Lp2Ta_Curl M_H;
					M_H.set_num_exp(num_exp);
					M_H.calc_matrix(-dRt_.cols(first_target(i),last_target(i)));

					// calculate and add vector potential
					H.cols(first_target(i),last_target(i)) += 
						M_H.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1))/(4*arma::Datum<fltp>::pi);
				});

				// add to self
				if(has('H'))add_field('H',H,true);
				if(has('B'))add_field('B',arma::Datum<fltp>::mu_0*H,true);
			}
		}

		// maximize computation speed using pre-calculated matrix
		else{
			// vector potential
			if(has('A')){
				// check if localpole to target matrix was set
				assert(!M_A_.is_empty());

				// create temporary storage for vector potential
				arma::Mat<fltp> A(num_dim,num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate and add vector potential
					A.cols(first_target(i),last_target(i)) += 
						RAT_CONST(1e-7)*M_A_.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1),
						first_target(i),last_target(i));
				});

				// add to self
				add_field('A',A,true);
			}

			// magnetic field
			if(has('H') || has('B')){
				// check if localpole to target matrix was set
				assert(!M_H_.is_empty());

				// create temporary storage for magnetic field
				arma::Mat<fltp> H(num_dim,num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate and add vector potential
					H.cols(first_target(i),last_target(i)) += M_H_.apply(
						Lp.cols(i*num_dim, (i+1)*num_dim-1),first_target(i),
						last_target(i))/(4*arma::Datum<fltp>::pi);
				});

				// add to self
				if(has('H'))add_field('H',H,true);
				if(has('B'))add_field('B',arma::Datum<fltp>::mu_0*H,true);
			}
		}
	}

	// add field contribution of supplied localpole to targets with indices
	void MgnTargets::localpole_to_target(
		const arma::Mat<std::complex<fltp> > &Lp,
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target,
		const arma::uword num_dim, 
		const ShSettingsPr &stngs){

		// scalar potential
		if(num_dim==1)localpole_to_target_scalar_potential(Lp,first_target,last_target,num_dim,stngs);

		// vector potential
		else localpole_to_target_vector_potential(Lp,first_target,last_target,num_dim,stngs);
	}


	// // get field from internal storage
	// // in this case the B and H field are implemented
	// // as special cases
	// arma::Mat<fltp> MgnTargets::get_field(
	// 	const char type) const{

	// 	// make sure the field is calculated
	// 	if(M_.empty())rat_throw_line("field is not calculated on this target");

	// 	// allocate output
	// 	arma::Mat<fltp> Mout;

	// 	// magnetic flux density (special case)
	// 	if(type=='B'){
	// 		// check
	// 		if(!has('H'))rat_throw_line("need H to calculate B");

	// 		// recursive call
	// 		Mout = arma::Datum<fltp>::mu_0*TargetPoints::get_field('H');
	// 		if(has('M')){
	// 			Mout += arma::Datum<fltp>::mu_0*TargetPoints::get_field('M');
	// 		}
	// 	}

	// 	// normal case
	// 	else{
	// 		Mout = TargetPoints::get_field(type);

	// 		// subtract magnetisation
	// 		// if(type=='H' && has('M')){
	// 		// 	Mout -= TargetPoints::get_field('M');
	// 		// }
	// 	}

	// 	// add field to matrix
	// 	return Mout; 
	// }

	// get magnetic field
	arma::Mat<fltp> MgnTargets::get_magnetic_field()const{
		if(!has('H'))rat_throw_line("there is no magnetic field stored");
		return get_field('H');
	}

	// get magnetic vector potential
	arma::Mat<fltp> MgnTargets::get_magnetic_vector_potential()const{
		if(!has('A'))rat_throw_line("there is no vector potential");
		return get_field('A');
	}

	// get magnetization
	arma::Mat<fltp> MgnTargets::get_magnetization()const{
		if(!has('M'))rat_throw_line("there is no magnetization");
		return get_field('M');
	}

	// get magnetization
	arma::Mat<fltp> MgnTargets::get_magnetic_flux_density()const{
		if(!has('B'))rat_throw_line("there is no flux density stored");
		return get_field('B');
	}

	// calculate force
	arma::Mat<fltp> MgnTargets::calc_force()const{
		return arma::Mat<fltp>(3,num_targets_,arma::fill::zeros);
	}

}}
