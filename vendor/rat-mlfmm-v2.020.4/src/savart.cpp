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
#include "savart.hh"

// specific headers
#include "rat/common/extra.hh"
#include "rat/common/parfor.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// calculate A directly without softening
	// approximating the element as a source point
	arma::Mat<fltp> Savart::calc_I2A(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &Ieff,
		const arma::Mat<fltp> &Rt,
		const bool use_parallel){

		// get number of dimensions
		const arma::uword num_dim = Ieff.n_rows;

		// check input
		assert(Rs.n_rows==3); //assert(Ieff.n_rows==3);  
		assert(Rt.n_rows==3); assert(Rs.n_cols==Ieff.n_cols);

		// allocate output
		arma::Mat<fltp> A(num_dim,Rt.n_cols);

		// walk over targets
		//for(int i=0;i<(int)Rt.n_cols;i++){
		cmn::parfor(0,Rt.n_cols,use_parallel,[&](arma::uword i, int) { // second int is CPU number
			// relative position
			const arma::Mat<fltp> dR = Rt.col(i) - Rs.each_col();

			// distance
			const arma::Row<fltp> rho = cmn::Extra::vec_norm(dR);

			// calculate field
			arma::Mat<fltp> Acon = Ieff.each_row()/rho;

			// fix self field
			Acon.cols(arma::find(rho<RAT_CONST(1e-6))).fill(RAT_CONST(0.0));

			// calculate vector potential and return output
			A.col(i) = arma::sum(Acon,1);
		});

		// check A
		assert(A.n_rows==num_dim); 
		assert(A.n_cols==Rt.n_cols);

		// apply scaling
		A*=RAT_CONST(1e-7);

		// return vector potential at each target point
		return A;
	}

	// calculate B directly using Biot-Savart without softening
	// approximating the element as a source point
	arma::Mat<fltp> Savart::calc_I2H(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &Ieff,
		const arma::Mat<fltp> &Rt,
		const bool use_parallel){

		// check input
		assert(Rs.n_rows==3); assert(Ieff.n_rows==3); 
		assert(Rt.n_rows==3); assert(Rs.n_cols==Ieff.n_cols);

		// allocate output
		arma::Mat<fltp> H(3,Rt.n_cols);

		// walk over targets
		// for(int i=0;i<(int)Rt.n_cols;i++){
		cmn::parfor(0,Rt.n_cols,use_parallel,[&](arma::uword i, int) { // second int is CPU number
			// relative position
			const arma::Mat<fltp> dR = Rt.col(i) - Rs.each_col();

			// distance
			const arma::Row<fltp> rho = cmn::Extra::vec_norm(dR);

			// powers of rho
			const arma::Row<fltp> rho3 = rho%rho%rho;
			
			// calculate magnetic field
			// https://en.wikipedia.org/wiki/Magnetic_moment
			arma::Mat<fltp> Hcon = cmn::Extra::cross(Ieff,dR).each_row()/rho3;

			// fix self field
			Hcon.cols(arma::find(rho<RAT_CONST(1e-6))).fill(0.0);

			// calculate vector potential and return output
			H.col(i) = arma::sum(Hcon,1);
		});

		// check H
		assert(H.n_rows==3); assert(H.n_cols==Rt.n_cols);

		// apply scaling
		H/=4*arma::Datum<fltp>::pi;

		// return magnetic field in [A/m] at each target point
		return H;
	}

	// calculate A directly without softening
	// approximating the element as a source point
	arma::Mat<fltp> Savart::calc_M2A(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &Meff,
		const arma::Mat<fltp> &Rt,
		const bool use_parallel){

		// check input
		assert(Rs.n_rows==3); assert(Meff.n_rows==3);  
		assert(Rt.n_rows==3); assert(Rs.n_cols==Meff.n_cols);

		// allocate output
		arma::Mat<fltp> A(3,Rt.n_cols);

		// walk over targets
		//for(int i=0;i<(int)Rt.n_cols;i++){
		cmn::parfor(0,Rt.n_cols,use_parallel,[&](arma::uword i, int) { // second int is CPU number
			// relative position
			const arma::Mat<fltp> dR = Rt.col(i) - Rs.each_col();

			// distance
			const arma::Row<fltp> rho = cmn::Extra::vec_norm(dR);
			
			// calculate field
			arma::Mat<fltp> Acon = 
				cmn::Extra::cross(Meff,dR).each_row()/(rho%rho%rho);

			// fix self field
			Acon.cols(arma::find(rho<1e-6)).fill(0.0);

			// calculate vector potential and return output
			A.col(i) = arma::sum(Acon,1);
		});

		// check A
		assert(A.n_rows==3); assert(A.n_cols==Rt.n_cols);

		// apply scaling
		A*=RAT_CONST(1e-7);

		// return vector potential at each target point
		return A;
	}

	// calculate B directly using Biot-Savart without softening
	// approximating the element as a source point
	arma::Mat<fltp> Savart::calc_M2H(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &Meff,
		const arma::Mat<fltp> &Rt,
		const bool use_parallel){

		// check input
		assert(Rs.n_rows==3); assert(Meff.n_rows==3); 
		assert(Rt.n_rows==3); assert(Rs.n_cols==Meff.n_cols);

		// allocate output
		arma::Mat<fltp> H(3,Rt.n_cols);

		// walk over targets
		//for(int i=0;i<(int)Rt.n_cols;i++){
		cmn::parfor(0,Rt.n_cols,use_parallel,[&](arma::uword i, int) { // second int is CPU number
			// relative position
			const arma::Mat<fltp> dR = Rt.col(i) - Rs.each_col();

			// distance
			const arma::Row<fltp> rho = cmn::Extra::vec_norm(dR);

			// powers of rho
			const arma::Row<fltp> rho3 = rho%rho%rho;
			const arma::Row<fltp> rho5 = rho3%rho%rho;

			// calculate magnetic field
			// https://en.wikipedia.org/wiki/Magnetic_moment
			arma::Mat<fltp> Hcon = -(Meff.each_row()/rho3) + 
				3*(cmn::Extra::dot(Meff,dR)/rho5)%dR.each_row();

			// arma::Mat<fltp> Hcon = -(Meff.each_row()/rho3);

			// fix self field
			Hcon.cols(arma::find(rho<RAT_CONST(1e-6))).fill(0);

			// calculate vector potential and return output
			H.col(i) = arma::sum(Hcon,1);
		});
		//}

		// check H
		assert(H.n_rows==3); assert(H.n_cols==Rt.n_cols);

		// apply scaling
		H /= 4*arma::Datum<fltp>::pi;

		// return magnetic field in [A/m] at each target point
		return H;
	}

	// calculate A directly without softening
	// approximating the element as a source point
	arma::Mat<fltp> Savart::calc_I2A_s(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &Ieff,
		const arma::Row<fltp> &eps,
		const arma::Mat<fltp> &Rt,
		const bool use_parallel){

		// get number of dimensions
		arma::uword num_dim = Ieff.n_rows;

		// check input
		assert(Rs.n_rows==3); //assert(Ieff.n_rows==3);  
		assert(Rt.n_rows==3); assert(Rs.n_cols==Ieff.n_cols);

		// allocate output
		arma::Mat<fltp> A(num_dim,Rt.n_cols);

		// power of eps
		const arma::Row<fltp> eps3 = eps%eps%eps;	

		// walk over targets
		//for(arma::uword i=0;i<Rt.n_cols;i++){
		cmn::parfor(0,Rt.n_cols,use_parallel,[&](arma::uword i, int) { // second int is CPU number
			// relative position
			const arma::Mat<fltp> dR = Rt.col(i) - Rs.each_col();

			// distance
			const arma::Row<fltp> rho = cmn::Extra::vec_norm(dR);

			// powers of distance
			const arma::Row<fltp> rho3 = rho%rho%rho;
			
			// scale current
			const arma::Mat<fltp> Ieff_s = Ieff.each_row()%arma::clamp(rho3/eps3,RAT_CONST(0.0),RAT_CONST(1.0));

			// calculate field
			arma::Mat<fltp> Acon = Ieff_s.each_row()/rho;

			// fix self field
			Acon.cols(arma::find(rho<RAT_CONST(1e-6))).fill(RAT_CONST(0.0));

			// calculate vector potential and return output
			A.col(i) = arma::sum(Acon,1);
		//}
		});

		// check A
		assert(A.n_rows==num_dim); 
		assert(A.n_cols==Rt.n_cols);

		// apply scaling
		A*=RAT_CONST(1e-7);

		// return vector potential at each target point
		return A;
	}

	// calculate B directly using Biot-Savart with softening
	// approximating the element as a source point
	arma::Mat<fltp> Savart::calc_I2H_vl(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &dRs, 
		const arma::Row<fltp> &Is, 
		const arma::Row<fltp> &eps, 
		const arma::Mat<fltp> &Rt, 
		const bool use_parallel){

		// check input
		assert(Rs.n_rows==3); assert(dRs.n_rows==3); 
		assert(Rt.n_rows==3); assert(Rs.n_cols==dRs.n_cols); assert(Rs.n_cols==Is.n_cols);

		// allocate output
		arma::Mat<fltp> H(3,Rt.n_cols);

		// walk over targets
		//for(arma::uword i=0;i<Rt.n_cols;i++){
		cmn::parfor(0,Rt.n_cols,use_parallel,[&](arma::uword i, int) { // second int is CPU number
			// left and right end of source line
			const arma::Mat<fltp> R1 = Rs - dRs/2;
			const arma::Mat<fltp> R2 = Rs + dRs/2;

			// calculate length
			const arma::Row<fltp> ell = rat::cmn::Extra::vec_norm(dRs);

			// distance between left end source line and target point 
			const arma::Mat<fltp> dR = R1.each_col() - Rt.col(i);

			// minimum distance between point and source line
			const arma::Mat<fltp> Q = rat::cmn::Extra::cross(-dR, Rt.col(i) - R2.each_col());
			const arma::Row<fltp> q = rat::cmn::Extra::vec_norm(Q)/ell;

			// line parameter closest to target point
			const arma::Row<fltp> t0 = -rat::cmn::Extra::dot(dR, dRs)/(ell%ell);

			// point on line closest to target point
			const arma::Mat<fltp> R01 = R1 + dRs.each_row()%t0;

			// sign of left end of source line 
			const arma::Row<arma::sword> sgn1 = 1 - 2*arma::conv_to<arma::Row<arma::sword> >::from(t0>=0);
			const arma::Row<arma::sword> sgn2 = 1 - 2*arma::conv_to<arma::Row<arma::sword> >::from(t0>=1);

			// sign of left end of source line 
			const arma::Row<fltp> p1 = sgn1%rat::cmn::Extra::vec_norm(R1 - R01);
			const arma::Row<fltp> p2 = sgn2%rat::cmn::Extra::vec_norm(R2 - R01);

			// solution of Biot-Savart law, absolute value of magnetic field
			const arma::Row<fltp> Ba = p2/(q%arma::sqrt(q%q + p2%p2)) - p1/(q%arma::sqrt(q%q + p1%p1));


			// field vector, to be normalised
			const arma::Mat<fltp> v = rat::cmn::Extra::cross(dRs,dR);

			// norm of v
			const arma::Row<fltp> lv = rat::cmn::Extra::vec_norm(v);

			// cylinder ratio
			const arma::Row<fltp> fi = arma::clamp(q%q/(eps%eps),RAT_CONST(0.0),RAT_CONST(1.0));

			// calculate magnetic field
			const arma::Mat<fltp> Hcon = v.each_row()%(-fi%Is%Ba/lv);

			// add field
			H.col(i) = arma::sum(Hcon.cols(arma::find(q>RAT_CONST(1e-3)*ell)),1);
		//}
		});

		// apply scaling
		H/=4*arma::Datum<fltp>::pi;

		// check H
		assert(H.n_rows==3); assert(H.n_cols==Rt.n_cols);

		// return magnetic field in [A/m] at each target point
		return H;
	}


	// calculate B directly using Biot-Savart with softening
	// approximating the element as a source point
	arma::Mat<fltp> Savart::calc_I2H_s(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &Ieff,
		const arma::Row<fltp> &eps,
		const arma::Mat<fltp> &Rt,
		const bool use_parallel){

		// check input
		assert(Rs.n_rows==3); assert(Ieff.n_rows==3); 
		assert(Rt.n_rows==3); assert(Rs.n_cols==Ieff.n_cols);

		// allocate output
		arma::Mat<fltp> H(3,Rt.n_cols);

		// power of eps
		const arma::Row<fltp> eps3 = eps%eps%eps;

		// walk over targets
		//for(arma::uword i=0;i<Rt.n_cols;i++){
		cmn::parfor(0,Rt.n_cols,use_parallel,[&](arma::uword i, int) { // second int is CPU number
			// relative position
			const arma::Mat<fltp> dR = Rt.col(i) - Rs.each_col();

			// distance
			const arma::Row<fltp> rho = rat::cmn::Extra::vec_norm(dR);

			// powers of rho
			const arma::Row<fltp> rho3 = rho%rho%rho;
			
			// scale current
			const arma::Mat<fltp> Ieff_s = Ieff.each_row()%arma::clamp(rho3/eps3,RAT_CONST(0.0),RAT_CONST(1.0));

			// calculate magnetic field
			// https://en.wikipedia.org/wiki/Magnetic_moment
			arma::Mat<fltp> Hcon = cmn::Extra::cross(Ieff_s,dR).each_row()/rho3;

			// calculate vector potential and return output
			H.col(i) = arma::sum(Hcon.cols(arma::find(rho>RAT_CONST(1e-6))),1);
		//}
		});

		// check H
		assert(H.n_rows==3); assert(H.n_cols==Rt.n_cols);

		// apply scaling
		H/=4*arma::Datum<fltp>::pi;

		// return magnetic field in [A/m] at each target point
		return H;
	}

	// van Lanen kernel more accurate 
	// version for calculating vector potential
	arma::Mat<fltp> Savart::calc_I2A_vl(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &dRs, 
		const arma::Row<fltp> &Is, 
		const arma::Row<fltp> &eps, 
		const arma::Mat<fltp> &Rt, 
		const bool use_parallel){

		// check input
		assert(Rs.n_rows==3); //assert(Ieff.n_rows==3);  
		assert(Rt.n_rows==3); assert(Rs.n_cols==Is.n_cols);

		// allocate output
		arma::Mat<fltp> A(3,Rt.n_cols);

		// power of eps
		const arma::Row<fltp> eps3 = eps%eps%eps;	
		// const arma::Row<fltp> eps2 = eps%eps;	

		// walk over targets
		//for(arma::uword i=0;i<Rt.n_cols;i++){
		cmn::parfor(0,Rt.n_cols,use_parallel,[&](arma::uword i, int) { // second int is CPU number
			// relative position
			const arma::Mat<fltp> dR = Rt.col(i) - Rs.each_col();

			// calculate coeffcicients
			const arma::Row<fltp> a = cmn::Extra::dot(dRs,dRs); // squared length of the current element 
			const arma::Row<fltp> b = arma::abs(RAT_CONST(2.0)*cmn::Extra::dot(dRs,dR));
			const arma::Row<fltp> c = cmn::Extra::dot(dR,dR); // squared distance

			// distance
			const arma::Row<fltp> rho = arma::sqrt(c);

			// distance
			// const arma::Row<fltp> rho2 = c - arma::square(b/(2*arma::sqrt(a)));

			// third power of distance
			const arma::Row<fltp> rho3 = c%rho;

			// calculate effective current
			// const arma::Row<fltp> cylinder_ratio = arma::clamp(rho2/eps2,0.0,1.0); 
			const arma::Row<fltp> sphere_ratio = arma::clamp(rho3/eps3,RAT_CONST(0.0),RAT_CONST(1.0)); 

			// subexpression elimination in the compiler will hopefully eliminate all fltp calculations here
			const arma::Row<fltp> k1 = (b+a)/(RAT_CONST(2.0)*arma::sqrt(a))+arma::sqrt(a/RAT_CONST(4.0) + b/RAT_CONST(2.0) + c); 
			const arma::Row<fltp> k2 = (b-a)/(RAT_CONST(2.0)*arma::sqrt(a))+arma::sqrt(a/RAT_CONST(4.0) - b/RAT_CONST(2.0) + c); //with softening factor!!

			// calculate vector potential
			const arma::Row<fltp> scale = (arma::log(k1/k2)/arma::sqrt(a))%sphere_ratio%Is;
			const arma::Mat<fltp> Acon = dRs.each_row()%scale;

	  		// add source contributions
	  		const arma::Row<arma::uword> idx = arma::find(k1>RAT_CONST(1e-7) && k2>RAT_CONST(1e-7) && rho>RAT_CONST(1e-7)).t();
	  		if(!idx.is_empty())A.col(i) = arma::sum(Acon.cols(idx),1); else A.col(i).fill(RAT_CONST(0.0));
			//}
		});

		// check A
		assert(A.n_rows==3); 
		assert(A.n_cols==Rt.n_cols);

		// apply scaling
		A*=RAT_CONST(1e-7);

		// return vector potential at each target point
		return A;
	}


	// calculate B directly using Biot-Savart with softening
	// approximating the element as a source point
	arma::Mat<fltp> Savart::calc_I2H_vl_dir(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &dRs, 
		const arma::Row<fltp> &Is, 
		const arma::Row<fltp> &eps, 
		const arma::Mat<fltp> &Rt, 
		const bool use_parallel,
		const cmn::ShLogPr&lg){

		// check input
		assert(Rs.n_rows==3); assert(dRs.n_rows==3); 
		assert(Rt.n_rows==3); assert(Rs.n_cols==dRs.n_cols); assert(Rs.n_cols==Is.n_cols);

		// allocate output
		arma::Mat<fltp> H(3,Rt.n_cols);

		// walk over targets
		// for(arma::uword i=0;i<Rt.n_cols;i++){
		cmn::parfor(0,Rt.n_cols,use_parallel,[&](arma::uword i, int) { // second int is CPU number
			// check cancel flag
			if(lg->is_cancelled())return;

			// accumulator
			arma::Col<fltp>::fixed<3> Hsum(arma::fill::zeros);

			// walk over sources
			for(arma::uword j=0;j<Rs.n_cols;j++){
				// check cancel flag
				if(lg->is_cancelled())return;

				// left and right end of source line
				const arma::Col<fltp>::fixed<3> R1 = Rs.col(j) - dRs.col(j)/2;
				const arma::Col<fltp>::fixed<3> R2 = Rs.col(j) + dRs.col(j)/2;

				// calculate length
				const fltp ell = arma::as_scalar(rat::cmn::Extra::vec_norm(dRs.col(j)));

				// distance between left end source line and target point 
				const arma::Col<fltp>::fixed<3> dR = R1 - Rt.col(i);

				// minimum distance between point and source line
				const arma::Col<fltp>::fixed<3> Q = rat::cmn::Extra::cross(-dR, Rt.col(i) - R2);
				const fltp q = arma::as_scalar(rat::cmn::Extra::vec_norm(Q))/ell;

				// check
				if(q<=RAT_CONST(1e-3)*ell)continue;

				// line parameter closest to target point
				const fltp t0 = -arma::as_scalar(rat::cmn::Extra::dot(dR, dRs.col(j))/(ell*ell));

				// point on line closest to target point
				const arma::Col<fltp>::fixed<3> R01 = R1 + dRs.col(j)*t0;

				// sign of left end of source line 
				const arma::sword sgn1 = 1 - 2*(t0>=0);
				const arma::sword sgn2 = 1 - 2*(t0>=1);

				// sign of left end of source line 
				const fltp p1 = sgn1*arma::as_scalar(rat::cmn::Extra::vec_norm(R1 - R01));
				const fltp p2 = sgn2*arma::as_scalar(rat::cmn::Extra::vec_norm(R2 - R01));

				// solution of Biot-Savart law, absolute value of magnetic field
				const fltp Ba = p2/(q*std::sqrt(q*q + p2*p2)) - p1/(q*std::sqrt(q*q + p1*p1));

				// field vector, to be normalised
				const arma::Col<fltp>::fixed<3> v = rat::cmn::Extra::cross(dRs.col(j),dR);

				// norm of v
				const fltp lv = arma::as_scalar(rat::cmn::Extra::vec_norm(v));

				// cylinder ratio
				const fltp fi = std::min(std::max(q*q/(eps(j)*eps(j)),RAT_CONST(0.0)),RAT_CONST(1.0));

				// calculate magnetic field
				const arma::Col<fltp>::fixed<3> Hcon = v*(-fi*Is(j)*Ba/lv);

				// add field
				Hsum += Hcon;
			}

			// add to array
		  	H.col(i) = Hsum;
		// }	
		});

		// apply scaling
		H/=4*arma::Datum<fltp>::pi;

		// check H
		assert(H.n_rows==3); assert(H.n_cols==Rt.n_cols);

		// return magnetic field in [A/m] at each target point
		return H;
	}


	// van Lanen kernel more accurate 
	// version for calculating vector potential
	arma::Mat<fltp> Savart::calc_I2A_vl_dir(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &dRs, 
		const arma::Row<fltp> &Is, 
		const arma::Row<fltp> &eps, 
		const arma::Mat<fltp> &Rt, 
		const bool use_parallel,
		const cmn::ShLogPr&lg){

		// check input
		assert(Rs.n_rows==3); //assert(Ieff.n_rows==3);  
		assert(Rt.n_rows==3); assert(Rs.n_cols==Is.n_cols);

		// allocate output
		arma::Mat<fltp> A(3,Rt.n_cols,arma::fill::zeros);

		// walk over targets
		cmn::parfor(0,Rt.n_cols,use_parallel,[&](arma::uword i, int) { // second int is CPU number
			// check cancel flag
			if(lg->is_cancelled())return;

			// accumulator
			arma::Col<fltp>::fixed<3> Asum(arma::fill::zeros);

			// walk over sources
			for(arma::uword j=0;j<Rs.n_cols;j++){
				// check cancel flag
				if(lg->is_cancelled())return;

				// relative position
				const arma::Col<fltp>::fixed<3> dR = Rt.col(i) - Rs.col(j);

				// calculate coeffcicients
				const fltp a = arma::as_scalar(cmn::Extra::dot(dRs.col(j),dRs.col(j))); // squared length of the current element 
				const fltp b = arma::as_scalar(arma::abs(2.0*cmn::Extra::dot(dRs.col(j),dR)));
				const fltp c = arma::as_scalar(cmn::Extra::dot(dR,dR)); // squared distance

				// distance
				const fltp rho = std::sqrt(c);

				// third power of distance
				const fltp rho3 = c*rho;
				const fltp eps3 = eps(j)*eps(j)*eps(j);

				// calculate effective current
				const fltp sphere_ratio = std::min(std::max(rho3/eps3,0.0),1.0);

				// subexpression elimination in the compiler will hopefully eliminate all fltp calculations here
				const fltp k1 = (b+a)/(RAT_CONST(2.0)*std::sqrt(a))+std::sqrt(a/RAT_CONST(4.0) + b/RAT_CONST(2.0) + c); 
				const fltp k2 = (b-a)/(RAT_CONST(2.0)*std::sqrt(a))+std::sqrt(a/RAT_CONST(4.0) - b/RAT_CONST(2.0) + c); //with softening factor!!

				// calculate vector potential
				const fltp scale = (std::log(k1/k2)/std::sqrt(a))*sphere_ratio*Is(j);
				const arma::Col<fltp>::fixed<3> Acon = dRs.col(j)*scale;

				// add source contributions
				if(k1>RAT_CONST(1e-7) && k2>RAT_CONST(1e-7) && rho>RAT_CONST(1e-7))Asum += Acon;
			}

			// add to array
			A.col(i) = Asum;
		});

		// check A
		assert(A.n_rows==3); 
		assert(A.n_cols==Rt.n_cols);

		// apply scaling
		A*=RAT_CONST(1e-7);

		// return vector potential at each target point
		return A;
	}

}}