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
#include "brick8node.hh"

// common includes
#include "rat/common/extra.hh"
#include "rat/common/elements.hh"

// mlfmm headers
#include "chargedpolyhedron.hh"

// code specific to Raccoon
namespace rat{namespace fmm{

	// calculate field from hexahedron
	// the current enters in plane 0123 and exits in 4567
	// it is assumed that current is preserved
	// this means that the planes all carry the same current
	// but a different current density
	arma::Mat<fltp> Brick8Node::calc_magnetic_field_core(
		const arma::Mat<fltp>::fixed<3,4> &R1,
		const arma::Mat<fltp>::fixed<3,4> &R2,
		const arma::Mat<fltp> &Rt,
		const fltp current,
		const arma::uword num_planes){

		// check input
		assert(num_planes>0);
		assert(Rt.n_rows==3); 

		// quadrilaterl positions of the surfaces
		const arma::Row<fltp> t = arma::linspace<arma::Row<fltp> >(
			RAT_CONST(0.5)/num_planes, RAT_CONST(1.0) - RAT_CONST(0.5)/num_planes, num_planes);

		// allocate H
		arma::Mat<fltp> H(3, Rt.n_cols, arma::fill::zeros);

		// get overal direction of current
		const arma::Col<fltp>::fixed<3> Rc1 = arma::mean(R1,1);
		const arma::Col<fltp>::fixed<3> Rc2 = arma::mean(R2,1);
		const arma::Col<fltp>::fixed<3> L = cmn::Extra::normalize(Rc2 - Rc1);

		// distance between nodes
		const arma::Row<fltp> ell_node = cmn::Extra::vec_norm(R2 - R1)/static_cast<fltp>(num_planes);

		// walk over surfaces
		for(arma::uword i=0;i<num_planes;i++){

			// extract plane
			const arma::Mat<fltp>::fixed<3,4> Rq = t(i)*R1 + (RAT_CONST(1.0)-t(i))*R2;

			// plane area
			const fltp area = cmn::Quadrilateral::calc_area(Rq);

			// scaling due to angle mismatch
			const fltp current_density = current/area;

			// calculate magnetic field using planes with linearly 
			// varying charge distribution (work in progress)
			H += cmn::Extra::cross(arma::repmat(L,1,Rt.n_cols), ChargedPolyhedron::calc_magnetic_field(Rq,current_density*ell_node,Rt));
		}

		// add contribution of this surface
		return H/(4*arma::Datum<fltp>::pi);
	}

	// calculate field from hexahedron
	arma::Mat<fltp> Brick8Node::calc_magnetic_field_richardson_adaptive(
		const arma::Mat<fltp>::fixed<3,4> &R1,
		const arma::Mat<fltp>::fixed<3,4> &R2,
		const arma::Mat<fltp> &Rt,
		const fltp current,
		const arma::uword num_planes_min,
		const fltp reltol,
		const fltp abstol,
		const arma::uword max_planes){

		// check input
		assert(num_planes_min>0);
		assert(max_planes>num_planes_min);

		// Adaptive refinement using First-Order Richardson Extrapolation (Standard Version)
		arma::Mat<fltp> H, H1, H2;
		H1 = calc_magnetic_field_core(R1, R2, Rt, current, num_planes_min);

		for (arma::uword num_planes=2*num_planes_min;num_planes<=max_planes;num_planes*=2) {
			// Compute for the new number of planes
			H2 = calc_magnetic_field_core(R1, R2, Rt, current, num_planes);

			// First-order Richardson extrapolation
			H = H2 + (H2 - H1)/3;  // Simple extrapolation with assumed O(h^2) error

			// Compute errors comparing H (extrapolated) and H2 (latest direct calculation)
			const fltp relative_error = arma::norm(H - H2, "inf") / std::max(arma::norm(H, "inf"), abstol);
			const fltp absolute_error = arma::norm(H - H2, "inf");

			// Check convergence criteria
			if (relative_error < reltol || absolute_error < abstol) break;

			// Prepare for next iteration
			H1 = std::move(H2);  // Move H2 to H1 to avoid redundant computation
		}

		// Return the final extrapolated value
		return H;
	}

	// calculate field from hexahedron
	arma::Mat<fltp> Brick8Node::calc_magnetic_field_richardson(
		const arma::Mat<fltp>::fixed<3,4> &R1,
		const arma::Mat<fltp>::fixed<3,4> &R2,
		const arma::Mat<fltp> &Rt,
		const fltp current,
		const arma::uword num_planes){

		// adaptive mode
		if(num_planes==0)return calc_magnetic_field_richardson_adaptive(R1,R2,Rt,current,1llu,RAT_CONST(5e-3),RAT_CONST(1e-4),128llu);

		// calculate for number of planes and double number of planes
		const arma::Mat<fltp> H1 = calc_magnetic_field_core(R1,R2,Rt,current,num_planes);
		const arma::Mat<fltp> H2 = calc_magnetic_field_core(R1,R2,Rt,current,2*num_planes);

		// richardson extrapolation
		const arma::Mat<fltp> H = H2 + (H2 - H1)/3;

		// return field
		return H;
	}

	// calculate field from hexahedron with richardson extrapolation
	// the current enters in plane 0123 and exits in 4567
	// also deals with the singularity by splitting the element
	arma::Mat<fltp> Brick8Node::calc_magnetic_field_singularity(
		const arma::Mat<fltp>::fixed<3,8> &Rn, 
		const arma::Mat<fltp> &Rt, 
		const fltp current,
		const arma::uword num_planes){

		// get quadrilateral coordinates (iteratively)
		const fltp tolerance = RAT_CONST(1e-4);
		const arma::Mat<fltp> Rqt = cmn::Hexahedron::cart2quad(Rn, Rt, tolerance);
		arma::Row<fltp> xi = Rqt.row(2); xi.cols(arma::find_nonfinite(xi)).fill(1.0); // treat non-finite values as outside
		const arma::Col<arma::uword> idx_inside = arma::find(xi>(-1.0+tolerance) && xi<(1.0-tolerance));
		const arma::Col<arma::uword> idx_outside = arma::find(xi<=(-1.0+tolerance) || xi>=(1.0-tolerance));

		// get two planes
		const arma::Mat<fltp>::fixed<3,4> R1 = Rn.cols(0,3);
		const arma::Mat<fltp>::fixed<3,4> R2 = Rn.cols(4,7);

		// quadrilateral
		arma::Mat<fltp>::fixed<2,4> Q{
			-RAT_CONST(1.0), -RAT_CONST(1.0), +RAT_CONST(1.0), -RAT_CONST(1.0), 
			+RAT_CONST(1.0), +RAT_CONST(1.0), -RAT_CONST(1.0), +RAT_CONST(1.0)}; 

		// allocate output field
		arma::Mat<fltp> H(3,Rt.n_cols);

		// no singularity
		if(!idx_outside.empty())H.cols(idx_outside) = calc_magnetic_field_richardson(R1,R2,Rt.cols(idx_outside),current,num_planes);

		// target points with singularity
		if(!idx_inside.empty()){
			// walk over targets insinde and treat each of them separately
			for(arma::uword i=0;i<idx_inside.n_elem;i++){
				// get index
				const arma::uword idx = idx_inside(i);

				// setup planes around singularity
				const arma::Mat<fltp>::fixed<3,4> Ri = cmn::Hexahedron::quad2cart(
					Rn,arma::join_vert(Q,arma::Row<fltp>::fixed<4>(arma::fill::value(xi(idx)))));

				// check
				assert(arma::as_scalar(cmn::Extra::dot(cmn::Extra::normalize(cmn::Extra::cross(Ri.col(2)-Ri.col(1),Ri.col(1)-Ri.col(0))),Rt.col(idx)-Ri.col(0)))<1e-6);

				// add contributions of two regions to field
				H.col(idx) = 
					calc_magnetic_field_richardson(R1, Ri, Rt.col(idx),current,num_planes) + 
					calc_magnetic_field_richardson(Ri, R2, Rt.col(idx),current,num_planes);
			}
		}

		// return field
		return H;
	}

	// full calculation of an 8 node brick including check for near/far field
	// the current enters in plane 0123 and exits in 4567
	arma::Mat<fltp> Brick8Node::calc_magnetic_field(
		const arma::Mat<fltp>::fixed<3,8> &Rn, 
		const arma::Mat<fltp> &Rt, 
		const fltp current,
		const arma::uword num_planes,
		const fltp near_far_factor){

		// check input
		assert(near_far_factor>=1.0);

		// find target points far away
		const arma::Col<fltp>::fixed<3> Rc = arma::mean(Rn,1);
		const arma::Row<fltp>::fixed<8> rhon = cmn::Extra::vec_norm(Rn.each_col() - Rc);
		const arma::Row<fltp> rhot = cmn::Extra::vec_norm(Rt.each_col() - Rc);
		const fltp rhon_max = near_far_factor*rhon.max();
		const arma::Col<arma::uword> idx_far = arma::find(rhot>rhon_max);
		const arma::Col<arma::uword> idx_near = arma::find(rhot<=rhon_max);

		// allocate output and calculate field 
		arma::Mat<fltp> H(3,Rt.n_cols);
		if(!idx_near.empty())H.cols(idx_near) = calc_magnetic_field_singularity(Rn,Rt.cols(idx_near),current,num_planes);
		if(!idx_far.empty())H.cols(idx_far) = calc_magnetic_field_richardson(Rn.cols(0,3),Rn.cols(4,7),Rt.cols(idx_far),current,num_planes);

		// return the field
		return H;
	}



	// calculate field from hexahedron
	// the current enters in plane 0123 and exits in 4567
	// it is assumed that current is preserved
	// this means that the planes all carry the same current
	// but a different current density
	arma::Mat<fltp> Brick8Node::calc_vector_potential_core(
		const arma::Mat<fltp>::fixed<3,4> &R1,
		const arma::Mat<fltp>::fixed<3,4> &R2,
		const arma::Mat<fltp> &Rt,
		const fltp current,
		const arma::uword num_planes){

		// check input
		assert(num_planes>0);
		assert(Rt.n_rows==3);

		// quadrilaterl positions of the surfaces
		const arma::Row<fltp> t = arma::linspace<arma::Row<fltp> >(
			RAT_CONST(0.5)/num_planes, RAT_CONST(1.0) - RAT_CONST(0.5)/num_planes, num_planes);

		// allocate H
		arma::Mat<fltp> A(3, Rt.n_cols, arma::fill::zeros);

		// get overal direction of current
		const arma::Col<fltp>::fixed<3> Rc1 = arma::mean(R1,1);
		const arma::Col<fltp>::fixed<3> Rc2 = arma::mean(R2,1);
		const arma::Col<fltp>::fixed<3> L = cmn::Extra::normalize(Rc2 - Rc1);

		// distance between nodes
		const arma::Row<fltp> ell_node = cmn::Extra::vec_norm(R2 - R1)/static_cast<fltp>(num_planes);

		// walk over surfaces
		for(arma::uword i=0;i<num_planes;i++){

			// extract plane
			const arma::Mat<fltp>::fixed<3,4> Rq = t(i)*R1 + (RAT_CONST(1.0)-t(i))*R2;

			// plane area
			const fltp area = cmn::Quadrilateral::calc_area(Rq);

			// scaling due to angle mismatch
			const fltp current_density = current/area;

			// calculate magnetic field using planes with linearly 
			// varying charge distribution (work in progress)
			A += arma::repmat(L,1,Rt.n_cols).eval().each_row()%ChargedPolyhedron::calc_scalar_potential(Rq,current_density*ell_node,Rt);
		}

		// add contribution of this surface
		return A*RAT_CONST(1e-7);
	}

	// calculate field from hexahedron
	arma::Mat<fltp> Brick8Node::calc_vector_potential_richardson_adaptive(
		const arma::Mat<fltp>::fixed<3,4> &R1,
		const arma::Mat<fltp>::fixed<3,4> &R2,
		const arma::Mat<fltp> &Rt,
		const fltp current,
		const arma::uword num_planes_min,
		const fltp reltol,
		const fltp abstol,
		const arma::uword max_planes){

		// check input
		assert(num_planes_min>0);
		assert(max_planes>num_planes_min);

		// Adaptive refinement using First-Order Richardson Extrapolation (Standard Version)
		arma::Mat<fltp> A, A1, A2;
		A1 = calc_vector_potential_core(R1, R2, Rt, current, num_planes_min);

		for (arma::uword num_planes=2*num_planes_min;num_planes<=max_planes;num_planes*=2) {
			// Compute for the new number of planes
			A2 = calc_vector_potential_core(R1, R2, Rt, current, num_planes);

			// First-order Richardson extrapolation
			A = A2 + (A2 - A1)/3;  // Simple extrapolation with assumed O(h^2) error

			// Compute errors comparing A (extrapolated) and A2 (latest direct calculation)
			const fltp relative_error = arma::norm(A - A2, "inf") / std::max(arma::norm(A, "inf"), abstol);
			const fltp absolute_error = arma::norm(A - A2, "inf");

			// Check convergence criteria
			if (relative_error < reltol || absolute_error < abstol) break;

			// Prepare for next iteration
			A1 = std::move(A2);  // Move A2 to A1 to avoid redundant computation
		}

		// Return the final extrapolated value
		return A;
	}

	// calculate vector potential from hexahedron
	// using richardson extrapolation
	arma::Mat<fltp> Brick8Node::calc_vector_potential_richardson(
		const arma::Mat<fltp>::fixed<3,4> &R1,
		const arma::Mat<fltp>::fixed<3,4> &R2,
		const arma::Mat<fltp> &Rt,
		const fltp current,
		const arma::uword num_planes){

		// adaptive mode
		if(num_planes==0)return calc_vector_potential_richardson_adaptive(R1,R2,Rt,current,1llu,RAT_CONST(5e-3),RAT_CONST(1e-12),128llu);

		// calculate for number of planes and double number of planes
		const arma::Mat<fltp> A1 = calc_vector_potential_core(R1,R2,Rt,current,num_planes);
		const arma::Mat<fltp> A2 = calc_vector_potential_core(R1,R2,Rt,current,2*num_planes);

		// richardson extrapolation
		const arma::Mat<fltp> A = A2 + (A2 - A1)/3;

		// return field
		return A;
	}


	// calculate vector potential from hexahedron with richardson extrapolation
	// the current enters in plane 0123 and exits in 4567
	// also deals with the singularity by splitting the element
	arma::Mat<fltp> Brick8Node::calc_vector_potential_singularity(
		const arma::Mat<fltp>::fixed<3,8> &Rn, 
		const arma::Mat<fltp> &Rt, 
		const fltp current,
		const arma::uword num_planes){

		// get quadrilateral coordinates (iteratively)
		const fltp tolerance = RAT_CONST(1e-4);
		const arma::Mat<fltp> Rqt = cmn::Hexahedron::cart2quad(Rn, Rt, tolerance);
		arma::Row<fltp> xi = Rqt.row(2); xi.cols(arma::find_nonfinite(xi)).fill(1.0); // treat non-finite values as outside
		const arma::Col<arma::uword> idx_inside = arma::find(xi>(-1.0+tolerance) && xi<(1.0-tolerance));
		const arma::Col<arma::uword> idx_outside = arma::find(xi<=(-1.0+tolerance) || xi>=(1.0-tolerance));

		// get two planes
		const arma::Mat<fltp>::fixed<3,4> R1 = Rn.cols(0,3);
		const arma::Mat<fltp>::fixed<3,4> R2 = Rn.cols(4,7);

		// quadrilateral
		arma::Mat<fltp>::fixed<2,4> Q{
			-RAT_CONST(1.0), -RAT_CONST(1.0), +RAT_CONST(1.0), -RAT_CONST(1.0), 
			+RAT_CONST(1.0), +RAT_CONST(1.0), -RAT_CONST(1.0), +RAT_CONST(1.0)}; 

		// allocate output field
		arma::Mat<fltp> A(3,Rt.n_cols);

		// no singularity
		if(!idx_outside.empty())A.cols(idx_outside) = calc_vector_potential_richardson(R1,R2,Rt.cols(idx_outside),current,num_planes);

		// target points with singularity
		if(!idx_inside.empty()){
			// walk over targets inside and treat each of them separately
			for(arma::uword i=0;i<idx_inside.n_elem;i++){
				// get index
				const arma::uword idx = idx_inside(i);

				// setup planes around singularity
				const arma::Mat<fltp>::fixed<3,4> Ri = cmn::Hexahedron::quad2cart(Rn,arma::join_vert(Q,arma::Row<fltp>::fixed<4>(arma::fill::value(xi(idx)))));

				// add contributions of two regions to field
				A.col(idx) = calc_vector_potential_richardson(R1, Ri, Rt.col(idx),current,num_planes) + 
					calc_vector_potential_richardson(Ri, R2, Rt.col(idx),current,num_planes);
			}
		}

		// return field
		return A;
	}

	// full calculation of an 8 node brick including check for near/far field
	// the current enters in plane 0123 and exits in 4567
	arma::Mat<fltp> Brick8Node::calc_vector_potential(
		const arma::Mat<fltp>::fixed<3,8> &Rn, 
		const arma::Mat<fltp> &Rt, 
		const fltp current,
		const arma::uword num_planes,
		const fltp near_far_factor){

		// find target points far away
		const arma::Col<fltp>::fixed<3> Rc = arma::mean(Rn,1);
		const arma::Row<fltp>::fixed<8> rhon = cmn::Extra::vec_norm(Rn.each_col() - Rc);
		const arma::Row<fltp> rhot = cmn::Extra::vec_norm(Rt.each_col() - Rc);
		const fltp rhon_max = near_far_factor*rhon.max();
		const arma::Col<arma::uword> idx_far = arma::find(rhot>rhon_max);
		const arma::Col<arma::uword> idx_near = arma::find(rhot<=rhon_max);

		// allocate output and calculate field 
		arma::Mat<fltp> A(3,Rt.n_cols);
		if(!idx_near.empty())A.cols(idx_near) = calc_vector_potential_singularity(Rn,Rt.cols(idx_near),current,num_planes);
		if(!idx_far.empty())A.cols(idx_far) = calc_vector_potential_richardson(Rn.cols(0,3),Rn.cols(4,7),Rt.cols(idx_far),current,num_planes);

		// return the field
		return A;
	}

}}