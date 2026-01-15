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
#include "chargedpolyhedron.hh"

// common includes
#include "rat/common/extra.hh"
#include "rat/common/elements.hh"

// code specific to Raccoon
namespace rat{namespace fmm{

	// analytical integral over the right triangle
	// note there is a factor charges/4*pi*mu0 not accounted for here
	arma::Row<fltp> ChargedPolyhedron::right_triangle_scalar_potential(
		const arma::Row<fltp>& a, const arma::Row<fltp>& b, const arma::Row<fltp>& c){
		
		// check input
		assert(a.n_elem == b.n_elem); assert(a.n_elem == c.n_elem);
		assert(a.is_finite()); assert(b.is_finite()); assert(c.is_finite());

		// pythagoras
		const arma::Row<fltp> Dabc = arma::sqrt(a%a + b%b + c%c);
		// const arma::Row<fltp> Dab = arma::hypot(a,b);
		const arma::Row<fltp> Dac = arma::hypot(a,c);

		// check
		assert(Dabc.is_finite()); assert(Dac.is_finite());

		// calculate potential
		arma::Row<fltp> phi = (a/2)%arma::log((Dabc+b)/(Dabc-b)) - 
			arma::abs(c)%arma::atan((a%b)/(Dac%Dac + arma::abs(c)%Dabc));

		// check for sliver triangles
		phi.cols(arma::find(arma::abs(a)<RAT_CONST(1e-9) || arma::abs(b)<RAT_CONST(1e-9) || arma::abs(Dabc-b)<RAT_CONST(1e-9))).fill(RAT_CONST(0.0));

		// check output
		assert(phi.is_finite());

		// return potential
		return phi;
	}

	// analytical integral over the right triangle
	// note there is a factor charges/4*pi*mu0 not accounted for here
	fltp ChargedPolyhedron::right_triangle_scalar_potential(
		const fltp a, const fltp b, const fltp c){
		
		// check input
		assert(std::isfinite(a)); assert(std::isfinite(b)); assert(std::isfinite(c));

		// sliver element
		if(a==0 || b==0)return RAT_CONST(0.0);

		// pythagoras
		const fltp Dabc = std::sqrt(a*a + b*b + c*c);
		// const fltp Dab = std::hypot(a,b);
		const fltp Dac = std::hypot(a,c);

		// check special condition
		if(Dabc-b==0)return RAT_CONST(0.0);

		// calculate potential
		fltp phi = (a/2)*std::log((Dabc+b)/(Dabc-b)) - 
			std::abs(c)*std::atan((a*b)/(Dac*Dac + std::abs(c)*Dabc));

		// check output
		assert(std::isfinite(phi));

		// return potential
		return phi;
	}

	// analytical integral over the right triangle
	// note there is a factor charges/4*pi*mu0 not accounted for here
	// this computes the potential for both b1 and b2, while a and c are the same for both
	std::pair<arma::Row<fltp>, arma::Row<fltp> > ChargedPolyhedron::right_triangle_scalar_potential(
		const arma::Row<fltp>&a, const arma::Row<fltp>&b1, 
		const arma::Row<fltp>&b2, const arma::Row<fltp>&c){

		// precompute common terms
		const arma::Row<fltp> asq = arma::square(a);
		const arma::Row<fltp> csq = arma::square(c);
		const arma::Row<fltp> Dacsq = asq + csq;
		// const arma::Row<fltp> Dac = arma::sqrt(Dacsq);
		const arma::Row<fltp> absa = arma::abs(a);
		const arma::Row<fltp> absc = arma::abs(c);
		const arma::Row<fltp> halfa = a/2;

		// compute Dabc for both b1 and b2
		const arma::Row<fltp> Dabc1 = arma::sqrt(asq + arma::square(b1) + csq);
		const arma::Row<fltp> Dabc2 = arma::sqrt(asq + arma::square(b2) + csq);

		// calculate phi1
		arma::Row<fltp> phi1 = halfa%arma::log((Dabc1+b1)/(Dabc1-b1)) - 
			absc%arma::atan((a%b1)/(Dacsq + absc%Dabc1));

		// calculate phi2
		arma::Row<fltp> phi2 = halfa%arma::log((Dabc2+b2)/(Dabc2-b2)) - 
			absc%arma::atan((a%b2)/(Dacsq + absc%Dabc2));

		// handle sliver triangles and other edge cases
		phi1.cols(arma::find(absa<RAT_CONST(1e-9) || arma::abs(b1)<RAT_CONST(1e-9) || arma::abs(Dabc1-b1)<RAT_CONST(1e-9))).fill(RAT_CONST(0.0));
		phi2.cols(arma::find(absa<RAT_CONST(1e-9) || arma::abs(b2)<RAT_CONST(1e-9) || arma::abs(Dabc2-b2)<RAT_CONST(1e-9))).fill(RAT_CONST(0.0));

		// return the potential pair
		return {phi1, phi2};
	}

	// analytical integral of magnetic field over the right triangle
	// note there is a factor charges/4*pi*mu0 not accounted for here
	arma::Mat<fltp> ChargedPolyhedron::right_triangle_magnetic_field(
		const arma::Row<fltp>&a, const arma::Row<fltp>&b, const arma::Row<fltp>&c){
		// check input
		assert(a.n_elem == b.n_elem); assert(a.n_elem == c.n_elem);
		assert(a.is_finite()); assert(b.is_finite()); assert(c.is_finite());

		// pythagoras
		const arma::Row<fltp> asq = arma::square(a);
		const arma::Row<fltp> bsq = arma::square(b);
		const arma::Row<fltp> csq = arma::square(c);
		const arma::Row<fltp> Dabc = arma::sqrt(asq + bsq + csq);
		const arma::Row<fltp> Dab = arma::sqrt(asq + bsq);
		const arma::Row<fltp> Dac = arma::sqrt(asq + csq);

		// check output
		assert(Dabc.is_finite()); assert(Dab.is_finite()); assert(Dac.is_finite());

		// compute non-divergent terms
		const arma::Row<fltp> factor = arma::log((Dabc+Dab)/(Dabc-Dab)); // this diverges when c = 0
		arma::Row<fltp> Hx1 = (-b/(2*Dab))%factor;
		arma::Row<fltp> Hy1 = (a/(2*Dab))%factor;
		arma::Row<fltp> Hy2 = -arma::log((Dac+a)/(Dac-a))/2; // this diverges when c = 0
		arma::Row<fltp> Hx2 = arma::atanh(b/Dabc);
		arma::Row<fltp> Hz1 = arma::atan((a%Dabc)/(b%c));
		arma::Row<fltp> Hz2 = -arma::sign(c)%arma::atan(a/b);

		// corrections to field from paper
		// Note: The normal component of the magnetic field is discontinuous in crossing the surface. It is reasonable to take Hz = 0
		// for [11]. It must also be pointed out that the field components Hx and Hy are undefined for (the plane ofthe polygon). 
		// The divergent terms cancel analytically when all the right triangles of a polygon are combined together. 
		// It was an-alytically verified on the case of the rectangle and numerically for other polygonal shapes (the 
		// divergent terms are simply not calculated). 
		const fltp tol = RAT_CONST(1e-9);
		
		// Identify indices where a or b are "sliver" elements.
		const arma::Col<arma::uword> idx_sliver = arma::find(arma::abs(a) < tol || arma::abs(b) < tol);
		// For these indices, set all fields to zero.
		Hz1(idx_sliver).zeros();
		Hz2(idx_sliver).zeros();
		Hx2(idx_sliver).zeros();
		Hx1(idx_sliver).zeros();
		Hy1(idx_sliver).zeros();
		Hy2(idx_sliver).zeros();

		// Identify indices where c is nearly zero (in-plane), but note that Hx2 should remain intact.
		const arma::Col<arma::uword> idx_in_plane = arma::find(arma::abs(c) < tol);
		Hz1(idx_in_plane).zeros();
		Hz2(idx_in_plane).zeros();
		Hx1(idx_in_plane).zeros();
		Hy1(idx_in_plane).zeros();
		Hy2(idx_in_plane).zeros();

		// special case
		const arma::Col<arma::uword> idx_log_fail = arma::find(arma::abs(a) < tol && arma::abs(c) < tol);
		Hx1(idx_log_fail).zeros();

		// check output
		assert(Hx1.is_finite()); assert(Hx2.is_finite()); assert(Hy1.is_finite());
		assert(Hy2.is_finite()); assert(Hz1.is_finite()); assert(Hz2.is_finite());

		// combine
		const arma::Mat<fltp> H = arma::join_vert(Hx1 + Hx2, Hy1 + Hy2, Hz1 + Hz2);

		// check output
		assert(H.is_finite());

		// return potential
		return H;
	}

	// // we are using this perfect line segment intersection algorithm from:
	// // https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
	// // check if points are listed in counter clockwise order
	// fltp ChargedPolyhedron::ccw(const arma::Col<fltp>::fixed<2>&A, const arma::Col<fltp>::fixed<2>&B, const arma::Col<fltp>::fixed<2>&C){
	// 	return (C(1)-A(1))*(B(0)-A(0)) > (B(1)-A(1))*(C(0)-A(0));
	// }

	// // check if two line segments intersect, the first segment is between vertices A and B and the second segment betwen C and D
	// bool ChargedPolyhedron::intersect(const arma::Col<fltp>::fixed<2>&A, const arma::Col<fltp>::fixed<2>&B, const arma::Col<fltp>::fixed<2>&C, const arma::Col<fltp>::fixed<2>&D){
	// 	return ccw(A,C,D) != ccw(B,C,D) && ccw(A,B,C) != ccw(A,B,D);
	// }

	// check if two polygons intersect
	// this is using Separating Axis Theorem (SAT)
	// note that these polygons must be concave
	bool ChargedPolyhedron::intersect_polygons(
		const arma::Mat<fltp>&Rn1, 
		const arma::Mat<fltp>&Rn2, 
		const fltp tol){

		// axes (perpendicular to surface elements of both shapes)
		arma::Mat<fltp> axes = arma::flipud(arma::join_horiz(
			arma::diff(Rn2,1,1), Rn2.col(0) - Rn2.col(Rn2.n_cols-1),
			arma::diff(Rn1,1,1), Rn1.col(0) - Rn1.col(Rn1.n_cols-1))); 
		axes.row(0)*=-1;

		// walk over axes
		for(arma::uword i=0;i<axes.n_cols;i++){
			// project shape into this axis using dot product
			const arma::Row<fltp> projection1 = arma::sum(Rn1.each_col()%axes.col(i));
			const arma::Row<fltp> projection2 = arma::sum(Rn2.each_col()%axes.col(i));

			// if there is no overlap on this axis
			if(projection1.max()-tol<projection2.min() || projection2.max()-tol<projection1.min())return false;
		}

		// there was always overlap
		return true; 
	}


	// // calculate the scalar potential of a polyhedron defined by vertices Rn at target points Rt
	// // the polyhedron must be planar
	// arma::Row<fltp> ChargedPolyhedron::polyhedron_scalar_potential(const arma::Mat<fltp>&Rn, const arma::Mat<fltp>&Rt){
	// 	// check that it is at least a triangle
	// 	assert(Rn.n_rows==3); assert(Rn.n_cols>=3); assert(Rt.n_rows==3);
	// 	assert(Rn.is_finite()); assert(Rt.is_finite());
 
	// 	// vector spanning first triangle
	// 	const arma::Col<fltp>::fixed<3> V1 = Rn.col(1) - Rn.col(0);
	// 	const arma::Col<fltp>::fixed<3> V2 = Rn.col(2) - Rn.col(0);

	// 	// create coordinate system where e1 and e2 are in plane with the polyhedron
	// 	// and e3 is normal to the polyhedron
	// 	const arma::Col<fltp>::fixed<3> e3 = cmn::Extra::normalize(cmn::Extra::cross(V1, V2));
	// 	const arma::Col<fltp>::fixed<3> e1 = cmn::Extra::normalize(V1);
	// 	const arma::Col<fltp>::fixed<3> e2 = cmn::Extra::cross(e1,e3);
	// 	assert(std::abs(arma::as_scalar(cmn::Extra::vec_norm(e2))-1.0)<1e-9);

	// 	// calculate Rn and T in coordinate system in 2D
	// 	const arma::Mat<fltp> Rn_trans = arma::join_vert(arma::sum(Rn.each_col()%e1), arma::sum(Rn.each_col()%e2));
	// 	const arma::Mat<fltp> Rt_trans = arma::join_vert(arma::sum(Rt.each_col()%e1), arma::sum(Rt.each_col()%e2));

	// 	// allocate output potential at each target point
	// 	arma::Row<fltp> phi(Rt.n_cols, arma::fill::zeros);

	// 	// walk over target points
	// 	for(arma::uword i=0;i<Rt_trans.n_cols;i++){
	// 		// calculate elevation of the target point with resepect to the polyhedron plane
	// 		const fltp c12 = arma::as_scalar(cmn::Extra::dot(Rt.col(i) - Rn.col(0), e3));

	// 		// get target point projected onto plane
	// 		const arma::Col<fltp>::fixed<2> O = Rt_trans.col(i);

	// 		// walk over edges
	// 		for(arma::uword j=0;j<Rn_trans.n_cols;j++){
	// 			// get indices of vertices
	// 			const arma::uword idx1 = j, idx2 = (j+1)%Rn_trans.n_cols;

	// 			// get vertices for this edge
	// 			const arma::Col<fltp>::fixed<2> Rn1 = Rn_trans.col(idx1);
	// 			const arma::Col<fltp>::fixed<2> Rn2 = Rn_trans.col(idx2);

	// 			// perpendicular projection
	// 			const arma::Col<fltp>::fixed<2> dRn12 = cmn::Extra::normalize(Rn2 - Rn1);
	// 			const fltp ell12 = arma::as_scalar(cmn::Extra::vec_norm(Rn2 - Rn1));
	// 			const fltp t = arma::as_scalar(cmn::Extra::dot(dRn12, O-Rn1));
	// 			const arma::Col<fltp>::fixed<2> Rpp = Rn1 + t*dRn12;

	// 			// check if Rpp and O coinside if so triangles have area zero and we can just continue
	// 			if(arma::as_scalar(cmn::Extra::vec_norm(Rpp - O))<1e-9)continue;

	// 			// this now gives two right triangles [Rn1,O,Rpp] and [Rn2,O,Rpp]
	// 			const fltp b1 = arma::as_scalar(cmn::Extra::vec_norm(Rpp - Rn1)); // distance right angle and first corner
	// 			const fltp b2 = arma::as_scalar(cmn::Extra::vec_norm(Rpp - Rn2)); // distance right angle and second corner
	// 			const fltp a12 = arma::as_scalar(cmn::Extra::vec_norm(O - Rpp)); // distance between O and right angle

	// 			// calculate their fields
	// 			const fltp phi1 = right_triangle_scalar_potential(a12,b1,c12);
	// 			const fltp phi2 = right_triangle_scalar_potential(a12,b2,c12);
	// 			assert(phi1>=-1e-14); assert(phi2>=-1e-14);

	// 			// check the projection interval
	// 			const fltp phi12 = t<RAT_CONST(0.0) ? phi2 - phi1 : (t<ell12 ? phi1 + phi2 : phi1 - phi2);

	// 			// check for intersection with the triangle to the polyhedron
	// 			// [Rpp,Rn1] and [Rpp,Rn2] will not intersect and neither will [O,Rpp]
	// 			// so we only need to check for [O,Rn1] and [O,Rn2} for the first
	// 			// and second triangle respectively
	// 			const bool intersect = intersect_polygons(Rn_trans, arma::join_horiz(O,Rn1,Rn2));

	// 			// add signed contributions
	// 			phi(i) += intersect ? phi12 : -phi12;
	// 		}
	// 	}

	// 	// return potential
	// 	return phi;
	// } 

	// calculate the scalar potential of a polyhedron defined by vertices Rn at target points Rt
	// the polyhedron must be planar
	arma::Row<fltp> ChargedPolyhedron::calc_scalar_potential(
		const arma::Mat<fltp>&Rn, const arma::Mat<fltp>&Rt){

		// check that it is at least a triangle
		assert(Rn.n_rows==3); assert(Rn.n_cols>=3); assert(Rt.n_rows==3);
		assert(Rn.is_finite()); assert(Rt.is_finite());

		// vector spanning first triangle
		const arma::Col<fltp>::fixed<3> V1 = Rn.col(1) - Rn.col(0);
		const arma::Col<fltp>::fixed<3> V2 = Rn.col(2) - Rn.col(0);

		// create coordinate system where e1 and e2 are in plane with the polyhedron
		// and e3 is normal to the polyhedron
		const arma::Col<fltp>::fixed<3> e3 = cmn::Extra::normalize(cmn::Extra::cross(V1, V2));
		const arma::Col<fltp>::fixed<3> e1 = cmn::Extra::normalize(V1);
		const arma::Col<fltp>::fixed<3> e2 = cmn::Extra::cross(e1,e3);
		assert(std::abs(arma::as_scalar(cmn::Extra::vec_norm(e2))-1.0)<RAT_CONST(1e-9));

		// calculate Rn and T in coordinate system in 2D
		const arma::Mat<fltp>::fixed<3,2> e12 = arma::join_horiz(e1,e2);
		const arma::Mat<fltp> Rn_trans = e12.t()*Rn;
		const arma::Mat<fltp> Rt_trans = e12.t()*Rt;
		assert(Rn_trans.is_finite());
		assert(Rt_trans.is_finite());

		// allocate output potential at each target point
		arma::Row<fltp> phi(Rt.n_cols, arma::fill::zeros);

		// calculate elevation of the target point with resepect to the polyhedron plane
		// const arma::Row<fltp> c12 = arma::sum((Rt.each_col() - Rn.col(0)).eval().each_col()%e3);
		const arma::Row<fltp> c12 = e3.t()*(Rt.each_col() - Rn.col(0));

		// walk over edges
		for(arma::uword j=0;j<Rn_trans.n_cols;j++){
			// get indices of vertices
			const arma::uword idx1 = j, idx2 = (j+1)%Rn_trans.n_cols;

			// get vertices for this edge
			const arma::Col<fltp>::fixed<2> Rn1 = Rn_trans.col(idx1), Rn2 = Rn_trans.col(idx2);

			// check if vertices are different
			assert(arma::all(cmn::Extra::vec_norm(Rn2-Rn1)>1e-14));

			// Compute norm and normalized vector in one step
			// const arma::Col<fltp>::fixed<2> dRn12 = cmn::Extra::normalize(Rn2 - Rn1);
			// const fltp ell12 = arma::as_scalar(cmn::Extra::vec_norm(Rn2 - Rn1));
			const arma::Col<fltp>::fixed<2> Rn12 = Rn2 - Rn1;
			const fltp ell12 = arma::norm(Rn12, 2);
			const arma::Col<fltp>::fixed<2> dRn12 = Rn12/ell12;

			// compute projection distances (dot product of Rt_trans with dRn12)
			// const arma::Row<fltp> t = arma::sum((Rt_trans.each_col()-Rn1).eval().each_col()%dRn12);
			arma::Row<fltp> t = dRn12.t()*(Rt_trans.each_col() - Rn1);

			// compute perpendicular projections
			// const arma::Mat<fltp> Rpp = (arma::repmat(t,2,1).eval().each_col()%dRn12).eval().each_col() + Rn1;
			const arma::Mat<fltp> Rpp = (arma::repmat(t,2,1).eval().each_col()%dRn12).eval().each_col() + Rn1;

			// this now gives two right triangles [Rn1,O,Rpp] and [Rn2,O,Rpp]
			const arma::Row<fltp> b1 = cmn::Extra::vec_norm(Rpp.each_col() - Rn1); // distance right angle and first corner
			const arma::Row<fltp> b2 = cmn::Extra::vec_norm(Rpp.each_col() - Rn2); // distance right angle and second corner
			const arma::Row<fltp> a12 = cmn::Extra::vec_norm(Rt_trans - Rpp); // distance between O and right angle
			assert(b1.is_finite()); assert(b2.is_finite()); assert(a12.is_finite());

			// calculate their fields
			const std::pair<arma::Row<fltp>, arma::Row<fltp> > phi12pair = right_triangle_scalar_potential(a12,b1,b2,c12);
			const arma::Row<fltp>& phi1 = phi12pair.first, phi2 = phi12pair.second; // take references

			// check calculated potentials
			assert(phi1.is_finite()); assert(phi2.is_finite());
			assert(arma::all(phi1>=-RAT_CONST(1e-14))); 
			assert(arma::all(phi2>=-RAT_CONST(1e-14)));

			// Precompute possible phi12 values for all points
			// const arma::Row<fltp> phi12 = 
			// 	(t<RAT_CONST(0.0))%(phi2 - phi1) + 
			// 	(t>=RAT_CONST(0.0) && t<ell12)%(phi1 + phi2) + 
			// 	(t>=ell12)%(phi1 - phi2);
			
			// walk over target points
			arma::Row<fltp> phi12(Rt_trans.n_cols);
			for(arma::uword i=0;i<Rt_trans.n_cols;i++){
				if(t(i)<RAT_CONST(0.0))phi12(i) = phi2(i) - phi1(i);
				else if(t(i)<ell12)phi12(i) = phi1(i) + phi2(i);
				else phi12(i) = phi1(i) - phi2(i);
			}

			// walk over target points
			for(arma::uword i=0;i<Rt_trans.n_cols;i++){
				// check the projection interval
				// const fltp phi12 = t(i)<RAT_CONST(0.0) ? phi2(i) - phi1(i) : (t(i)<ell12 ? phi1(i) + phi2(i) : phi1(i) - phi2(i));

				// check for intersection with the triangle to the polyhedron
				// [Rpp,Rn1] and [Rpp,Rn2] will not intersect and neither will [O,Rpp]
				// so we only need to check for [O,Rn1] and [O,Rn2} for the first
				// and second triangle respectively
				const bool intersect = intersect_polygons(Rn_trans, arma::join_horiz(Rt_trans.col(i),Rn1,Rn2));

				// add signed contributions
				phi(i) += intersect ? phi12(i) : -phi12(i);
			}
		}

		// return potential
		return phi;
	} 

	// calculate the magnetic field of a charged polyhedron defined by vertices Rn at target points Rt
	// the polyhedron must be planar
	arma::Mat<fltp> ChargedPolyhedron::calc_magnetic_field(
		const arma::Mat<fltp>&Rn, const arma::Mat<fltp>&Rt){

		// check that it is at least a triangle
		assert(Rn.n_rows==3); assert(Rn.n_cols>=3); assert(Rt.n_rows==3);
		assert(Rn.is_finite()); assert(Rt.is_finite());

		// vector spanning first triangle
		const arma::Col<fltp>::fixed<3> V1 = Rn.col(1) - Rn.col(0);
		const arma::Col<fltp>::fixed<3> V2 = Rn.col(2) - Rn.col(0);

		// create coordinate system where e1 and e2 are in plane with the polyhedron
		// and e3 is normal to the polyhedron
		const arma::Col<fltp>::fixed<3> e3 = cmn::Extra::normalize(cmn::Extra::cross(V1, V2));
		const arma::Col<fltp>::fixed<3> e1 = cmn::Extra::normalize(V1);
		const arma::Col<fltp>::fixed<3> e2 = cmn::Extra::cross(e1,e3);
		assert(std::abs(arma::as_scalar(cmn::Extra::vec_norm(e2))-1.0)<1e-9);

		// calculate Rn and T in coordinate system in 2D
		const arma::Mat<fltp> Rn_trans = arma::join_vert(arma::sum(Rn.each_col()%e1), arma::sum(Rn.each_col()%e2));
		const arma::Mat<fltp> Rt_trans = arma::join_vert(arma::sum(Rt.each_col()%e1), arma::sum(Rt.each_col()%e2));

		// allocate output potential at each target point
		arma::Mat<fltp> H(3, Rt.n_cols, arma::fill::zeros);

		// calculate elevation of the target point with respect to the polyhedron plane
		arma::Row<fltp> c12 = arma::sum((Rt.each_col() - Rn.col(0)).eval().each_col()%e3);
		assert(c12.is_finite());

		// walk over edges
		for(arma::uword j=0;j<Rn_trans.n_cols;j++){
			// get indices of vertices
			const arma::uword idx1 = j, idx2 = (j+1)%Rn_trans.n_cols;

			// get vertices for this edge
			const arma::Col<fltp>::fixed<2> Rn1 = Rn_trans.col(idx1);
			const arma::Col<fltp>::fixed<2> Rn2 = Rn_trans.col(idx2);

			// check if vertices are different
			assert(arma::all(cmn::Extra::vec_norm(Rn2-Rn1)>1e-14));

			// perpendicular projection
			const arma::Col<fltp>::fixed<2> dRn12 = cmn::Extra::normalize(Rn2 - Rn1);
			const arma::Col<fltp>::fixed<2> dRpp{-dRn12(1),dRn12(0)};

			// project
			const fltp ell12 = arma::as_scalar(cmn::Extra::vec_norm(Rn2 - Rn1));
			const arma::Row<fltp> t = arma::sum((Rt_trans.each_col()-Rn1).eval().each_col()%dRn12);
			const arma::Mat<fltp> Rpp = (arma::repmat(t,2,1).eval().each_col()%dRn12).eval().each_col() + Rn1;
			// arma::Mat<fltp> dRpp = cmn::Extra::normalize(Rpp - Rt_trans);

			// this now gives two right triangles [Rn1,O,Rpp] and [Rn2,O,Rpp]
			const arma::Row<fltp> b1 = arma::sum((Rn1 - Rpp.each_col()).eval().each_col()%dRn12); // signed distance right angle and first corner
			const arma::Row<fltp> b2 = arma::sum((Rn2 - Rpp.each_col()).eval().each_col()%dRn12); // signed distance right angle and second corner
			const arma::Row<fltp> a12 = arma::sum((Rpp - Rt_trans).eval().each_col()%dRpp); // signed distance between O and right angle

			// calculate their fields (the benefit of joining this calculation is minimal)
			arma::Mat<fltp> H1 = right_triangle_magnetic_field(arma::abs(a12),arma::abs(b1),c12);
			arma::Mat<fltp> H2 = right_triangle_magnetic_field(arma::abs(a12),arma::abs(b2),c12);

			// check calculated potentials
			assert(H1.is_finite()); assert(H2.is_finite());

			// walk over target points
			for(arma::uword i=0;i<Rt_trans.n_cols;i++){
				// when the x-coordinate a is negative the Hx component needs flipping
				if(a12(i)<0.0){
					H1(0,i)*=-1; H2(0,i)*=-1;
				} 
				
				// when the y-coordinate b is negative the Hy component needs flipping
				if(b1(i)<0.0)H1(1,i)*=-1;
				if(b2(i)<0.0)H2(1,i)*=-1;

				// check the projection interval
				arma::Col<fltp>::fixed<3> H12 = t(i)<RAT_CONST(0.0) ? (H2.col(i) - H1.col(i)).eval() : 
					(t(i)<=ell12 ? (H1.col(i) + H2.col(i)).eval() : (H1.col(i) - H2.col(i)).eval());

				// early out
				if(arma::all(H12==RAT_CONST(0.0)))continue;

				// coordinate transformation to global coordinate system
				H12 = e1*(dRn12(0)*H12(1) + dRpp(0)*H12(0)) + 
					e2*(dRn12(1)*H12(1) + dRpp(1)*H12(0)) + 
					e3*H12(2);

				// check for intersection with the triangle to the polyhedron
				// [Rpp,Rn1] and [Rpp,Rn2] will not intersect and neither will [O,Rpp]
				// so we only need to check for [O,Rn1] and [O,Rn2} for the first
				// and second triangle respectively
				bool intersect=intersect_polygons(Rn_trans, arma::join_horiz(Rt_trans.col(i),Rn1,Rn2),RAT_CONST(1e-15));

				// add signed contributions
				H.col(i) += intersect ? H12 : -H12;
			}
		}

		// check output
		assert(H.is_finite());

		// return potential
		return H;
	} 

	// integrate from gauss points
	arma::Mat<fltp> ChargedPolyhedron::calc_magnetic_field_gauss(
		const arma::Mat<fltp>&Rn, const arma::Mat<fltp>&Rt, const arma::sword num_gauss){

		// check input
		assert(Rn.n_cols==4); assert(Rn.n_rows==3); // only quad elements allowed for this test
		assert(Rt.n_rows==3); assert(num_gauss!=0);

		// create gauss points
		const arma::Mat<fltp> gp = cmn::Quadrilateral::create_gauss_points(num_gauss);
		const arma::Mat<fltp> Rqg = gp.rows(0,1);
		const arma::Row<fltp> wg = gp.row(2);

		// convert to carthesian coordinates
		const arma::Mat<fltp> Rcg = cmn::Quadrilateral::quad2cart(Rn,Rqg);

		// calculate jacobian for each gauss point
		const arma::Mat<fltp> dN = cmn::Quadrilateral::shape_function_derivative(Rqg);
		const arma::Mat<fltp> J = cmn::Quadrilateral::shape_function_jacobian(Rn,dN);
		const arma::Row<fltp> Jdet = cmn::Quadrilateral::jacobian2determinant(J);

		// allocate output
		arma::Mat<fltp> H(3,Rt.n_cols);

		// walk over target points
		for(arma::uword i=0;i<Rt.n_cols;i++){
			// relative position
			const arma::Mat<fltp> dR = Rcg.each_col() - Rt.col(i);

			// distance between target and sources
			const arma::Row<fltp> rho = cmn::Extra::vec_norm(dR);

			// add contributions from source gauss points
			H.col(i) += -arma::sum(dR.each_row()%(wg%Jdet/(rho%rho%rho)),1);
		}

		// return magnetic field
		return H;
	}


	// analytical integral of magnetic field over the right triangle
	// note there is a factor 1/4*pi*mu0 not accounted for here
	// the charge distribution on the triangle is given as:
	// sigma(x,y) = sigma0 + sigma1*x + sigma2*y
	arma::Mat<fltp> ChargedPolyhedron::right_triangle_magnetic_field(
		const arma::Row<fltp>&a, const arma::Row<fltp>&b, const arma::Row<fltp>&c, 
		const arma::Row<fltp>&sigma0, const arma::Row<fltp>&sigma1, const arma::Row<fltp>&sigma2){

		// check input
		assert(a.n_elem == b.n_elem); assert(a.n_elem == c.n_elem);
		assert(a.is_finite()); assert(b.is_finite()); assert(c.is_finite());

		// pythagoras
		const arma::Row<fltp> asq = arma::square(a);
		const arma::Row<fltp> bsq = arma::square(b);
		const arma::Row<fltp> csq = arma::square(c);
		const arma::Row<fltp> Dabc = arma::sqrt(asq + bsq + csq);
		const arma::Row<fltp> Dab = arma::sqrt(asq + bsq);
		const arma::Row<fltp> Dac = arma::sqrt(asq + csq);
		const arma::Row<fltp> Dbc = arma::sqrt(bsq + csq);

		// check output
		assert(Dabc.is_finite()); assert(Dab.is_finite()); assert(Dac.is_finite());

		// contributions to Hx
		arma::Row<fltp> Hx_sigma0_a = sigma0 % ((-b/(2*Dab))%arma::log((Dabc+Dab)/(Dabc-Dab)));
		arma::Row<fltp> Hx_sigma0_b = sigma0 % ((1.0/2)*arma::log((Dabc+b)/(Dabc-b)));
		arma::Row<fltp> Hx_sigma1 = sigma1 % (c%atan(a%b/(arma::square(Dac)+c%Dabc)) + (a%b/(Dabc%(Dabc-b)))%((b+c-Dabc)+(c%(b+c)%(c-Dabc))/arma::square(Dab)));
		arma::Row<fltp> Hx_sigma2 = sigma2 % (asq%Dabc - (asq+bsq)%Dac + bsq%c) / (asq+bsq);

		// contributions to Hy
		arma::Row<fltp> Hy_sigma0 = sigma0 % ((a/(2*Dab))%arma::log((Dabc+Dab)/(Dabc-Dab)) - (1.0/2)*arma::log((Dac+a)/(Dac-a)));
		arma::Row<fltp> Hy_sigma1 = sigma1 % (asq%Dabc - (asq+bsq)%Dac + bsq%c) / (asq+bsq);
		arma::Row<fltp> Hy_sigma2 = -sigma2 % (1.0/(2*arma::square(Dab)))%(a%arma::square(Dab)%arma::log((b+Dabc)/(Dabc-b)) - 2*c%arma::square(Dab)%arma::atan((a%b)/(arma::square(Dac)+c%Dabc)) + 2*a%b%(c-Dabc));

		// contributions to Hz
		arma::Row<fltp> Hz_sigma0 = sigma0%(arma::atan((a%Dabc)/(b%c)) - arma::sign(c)%arma::atan(a/b) );
		arma::Row<fltp> Hz_sigma1 = -sigma1%((c/2)%arma::log((Dabc+b)/(Dabc-b)) - (b%c)/(2*Dabc) - (b/(2*Dab))%( 2*c%arma::log((Dab+Dabc)/arma::abs(c)) + arma::pow(c,3)/(Dabc%(Dab+Dabc)) - c));
		arma::Row<fltp> Hz_sigma2 = -sigma2%((a/2)%(c/Dabc - c/Dac) + (a/(2*Dab))%(2*c%arma::log((Dab+Dabc)/arma::abs(c)) + arma::pow(c,3)/(Dabc%(Dab+Dabc)) - c) - c%arma::log((Dac+a)/abs(c)) - arma::pow(c,3)/(2*Dac%(Dac+a)) + c/2);

		// corrections to field from paper
		const fltp tol = RAT_CONST(1e-14);
		for(arma::uword i=0;i<c.n_elem;i++){
			// sliver element in a or b
			if(std::abs(a(i))<tol || std::abs(b(i))<tol){
				Hx_sigma0_a(i) = 0.0; Hx_sigma0_b(i) = 0.0; Hx_sigma1(i) = 0.0; Hx_sigma2(i) = 0.0; 
				Hy_sigma0(i) = 0.0; Hy_sigma1(i) = 0.0; Hy_sigma2(i) = 0.0; 
				Hz_sigma0(i) = 0.0; Hz_sigma1(i) = 0.0; Hz_sigma2(i) = 0.0; 
			}

			// in plane divergent terms cancel and z-field is zero
			if(std::abs(c(i))<tol){
				// assert(std::isfinite(Hz_sigma2(i)));
				Hz_sigma0(i) = 0.0; Hz_sigma1(i) = 0.0; Hz_sigma2(i) = 0.0; // z contribution should be taken zero in plane according to paper
				Hy_sigma0(i) = 0.0; // this is also divergent in the regular version
				Hx_sigma0_a(i) = 0.0; // this is also divergent in the regular version
				assert(std::isfinite(Hx_sigma0_b(i)));
				assert(std::isfinite(Hx_sigma1(i)));
				assert(std::isfinite(Hx_sigma2(i)));
				assert(std::isfinite(Hy_sigma1(i)));
				assert(std::isfinite(Hy_sigma2(i)));
			}
		}

		// combine
		const arma::Mat<fltp> H = arma::join_vert(
			Hx_sigma0_a + Hx_sigma0_b + Hx_sigma1 + Hx_sigma2,
			Hy_sigma0 + Hy_sigma1 + Hy_sigma2,
			Hz_sigma0 + Hz_sigma1 + Hz_sigma2);

		// H.cols(arma::find_nan(cmn::Extra::vec_norm(H))).fill(0.0);
		// H.cols(arma::find_nonfinite(cmn::Extra::vec_norm(H))).fill(0.0);

		// check output
		assert(H.is_finite());

		// return potential
		return H;
	}

	// calculate the magnetic field of a charged polyhedron defined by vertices Rn at target points Rt
	// the polyhedron must be planar
	arma::Mat<fltp> ChargedPolyhedron::calc_magnetic_field(
		const arma::Mat<fltp>&Rn, 
		const arma::Row<fltp>&sigma,
		const arma::Mat<fltp>&Rt){

		// check that it is at least a triangle
		assert(Rn.n_rows==3); assert(Rn.n_cols>=3); assert(Rt.n_rows==3);
		assert(Rn.is_finite()); assert(Rt.is_finite());

		// vector spanning first triangle
		const arma::Col<fltp>::fixed<3> V1 = Rn.col(1) - Rn.col(0);
		const arma::Col<fltp>::fixed<3> V2 = Rn.col(2) - Rn.col(0);

		// create coordinate system where e1 and e2 are in plane with the polyhedron
		// and e3 is normal to the polyhedron
		const arma::Col<fltp>::fixed<3> e3 = cmn::Extra::normalize(cmn::Extra::cross(V1, V2));
		const arma::Col<fltp>::fixed<3> e1 = cmn::Extra::normalize(V1);
		const arma::Col<fltp>::fixed<3> e2 = cmn::Extra::cross(e1,e3);
		assert(std::abs(arma::as_scalar(cmn::Extra::vec_norm(e2))-1.0)<1e-9);

		// calculate Rn and T in coordinate system in 2D
		const arma::Mat<fltp> Rn_trans = arma::join_vert(arma::sum(Rn.each_col()%e1), arma::sum(Rn.each_col()%e2));
		const arma::Mat<fltp> Rt_trans = arma::join_vert(arma::sum(Rt.each_col()%e1), arma::sum(Rt.each_col()%e2));

		// allocate output potential at each target point
		arma::Mat<fltp> H(3, Rt.n_cols, arma::fill::zeros);

		// calculate elevation of the target point with respect to the polyhedron plane
		arma::Row<fltp> c12 = arma::sum((Rt.each_col() - Rn.col(0)).eval().each_col()%e3);
		assert(c12.is_finite());

		// calculate sigma distribution by fitting a plane
		const arma::Mat<fltp>::fixed<3,3> Ms = arma::join_vert(arma::Row<fltp>::fixed<3>(arma::fill::ones), Rn_trans.cols(0,2)).t();
		const arma::Col<fltp>::fixed<3> X = arma::inv(Ms)*sigma.cols(0,2).t(); // only use first 3 values (assume the rest is linear)
		const arma::Row<fltp> sigma0 = X(0) + X(1)*Rt_trans.row(0) + X(2)*Rt_trans.row(1); // at position of target point

		// walk over edges
		for(arma::uword j=0;j<Rn_trans.n_cols;j++){
			// get indices of vertices
			const arma::uword idx1 = j, idx2 = (j+1)%Rn_trans.n_cols;

			// get vertices for this edge
			const arma::Col<fltp>::fixed<2> Rn1 = Rn_trans.col(idx1);
			const arma::Col<fltp>::fixed<2> Rn2 = Rn_trans.col(idx2);

			// check if vertices are different
			assert(arma::all(cmn::Extra::vec_norm(Rn2-Rn1)>1e-14));

			// perpendicular projection
			const arma::Col<fltp>::fixed<2> dRn12 = cmn::Extra::normalize(Rn2 - Rn1);
			const arma::Col<fltp>::fixed<2> dRpp{-dRn12(1),dRn12(0)};

			// project
			const fltp ell12 = arma::as_scalar(cmn::Extra::vec_norm(Rn2 - Rn1));
			const arma::Row<fltp> t = arma::sum((Rt_trans.each_col()-Rn1).eval().each_col()%dRn12);
			const arma::Mat<fltp> Rpp = (arma::repmat(t,2,1).eval().each_col()%dRn12).eval().each_col() + Rn1;
			// arma::Mat<fltp> dRpp = cmn::Extra::normalize(Rpp - Rt_trans);

			// this now gives two right triangles [Rn1,O,Rpp] and [Rn2,O,Rpp]
			const arma::Row<fltp> b1 = arma::sum((Rn1 - Rpp.each_col()).eval().each_col()%dRn12); // signed distance right angle and first corner
			const arma::Row<fltp> b2 = arma::sum((Rn2 - Rpp.each_col()).eval().each_col()%dRn12); // signed distance right angle and second corner
			const arma::Row<fltp> a12 = arma::sum((Rpp - Rt_trans).eval().each_col()%dRpp); // signed distance between O and right angle

			// rotate sigma along edge
			const fltp sigma1 = (X(1)*dRpp(0) + X(2)*dRpp(1)); // sigma 1 should be perpendicular to the projection
			const fltp sigma2 = (X(1)*dRn12(0) + X(2)*dRn12(1)); // sigma 2 is parallel to the edges

			// calculate their fields (the benefit of joining this calculation is minimal)
			arma::Mat<fltp> H1 = right_triangle_magnetic_field(
				arma::abs(a12),arma::abs(b1),arma::abs(c12),sigma0,
				sigma1*arma::sign(a12),sigma2*arma::sign(b1));
			arma::Mat<fltp> H2 = right_triangle_magnetic_field(
				arma::abs(a12),arma::abs(b2),arma::abs(c12),sigma0,
				sigma1*arma::sign(a12),sigma2*arma::sign(b2));

			// fix sign in z
			H1.row(2)%=arma::sign(c12);
			H2.row(2)%=arma::sign(c12);

			// check calculated potentials
			assert(H1.is_finite()); assert(H2.is_finite());

			// walk over target points
			for(arma::uword i=0;i<Rt_trans.n_cols;i++){
				// when the x-coordinate a is negative the Hx component needs flipping
				if(a12(i)<0.0){
					H1(0,i)*=-1; H2(0,i)*=-1;
				} 
				
				// when the y-coordinate b is negative the Hy component needs flipping
				if(b1(i)<0.0)H1(1,i)*=-1;
				if(b2(i)<0.0)H2(1,i)*=-1;

				// check the projection interval
				arma::Col<fltp>::fixed<3> H12 = t(i)<RAT_CONST(0.0) ? (H2.col(i) - H1.col(i)).eval() : 
					(t(i)<=ell12 ? (H1.col(i) + H2.col(i)).eval() : (H1.col(i) - H2.col(i)).eval());

				// early out
				if(arma::all(H12==RAT_CONST(0.0)))continue;

				// coordinate transformation to global coordinate system
				H12 = e1*(dRn12(0)*H12(1) + dRpp(0)*H12(0)) + 
					e2*(dRn12(1)*H12(1) + dRpp(1)*H12(0)) + 
					e3*H12(2);

				// check for intersection with the triangle to the polyhedron
				// [Rpp,Rn1] and [Rpp,Rn2] will not intersect and neither will [O,Rpp]
				// so we only need to check for [O,Rn1] and [O,Rn2} for the first
				// and second triangle respectively
				bool intersect=intersect_polygons(Rn_trans, arma::join_horiz(Rt_trans.col(i),Rn1,Rn2),RAT_CONST(1e-15));

				// add signed contributions
				H.col(i) += intersect ? H12 : -H12;
			}
		}

		// check output
		assert(H.is_finite());

		// return potential
		return H;
	} 


	// analytical integral over the right triangle
	// note there is a factor charges/4*pi*mu0 not accounted for here
	// this computes the potential for both b1 and b2, while a and c are the same for both
	arma::Row<fltp> ChargedPolyhedron::right_triangle_scalar_potential(
		const arma::Row<fltp>&a, 
		const arma::Row<fltp>&b,
		const arma::Row<fltp>&c,
		const arma::Row<fltp>&sigma0, 
		const arma::Row<fltp>&sigma1, 
		const arma::Row<fltp>&sigma2){

		// precompute common terms
		const arma::Row<fltp> asq = arma::square(a);
		const arma::Row<fltp> csq = arma::square(c);
		const arma::Row<fltp> bsq = arma::square(b);

		// compute Dabc for both b1 and b2
		const arma::Row<fltp> Dac = arma::sqrt(asq + csq);
		const arma::Row<fltp> Dabc = arma::sqrt(asq + bsq + csq);
		const arma::Row<fltp> Dab = arma::sqrt(asq + bsq);

		// perform calculation
		arma::Row<fltp> phi_sigma01 = sigma0 % ( (a/2)%log((Dabc+b)/(Dabc-b)));
		arma::Row<fltp> phi_sigma02 = sigma0 % ( -arma::abs(c)%arma::atan((a%b)/(arma::square(Dac) + arma::abs(c)%Dabc) ) );
		arma::Row<fltp> phi_sigma11 = sigma1 % ( ((asq+csq)/4)%log((Dabc+b)/(Dabc-b)) );
		arma::Row<fltp> phi_sigma12 = sigma1 % ( -(b%csq/(2*Dab))%arma::log((Dab+Dabc)/abs(c)) );
		arma::Row<fltp> phi_sigma21 = sigma2 % ( (a/2)%(Dabc-Dac) );
		arma::Row<fltp> phi_sigma22 = sigma2 % ( (a%csq/(2*Dab))%arma::log((Dab+Dabc)/arma::abs(c)) );
		arma::Row<fltp> phi_sigma23 = sigma2 % ( -(csq/2)%arma::log((Dac+a)/arma::abs(c)) );

		// in plane
		const arma::Col<arma::uword> idx_in_plane = arma::find(arma::abs(c)<1e-9);
		phi_sigma12.cols(idx_in_plane).fill(0.0);
		phi_sigma22.cols(idx_in_plane).fill(0.0);
		phi_sigma23.cols(idx_in_plane).fill(0.0);

		// calculate phi1
		arma::Row<fltp> phi = phi_sigma01 + phi_sigma02 + phi_sigma11 + phi_sigma12 + phi_sigma21 + phi_sigma22 + phi_sigma23;

		// handle sliver triangles and other edge cases
		phi.cols(arma::find(
			arma::abs(a)<RAT_CONST(1e-9) || 
			arma::abs(b)<RAT_CONST(1e-9))).fill(RAT_CONST(0.0));

		// return the potential pair
		return phi;
	}

	// calculate the scalar potential of a polyhedron defined by vertices Rn at target points Rt
	// the polyhedron must be planar
	arma::Row<fltp> ChargedPolyhedron::calc_scalar_potential(
		const arma::Mat<fltp>&Rn, 
		const arma::Row<fltp>&sigma,
		const arma::Mat<fltp>&Rt){

		// check that it is at least a triangle
		assert(Rn.n_rows==3); assert(Rn.n_cols>=3); assert(Rt.n_rows==3);
		assert(Rn.is_finite()); assert(Rt.is_finite());

		// vector spanning first triangle
		const arma::Col<fltp>::fixed<3> V1 = Rn.col(1) - Rn.col(0);
		const arma::Col<fltp>::fixed<3> V2 = Rn.col(2) - Rn.col(0);

		// create coordinate system where e1 and e2 are in plane with the polyhedron
		// and e3 is normal to the polyhedron
		const arma::Col<fltp>::fixed<3> e3 = cmn::Extra::normalize(cmn::Extra::cross(V1, V2));
		const arma::Col<fltp>::fixed<3> e1 = cmn::Extra::normalize(V1);
		const arma::Col<fltp>::fixed<3> e2 = cmn::Extra::cross(e1,e3);
		assert(std::abs(arma::as_scalar(cmn::Extra::vec_norm(e2))-1.0)<1e-9);

		// calculate Rn and T in coordinate system in 2D
		const arma::Mat<fltp> Rn_trans = arma::join_vert(arma::sum(Rn.each_col()%e1), arma::sum(Rn.each_col()%e2));
		const arma::Mat<fltp> Rt_trans = arma::join_vert(arma::sum(Rt.each_col()%e1), arma::sum(Rt.each_col()%e2));


		// calculate elevation of the target point with respect to the polyhedron plane
		arma::Row<fltp> c12 = arma::sum((Rt.each_col() - Rn.col(0)).eval().each_col()%e3);
		assert(c12.is_finite());

		// allocate output potential at each target point
		arma::Row<fltp> phi(Rt.n_cols, arma::fill::zeros);

		// calculate sigma distribution by fitting a plane
		const arma::Mat<fltp>::fixed<3,3> Ms = arma::join_vert(arma::Row<fltp>::fixed<3>(arma::fill::ones), Rn_trans.cols(0,2)).t();
		const arma::Col<fltp>::fixed<3> X = arma::inv(Ms)*sigma.cols(0,2).t(); // only use first 3 values (assume the rest is linear)
		const arma::Row<fltp> sigma0 = X(0) + X(1)*Rt_trans.row(0) + X(2)*Rt_trans.row(1); // at position of target point

		// walk over edges
		for(arma::uword j=0;j<Rn_trans.n_cols;j++){
			// get indices of vertices
			const arma::uword idx1 = j, idx2 = (j+1)%Rn_trans.n_cols;

			// get vertices for this edge
			const arma::Col<fltp>::fixed<2> Rn1 = Rn_trans.col(idx1);
			const arma::Col<fltp>::fixed<2> Rn2 = Rn_trans.col(idx2);

			// check if vertices are different
			assert(arma::all(cmn::Extra::vec_norm(Rn2-Rn1)>1e-14));

			// perpendicular projection
			const arma::Col<fltp>::fixed<2> dRn12 = cmn::Extra::normalize(Rn2 - Rn1);
			const arma::Col<fltp>::fixed<2> dRpp{-dRn12(1),dRn12(0)};

			// project
			const fltp ell12 = arma::as_scalar(cmn::Extra::vec_norm(Rn2 - Rn1));
			const arma::Row<fltp> t = arma::sum((Rt_trans.each_col()-Rn1).eval().each_col()%dRn12);
			const arma::Mat<fltp> Rpp = (arma::repmat(t,2,1).eval().each_col()%dRn12).eval().each_col() + Rn1;
			// arma::Mat<fltp> dRpp = cmn::Extra::normalize(Rpp - Rt_trans);

			// this now gives two right triangles [Rn1,O,Rpp] and [Rn2,O,Rpp]
			const arma::Row<fltp> b1 = arma::sum((Rn1 - Rpp.each_col()).eval().each_col()%dRn12); // signed distance right angle and first corner
			const arma::Row<fltp> b2 = arma::sum((Rn2 - Rpp.each_col()).eval().each_col()%dRn12); // signed distance right angle and second corner
			const arma::Row<fltp> a12 = arma::sum((Rpp - Rt_trans).eval().each_col()%dRpp); // signed distance between O and right angle

			// rotate sigma along edge
			const fltp sigma1 = (X(1)*dRpp(0) + X(2)*dRpp(1)); // sigma 1 should be perpendicular to the projection
			const fltp sigma2 = (X(1)*dRn12(0) + X(2)*dRn12(1)); // sigma 2 is parallel to the edges

			// calculate their fields
			const arma::Row<fltp> phi1 = right_triangle_scalar_potential(
				arma::abs(a12),arma::abs(b1),arma::abs(c12),sigma0,sigma1*arma::sign(a12),sigma2*arma::sign(b1));
			const arma::Row<fltp> phi2 = right_triangle_scalar_potential(
				arma::abs(a12),arma::abs(b2),arma::abs(c12),sigma0,sigma1*arma::sign(a12),sigma2*arma::sign(b2));

			// check calculated potentials
			assert(phi1.is_finite()); assert(phi2.is_finite());

			// walk over target points
			arma::Row<fltp> phi12(Rt_trans.n_cols);
			for(arma::uword i=0;i<Rt_trans.n_cols;i++){
				if(t(i)<RAT_CONST(0.0))phi12(i) = phi2(i) - phi1(i);
				else if(t(i)<ell12)phi12(i) = phi1(i) + phi2(i);
				else phi12(i) = phi1(i) - phi2(i);
			}

			// walk over target points
			for(arma::uword i=0;i<Rt_trans.n_cols;i++){
				// check for intersection with the triangle to the polyhedron
				// [Rpp,Rn1] and [Rpp,Rn2] will not intersect and neither will [O,Rpp]
				// so we only need to check for [O,Rn1] and [O,Rn2} for the first
				// and second triangle respectively
				const bool intersect = intersect_polygons(Rn_trans, arma::join_horiz(Rt_trans.col(i),Rn1,Rn2));

				// add signed contributions
				phi(i) += intersect ? phi12(i) : -phi12(i);
			}
		}

		// return potential
		return phi;
	} 


}}