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
#include "tetrahedron.hh"



// hexahedron face matrix
arma::Mat<arma::uword>::fixed<4,3> Tetrahedron::get_faces(){
	// setup matrix
	arma::Mat<arma::uword>::fixed<4,3> M; 

	// faces is counter clockwise order (as seen from outside)
	M.row(0) = arma::Row<arma::uword>{0,2,1}; 
	M.row(1) = arma::Row<arma::uword>{0,1,3}; 
	M.row(2) = arma::Row<arma::uword>{0,3,2}; 
	M.row(3) = arma::Row<arma::uword>{1,2,3}; 

	// return matrix
	return M;
}


// vector wise determinants from matrix
// |x1,y1,z1,1;x2,y2,z2,1;x3,y3,z3,1;x4,y4,z4,1| = 
// -x3 y2 z1 + x4 y2 z1 + x2 y3 z1 - x4 y3 z1 - x2 y4 z1 + x3 y4 z1 + 
//  x3 y1 z2 - x4 y1 z2 - x1 y3 z2 + x4 y3 z2 + x1 y4 z2 - x3 y4 z2 - 
//  x2 y1 z3 + x4 y1 z3 + x1 y2 z3 - x4 y2 z3 - x1 y4 z3 + x2 y4 z3 + 
//  x2 y1 z4 - x3 y1 z4 - x1 y2 z4 + x3 y2 z4 + x1 y3 z4 - x2 y3 z4
arma::Row<double> Tetrahedron::special_determinant(
	const arma::Mat<double> &R0, 
	const arma::Mat<double> &R1, 
	const arma::Mat<double> &R2,
	const arma::Mat<double> &R3){

	// calculate determinants
	arma::Row<double> det = 
	   -R2.row(0)%R1.row(1)%R0.row(2) + R3.row(0)%R1.row(1)%R0.row(2) + 
		R1.row(0)%R2.row(1)%R0.row(2) - R3.row(0)%R2.row(1)%R0.row(2) - 
		R1.row(0)%R3.row(1)%R0.row(2) + R2.row(0)%R3.row(1)%R0.row(2) + 
		R2.row(0)%R0.row(1)%R1.row(2) - R3.row(0)%R0.row(1)%R1.row(2) - 
		R0.row(0)%R2.row(1)%R1.row(2) + R3.row(0)%R2.row(1)%R1.row(2) + 
		R0.row(0)%R3.row(1)%R1.row(2) - R2.row(0)%R3.row(1)%R1.row(2) - 
		R1.row(0)%R0.row(1)%R2.row(2) + R3.row(0)%R0.row(1)%R2.row(2) + 
		R0.row(0)%R1.row(1)%R2.row(2) - R3.row(0)%R1.row(1)%R2.row(2) - 
		R0.row(0)%R3.row(1)%R2.row(2) + R1.row(0)%R3.row(1)%R2.row(2) + 
		R1.row(0)%R0.row(1)%R3.row(2) - R2.row(0)%R0.row(1)%R3.row(2) - 
		R0.row(0)%R1.row(1)%R3.row(2) + R2.row(0)%R1.row(1)%R3.row(2) + 
		R0.row(0)%R2.row(1)%R3.row(2) - R1.row(0)%R2.row(1)%R3.row(2);

	// return determinants
	return det;
}

// function for determining if points are inside a thetahedron
arma::Row<arma::uword> Tetrahedron::is_inside(
	const arma::Mat<double> &R,
	const arma::Mat<double> &Rn,
	const arma::Mat<arma::uword> &n){

	// check input
	assert(R.n_rows==3); 
	assert(Rn.n_rows==3);
	assert(n.n_rows==4);
	assert(n.n_elem>0);
	assert(arma::max(arma::max(n))<Rn.n_cols);

	// get number of elements
	arma::uword num_elem = n.n_cols;
	arma::uword num_points = R.n_cols;

	// allocate output (outside is nan)
	arma::Row<arma::uword> isinside(1,num_points,arma::fill::zeros);

	// get four corners
	arma::Mat<double> R0 = Rn.cols(n.row(0)); 
	arma::Mat<double> R1 = Rn.cols(n.row(1));
	arma::Mat<double> R2 = Rn.cols(n.row(2)); 
	arma::Mat<double> R3 = Rn.cols(n.row(3));

	// walk over target points
	for(arma::uword k=0;k<num_points;k++){
	//parfor(0,num_points,true,[&](int k, int){
		// allocate determinant
		arma::Mat<double> determinants(5,num_elem);
		determinants.row(0) = Tetrahedron::special_determinant(R0,R1,R2,R3);

		// create array from point
		arma::Mat<double> Rp(3,num_elem);
		Rp.row(0).fill(R(0,k)); Rp.row(1).fill(R(1,k)); Rp.row(2).fill(R(2,k));

		// calculate determinants
		determinants.row(1) = Tetrahedron::special_determinant(Rp,R1,R2,R3);
		determinants.row(2) = Tetrahedron::special_determinant(R0,Rp,R2,R3);
		determinants.row(3) = Tetrahedron::special_determinant(R0,R1,Rp,R3);
		determinants.row(4) = Tetrahedron::special_determinant(R0,R1,R2,Rp);

		// get index
		// arma::Row<arma::uword> idx = 
		// 	arma::find(arma::all(determinants>-1e-12,0) || 
		// 	arma::all(determinants<1e-12,0)).t();

		// check that point is not in more than one tetrahedron
		// this can happen now that we have set 1e-12 tolerance
		// assert(idx.n_elem<=1);
		
		// store
		// if(!idx.is_empty())isinside(k) = true;

		// check if inside
		isinside(k) = arma::as_scalar(arma::any(
			arma::all(determinants>-1e-8,0) || 
			arma::all(determinants<1e-8,0),1));

	//});
	}

	// return array of 
	return isinside;
}

// volume of a tetrahedron from its four corners
// https://en.wikipedia.org/wiki/Tetrahedron#Volume
arma::Row<double> Tetrahedron::calc_volume(
	const arma::Mat<double> &Rn,
	const arma::Mat<arma::uword> &n){

	// check that input is in three dimensional space
	assert(n.n_rows==4); assert(Rn.n_rows==3);
	assert(arma::max(arma::max(n))<Rn.n_cols);

	// get points
	arma::Mat<double> R0 = Rn.cols(n.row(0)); arma::Mat<double> R1 = Rn.cols(n.row(1));
	arma::Mat<double> R2 = Rn.cols(n.row(2)); arma::Mat<double> R3 = Rn.cols(n.row(3));

	// calculate
	arma::Row<double> V = arma::abs(Extra::dot((R0-R3),Extra::cross((R1-R3),(R2-R3))))/6;
	
	// return volume array
	return V;
}

// tetrahedron corner node positions 
// in quadrilateral coordinates
// matrix columns are given as [dz1,dz2,dz3,dz4]
arma::Mat<arma::sword>::fixed<4,4> Tetrahedron::get_corner_nodes(){
	// setup matrix
	arma::Mat<arma::sword>::fixed<4,4> M; 
	M.row(0) = arma::Row<arma::sword>{ 1,0,0,0};
	M.row(1) = arma::Row<arma::sword>{-1,1,0,0};
	M.row(2) = arma::Row<arma::sword>{-1,0,1,0};
	M.row(3) = arma::Row<arma::sword>{-1,0,0,1};

	// return matrix
	return M;
}

// quadrilateral coordinates to carthesian coordinates
arma::Mat<double> Tetrahedron::quad2cart(
	const arma::Mat<double> &Rn, 
	const arma::Mat<double> &Rq){
	
	// convert 
	arma::Mat<double> Rc(3,Rq.n_cols);
	Rc.row(0) = 1.0-Rq.row(0); 
	Rc.row(1) = (1.0-Rq.row(1))%Rq.row(0);   
	Rc.row(2) = Rq.row(0)%Rq.row(1)%Rq.row(2);

	// convert node matrix
	arma::Mat<arma::sword>::fixed<4,4> M = get_corner_nodes();
	arma::Mat<double> c = M*Rn.t();

	// calculate coordinates
	arma::Mat<double> Rcc = arma::join_horiz(
		arma::Col<double>(Rq.n_cols,arma::fill::ones),Rc.t())*c;

	// return coords
	return Rcc;
}
