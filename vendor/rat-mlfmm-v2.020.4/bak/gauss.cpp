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
#include "common/gauss.hh"

// default constructor
Gauss::Gauss(){
	// setup points
	calculate();
}

// constructor
Gauss::Gauss(arma::sword num_gauss){
	// check input
	assert(num_gauss!=0);

	// set number of gauss points
	num_gauss_ = num_gauss;

	// setup points
	calculate();
}


// Generates the abscissa and weights for a Gauss-Legendre quadrature.
// Reference:  Numerical Recipes in Fortran 77, Cornell press.
void Gauss::calculate(){
	// can not be zero
	assert(num_gauss_!=0);

	// get number of gauss points
	arma::uword ng = (arma::uword)std::abs(num_gauss_);

	// gauss point distribution
	if(num_gauss_>0){
		// allocate output
		xg_.zeros(ng); wg_.zeros(ng);

		// calculate
		double m = (ng+1)/2;
		for(arma::uword ii=0;ii<m;ii++){
			// initial estimate
			double z = std::cos(arma::datum::pi*(ii+1-0.25)/(ng+0.5));                      
			double z1 = z+1;
			double pp = 0; // not used
			while(std::abs(z-z1)>tol_){
				double p1 = 1, p2 = 0;
				for(arma::uword jj=1;jj<=ng;jj++){
					double p3 = p2;
					p2 = p1;

					// The Legendre polynomial.
					p1 = ((2*jj-1)*z*p2-(jj-1)*p3)/jj;       
				}

				// The L.P. derivative.
				pp = ng*(z*p1-p2)/(z*z-1);                       
				z1 = z;
				z = z1-p1/pp;
			}

			// Build up the abscissas.
			xg_(ii) = -z;
			xg_(ng-1-ii) = z;

			// Build up the weights.
			wg_(ii) = 2/((1-z*z)*(pp*pp));
			wg_(ng-1-ii) = wg_(ii);
		}
	}

	// regular distribution
	if(num_gauss_<0){
		xg_ = arma::linspace<arma::Row<double> >(-1.0 + 1.0/ng, 1.0 - 1.0/ng,ng);
		wg_ = arma::Row<double>(ng,arma::fill::ones)*2.0/ng;
	}
}

// function for getting absissas
arma::Row<double> Gauss::get_abscissae() const{
	return xg_;
}

// function for getting absissas
arma::Row<double> Gauss::get_weights() const{
	return wg_;
}

// make grid
// arma::Mat<arma::uword> Gauss::setup_grid(
// 	const arma::uword N0,
// 	const arma::uword N1,
// 	const arma::uword N2){
	
// 	// setup grid
// 	arma::Mat<arma::uword> grd(3,N0*N1*N2);

// 	// walk over grid
// 	arma::uword p = 0;
// 	for(arma::uword i=0;i<N0;i++){
// 		for(arma::uword j=0;j<N1;j++){
// 			for(arma::uword k=0;k<N2;k++){
// 				// set value
// 				grd(0,p) = i; grd(1,p) = j; grd(2,p) = k;

// 				// increment counter
// 				p++;
// 			}
// 		}
// 	}

// 	// return grid coordinates
// 	return grd;
// }

// gaussian grid
// outputs coordinates in quadrilateral coordinates
// and the corresponding weights (in one joined matrix)
// arma::Mat<double> Gauss::setup_gauss_grid(
// 	const arma::uword N){
// 	// check input 
// 	assert(N>=1);

// 	// calculate gauss points
// 	arma::Mat<double> v = Gauss::calc_gauss_points(N);
		
// 	// extract abscissae and weights
// 	arma::Row<double> x = v.row(0); 
// 	arma::Row<double> w = v.row(1);

// 	// walk over grid
// 	arma::Mat<arma::uword> grd = Gauss::setup_grid(N,N,N);

// 	// get grid coordinates
// 	arma::Mat<double> Rgrd = arma::reshape(
// 		x(arma::reshape(grd,3*N*N*N,1)),3,N*N*N);

// 	// get grid weights
// 	arma::Row<double> wgrd = arma::prod(
// 		arma::reshape(w(arma::reshape(
// 		grd,3*N*N*N,1)),3,N*N*N),0)/8;	

// 	// check weights
// 	assert(std::abs(arma::sum(wgrd)-1)<1e-6);

// 	// return gauss grid
// 	return arma::join_vert(Rgrd,wgrd);
// }

// gaussian grid
// outputs coordinates in quadrilateral coordinates
// and the corresponding weights (in one joined matrix)
// arma::Mat<double> Gauss::setup_regular_grid(
// 	const arma::uword N){
// 	// check input 
// 	assert(N>=1);

// 	// extract abscissae and weights
// 	arma::Row<double> x = arma::linspace<arma::Row<double> >(-1.0+1.0/N,1.0-1.0/N,N); 
// 	arma::Row<double> w = 2*arma::Row<double>(N,arma::fill::ones)/N;

// 	// walk over grid
// 	arma::Mat<arma::uword> grd = Gauss::setup_grid(N,N,N);

// 	// get grid coordinates
// 	arma::Mat<double> Rgrd = arma::reshape(
// 		x(arma::reshape(grd,3*N*N*N,1)),3,N*N*N);

// 	// get grid weights
// 	arma::Row<double> wgrd = arma::prod(
// 		arma::reshape(w(arma::reshape(
// 		grd,3*N*N*N,1)),3,N*N*N),0)/8;	

// 	// check weights
// 	assert(std::abs(arma::sum(wgrd)-1)<1e-6);

// 	// return gauss grid
// 	return arma::join_vert(Rgrd,wgrd);
// }





