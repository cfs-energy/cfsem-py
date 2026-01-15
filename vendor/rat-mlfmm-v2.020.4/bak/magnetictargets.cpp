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
#include "magnetictargets.hh"

// default constructor
MagneticTargets::MagneticTargets(){

}

// factory
ShMagneticTargetsPr MagneticTargets::create(){
	//return ShMagneticTargetsPr(new MagneticTargets);
	return std::make_shared<MagneticTargets>();
}

// set coordinates
void MagneticTargets::set_coords(
	const arma::Mat<double> &Rt){

	// check input
	assert(Rt.n_rows==3);
	assert(Rt.is_finite());

	// set coordinates
	Rt_ = Rt;

	// set number of points
	num_targets_ = Rt.n_cols;
}

// get target point coordinates
arma::Mat<double> MagneticTargets::get_target_coords() const{
 	return Rt_;
}

// get target point coordinates
arma::Mat<double> MagneticTargets::get_target_coords(
	const arma::Row<arma::uword> &indices) const{
 	return Rt_.cols(indices);
}

// number of points stored
arma::uword MagneticTargets::num_targets() const{
	return num_targets_;
}

// set plane coordinates
void MagneticTargets::set_xy_plane(double ellx, double elly, 
	double xoff, double yoff, double zoff, 
	arma::uword nx, arma::uword ny){

	// allocate coordinates
	arma::Mat<double> x(ny,nx); arma::Mat<double> y(ny,nx); 
	arma::Mat<double> z(ny,nx,arma::fill::zeros);

	// set coordinates to matrix
	x.each_row() = arma::linspace<arma::Row<double> >(-ellx/2,ellx/2,nx) + xoff;
	y.each_col() = arma::linspace<arma::Col<double> >(-elly/2,elly/2,ny) + yoff;
	z += zoff;
	
	// store coordinates
	arma::Mat<double> Rt(3,nx*ny);
	Rt.row(0) = arma::reshape(x,1,nx*ny);
	Rt.row(1) = arma::reshape(y,1,nx*ny);
	Rt.row(2) = arma::reshape(z,1,nx*ny);

	// set coordinates
	set_coords(Rt);
}

// set plane coordinates
void MagneticTargets::set_xz_plane(double ellx, double ellz, 
	double xoff, double yoff, double zoff, 
	arma::uword nx, arma::uword nz){

	// allocate coordinates
	arma::Mat<double> x(nz,nx); arma::Mat<double> z(nz,nx); 
	arma::Mat<double> y(nz,nx,arma::fill::zeros);

	// set coordinates to matrix
	x.each_row() = arma::linspace<arma::Row<double> >(-ellx/2,ellx/2,nx) + xoff;
	y += yoff;
	z.each_col() = arma::linspace<arma::Col<double> >(-ellz/2,ellz/2,nz) + zoff;
	
	// store coordinates
	arma::Mat<double> Rt(3,nx*nz);
	Rt.row(0) = arma::reshape(x,1,nx*nz);
	Rt.row(1) = arma::reshape(y,1,nx*nz);
	Rt.row(2) = arma::reshape(z,1,nx*nz);

	// set coordinates
	set_coords(Rt);
}
