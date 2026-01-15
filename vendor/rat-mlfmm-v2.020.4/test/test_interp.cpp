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

// general headers
#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>

#include "rat/common/extra.hh"
#include "rat/common/log.hh"

#include "interp.hh"
#include "targetpoints.hh"
#include "mlfmm.hh"
#include "settings.hh"

// main
int main(){
	// settings
	const rat::fltp Rin = 0.04;
	const rat::fltp Rout = 0.08;
	const rat::fltp height = 0.04;
	const rat::fltp dl = 3e-3;
	const arma::uword Nt = 1000; // number of target points

	// set seed this is recommended to always have same unit test
	arma::arma_rng::set_seed(1002);

	// calculate number of elements
	const arma::uword nr = std::max(2,(int)std::ceil((Rout-Rin)/dl));
	const arma::uword nz = std::max(2,(int)std::ceil(height/dl));
	const arma::uword nl = std::max(2,(int)std::ceil(2*arma::Datum<rat::fltp>::pi*Rout/dl));

	// create random target coordinates
	const arma::Mat<rat::fltp> Rt = rat::cmn::Extra::random_coordinates(
		0, 0, 0, 1.1*Rout, Nt); // x,y,z,size,N

	// create interpolation mesh
	rat::fmm::ShInterpPr msh = rat::fmm::Interp::create();

	// set mesh
	msh->setup_cylinder(Rin,Rout,height,nr,nz,nl);

	// get coordinates
	const arma::Mat<rat::fltp> Rn = msh->get_node_coords();
	const arma::Mat<arma::uword> n = msh->get_elements();

	// calculate distance to origin
	const arma::Row<rat::fltp> dist = rat::cmn::Extra::vec_norm(Rn);
	assert(Rn.n_cols==dist.n_cols);

	// set the origin distance as interpolation value
	msh->set_interpolation_values('d',dist);

	// subdivide for additional testing
	const rat::fmm::ShSourcesPr src = msh->subdivide(height/2);

	// create targets with type "d" and dim 1
	rat::fmm::ShTargetPointsPr tar = rat::fmm::TargetPoints::create(Rt);
	tar->set_field_type('d',1);

	// create settings and set to S2T only
	rat::fmm::ShSettingsPr stngs = rat::fmm::Settings::create();
	stngs->set_enable_fmm(false);
	stngs->set_enable_s2t(true);

	// create mlfmm object
	rat::fmm::ShMlfmmPr fmm = rat::fmm::Mlfmm::create(src,tar);
	fmm->set_settings(stngs);

	// create logger
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();	

	// setup fmm and calculate
	fmm->setup(lg);
	fmm->calculate(lg);

	// get calculate distances
	const arma::Row<rat::fltp> dist_interp = tar->get_field('d');

	// calculate expected answer
	const arma::Row<rat::fltp> dist_direct = rat::cmn::Extra::vec_norm(Rt);

	// find target positions inside the mesh
	arma::Row<arma::uword> is_inside = 
		Rt.row(2)<=height/2 && Rt.row(2)>=-height/2 && 
		rat::cmn::Extra::vec_norm(Rt.rows(0,1))>=Rin+1e-4 && 
		rat::cmn::Extra::vec_norm(Rt.rows(0,1))<=Rout-1e-4;

	// get mesh
	// arma::Row<arma::uword> is_inside = 
	// 	rat::cmn::Hexahedron::is_inside(Rt,Rn,n);

	// std::cout<<Rt.t()<<std::endl;

	// display
	std::cout<<arma::join_horiz(
		arma::conv_to<arma::Row<rat::fltp> >::from(is_inside).t(),
		dist_direct.t(),dist_interp.t())<<std::endl;

	// // check whether all points outside are zero
	// if(!dist_interp(arma::find(is_inside==false)).is_zero(1e-14))
	// 	rat_throw_line("points outside mesh are not all set to zero");

	// check if points inside are correctly interpolated
	arma::Row<arma::uword> idx_inside = arma::find(is_inside).t();
	if(arma::any(arma::abs(dist_interp.cols(idx_inside)-
		rat::cmn::Extra::vec_norm(Rt.cols(idx_inside)))>1e-4))
		rat_throw_line("interpolation is outside of tolerance");	
}