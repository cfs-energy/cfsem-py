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

// headers for common
#include "rat/common/error.hh"
#include "rat/common/extra.hh"
#include "rat/common/gmshfile.hh"

// headers for fmm
#include "mgntargets.hh"
#include "currentsources.hh"
#include "soleno.hh"
#include "mlfmm.hh"
#include "sourcecoil.hh"

// main
int main(){
	// settings
	const rat::fltp radius = 0.1;
	const rat::fltp height = 0.01;
	const rat::fltp thickness = 0.01;
	const arma::uword num_elem_theta = 64llu;
	const rat::fltp current = 100*1000;

	// azymuthal planes
	const arma::Row<rat::fltp> theta = arma::linspace<arma::Row<rat::fltp> >(
		RAT_CONST(0.0),arma::Datum<rat::fltp>::tau,num_elem_theta+1);

	// cross section edges
	const arma::Row<rat::fltp> radii{radius,radius+thickness,radius+thickness,radius};
	const arma::Row<rat::fltp> z{-height/2,-height/2,height/2,height/2};

	// allocate edges
	arma::field<arma::Mat<rat::fltp> > Re(radii.n_elem);

	// walk over edges
	for(arma::uword i=0;i<radii.n_elem;i++){
		Re(i) = arma::join_vert(
			radii(i)*arma::sin(theta), 
			radii(i)*arma::cos(theta), 
			arma::Row<rat::fltp>(theta.n_elem,arma::fill::value(z(i))));
	}

	// create nodes
	const arma::Mat<rat::fltp> Rn = arma::reshape(rat::cmn::Extra::field2mat(Re),3,4*(num_elem_theta+1));

	// elements
	arma::Mat<arma::uword> n(2*radii.n_elem, num_elem_theta);

	for(arma::uword i=0;i<num_elem_theta;i++){
		n.col(i) = arma::join_vert(
			arma::regspace<arma::Col<arma::uword> >(i*4,1,i*4+3),
			arma::regspace<arma::Col<arma::uword> >((i+1)*4,1,(i+1)*4+3));
	}

	// current
	const arma::Row<rat::fltp> Ie(n.n_cols,arma::fill::value(current));

	// create gmsh
	// const rat::cmn::ShGmshFilePr gmsh = rat::cmn::GmshFile::create("solenoid.gmsh");
	// gmsh->write_nodes(Rn);
	// gmsh->write_elements(n);

	// create targets on a plane
	const arma::Col<rat::fltp> xa = arma::linspace<arma::Col<rat::fltp> >(-2*(radius+thickness),2*(radius+thickness),500);
	const arma::Col<rat::fltp> za = arma::linspace<arma::Col<rat::fltp> >(-2*(radius+thickness),2*(radius+thickness),501);
	arma::Mat<rat::fltp> Rt(3,xa.n_elem*za.n_elem); arma::uword cnt = 0;
	for(arma::uword i=0;i<za.n_elem;i++){
		for(arma::uword j=0;j<xa.n_elem;j++){
			Rt.col(cnt++) = arma::Col<rat::fltp>::fixed<3>{xa(j),RAT_CONST(0.0),za(i)};
		}
	}

	// create coil
	const rat::fmm::ShSourceCoilPr src = rat::fmm::SourceCoil::create(Rn,n,Ie);

	// create target plane
	const rat::fmm::ShMgnTargetsPr tar = rat::fmm::MgnTargets::create(Rt);
	tar->set_field_type("AB",{3,3});

	// create logger
	const rat::cmn::ShLogPr lg = rat::cmn::Log::create();

	// create mlfmm
	const rat::fmm::ShMlfmmPr myfmm = rat::fmm::Mlfmm::create(src,tar);

	// create settings
	const rat::fmm::ShSettingsPr settings = myfmm->settings();
	settings->set_min_levels(2);
	settings->set_num_exp(7);
	settings->set_num_refine(50);
	settings->set_allow_subdivision(true);
	// settings->set_enable_fmm(false);

	// setup mlfmm
	myfmm->setup(lg);

	// run multipole method
	myfmm->calculate(lg);
	// myfmm->calculate_direct();

	// get field
	const arma::Mat<rat::fltp> A = tar->get_field('A');
	const arma::Mat<rat::fltp> B = tar->get_field('B');



	std::cout<<arma::join_vert(A,B).t().eval().save("output.csv",arma::csv_ascii)<<std::endl;


	// return
	return 0;
}