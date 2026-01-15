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

// rat common headers
#include "rat/common/error.hh"
#include "rat/common/extra.hh"

// rat mlfmm headers
#include "mgntargets.hh"
#include "currentsources.hh"
#include "soleno.hh"
#include "mlfmm.hh"
#include "currentmesh.hh"

// main
int main(){
	// settings
	const arma::uword Nt = 1000;
	const arma::uword num_exp = 7;
	const arma::uword num_refine = 240;

	// geometry
	const rat::fltp Rin = 0.1;
	const rat::fltp Rout = 0.11;
	const rat::fltp height = 0.1;
	const rat::fltp J = 400e6;
	const arma::uword num_turns = 1000;

	// tolerance
	const rat::fltp tol = 2e-2;

	// number of elements for mlfmm
	//rat::fltp dl = 1.2e-3;
	const rat::fltp dl = 1.2e-3;
	const arma::uword num_rad = std::max(2,(int)std::ceil((Rout-Rin)/dl));
	const arma::uword num_height = std::max(2,(int)std::ceil(height/dl));
	const arma::uword num_azym = std::max(2,(int)std::ceil((2*arma::Datum<rat::fltp>::pi*Rin)/dl));

	// number of elements for soleno
	const arma::uword num_layer = 5;

	// create logger
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();	

	lg->newl();
	lg->msg(2,"%s%sDESCRIPTION%s\n",KBLD,KGRN,KNRM);
	lg->msg("%sValidation script comparing the results%s\n",KCYN,KNRM);
	lg->msg("%sof the mlfmm to that of Soleno. A software%s\n",KCYN,KNRM);
	lg->msg("%spackage for magnetic field calculation%s\n",KCYN,KNRM);
	lg->msg("%sin solenoids developed at the University%s\n",KCYN,KNRM);
	lg->msg("%sof Twente.%s\n",KCYN,KNRM);
	lg->msg(-2,"\n");

	// start setup 
	lg->msg(2,"%s%sGEOMETRY%s\n",KBLD,KGRN,KNRM);
	lg->msg("solenoid with inner radius %s%.2f%s [m]\n",KYEL,Rin,KNRM);
	lg->msg("outer radius %s%.2f%s [m] and height %s%.2f [m]%s\n",KYEL,Rout,KNRM,KYEL,height,KNRM);
	lg->msg("with current density %s%.2e [A/mm^2]%s\n",KYEL,J/1e6,KNRM);
	lg->msg(-2,"\n");

	// radial target coordinates
	arma::Mat<rat::fltp> Rtr(3,Nt,arma::fill::zeros), Rta(3,Nt,arma::fill::zeros);
	Rtr.row(0) = arma::linspace<arma::Row<rat::fltp> >(0,0.2,Nt);

	// axial target coordinates
	Rta.row(0).fill(0.09);
	Rta.row(2) = arma::linspace<arma::Row<rat::fltp> >(-0.1,0.1,Nt);

	// combine coords
	arma::Mat<rat::fltp> Rt = arma::join_horiz(Rtr,Rta);

	// targets on line in radial direction
	rat::fmm::ShTargetPointsPr tar = rat::fmm::MgnTargets::create(Rt);
	tar->set_field_type('B',3);

	// create sources
	rat::fmm::ShCurrentSourcesPr src = rat::fmm::CurrentSources::create();
	src->setup_solenoid(Rin,Rout,height,num_rad,num_height,num_azym,J);

	// create mlfmm
	rat::fmm::ShMlfmmPr myfmm = rat::fmm::Mlfmm::create(src,tar);

	// create settings
	rat::fmm::ShSettingsPr settings = myfmm->settings();
	settings->set_num_exp(num_exp);
	settings->set_num_refine(num_refine);

	// setup mlfmm
	myfmm->setup(lg);

	// run multipole method
	myfmm->calculate(lg);

	//myfmm->calculate_direct();

	// get results
	arma::Mat<rat::fltp> Bfmm = tar->get_field('B');

	// with soleno
	lg->msg(2,"%s%sSOLENO COMPARISON%s\n",KBLD,KGRN,KNRM);

	// setup soleno calculation
	rat::fmm::ShSolenoPr sol = rat::fmm::Soleno::create();
	sol->set_solenoid(Rin,Rout,-height/2,height/2,J*(Rout-Rin)*height/num_turns,num_turns,num_layer);

	// calculate at target points
	arma::Mat<rat::fltp> Bsol = sol->calc_B(Rt);

	// compare soleno to direct
	rat::fltp sol2fmm = arma::max(arma::max(arma::abs(Bsol-Bfmm)/arma::max(rat::cmn::Extra::vec_norm(Bsol)),1));

	// output difference
	lg->msg("difference is %s%2.2f [pct]%s\n",KYEL,sol2fmm*100,KYEL);
	lg->msg(-2,"\n");

	// display results
	lg->msg(2,"%s%sSUMMARY%s\n",KBLD,KGRN,KNRM);

	// checking if vector potential descending
	if(sol2fmm<tol){
		lg->msg("= accuracy magnetic field: %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy magnetic field: %sNOT OK%s\n",KRED,KNRM);
		rat_throw_line("difference in magnetic field exceeds tolerance");
	}

	// go back
	lg->msg(-2,"\n");

	// statistics
	//myfmm->display(lg);

	// return
	return 0;
}