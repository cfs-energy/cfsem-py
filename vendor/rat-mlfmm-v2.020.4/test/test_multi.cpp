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

// specific headers
#include "rat/common/error.hh"
#include "rat/common/extra.hh"
#include "mgntargets.hh"
#include "currentsources.hh"
#include "soleno.hh"
#include "mlfmm.hh"
#include "currentmesh.hh"
#include "multisources.hh"
#include "multitargets2.hh"

// main
int main(){
	// settings
	const arma::uword Nta = 1000;
	const arma::uword Ntr = 1200;
	const arma::uword num_exp = 5;
	const arma::uword num_refine = 240;

	// geometry
	const rat::fltp Rin = 0.08;
	const rat::fltp Rout = 0.1;
	const rat::fltp height = 0.02;
	const rat::fltp hsplit = 0.03;
	const rat::fltp J = 400e6;
	const arma::uword num_turns = 1000;
	const rat::fltp dl = 2e-3;

	// tolerance
	const rat::fltp tol = 2e-2;
	
	// number of elements for soleno
	const arma::uword num_layer = 5;

	// number of elements for mlfmm
	const arma::uword num_rad = std::max(2,(int)std::ceil((Rout-Rin)/dl));
	const arma::uword num_height = std::max(2,(int)std::ceil(height/dl));
	const arma::uword num_azym = std::max(2,(int)std::ceil((2*arma::Datum<rat::fltp>::pi*Rout)/dl));

	// create logger
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();	

	// radial target coordinates
	arma::Mat<rat::fltp> Rtr(3,Ntr,arma::fill::zeros);
	Rtr.row(0) = arma::linspace<arma::Row<rat::fltp> >(0,0.2,Ntr);

	// axial target coordinates
	arma::Mat<rat::fltp> Rta(3,Nta,arma::fill::zeros);
	Rta.row(0).fill(0.07);
	Rta.row(2) = arma::linspace<arma::Row<rat::fltp> >(-0.15,0.15,Nta);

	// targets on line in radial direction
	rat::fmm::ShTargetPointsPr tar1 = rat::fmm::MgnTargets::create(Rta);
	tar1->set_field_type('B',3);
	rat::fmm::ShTargetPointsPr tar2 = rat::fmm::MgnTargets::create(Rtr);
	tar2->set_field_type('B',3);

	// create source solenoids
	rat::fmm::ShCurrentSourcesPr src1 = rat::fmm::CurrentSources::create();
	src1->setup_solenoid(Rin,Rout,height,num_rad,num_height,num_azym,J);
	src1->translate(0,0,-hsplit/2);
	rat::fmm::ShCurrentSourcesPr src2 = rat::fmm::CurrentSources::create();
	src2->setup_solenoid(Rin,Rout,height,num_rad,num_height,num_azym,J);
	src2->translate(0,0,hsplit/2);

	// create multi-targets
	rat::fmm::ShMultiTargets2Pr tar = rat::fmm::MultiTargets2::create();
	tar->add_targets(tar1);
	tar->add_targets(tar2);

	// create multisource
	rat::fmm::ShMultiSourcesPr src = rat::fmm::MultiSources::create();
	src->add_sources(src1);
	src->add_sources(src2);

	// create mlfmm
	rat::fmm::ShMlfmmPr myfmm = rat::fmm::Mlfmm::create(src,tar);
	myfmm->settings()->set_num_exp(num_exp);
	myfmm->settings()->set_num_refine(num_refine);
	myfmm->settings()->set_parallel_s2t(false);

	// call setup function on sources and targets
	src->setup_sources(); 
	tar->setup_targets();

	// setup mlfmm
	myfmm->setup(lg);

	// run multipole method
	myfmm->calculate(lg);

	// get results
	arma::Mat<rat::fltp> Bfmma = tar1->get_field('B');
	arma::Mat<rat::fltp> Bfmmb = tar2->get_field('B');

	// with soleno
	lg->msg(2,"%s%sSOLENO COMPARISON%s\n",KBLD,KGRN,KNRM);

	// setup soleno calculation
	rat::fmm::ShSolenoPr sol = rat::fmm::Soleno::create();
	sol->add_solenoid(Rin,Rout,-height/2-hsplit/2,height/2-hsplit/2,J*(Rout-Rin)*height/num_turns,num_turns,num_layer);
	sol->add_solenoid(Rin,Rout,-height/2+hsplit/2,height/2+hsplit/2,J*(Rout-Rin)*height/num_turns,num_turns,num_layer);

	// calculate at target points
	arma::Mat<rat::fltp> Bsola = sol->calc_B(Rta);
	arma::Mat<rat::fltp> Bsolb = sol->calc_B(Rtr);

	// compare soleno to direct
	rat::fltp sol2fmma = arma::max(arma::max(arma::abs(Bsola-Bfmma)/arma::max(rat::cmn::Extra::vec_norm(Bsola)),1));
	rat::fltp sol2fmmb = arma::max(arma::max(arma::abs(Bsolb-Bfmmb)/arma::max(rat::cmn::Extra::vec_norm(Bsolb)),1));

	// output difference
	lg->msg("difference tar1 is %s%2.2f [pct]%s\n",KYEL,sol2fmma*100,KNRM);
	lg->msg("difference tar2 is %s%2.2f [pct]%s\n",KYEL,sol2fmmb*100,KNRM);
	lg->msg(-2,"\n");

	// display results
	lg->msg(2,"%s%sSUMMARY%s\n",KBLD,KGRN,KNRM);

	// checking if vector potential descending
	if(sol2fmma<tol && sol2fmmb<tol){
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