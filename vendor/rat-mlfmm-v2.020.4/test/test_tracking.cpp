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
#include "currentmesh.hh"

#include "ilist.hh"
#include "trackinggrid.hh"
#include "nodelevel.hh"
#include "settings.hh"

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

	// radial target coordinates
	arma::Mat<rat::fltp> Rtr(3,Nt,arma::fill::zeros), Rta(3,Nt,arma::fill::zeros);
	Rtr.row(0) = arma::linspace<arma::Row<rat::fltp> >(0,0.2,Nt);

	// axial target coordinates
	Rta.row(0).fill(0.09);
	Rta.row(2) = arma::linspace<arma::Row<rat::fltp> >(-0.1,0.1,Nt);

	// combine coords
	arma::Mat<rat::fltp> Rt = arma::join_horiz(Rtr,Rta);

	// targets used to setup the tracking grid
	rat::fmm::ShTargetPointsPr tar = rat::fmm::MgnTargets::create(Rt);
	tar->set_field_type('B',3);


	// create sources
	rat::fmm::ShCurrentSourcesPr src = rat::fmm::CurrentSources::create();
	src->setup_solenoid(Rin,Rout,height,num_rad,num_height,num_azym,J);

	// timer
	arma::wall_clock timer;


	// SETUP TRACKING GRID
	// display results
	lg->msg(2,"%s%sSETUP TRACKING%s\n",KBLD,KGRN,KNRM);

	// set timer
	timer.tic();

	// create settings
	rat::fmm::ShSettingsPr stngs = rat::fmm::Settings::create();
	stngs->set_num_exp(num_exp);
	stngs->set_num_refine(num_refine);
	stngs->set_direct(rat::fmm::DirectMode::NEVER); // if using direct be carefull with fmm_setup
	stngs->set_single_threaded();

	// call setup function on sources and targets
	src->setup_sources(); tar->setup_targets();

	// create interaction list object
	rat::fmm::ShIListPr ilist = rat::fmm::IList::create(stngs);

	// setup oct-tree grid
	rat::fmm::ShTrackingGridPr grid = rat::fmm::TrackingGrid::create();
	grid->set_settings(stngs);
	grid->set_ilist(ilist);
	grid->set_sources(src);
	grid->set_targets(tar);

	// create root nodelevel object
	rat::fmm::ShNodeLevelPr root = rat::fmm::NodeLevel::create(stngs,ilist,grid);

	// setup interaction lists
	ilist->setup(); ilist->display(lg);
	
	// call setup functions
	grid->setup(lg);
	grid->setup_matrices(); 
	grid->display(lg);

	// setup tree
	root->setup(lg);

	// run multipole method recursively from root
	grid->sort_srctar();
	root->run_mlfmm(lg);
	grid->unsort_srctar();

	// time
	rat::fltp setup_time = timer.toc();

	// go back
	lg->msg(-2,"\n");

	// TRACKING
	lg->msg(2,"%s%sTRACK%s\n",KBLD,KGRN,KNRM);

	// set timer
	timer.tic();

	// targets used to test the tracking grid
	rat::fmm::ShTargetPointsPr tar2 = rat::fmm::MgnTargets::create(Rta);
	tar2->set_field_type('B',3);

	// calculate field at new set of target points
	tar2->allocate();
	grid->fmm_setup();// setup fmm (sorts sources and targets)
	grid->sort_srctar();
	grid->calc_field(tar2);
	grid->unsort_srctar();
	grid->fmm_post();// finish tracking (unsorts sources and targets)

	// get results
	arma::Mat<rat::fltp> Bfmm = tar2->get_field('B');
	rat::fltp interp_time = timer.toc();

	// go back
	lg->msg(-2,"\n");


	// RESULTS
	// display results
	lg->msg(2,"%s%sSUMMARY%s\n",KBLD,KGRN,KNRM);

	// setup soleno calculation
	rat::fmm::ShSolenoPr sol = rat::fmm::Soleno::create();
	sol->set_solenoid(Rin,Rout,-height/2,height/2,J*(Rout-Rin)*height/num_turns,num_turns,num_layer);

	// calculate at target points
	arma::Mat<rat::fltp> Bsol = sol->calc_B(Rta);

	// compare soleno to direct
	rat::fltp sol2fmm = arma::max(arma::max(arma::abs(Bsol-Bfmm)/arma::max(rat::cmn::Extra::vec_norm(Bsol)),1));

	// checking if vector potential descending
	if(sol2fmm<tol){
		lg->msg("= accuracy magnetic field: %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy magnetic field: %sNOT OK%s\n",KRED,KNRM);
	}
	lg->msg("= difference with Soleno %s%2.2f%s [pct]\n",KYEL,sol2fmm*100,KNRM);
	lg->msg("= setup time: %s%2.2f%s [s]\n", KYEL, setup_time, KNRM);
	lg->msg("= interpolation time: %s%2.2f%s [s]\n", KYEL, interp_time, KNRM);

	// final check
	if(sol2fmm>=tol)rat_throw_line("difference in magnetic field exceeds tolerance");

	// go back
	lg->msg(-2,"\n");

	// statistics
	//myfmm->display(lg);

	// return
	return 0;
}