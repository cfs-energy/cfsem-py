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

/* The code in this file tests all fmm matrices by performing all
the necessary steps in the fast multipole method.

	Sources->Mp1->Mp2->Lp1->Lp2->Targets

After each step the field of the Multipole or Localpole is 
evaluated at the target points and cross-checked with 
the directly calculated field (i.e. Sources->Targets). 
This code also serves as an example to illustrate the matrices 
and steps of the multipole method. */

// general headers
#include <armadillo>
#include <iostream>
#include <cmath>

// specific headers
#include "rat/common/error.hh"
#include "rat/common/extra.hh"
#include "spharm.hh" 
#include "fmmmat.hh"
#include "stmat.hh"
#include "savart.hh"

// form multipole from sources
arma::Mat<std::complex<rat::fltp> > So2Mp(
	const int num_exp,
	const arma::Mat<rat::fltp> &Rmp, 
	const arma::Mat<rat::fltp> &Rs, 
	const arma::Mat<rat::fltp> &Ieff,
	const arma::Mat<rat::fltp> &Meff){

	// calculate relative position
	const arma::Mat<rat::fltp> rel_R = Rs.each_col()-Rmp;

	// setup So2Mp matrix
	rat::fmm::StMat_So2Mp_J Ms2m(num_exp,rel_R);
	rat::fmm::StMat_So2Mp_M Ms2m_m(num_exp,rel_R);

	// Apply matrix to sources and return multipole
	return Ms2m.apply(Ieff) + Ms2m_m.apply(Meff);
}

// translate multipole to new location
arma::Mat<std::complex<rat::fltp> > Mp2Mp(
	const int num_exp,
	const arma::Mat<rat::fltp> &Rs, 
	const arma::Mat<rat::fltp> &Rt, 
	const arma::Mat<std::complex<rat::fltp> > &Mp){

	// calculate relative position
	const arma::Mat<rat::fltp> rel_R = Rs-Rt;

	// setup Mp2MP matrix
	rat::fmm::FmmMat_Mp2Mp Mm2m(num_exp,rel_R);

	// Apply matrix to multipole and return new multipole
	return Mm2m.apply(0,Mp);
}

// convert to localpole
arma::Mat<std::complex<rat::fltp> > Mp2Lp(
	const int num_exp,
	const arma::Mat<rat::fltp> &Rs,
	const arma::Mat<rat::fltp> &Rt,
	const arma::Mat<std::complex<rat::fltp> > &Mp){

	// calculate relative position
	const arma::Mat<rat::fltp> rel_R = Rs-Rt;

	// setup Mp2MP matrix
	rat::fmm::FmmMat_Mp2Lp Mm2l(num_exp,rel_R);

	// Apply matrix to multipole and return new multipole
	return Mm2l.apply(0,Mp);
}

// convert to localpole
arma::Mat<std::complex<rat::fltp> > Lp2Lp(
	const int num_exp,
	const arma::Mat<rat::fltp> &Rs,
	const arma::Mat<rat::fltp> &Rt,
	const arma::Mat<std::complex<rat::fltp> > &Lp){

	// calculate relative position
	const arma::Mat<rat::fltp> rel_R = Rs-Rt;

	// setup Mp2MP matrix
	rat::fmm::FmmMat_Lp2Lp Ml2l(num_exp,rel_R);

	// Apply matrix to multipole and return new multipole
	return Ml2l.apply(0,Lp);
}

// calculate vector potential from multipole
arma::Mat<rat::fltp> Mp2Ta_A(
	const int num_exp,
	const arma::Col<rat::fltp>::fixed<3> &Rmp,
	const arma::Mat<std::complex<rat::fltp> > &Mp,
	const arma::Mat<rat::fltp> &Rt){

	// calculate relative position
	const arma::Mat<rat::fltp> rel_R = Rt.each_col()-Rmp;

	// setup Mp2Ta matrix for vector potential
	rat::fmm::StMat_Mp2Ta Mm2t_a(num_exp,rel_R);

	// Apply matrix and return vector potential
	return 1e-7*Mm2t_a.apply(Mp);
}

// calculate vector potential from localpole
arma::Mat<rat::fltp> Lp2Ta_A(
	const int num_exp,
	const arma::Col<rat::fltp>::fixed<3> &Rlp,
	const arma::Mat<std::complex<rat::fltp> > &Lp,
	const arma::Mat<rat::fltp> &Rt){

	// calculate relative position
	const arma::Mat<rat::fltp> rel_R = Rt.each_col()-Rlp;

	// setup Mp2Ta matrix for vector potential
	rat::fmm::StMat_Lp2Ta Ml2t_a(num_exp,rel_R);

	// Apply matrix and return vector potential
	return 1e-7*Ml2t_a.apply(Lp);
}

// calculate vector potential from localpole
arma::Mat<rat::fltp> Lp2Ta_H(
	const int num_exp,
	const arma::Col<rat::fltp>::fixed<3> &Rlp,
	const arma::Mat<std::complex<rat::fltp> > &Lp,
	const arma::Mat<rat::fltp> &Rt){

	// calculate relative position
	const arma::Mat<rat::fltp> rel_R = Rt.each_col()-Rlp;

	// setup Mp2Ta matrix for vector potential
	rat::fmm::StMat_Lp2Ta_Curl Ml2t_h(num_exp,rel_R);

	// Apply matrix and return magnetic field
	return Ml2t_h.apply(Lp)/(4*arma::Datum<rat::fltp>::pi);
}

// compare the results from the multipole method and the direct calculation
rat::fltp compare(
	const arma::Mat<rat::fltp> &Adir,
	const arma::Mat<rat::fltp> &Afmm){

	const arma::Mat<rat::fltp> Adiff = Adir-Afmm;
	const arma::Mat<rat::fltp> Adir_mag = rat::cmn::Extra::vec_norm(Adir);
	const arma::Mat<rat::fltp> Err = Adiff.each_row()/Adir_mag;

	const rat::fltp diff = 100*arma::max(arma::max(arma::abs(Err)));
	return diff;
}

// main
int main(){
	// settings
	const arma::uword num_sources = 20; // number of source elements 
	const arma::uword num_targets = 10; // number of target points
	const int num_exp_min = 2; // minimal number of expansions to test
	const rat::fltp radius = 10e-3; // radius of source sphere in [m]
	const rat::fltp current = 400; // element current [A]

	// heuristics for maximum number of expansions
	int num_exp_max;
	#ifdef RAT_DOUBLE_PRECISION
		num_exp_max = 10;
	#else
		num_exp_max = 8;
	#endif

	// set seed this is recommended to always have same unit test
	arma::arma_rng::set_seed(1001);

	// create logger
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();	

	// report to user
	lg->msg(2,"%s%sDESCRIPTION%s\n",KBLD,KGRN,KNRM);
	lg->msg("%sTest script performing basic MLFMM operations%s\n",KCYN,KNRM);
	lg->msg("%sSources -> Mp1 -> Mp2 -> Lp1 -> Lp2 -> Targets%s\n",KCYN,KNRM);
	lg->msg(-2,"\n");

	// create nodes on a circular path
	const arma::Row<rat::fltp> t = arma::linspace<arma::Row<rat::fltp> >(0,2*arma::Datum<rat::fltp>::pi,num_sources+1);
	const arma::Mat<rat::fltp> Rn = arma::join_vert(radius*arma::sin(t),radius*arma::cos(t),0.1*radius*arma::sin(t));

	// elements
	const arma::Mat<rat::fltp> Rs = (Rn.tail_cols(num_sources) + Rn.head_cols(num_sources))/2;
	const arma::Mat<rat::fltp> dRs = arma::diff(Rn,1,1);
	const arma::Row<rat::fltp> Is = arma::Row<rat::fltp>(num_sources,arma::fill::ones)*current;

	// make target positions
	arma::Mat<rat::fltp> Rt = 20e-3*arma::Mat<rat::fltp>(3,num_targets,arma::fill::randu)-10e-3; 
	Rt.each_col() += arma::Col<rat::fltp>::fixed<3>{0.1,0.05,0.02};

	// random magnetic moment
	const arma::Mat<rat::fltp> Meff = 1e-1*(arma::Mat<rat::fltp>(3,num_sources,arma::fill::randu)-0.5);

	// perform direct calculation for comparison
	const arma::Mat<rat::fltp> Adir = 
		rat::fmm::Savart::calc_I2A(Rs,dRs.each_row()%Is,Rt,false) + 
		rat::fmm::Savart::calc_M2A(Rs,Meff,Rt,false);
	const arma::Mat<rat::fltp> Hdir = 
		rat::fmm::Savart::calc_I2H(Rs,dRs.each_row()%Is,Rt,false) +
		rat::fmm::Savart::calc_M2H(Rs,Meff,Rt,false);

	// position of first multipole (near source positions)
	const arma::Col<rat::fltp>::fixed<3> Rmp1 = Rs.col(0); // deliberately on top of a source

	// position of second multipole (near source positions)
	arma::Col<rat::fltp>::fixed<3> Rmp2 = Rmp1; 
	Rmp2(0) += 0.01;

	// position of first localpole (near target positions)
	arma::Col<rat::fltp>::fixed<3> Rlp1 = arma::mean(Rt,1); 
	Rlp1(0) += 0.01; Rlp1(1) -= 0.005; Rlp1(2) += 0.001;
	
	// position of second localpole (near target positions)
	//arma::Mat<rat::fltp> Rlp2 = arma::mean(Rt,1); 
	const arma::Col<rat::fltp>::fixed<3> Rlp2 = Rt.col(0); // deliberately on top of a target

	// display positions
	lg->msg(2,"%s%sCOORDINATES%s\n",KBLD,KGRN,KNRM);
	lg->msg("Multipole 1 is at: %+02.2e, %+02.2e, %+02.2e\n",Rmp1(0),Rmp1(1),Rmp1(2));
	lg->msg("Multipole 2 is at: %+02.2e, %+02.2e, %+02.2e\n",Rmp2(0),Rmp2(1),Rmp2(2));
	lg->msg("Localpole 1 is at: %+02.2e, %+02.2e, %+02.2e\n",Rlp1(0),Rlp1(1),Rlp1(2));
	lg->msg("Localpole 2 is at: %+02.2e, %+02.2e, %+02.2e\n",Rlp2(0),Rlp2(1),Rlp2(2));
	lg->msg(-2,"\n");

	// setup number of expanions
	const arma::Row<int> num_exp = arma::regspace<arma::Row<int> >(num_exp_min,num_exp_max);

	// allocate accuracy arrays
	arma::Row<rat::fltp> accuracy_So2Mp_A(num_exp.n_elem);
	arma::Row<rat::fltp> accuracy_Mp2Mp_A(num_exp.n_elem);
	arma::Row<rat::fltp> accuracy_Mp2Lp_A(num_exp.n_elem);
	arma::Row<rat::fltp> accuracy_Mp2Lp_H(num_exp.n_elem);
	arma::Row<rat::fltp> accuracy_Lp2Lp_A(num_exp.n_elem);
	arma::Row<rat::fltp> accuracy_Lp2Lp_H(num_exp.n_elem);

	// run multipole method
	lg->msg(2,"%s%sRAT-MLFMM CALCULATIONS%s\n",KBLD,KGRN,KNRM);

	// increment number of expansions
	for (arma::uword i=0;i<num_exp.n_cols;i++){
		// header
		lg->msg(2,"%snumber of expansions: %s%i%s\n",KBLU,KGRN,num_exp(i),KNRM);

		// form multipole from sources
		const arma::Mat<std::complex<rat::fltp> > Mp1 = So2Mp(num_exp(i),Rmp1,Rs,dRs.each_row()%Is,Meff);

		// calculate field from multipole (1) at target positions
		const arma::Mat<rat::fltp> Afmm1 = Mp2Ta_A(num_exp(i),Rmp1,Mp1,Rt);

		// Intermediate comparison to direct calculation
		accuracy_So2Mp_A(i) = compare(Adir,Afmm1);
		lg->msg("So2Mp A: %s%02.4f%s [pct]\n",KYEL,accuracy_So2Mp_A(i),KNRM);

		// translate mulitpole to new position
		const arma::Mat<std::complex<rat::fltp> > Mp2 = Mp2Mp(num_exp(i),Rmp1,Rmp2,Mp1);

		// calculate field from multipole (2) at all target positions
		const arma::Mat<rat::fltp> Afmm2 = Mp2Ta_A(num_exp(i),Rmp2,Mp2,Rt);

		// Intermediate comparison to direct calculation
		accuracy_Mp2Mp_A(i) = compare(Adir,Afmm2);
		lg->msg("Mp2Mp A: %s%02.4f%s [pct]\n",KYEL,accuracy_Mp2Mp_A(i),KNRM);

		// convert to local pole
		const arma::Mat<std::complex<rat::fltp> > Lp1 = Mp2Lp(num_exp(i),Rmp2,Rlp1,Mp2);

		// calculate field from localpole (1) at all target positions
		const arma::Mat<rat::fltp> Afmm3 = Lp2Ta_A(num_exp(i),Rlp1,Lp1,Rt);
		const arma::Mat<rat::fltp> Hfmm3 = Lp2Ta_H(num_exp(i),Rlp1,Lp1,Rt);

		// Intermediate comparison to direct calculation
		accuracy_Mp2Lp_A(i) = compare(Adir,Afmm3);
		accuracy_Mp2Lp_H(i) = compare(Hdir,Hfmm3);
		lg->msg("Mp2Lp A: %s%02.4f%s [pct]\n",KYEL,accuracy_Mp2Lp_A(i),KNRM);
		lg->msg("Mp2Lp H: %s%02.4f%s [pct]\n",KYEL,accuracy_Mp2Lp_H(i),KNRM);

		// translate localpole
		const arma::Mat<std::complex<rat::fltp> > Lp2 = Lp2Lp(num_exp(i),Rlp1,Rlp2,Lp1);

		// calculate field from localpole (1) at all target positions
		const arma::Mat<rat::fltp> Afmm4 = Lp2Ta_A(num_exp(i),Rlp2,Lp2,Rt);
		const arma::Mat<rat::fltp> Hfmm4 = Lp2Ta_H(num_exp(i),Rlp2,Lp2,Rt);

		// Intermediate comparison to direct calculation
		accuracy_Lp2Lp_A(i) = compare(Adir,Afmm4);
		accuracy_Lp2Lp_H(i) = compare(Hdir,Hfmm4);
		lg->msg("Lp2Lp A: %s%02.4f%s [pct]\n",KYEL,accuracy_Lp2Lp_A(i),KNRM);
		lg->msg("Lp2Lp H: %s%02.4f%s [pct]\n",KYEL,accuracy_Lp2Lp_H(i),KNRM);

		// insert empty line
		lg->msg(-2,"\n");
	}

	// return
	lg->msg(-2);

	// display results
	// header
	lg->msg(2,"%s%sRESULTS%s\n",KBLD,KGRN,KNRM);

	// check source to multipole
	if(accuracy_So2Mp_A.is_sorted("descend")){
		lg->msg("= accuracy source to multipole (A)...: %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy source to multipole (A)...: %sNOT OK%s\n",KRED,KNRM);
	}	

	// check multipole to multipole
	if(accuracy_Mp2Mp_A.is_sorted("descend")){
		lg->msg("= accuracy multipole to multipole (A): %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy multipole to multipole (A): %sNOT OK%s\n",KRED,KNRM);
	}	

	// check multipole to localpole
	if(accuracy_Mp2Lp_A.is_sorted("descend")){
		lg->msg("= accuracy multipole to localpole (A): %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy multipole to localpole (A): %sNOT OK%s\n",KRED,KNRM);
	}

	// check multipole to localpole
	if(accuracy_Mp2Lp_H.is_sorted("descend")){
		lg->msg("= accuracy multipole to localpole (H): %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy multipole to localpole (H): %sNOT OK%s\n",KRED,KNRM);
	}

	// check multipole to localpole
	if(accuracy_Lp2Lp_A.is_sorted("descend")){
		lg->msg("= accuracy localpole to localpole (A): %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy localpole to localpole (A): %sNOT OK%s\n",KRED,KNRM);
	}

	// check multipole to localpole
	if(accuracy_Lp2Lp_H.is_sorted("descend")){
		lg->msg("= accuracy localpole to localpole (H): %sOK%s\n",KGRN,KNRM);
	}else{
		lg->msg("= accuracy localpole to localpole (H): %sNOT OK%s\n",KRED,KNRM);
	}	

	// new line
	lg->msg(-2,"\n");

	// check convergence
	if(!accuracy_So2Mp_A.is_sorted("descend"))rat_throw_line(
		"source to multipole does not converge with number of expansions");
	if(!accuracy_Mp2Mp_A.is_sorted("descend"))rat_throw_line(
		"multipole to multipole does not converge with number of expansions");
	if(!accuracy_Mp2Lp_A.is_sorted("descend"))rat_throw_line(
		"multipole to localpole for vector potential does not converge with number of expansions");
	if(!accuracy_Mp2Lp_H.is_sorted("descend"))rat_throw_line(
		"multipole to localpole for magnetic field does not converge with number of expansions");
	if(!accuracy_Lp2Lp_A.is_sorted("descend"))rat_throw_line(
		"localpole to localpole for vector potential does not converge with number of expansions");
	if(!accuracy_Lp2Lp_H.is_sorted("descend"))rat_throw_line(
		"localpole to localpole for magnetic field does not converge with number of expansions");

	// all okay return zero
	return 0;
}