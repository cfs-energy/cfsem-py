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

// system header
#include <armadillo>
#include <cmath>

// multipole method header
#include "mlfmm.hh"
#include "settings.hh"
#include "currentsources.hh"
#include "mgntargets.hh"

// MEX headers
#include "armamex.hh"

// MEX entry function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

	// check if there is at least one input
	if(nrhs<1)mexErrMsgTxt("at least one input required");

	// persistent storage for the mlfmm class
	static rat::fmm::ShMlfmmPr mlfmm;
	static rat::fmm::ShMgnTargetsPr targets;
	static rat::fmm::ShCurrentSourcesPr sources;
	static rat::cmn::ShLogPr lg;

	// get input
	const std::string assignment = mxArrayToString(prhs[0]);

	// setup step
	if(assignment.compare("setup")==0){
		// no further inputs and no outputs
		if(nrhs!=5)mexErrMsgTxt("3 inputs required");
		if(nlhs!=1)mexErrMsgTxt("1 output required");

		// create log
		lg = rat::cmn::NullLog::create();

		// create the required objects
		mlfmm = rat::fmm::Mlfmm::create();

		// some settings
		const rat::fltp num_refine = armaGetScalar<rat::fltp>(prhs[1]);
		const arma::uword num_exp = armaGetScalar<arma::uword>(prhs[2]);
		const bool large_ilist = armaGetScalar<bool>(prhs[3]);
		const bool allow_gpu = armaGetScalar<bool>(prhs[4]);

		// set to settings
		mlfmm->settings()->set_num_refine(num_refine);
		mlfmm->settings()->set_num_exp(num_exp);
		mlfmm->settings()->set_large_ilist(large_ilist);
		mlfmm->settings()->set_enable_gpu(allow_gpu);

		// define a cleanup function in case matlab exits
		mexAtExit([](){mlfmm.reset();targets.reset();sources.reset();lg.reset();});
	}

	// setup step
	else if(assignment.compare("enable_logging")==0){
		lg = rat::cmn::Log::create();
	}

	// set sources
	else if(assignment.compare("set_sources")==0){
		// check number of inputs and outputs
		if(nrhs!=6)mexErrMsgTxt("5 inputs required");
		if(nlhs!=1)mexErrMsgTxt("1 output required");

		// capture armadillo matrices
		const arma::Mat<rat::fltp> Rs = armaGetPr(prhs[1]);
		const arma::Mat<rat::fltp> dRs = armaGetPr(prhs[2]);
		const arma::Mat<rat::fltp> Is = armaGetPr(prhs[3]);
		const arma::Mat<rat::fltp> epss = armaGetPr(prhs[4]);
		const bool van_lanen = armaGetScalar<bool>(prhs[5]);

		// check sizes
		if(Rs.n_rows!=3)mexErrMsgTxt("source point coordinates Rs must have 3 rows");
		if(dRs.n_rows!=3)mexErrMsgTxt("source point vectors dRs must have 3 rows");
		if(Is.n_rows!=1)mexErrMsgTxt("current vectors Is must have 1 row");
		if(epss.n_rows!=1)mexErrMsgTxt("softening vectors epss must have 1 row");
		if(Rs.n_cols!=dRs.n_cols)mexErrMsgTxt("coordinate Rs and vectors dRs must have same number of columns");
		if(Rs.n_cols!=Is.n_cols)mexErrMsgTxt("coordinate Rs and currents Is must have same number of columns");
		if(Rs.n_cols!=Is.n_cols)mexErrMsgTxt("coordinate Rs and softening epss must have same number of columns");

		// create sources
		sources = rat::fmm::CurrentSources::create();
		sources->set_coords(Rs); 
		sources->set_direction(dRs);
		sources->set_currents(Is);
		sources->set_softening(epss);
		sources->set_van_Lanen(van_lanen);

		// add sources to mlfmm
		mlfmm->set_sources(sources);
	}

	// set sources
	else if(assignment.compare("set_targets")==0){
		// check number of inputs and outputs
		if(nrhs!=2)mexErrMsgTxt("2 inputs required");
		if(nlhs!=1)mexErrMsgTxt("1 output required");

		// get coordinates
		const arma::Mat<rat::fltp> Rt = armaGetPr(prhs[1]);

		// check input
		if(Rt.n_rows!=3)mexErrMsgTxt("target point coordinates Rs must have 3 rows");

		// create targets
		targets = rat::fmm::MgnTargets::create();
		targets->set_target_coords(Rt);
		targets->set_field_type("AB",arma::Row<arma::uword>{3,3});

		// add sources to mlfmm
		mlfmm->set_targets(targets);
	}

	// setup tree structure
	else if(assignment.compare("setup_tree")==0){
		// check number of inputs and outputs
		if(nrhs!=1)mexErrMsgTxt("1 inputs required");
		if(nlhs!=1)mexErrMsgTxt("1 output required");

		// run the setup function
		try{
			mlfmm->setup(lg);
		}catch(const rat_error& err){
			mexErrMsgTxt(err.what());
		}
	}

	// setup tree structure
	else if(assignment.compare("update_currents")==0){
		// check number of inputs and outputs
		if(nrhs!=2)mexErrMsgTxt("2 inputs required");
		if(nlhs!=1)mexErrMsgTxt("1 output required");

		// capture armadillo matrices
		const arma::Mat<rat::fltp> Is = armaGetPr(prhs[1]);
		
		// check sizes
		if(Is.n_rows!=1)mexErrMsgTxt("current vectors Is must have 1 row");

		// update current vector
		sources->set_currents(Is);
	}

	// setup tree structure
	else if(assignment.compare("update_direction")==0){
		// check number of inputs and outputs
		if(nrhs!=2)mexErrMsgTxt("2 inputs required");
		if(nlhs!=1)mexErrMsgTxt("1 output required");

		// capture armadillo matrices
		const arma::Mat<rat::fltp> dR = armaGetPr(prhs[1]);

		// check sizes
		if(dR.n_rows!=3)mexErrMsgTxt("direction vectors Is must have 3 rows");

		// update current vector
		sources->set_direction(dR);
	}

	// run the calculation
	else if(assignment.compare("calculate_mlfmm")==0){
		// check number of inputs and outputs
		if(nrhs!=1)mexErrMsgTxt("1 inputs required");
		if(nlhs!=1)mexErrMsgTxt("1 output required");

		// run the calculation
		try{
			mlfmm->calculate(lg);
		}catch(const rat_error& err){
			mexErrMsgTxt(err.what());
		}
	}

	// run the calculation direct for comparison
	else if(assignment.compare("calculate_direct")==0){
		// check number of inputs and outputs
		if(nrhs!=1)mexErrMsgTxt("1 inputs required");
		if(nlhs!=1)mexErrMsgTxt("1 output required");

		// run the calculation
		try{
			mlfmm->calculate_direct(lg);
		}catch(const rat_error& err){
			mexErrMsgTxt(err.what());
		}
	}

	// get the resulting field
	else if(assignment.compare("get_field")==0){
		// check number of inputs and outputs
		if(nrhs!=1)mexErrMsgTxt("1 inputs required");
		if(nlhs!=3)mexErrMsgTxt("3 outputs required");

		// get vector potential and flux density
		arma::Mat<double> A, B;
		try{
			A = targets->get_field('A');
			B = targets->get_field('B');
		}catch(const rat_error& err){
			mexErrMsgTxt(err.what());
		}

		// transfer output field
		plhs[1] = armaCreateMxMatrix(A.n_rows,A.n_cols);
		armaSetPr(plhs[1], A);
		plhs[2] = armaCreateMxMatrix(B.n_rows,B.n_cols);
		armaSetPr(plhs[2], B);
	}

	// cleanup
	else if(assignment.compare("cleanup")==0){
		// cleaning up
		mlfmm.reset();
		sources.reset();
		targets.reset();
		lg.reset();
	}

	// command not recognized
	else{
		mexErrMsgTxt("assignment not recognized");
	}

	// output code
	arma::Row<double> flag{0};
	plhs[0] = armaCreateMxMatrix(1,1);
	armaSetPr(plhs[0], flag);

	// return to matlab
	return;
}