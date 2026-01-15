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

// specific headers
#include "error.hh"
#include "rungekutta.hh"

// this test tests runge kutta:
// test ODE: dy/dt = -(1.0/tau)*y
// solution: y(t) = y0 * exp(-t/tau)

// main
int main(){
	// time constants
	const rat::fltp t1 = 0.0;
	const rat::fltp t2 = 20.0;
	const rat::fltp reltol = 1e-2;
	const rat::fltp yystop = 0.1;

	// time constants
	const arma::Row<rat::fltp> tau = {2.0,3.0,4.0,5.0};

	// create log
	const rat::cmn::ShLogPr lg = rat::cmn::Log::create();

	// header
	lg->msg(2,"%s%srunge-kutta test%s\n",KGRN,KBLD,KNRM);

	// create the solver object
	const rat::cmn::ShRungeKuttaPr runge_kutta = 
		rat::cmn::RungeKutta::create(rat::cmn::RungeKutta::IntegrationMethod::HAIRER_WANNER_EDIRK34);
	runge_kutta->set_reltol(1e-6);
	runge_kutta->set_abstol(1e-6);

	// display the method
	runge_kutta->display_butchers_tableau(lg);

	// build the system function
	runge_kutta->set_system_fun([&tau](
		const rat::fltp /*tt*/, const arma::Row<rat::fltp>&yy){

		// calculate change
		const arma::Row<rat::fltp> dy = -yy/tau;

		// return change
		return dy;
	});

	// build the system function
	runge_kutta->set_monitor_fun([&lg](const arma::uword iter, const rat::fltp tt, const rat::fltp dt, const arma::sword event_detected){
		if(iter%400==0){
			lg->msg("%s%s%4s %9s %9s%s\n",KCYN,KBLD,"iter","time","dstep",KNRM);
			lg->hline(4+9+9+2,'=',KCYN,KNRM);
		}
		if(iter%20==0 || event_detected>=0)lg->msg("%s%04llu %+9.2e %+9.2e %s%s%s\n",KCYN,iter,tt,dt,KYEL,event_detected>=0 ? "(root found)" : "", KNRM);
		
		// all ok
		return 0;
	});

	// super basic event fun
	runge_kutta->set_event_fun([yystop](const rat::fltp tt, const arma::Mat<rat::fltp>&yy){
		return arma::Row<rat::fltp>{arma::max(arma::vectorise(yy)) - yystop};
	});

	// start point
	arma::Row<rat::fltp> y0(tau.n_elem, arma::fill::ones);

	// run integration
	lg->msg(2,"%srunge-kutta integration%s\n",KBLU,KNRM);
	const rat::cmn::RKData solution = runge_kutta->integrate(t1,t2,y0);
	lg->msg(-2,"\n");

	// calculate decay analytically
	const arma::Mat<rat::fltp> M2 = arma::exp(-(arma::repmat(solution.tt,1,tau.n_cols).eval().each_row()/tau));

	// compare
	const rat::fltp error = arma::max(arma::vectorise(arma::abs(solution.yy - M2)/arma::abs(M2)));

	// results
	lg->msg(2,"%sresults%s\n",KBLU,KNRM);
	lg->msg("error: %s%2.3e%s %s%s%s\n",
		KYEL,error,KNRM,error>reltol ? KRED : KGRN, 
		error>reltol ? "NOT OK" : "OK", KNRM);

	// check if solution looks like exponential decay
	if(error>reltol)rat_throw_line("solution exceeds relative tolerance");

	// check if root finding stopped at the right location
	if(!arma::any(arma::abs(arma::vectorise(solution.yy.tail_rows(1)) - yystop)<runge_kutta->get_roottol()))
		rat_throw_line("did not find expected root");

	// done
	lg->msg(-2);
}