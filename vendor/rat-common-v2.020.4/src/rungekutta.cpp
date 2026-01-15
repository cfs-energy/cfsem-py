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

// include header
#include "rungekutta.hh"

// common headers
#include "error.hh"
#include "parfor.hh"
#include "log.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// constructor
	RungeKutta::RungeKutta(){
		set_butcher_tableau(IntegrationMethod::RUNGE_KUTTA4);
	}

	// constructor
	RungeKutta::RungeKutta(const RungeKutta::IntegrationMethod integration_method){
		set_butcher_tableau(integration_method);
	}

	// factory
	ShRungeKuttaPr RungeKutta::create(){
		return std::make_shared<RungeKutta>();
	}

	// factory
	ShRungeKuttaPr RungeKutta::create(const RungeKutta::IntegrationMethod integration_method){
		return std::make_shared<RungeKutta>(integration_method);
	}

	// set init ial time step
	void RungeKutta::set_dtini(const fltp dtini){
		dtini_ = dtini;
	}

	// set system function
	void RungeKutta::set_system_fun(const RKSysFun& system_fun){
		system_fun_ = system_fun;
	}

	// set monitor function
	void RungeKutta::set_monitor_fun(const RKMonitorFun& monitor_fun){
		monitor_fun_ = monitor_fun;
	}

	// set event function
	void RungeKutta::set_event_fun(const RKEventFun& event_fun){
		event_fun_ = event_fun;
	}

	// setters
	void RungeKutta::set_abstol(const fltp abstol){
		abstol_ = abstol;
	}

	void RungeKutta::set_reltol(const fltp reltol){
		reltol_ = reltol;
	}

	void RungeKutta::set_roottol(const fltp roottol){
		roottol_ = roottol;
	}

	void RungeKutta::set_max_iter(const arma::uword max_iter){
		max_iter_ = max_iter;
	}

	// getters
	fltp RungeKutta::get_abstol()const{
		return abstol_;
	}

	fltp RungeKutta::get_reltol()const{
		return reltol_;
	}

	fltp RungeKutta::get_roottol()const{
		return roottol_;
	}

	arma::uword RungeKutta::get_max_iter()const{
		return max_iter_;
	}


	// number of solutions
	bool RungeKutta::is_embedded()const{
		return b_.n_rows==2;
	}

	// get order
	arma::uword RungeKutta::get_order()const{
		return order_;
	}


	// set butcher table directly
	void RungeKutta::set_butcher_tableau(
		const arma::Mat<fltp> &a,
		const arma::Row<fltp> &b,
		const arma::Col<fltp> &c,
		const std::string& methodname){

		assert(!a.empty()); assert(!b.empty()); assert(!c.empty());
		assert(a.is_square());
		assert(b.n_elem==a.n_cols);
		assert(c.n_elem==a.n_cols);
		assert(a.is_trimatl());

		// set to self
		a_ = a; b_ = b; c_ = c;
		methodname_ = methodname;
	}

	// set butcher table using available integration methods
	// for example classical fourth order runge kutta 
	// can be applied using this representation
	// 0   | 0   0   0   0
	// 1/2 | 1/2 0   0   0
	// 1/2 | 0   1/2 0   0
	// 1   | 0   0   1   0
	// ------------------
	//     | 1/6 1/3 1/3 1/6
	void RungeKutta::set_butcher_tableau(
		const RungeKutta::IntegrationMethod integration_method){

		// get matrices for the integration method
		std::string integration_methodname;
		switch(integration_method){

			// explicit runge kutta methods
			// ====
			// forward euler method
			case IntegrationMethod::EULER_FORWARD:{
				methodname_ = "Forward Euler";
				schemename_ = "Explicit Runge Kutta";
				order_ = 1llu;
				a_ = arma::Mat<fltp>::fixed<1,1>{0.0};
				b_ = arma::Row<fltp>::fixed<1>{1.0};
				c_ = arma::Col<fltp>::fixed<1>{0.0};
				break;
			}

			// Ralston second order method
			case IntegrationMethod::RALSTON:{
				methodname_ = "Ralston";
				schemename_ = "Explicit Runge Kutta";
				order_ = 2llu;
				a_ = arma::Mat<fltp>::fixed<2,2>{0.0,2.0/3,0.0,0.0};
				b_ = arma::Row<fltp>::fixed<2>{1.0/4,3.0/4};
				c_ = arma::Col<fltp>::fixed<2>{0.0,2.0/3};
				break;
			}

			// Runge-Kutta's third order method
			case IntegrationMethod::RUNGE_KUTTA3:{
				methodname_ = "Runge Kutta 3";
				schemename_ = "Embedded Runge Kutta";
				order_ = 3llu;
				a_ = arma::Mat<fltp>::fixed<3,3>{
					0.0,1.0/2,-1.0, 0.0,0.0,2.0, 0.0,0.0,0.0};
				b_ = arma::Row<fltp>::fixed<3>{1.0/6,2.0/6,1.0/6};
				c_ = arma::Col<fltp>::fixed<3>{0.0,0.5,1.0};
				break;
			}

			// classical fourth order runge kutta method
			case IntegrationMethod::RUNGE_KUTTA4:{
				methodname_ = "Runge Kutta 4";
				schemename_ = "Explicit Runge Kutta";
				order_ = 4llu;
				a_ = arma::Mat<fltp>::fixed<4,4>{
					0.0,1.0/2,0.0,0.0, 0.0,0.0,1.0/2,0.0, 
					0.0,0.0,0.0,1.0, 0.0,0.0,0.0,0.0};
				b_ = arma::Row<fltp>::fixed<4>{1.0/6,1.0/3,1.0/3,1.0/6};
				c_ = arma::Col<fltp>::fixed<4>{0.0,1.0/2,1.0/2,1.0};
				break;
			}

			// embedded explicit runge kutta methods
			// ====
			// runge kutta-fehlberg 45
			// by convention we want the first solution to be the most accurate
			case IntegrationMethod::RUNGE_KUTTA_FEHLBERG45:{
				methodname_ = "Runge Kutta Fehlberg 4/5";
				schemename_ = "Embedded Explicit Runge Kutta";
				order_ = 5llu;
				a_ = arma::Mat<fltp>::fixed<6,6>{
					0.0,1.0/4,3.0/32,1932.0/2197,439.0/216,-8.0/27, 
					0.0,0.0,9.0/32,-7200.0/2197,-8.0,2.0, 
					0.0,0.0,0.0,7296.0/2197,3680.0/513,-3544.0/2565, 
					0.0,0.0,0.0,0.0,-845.0/4104,1859.0/4104, 
					0.0,0.0,0.0,0.0,0.0,-11.0/40,
					0.0,0.0,0.0,0.0,0.0,0.0};
				b_ = arma::Mat<fltp>::fixed<2,6>{
					16.0/135,25.0/216, 0,0, 6656.0/12825,1408.0/2565, 
					28561.0/56430,2197.0/4104, -9.0/50,-1.0/5, 2.0/55,0.0};
				c_ = arma::Col<fltp>::fixed<6>{0.0,1.0/5,3.0/10,3.0/5,1.0,7.0/8};
				break;
			}

			// Dormand-Prince
			// Dormand, J.R.; Prince, P.J. (1980). "A family of embedded Runge-Kutta formulae". 
			// Journal of Computational and Applied Mathematics. 6 (1): 19–26. doi:10.1016/0771-050X(80)90013-3.
			case IntegrationMethod::DORMAND_PRINCE:{
				methodname_ = "Dormand Prince";
				schemename_ = "Embedded Explicit Runge Kutta";
				order_ = 5llu;
				a_ = arma::Mat<fltp>::fixed<7,7>{
					0.0,1.0/5,3.0/40,44.0/45,19372.0/6561,9017.0/3168,35.0/384,
					0.0,0.0,9.0/40,-56.0/15,-25360.0/2187,-355.0/33,0.0,
					0.0,0.0,0.0,32.0/9,64448.0/6561,46732.0/5247,500.0/1113,
					0.0,0.0,0.0,0.0,-212.0/729,49.0/176,125.0/192,
					0.0,0.0,0.0,0.0,0.0,-5103.0/18656,-2187.0/6784,
					0.0,0.0,0.0,0.0,0.0,0.0,11.0/84,
					0.0,0.0,0.0,0.0,0.0,0.0,0.0};
				b_ = arma::Mat<fltp>::fixed<2,7>{
					35.0/384,5179.0/57600, 0.0,0.0, 500.0/1113,7571.0/16695, 125.0/192,
					393.0/640, -2187.0/6784,-92097.0/339200, 11.0/84,187.0/2100, 0.0,1.0/40};
				c_ = arma::Col<fltp>::fixed<7>{0.0,1.0/5,3.0/10,4.0/5,8.0/9,1.0,1.0};
				break;
			}



			// Diagonally Implicit Runge–Kutta (DIRK)
			// this type of integration is more stable 
			// when used on stiff differential equations
			// https://www.igpm.rwth-aachen.de/Download/reports/schuetz/hybrid_embedded_dirk.pdf
			// ====
			// Crank-Nicolson
			case IntegrationMethod::CRANK_NICOLSON:{
				methodname_ = "Crank Nicolson";
				schemename_ = "Diagonally Implicit Runge Kutta (DIRK)";
				order_ = 2llu;
				a_ = arma::Mat<fltp>::fixed<2,2>{
					0.0,1.0/2, 0.0,1.0/2};
				b_ = arma::Mat<fltp>::fixed<1,2>{
					1.0/2, 1.0/2};
				c_ = arma::Col<fltp>::fixed<2>{
					0.0, 1.0};
				break;
			}

			
			// Kraaijevanger and spijker
			case IntegrationMethod::KRAAIJEVANGER_SPIJKER:{
				methodname_ = "Kraaijevanger Spijker";
				schemename_ = "Diagonally Implicit Runge Kutta (DIRK)";
				order_ = 2llu;
				a_ = arma::Mat<fltp>::fixed<2,2>{
					1.0/2,-1.0/2, 0.0,2.0};
				b_ = arma::Mat<fltp>::fixed<1,2>{
					-1.0/2, 3.0/2};
				c_ = arma::Col<fltp>::fixed<2>{
					1.0/2, 3.0/2};
				break;
			}

			// Cash
			// J. Cash, Diagonally implicit runge-kutta formulae with 
			// error estimates, J. Inst. Maths Applics 24 (1979) 293–301
			case IntegrationMethod::CASH_EDIRK23:{
				methodname_ = "Cash EDIRK 2/3";
				schemename_ = "Embedded Diagonally Implicit Runge Kutta (EDIRK)";
				order_ = 3llu;
				a_ = arma::Mat<fltp>::fixed<3,3>{
					0.435866521508, 0.2820667320, 1.208496649,
					0.0, 0.435866521508, -0.6443632015, 
					0.0, 0.0, 0.435866521508};
				b_ = arma::Mat<fltp>::fixed<2,3>{
					1.208496649,0.77263013745746, -0.6443632015,0.22736986254254, 0.435866521508,0.0};
				c_ = arma::Col<fltp>::fixed<3>{
					0.435866521508,0.717933260755,1.0};
				break;
			}

			// Al-Rabeh
			// A. H. Al-Rabeh, Embedded dirk methods for the numerical 
			// integration of stiff systems of odes, International 
			// Journal of Computer Mathematics 21 (1987) 65–84
			case IntegrationMethod::AL_RABEH_EDIRK34:{
				methodname_ = "Al-Rabeh EDIRK 3/4";
				schemename_ = "Embedded Diagonally Implicit Runge Kutta (EDIRK)";
				order_ = 4llu;
				a_ = arma::Mat<fltp>::fixed<4,4>{
					0.4358665,-0.4034943,-0.3298751,0.5575315,
					0.0,0.4358665,0.8616364,-0.1930865,
					0.0,0.0,0.4358665,-0.2361781,
					0.0,0.0,0.0,0.4358665};
				b_ = arma::Mat<fltp>::fixed<2,4>{
					0.3153914,0.6307827, 0.1846086,0.1413538, 0.1846086,0.2278634, 0.3153914,0.0};
				c_ = arma::Col<fltp>::fixed<4>{
					0.4358665,0.0323722,0.9676278,0.5641335 };
				break;
			}


			// Hairer and Wanner
			// E. Hairer, G. Wanner, Solving Ordinary Differential 
			// Equations II, Springer Series in Computational Mathematics, 1991
			case IntegrationMethod::HAIRER_WANNER_EDIRK34:{
				methodname_ = "Hairer Wanner EDIRK 3/4";
				schemename_ = "Embedded Diagonally Implicit Runge Kutta (EDIRK)";
				order_ = 4llu;
				a_ = arma::Mat<fltp>::fixed<5,5>{
					1.0/4,1.0/2,17.0/50,371.0/1360,25.0/24,
					0.0,1.0/4,-1.0/25,-137.0/2720,-49.0/48,
					0.0,0.0,1.0/4,15.0/544,125.0/16,
					0.0,0.0,0.0,1.0/4,-85.0/12,
					0.0,0.0,0.0,0.0,1.0/4};
				b_ = arma::Mat<fltp>::fixed<2,5>{
					25.0/24,59.0/48, -49.0/48,-17.0/96, 125.0/16,225.0/32, -85.0/12,-85.0/12, 1.0/4,0.0};
				c_ = arma::Col<fltp>::fixed<5>{1.0/4, 3.0/4, 11.0/20, 1.0/2, 1.0};
				break;
			}

			// other
			default:
				rat_throw_line("integration method not recognized");
		}

		// check matrix
		assert(!a_.empty()); assert(!b_.empty()); assert(!c_.empty());
		assert(c_.n_elem==a_.n_rows);
		assert(b_.n_cols==a_.n_cols);
		assert(a_.is_square());
		assert(a_.is_trimatl());
	}

	// display butcher's tableau
	void RungeKutta::display_butchers_tableau(const ShLogPr& lg){
		// display integration method
		lg->msg(2,"%sintegration method%s\n",KBLU,KNRM);
		lg->msg("method: %s%s%s\n",KYEL,methodname_.c_str(),KNRM);
		lg->msg("scheme: %s%s%s\n",KYEL,schemename_.c_str(),KNRM);
		lg->msg("order: %s%llu%s %s\n",KYEL,order_,KNRM,is_embedded() ? "(embedded)" : "(step doubling error control)");
		lg->msg("Butcher's tableau for this method:\n");

		// hline
		lg->msg("-");
		for(arma::uword i=0;i<(7+a_.n_cols*7);i++)
			lg->msg(0,"-");
		lg->msg(0,"\n");
		
		// display a and c
		for(arma::uword i=0;i<a_.n_rows;i++){
			lg->msg("%+06.2f | ",c_(i));
			for(arma::uword j=0;j<a_.n_cols;j++){
				lg->msg(0,"%+06.2f",a_(i,j));
				if(j!=a_.n_cols-1)lg->msg(0," ");
			}
			lg->msg(0,"\n");
		}

		// hline
		lg->msg("-");
		for(arma::uword i=0;i<(7+a_.n_cols*7);i++)
			lg->msg(0,"-");
		lg->msg(0,"\n");

		// display b
		for(arma::uword i=0;i<b_.n_rows;i++){
			lg->msg("       | ");
			for(arma::uword j=0;j<b_.n_cols;j++){
				lg->msg(0,"%+06.2f",b_(i,j));
				if(j!=b_.n_cols-1)lg->msg(0," ");
			}
			lg->msg(0,"\n");
		}
		lg->msg(-2,"\n");

		// done 
		return;
	}

	// perform step
	arma::field<arma::Mat<fltp> > RungeKutta::perform_step(
		const fltp tt, const fltp dt,
		const arma::Mat<fltp> &yy,
		const fltp step_tolerance){

		// check system function
		if(system_fun_==NULL)rat_throw_line("system function is not set");
		if(a_.empty())rat_throw_line("butcher's tableau not set");

		// allocate storage for k
		arma::field<arma::Mat<fltp> > k(a_.n_rows);

		// call system function
		for(arma::uword i=0;i<a_.n_rows;i++){
			// calculate the sum
			arma::Mat<fltp> kk(yy.n_rows,yy.n_cols, arma::fill::zeros);
			for(arma::uword j=0;j<i;j++)kk += a_(i,j)*k(j);
			assert(kk.is_finite());

			// explicit runge kutta
			if(a_(i,i)==0.0){
				k(i) = system_fun_(tt + c_(i)*dt, yy + dt*kk);
				if(k(i).n_elem!=yy.n_elem)rat_throw_line("system function output does not agree with number of equations");
				if(!k(i).is_finite())rat_throw_line("system function returned non-finite value");
			}

			// diagonally implicit runge kutta (DIRK)
			else{
				// solve for k(i) implicitly
				arma::Mat<fltp> k_i_guess(yy.n_rows, yy.n_cols, arma::fill::zeros); 
				if(i>0)k_i_guess = k(i-1);
				
				// solve for diagonal using "fixed point" method
				fltp last_error = arma::Datum<fltp>::inf;
				for(arma::uword iteration=0;iteration<dirk_max_iter_;iteration++) {
					// check kk
					assert(k_i_guess.is_finite());

					// update k
					arma::Mat<fltp> k_i_new = system_fun_(tt + c_(i)*dt, yy + dt*(kk+a_(i, i)*k_i_guess));
					if(!k_i_new.is_finite())rat_throw_line("system function returned non-finite value");

					// calculate error
					const fltp error = arma::norm(k_i_new - k_i_guess, "inf");

					// check convergence
					if(error<step_tolerance){
						k(i) = k_i_new; break;
					}

					// check diverging
					if(error>last_error)return{};

					// check max number of iterations
					if(iteration==dirk_max_iter_-1)return{}; // empty output means we could not find a solution

					// update guess
					k_i_guess = k_i_guess + dirk_damping_*(k_i_new - k_i_guess);

					// update last error
					last_error = error;
				}
			}
		}

		// allocate solutions
		arma::field<arma::Mat<fltp> > dy(b_.n_rows,1);

		// walk over solutions
		// by convention we want the first solution to be the most accurate
		for(arma::uword j=0;j<b_.n_rows;j++){
			// allocate output
			dy(j).zeros(yy.n_rows, yy.n_cols);

			// apply weights and add to solution matrix
			for(arma::uword i=0;i<a_.n_cols;i++)
				dy(j) += dt*b_(j,i)*k(i);

			// check output
			assert(dy(j).is_finite());
		}

		// return result
		return dy;
	}


	// perform integration with adaptive time stepping
	RKData RungeKutta::integrate(
		const fltp t1, const fltp t2, 
		const arma::Row<fltp> &y0,
		const arma::Row<fltp> &yp0){

		// check system function
		if(system_fun_==NULL)rat_throw_line("system function is not set");
		if(a_.empty())rat_throw_line("butcher's tableau not set");

		// set initial data
		std::list<RKTimeData> solver_data;

		// copy initial step
		fltp tt = t1; fltp dt = dtini_; arma::Row<fltp> yy = y0; 

		// step tolerance
		fltp step_tolerance = std::max(abstol_, reltol_*arma::norm(arma::vectorise(yy), "inf"));

		// insert initial step in dataset
		solver_data.push_back({tt,y0,yp0.empty() ? arma::Row<fltp>(y0.n_elem,arma::fill::zeros) : yp0});

		// event detected
		arma::sword event_detected = -1;
		bool cancelled = false;

		// integrate until stop criterion is reached
		for(arma::uword i=0;i<max_iter_;i++){
			// check for cancelling
			if(cancelled)break;

			// allocate solutions
			// dydt1 is assumed less accurate than dydt2
			arma::Row<fltp> dydt1, dydt2;

			// embedded method
			if(b_.n_rows==1){
				// calculate one step
				const arma::field<arma::Mat<fltp> > dydt1fld = perform_step(tt,dt,yy,step_tolerance);
				if(dydt1fld.empty()){dt/=2;i--;continue;}
				dydt1 = dydt1fld.front()/dt;

				// perform two half steps to find a more accurate result
				const arma::field<arma::Mat<fltp> > dydt2a = perform_step(tt,dt/2,yy,step_tolerance);
				if(dydt2a.empty()){dt/=2;i--;continue;}
				const arma::field<arma::Mat<fltp> > dydt2b = perform_step(tt+dt/2,dt/2,yy+dydt2a.front()/2,step_tolerance);
				if(dydt2b.empty()){dt/=2;i--;continue;}

				// add two separate step
				dydt2 = dydt2a.front()/dt + dydt2b.front()/dt;
			}

			// fehlberg style method
			else if(b_.n_rows==2){
				// perform steps
				const arma::field<arma::Mat<fltp> > ypfld = perform_step(tt,dt,yy,step_tolerance);
				if(ypfld.empty()){dt/=2;i--;continue;}
				assert(ypfld.n_rows==b_.n_rows);
				dydt1 = ypfld(1)/dt; dydt2 = ypfld(0)/dt; 
			}

			// safety check
			else rat_throw_line("b has too many rows");

			// check output
			assert(dydt1.n_elem==yy.n_elem);
			assert(dydt2.n_elem==yy.n_elem);

			// find new points
			const arma::Row<fltp> y1 = yy + dt*dydt1;
			const arma::Row<fltp> y2 = yy + dt*dydt2;

			// estimate local error
			const fltp scaled_error = arma::max(arma::abs(y2 - y1)/(arma::abs(y2) + abstol_));

			// step size scaling factor
			const fltp safety_factor = RAT_CONST(0.9);
			const fltp smin = RAT_CONST(0.1);
			const fltp smax = RAT_CONST(10.0);
			const fltp sreq = std::pow(reltol_/scaled_error,RAT_CONST(1.0)/(order_+1));
			const fltp s = std::max(smin,std::min(safety_factor*sreq,smax));

			// Adaptive tolerance for fixed-point solver
			step_tolerance = RAT_CONST(1e-2)*std::max(abstol_, reltol_*arma::norm(arma::vectorise(yy), "inf"))*s;

			// if error exceeds tolerance
			if(scaled_error>reltol_){dt = dt*s; i--; continue;}

			// check for overrunning the end
			if(tt+dt>t2)dt=t2-tt;

			// event detected
			if(event_fun_){
				// run event fun at both times
				const arma::Row<fltp> rf1 = event_fun_(tt,yy);
				const arma::Row<fltp> rf2 = event_fun_(tt + dt, yy + dt*dydt2);

				// find zero crossing of root finding function
				const arma::Col<arma::uword> idx_root = arma::find(rf1%rf2<0);

				// check if there are any roots
				if(!idx_root.empty()){
					// create bounds for bisection method
					// the root must be between these two points
					arma::Row<fltp> t1(idx_root.n_elem, arma::fill::value(tt));
					arma::Row<fltp> t2(idx_root.n_elem, arma::fill::value(tt + dt));

					// bisection method
					for(arma::uword k=0;k<root_finding_max_iter_;k++){
						// find midpoint
						arma::Row<fltp> tmid = (t1 + t2)/2;

						// calculate at tmid
						arma::Row<fltp> rfmid(idx_root.n_elem);
						for(arma::uword j=0;j<idx_root.n_elem;j++){
							const arma::field<arma::Mat<fltp> > dy = perform_step(tt,tmid(j)-tt,yy,step_tolerance);
							if(dy.empty())rat_throw_line("step function failed during root finding");
							arma::Row<fltp> rfef = event_fun_(tmid(j), yy + dy.front());
							rfmid(j) = rfef(idx_root(j));
						}

						// find which point to move
						const arma::Col<arma::uword> idx1 = arma::find(rfmid<0);
						const arma::Col<arma::uword> idx2 = arma::find(rfmid>=0);

						// update bounds
						t2(idx1) = tmid(idx1); t1(idx2) = tmid(idx2);

						// check tolerance
						if(arma::all(arma::abs(t2 - t1)<roottol_*dt))break;
					}

					// find smallest 
					const arma::uword idx_tmin = t1.index_min();

					// update time step
					dt = t1(idx_tmin) - tt;

					// flag for event
					event_detected = idx_tmin;
				}
			}

			// change
			const arma::Row<fltp> yp = dydt2;

			// make actual time step (we take the accurate result here)
			yy += dt*yp; tt += dt;

			// update step size
			dt *= s;

			// insert into dataset
			solver_data.push_back({tt,yy,yp});

			// show monitoring function
			if(monitor_fun_)
				if(monitor_fun_(i,tt,dt,event_detected))
					cancelled = true;

			// check for minimum allowable stepsize
			if(dt<min_allowable_step_size_)break;

			// check for roots
			if(event_detected>=0)break;

			// done
			if(tt>=t2)break;
		}

		// join data
		RKData data;
		data.tt.set_size(solver_data.size());
		data.yy.set_size(solver_data.size(), y0.n_elem);
		data.yp.set_size(solver_data.size(), y0.n_elem);
		data.event_detected = event_detected;

		// fill dataset
		arma::uword cnt = 0;
		for(auto it=solver_data.begin();it!=solver_data.end();it++,cnt++){
			data.tt(cnt) = (*it).tt;
			data.yy.row(cnt) = (*it).yy;
			data.yp.row(cnt) = (*it).yp;
		}

		// return matrix
		return data;
	}

}}