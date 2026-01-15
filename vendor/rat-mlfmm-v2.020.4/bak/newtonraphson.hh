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

// adapted from: https://ch.mathworks.com/matlabcentral/fileexchange/43097-newton-raphson-solver

#ifndef NEWTON_RAPHSON_HH
#define NEWTON_RAPHSON_HH

#include <armadillo> 
#include <memory>
#include <iomanip>
#include <functional>
#include <cassert>

#include "common/parfor.hh"

// shared pointer definition
typedef std::shared_ptr<class NewtonRaphson> ShNewtonRaphsonPr;

// report function
typedef std::function<void(const arma::uword, const double, const double, const double, const double, const double) > NRReportFun;

// jacobian function
typedef std::function<arma::Mat<double>(const arma::Col<double>&) > NRJacFun;

// system function
typedef std::function<arma::Col<double>(const arma::Col<double>&) > NRSysFun;

// newton raphson non-linear solver
class NewtonRaphson{
	// properties
	private:
		// settings
		bool display_ = true;
		bool use_parallel_ = false; // only possible if system function is thread safe
		double tolx_ = 1e-12; // x tolerance
		double tolfun_ = 1e-6; // function tolerance
		arma::uword num_iter_max_ = 150; // maximum number of iterations
		double alpha_ = 1e-4; // criteria for decrease
		double min_lambda_ = 0.1; // minimum lambda
		double max_lambda_ = 0.5; // maximum lambda
		double delta_ = 1e-6; // finite difference stepsize

		// initial guess vector
		arma::Col<double> x0_; 

		// solver functions
		NRSysFun systemfun_;
		NRJacFun jacfun_;
		
		// report function
		NRReportFun reportfun_;

		// solution vector
		arma::Col<double> x_; 

		// jacobian matrix
		arma::Mat<double> J_;

		// statistics
		arma::sword flag_ = 0;
		arma::uword num_iter_;
		double resnorm_; // residual norm
		double stepnorm_;
		double lambda_; // lambda
		double rc_; // reciprocal condition
		arma::Col<double> F_; // function value
		arma::Col<double> dx_; // stepsize

		// output message
		std::string message_ = "unsolved";

	// methods
	public:
		// constructor
		NewtonRaphson();

		// factory
		static ShNewtonRaphsonPr create();

		// setting of system function and start point
		void set_systemfun(NRSysFun fn);
		void set_jacfun();
		void set_jacfun(NRJacFun fn);
		void set_reportfun(NRReportFun fn);
		
		// set initial guess for x
		void set_initial(const arma::Col<double> &x0);

		// settings
		void set_use_parallel(const bool use_parallel);
		void set_display(const bool display);
		void set_tolx(const double tolx);
		void set_tolfun(const double tolfun);
		void set_delta(const double delta);
		void set_num_iter_max(const arma::uword num_iter_max);
		
		// solver functions
		void solve();
		arma::Col<double> solve_newton() const;

		// getting
		arma::Col<double> get_result() const;
		std::string get_message() const;

		// finite difference approximation of the jacobian matrix
		// this will be used if no viable jacobian function assigned
		static arma::Mat<double> approximate_jacobian(const arma::Col<double> &x, const double dx, const bool use_parallel, NRSysFun sysfn);

		// display function
		static void default_displayfun(const arma::uword n, const double r, const double s, const double l, const double rc, const double c);
};

#endif
