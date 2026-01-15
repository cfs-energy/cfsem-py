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

#ifndef GMRES_HH
#define GMRES_HH

#include <cmath> 
#include <functional>
#include <armadillo>
#include <cassert>
#include <iostream>
#include <memory>

// shared pointer definition
typedef std::shared_ptr<class GMRES> ShGMRESPr;

// harmonics class template
class GMRES{
	// properties
	private:
		// settings
		arma::uword num_restart_ = 32;
		arma::uword num_iter_max_ = 150;
		double tol_ = 1e-6;

		// functions
		std::function<arma::Col<double>(arma::Col<double>) > systemfun_;
		std::function<arma::Col<double>(arma::Col<double>) > precfun_;
		std::function<void(int,double) > reportfun_;

		// upper Hessenberg
		arma::Mat<double> H_;

		// other internal storage
		double normb_;
		arma::Col<double> s_;
		arma::Col<double> cs_;
		arma::Col<double> sn_;
		arma::field<arma::Col<double> > v_;

		// additional output
		int flag_;
		double relres_ = arma::datum::inf;
		arma::uword num_iter_ = arma::datum::inf;

	// methods
	public:
		// constructor
		GMRES();

		// factory
		static ShGMRESPr create();

		// setting of functions
		void set_systemfun(std::function<arma::Col<double>(const arma::Col<double>) > fn);
		void set_precfun(std::function<arma::Col<double>(const arma::Col<double>) > fn);
		void set_reportfun(std::function<void(int,double) > fn);

		// settings
		void set_tol(const double tol);
		void set_num_iter_max(const arma::uword num_iter_max);
		void set_num_restart(const arma::uword num_restart);

		// solve system (x is output but also contains initial guess)
		void iter_report() const;
		void solve(arma::Col<double> &x, const arma::Col<double> &b);
		void update(arma::Col<double> &x, const int k);

		// helper functions
		static void apply_plane_rot(double &dx, double &dy, double &cs, double &sn);
		static void gen_plane_rot(double &dx, double &dy, double &cs, double &sn);

		// display function
		void display() const;
};

#endif
