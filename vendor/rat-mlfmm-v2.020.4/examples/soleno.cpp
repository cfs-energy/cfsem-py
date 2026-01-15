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
#include <cassert>
#include <tclap/CmdLine.h>

// common headers
#include "rat/common/log.hh"
#include "rat/common/extra.hh"

// fmm specific headers
#include "mlfmm.hh"
#include "currentsources.hh"
#include "mgntargets.hh"
#include "soleno.hh"


// main
int main(int argc, char * argv[]){
	// create tclap object
	TCLAP::CmdLine cmd("Soleno");

	// input filename
	TCLAP::UnlabeledValueArg<std::string> input_file_argument(
		"ifname","Input filename",true,
		"solenoidlist.csv",
		"string", cmd);

	// input filename
	TCLAP::UnlabeledValueArg<std::string> output_file_argument(
		"ofname","Output filename",true,
		"inductancematrix.csv",
		"string", cmd);

	// parse the command line arguments
	cmd.parse(argc, argv);

	// get filename
	boost::filesystem::path ifpth = input_file_argument.getValue();
	boost::filesystem::path ofpth = output_file_argument.getValue();

	// read input file and store in a matrix
	arma::Mat<rat::fltp> dat;
	arma::field<std::string> header;
	dat.load(arma::csv_name(ifpth.string(), header));

	// get values from columns of matrix
	arma::Row<rat::fltp> R = dat.col(0).t();
	arma::Row<rat::fltp> Z = dat.col(1).t();
	arma::Row<rat::fltp> dR = dat.col(2).t();
	arma::Row<rat::fltp> dZ = dat.col(3).t();
	arma::Row<rat::fltp> nt = arma::conv_to<arma::Row<rat::fltp> >::from(dat.col(4).t());

	// convert to soleno definition
	arma::Row<rat::fltp> Rin = R-dR/2; arma::Row<rat::fltp> Rout = R+dR/2;
	arma::Row<rat::fltp> Zlow = Z-dZ/2; arma::Row<rat::fltp> Zhigh = Z+dZ/2;

	// number of layers used for calculation
	// determines accuracy 5 is usually good number
	arma::Row<arma::uword> nl = arma::Row<arma::uword>(Rin.n_elem, arma::fill::value(5));
	
	// current in solenoids (not relevant for inductance matrix)
	arma::Row<rat::fltp> current(Rin.n_elem, arma::fill::zeros);

	// create soleno object
	rat::fmm::ShSolenoPr sol = rat::fmm::Soleno::create();
	
	// add multiple solenoids
	sol->add_solenoid(Rin, Rout, Zlow, Zhigh, current, nt, nl);

	// set timer
	arma::wall_clock timer;
	timer.tic();

	// calculate inductance matrix
	arma::Mat<rat::fltp> M = sol->calc_M();


	// output calculation time
	std::cout<<"time: "<<timer.toc()<<" s"<<std::endl;


	// write inductance matrix to table
	M.save(ofpth.string(),arma::csv_ascii);

	// return
	return 0;
}