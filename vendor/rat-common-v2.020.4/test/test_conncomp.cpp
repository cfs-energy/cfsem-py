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

// specific headers
#include "error.hh"
#include "extra.hh"

// main
int main(){
	// create a graph
	const arma::uword num_nodes = 10;
	arma::Mat<arma::uword> M(9,2);
	M.row(0) = arma::Row<arma::uword>{0,1};
	M.row(1) = arma::Row<arma::uword>{1,2};
	M.row(2) = arma::Row<arma::uword>{2,0};
	M.row(3) = arma::Row<arma::uword>{3,4};
	M.row(4) = arma::Row<arma::uword>{4,6};
	M.row(5) = arma::Row<arma::uword>{6,9};
	M.row(6) = arma::Row<arma::uword>{9,8};
	M.row(7) = arma::Row<arma::uword>{8,7};
	M.row(8) = arma::Row<arma::uword>{6,7};

	// connect
	const arma::Row<arma::uword> graph_indices = 
		rat::cmn::Extra::conncomp(M,num_nodes);

	// expected output
	arma::Row<arma::uword> expected{0,0,0,1,1,2,1,1,1,1};

	// show
	if(arma::any(expected!=graph_indices))
		rat_throw_line("graph connectivity is not as expected");


	// return
	return 0;
}