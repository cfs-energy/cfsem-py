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

// specific headers
#include "parfor.hh"
#include "error.hh"

// main
int main(){
	// settings
	const arma::uword num_threads = 250;
	
	// run parfor loop
	arma::Mat<arma::uword> cpu(num_threads,num_threads,arma::fill::zeros);
	rat::cmn::parfor(0,num_threads,true,[&](arma::uword i, int local_cpu_i) {
		rat::cmn::parfor(0,num_threads,true,[&](arma::uword j, int local_cpu_j) {
			cpu(i,j) = local_cpu_i+1;
		});
	});

	// check if all threads were run
	if(arma::any(arma::vectorise(cpu)==0))
		rat_throw_line("not all threads were run");
}