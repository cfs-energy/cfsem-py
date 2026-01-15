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
#include "rat/common/error.hh"
#include "extra.hh"

// main
int main(){
	// run factorial calculation
	const arma::Col<rat::fltp> facs = rat::fmm::Extra::factorial(10);

	// check if number of elements is correct
	if(facs.n_elem!=10+1)rat_throw_line(
	"factorial output has different number of elements than requested");

	// check if first ten factorials are acceptable
	if(facs(0)!=1)rat_throw_line("0! is not equal to 1");
	if(facs(1)!=1)rat_throw_line("1! is not equal to 1");
	if(facs(2)!=2)rat_throw_line("2! is not equal to 2");
	if(facs(3)!=6)rat_throw_line("3! is not equal to 6");
	if(facs(4)!=24)rat_throw_line("4! is not equal to 24");
	if(facs(5)!=120)rat_throw_line("5! is not equal to 120");
	if(facs(6)!=720)rat_throw_line("6! is not equal to 720");
	if(facs(7)!=5040)rat_throw_line("7! is not equal to 5040");
	if(facs(8)!=40320)rat_throw_line("8! is not equal to 40320");
	if(facs(9)!=362880)rat_throw_line("9! is not equal to 362880");
	if(facs(10)!=3628800)rat_throw_line("10! is not equal to 3628800");

	// return
	return 0;
}