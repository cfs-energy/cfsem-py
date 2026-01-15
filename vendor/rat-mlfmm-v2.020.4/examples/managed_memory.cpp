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

#include "gpukernels.hh"
#include "typedefs.hh"

// main
int main(){
	// settings
	unsigned int num_elements = 1000;

	// create pointers
	rat::cufltp *a_ptr, *b_ptr;
	
	// create cuda managed memory
	// this memory is accessable both from CPU and GPU
	rat::fmm::GpuKernels::create_cuda_managed((void**)&a_ptr,num_elements*sizeof(rat::fltp));
	rat::fmm::GpuKernels::create_cuda_managed((void**)&b_ptr,num_elements*sizeof(rat::fltp));
	
	// wrap to armadillo vectors
	// these vectors do not destroy the 
	// cuda memory when they are destroyed
	arma::Col<rat::cufltp> A(a_ptr, num_elements, false, true);
	arma::Col<rat::cufltp> B(b_ptr, num_elements, false, true);

	// fill armadillo vectors with numbers
	A.fill(1.0); B.fill(2.0);
	
	// add two numbers
	rat::fmm::GpuKernels::test_kernel(A.memptr(), B.memptr(), num_elements);

	// std::cout<<A<<std::endl;

	// check numbers
	if(arma::any(A!=3))rat_throw_line("problem with gpu addition detected");

	// free cuda memory
	rat::fmm::GpuKernels::free_cuda_managed(a_ptr);
	rat::fmm::GpuKernels::free_cuda_managed(b_ptr);
}