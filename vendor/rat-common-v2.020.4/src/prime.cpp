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

// include header file
#include "prime.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// prime number generator
	// based on Sieve Of Eratosthenes algorithm
	arma::Row<arma::uword> Prime::eratosthenes(const arma::uword N){

		// create the sieve
	    arma::Row<unsigned char> sieve(N+1,arma::fill::ones);
	 	
	    // zero and one are not primes
	    sieve(0) = 0; sieve(1) = 0;

	 	// calculate 	
	    for (arma::uword p=2; p*p<=N; p++){
	        // If prime[p] is not changed, then it is a prime
	        if (sieve(p) == 1){
	            // Update all multiples of p
	            for (arma::uword i=p*2; i<=N; i += p)
	                sieve(i) = 0;
	        }
	    }

	    // get primes
	 	arma::Row<arma::uword> primes = arma::find(sieve==1).t();

	 	// return
	    return primes;
	}

	// get first N primes
	arma::Row<arma::uword> Prime::calculate(const arma::uword N){
		// estimate n-prime max using prime number theorem
		// to invert this theorem one can use "lambertw" functions
		// we use simple interpolation instead
		arma::Col<fltp> nlist = arma::logspace<arma::Col<fltp> >(0,12);
		arma::Col<fltp> pi = nlist/arma::log(nlist);
		arma::Col<fltp> x; arma::Col<fltp> Na = {static_cast<fltp>(N)};
		arma::interp1(pi,nlist,Na,x,"linear",0);
		arma::uword num_sieve = static_cast<arma::uword>(arma::as_scalar(x));

		// run algorithm to calculate list of prime numbers
		arma::Row<arma::uword> primes = Prime::eratosthenes(static_cast<arma::uword>(std::ceil(1.15*num_sieve)));
		//std::cout<<arma::as_scalar(primes.tail(1))/(1.15*num_sieve)<<std::endl;

		// cut list short
		assert(primes.n_elem>=N);
		primes = primes.cols(0,N-1);

		// return list
		return primes;
	}

}}