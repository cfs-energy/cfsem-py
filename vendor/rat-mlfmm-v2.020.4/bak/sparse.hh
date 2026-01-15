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

#ifndef SPARSE_HH
#define SPARSE_HH

#include <armadillo> 
#include <complex>
#include <cmath>
#include <cassert>

#include "common/parfor.hh"

// library of sparse matrix functions

// that are used throughout the code
class Sparse{
	public:
		// field to mat function
		static arma::SpMat<std::complex<double> > field2mat(const arma::field<arma::SpMat<std::complex<double> > > &fld, const bool sort);
		static arma::SpMat<double> field2mat(const arma::field<arma::SpMat<double> > &fld, const bool sort);
		static arma::SpMat<std::complex<double> > add_field(const arma::field<arma::SpMat<std::complex<double> > > &fld, const bool sort);

		// expand matrix
		static arma::SpMat<std::complex<double> > expand(const arma::SpMat<std::complex<double> > &M, const arma::uword num_dim);
		static arma::SpMat<std::complex<double> > migrate(const arma::SpMat<std::complex<double> > &M, const arma::uword n_cols, const arma::uword n_rows, const arma::uword idx_col, const arma::uword idx_row);

		// number of non-zeros in matrix
		static arma::uword num_nz(const arma::SpMat<double> &M);
		static arma::uword num_nz(const arma::SpMat<std::complex<double> > &M);

		// extract indexes and values from sparse matrix
		static void extract_elements(arma::Row<arma::uword> &row, arma::Row<arma::uword> &col, arma::Row<std::complex<double> > &val, const arma::SpMat<std::complex<double> > &M);
		static void extract_elements(arma::Row<arma::uword> &row, arma::Row<arma::uword> &col, arma::Row<double> &val, const arma::SpMat<double> &M);

		// element wise 
		static arma::Col<arma::uword> compress(const arma::Col<arma::uword> &idx, const arma::uword num_size);
		static void sort_columnwise(arma::Col<arma::uword> &row, arma::Col<arma::uword> &col, arma::Mat<double> &val);
		static void sort_rowwise(arma::Col<arma::uword> &row, arma::Col<arma::uword> &col, arma::Mat<double> &val);

		// matrix operations
		static void multiply_each_row(arma::SpMat<double> &M, const arma::Row<double> &v);
		static void multiply_each_col(arma::SpMat<double> &M, const arma::Col<double> &v);

		// diagonal matrix
		static arma::SpMat<double> diagmat(const arma::Col<double> &v);
		
};

#endif