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
#include "sparse.hh"

// field to mat for sparse complex matrix
arma::SpMat<std::complex<double> > Sparse::field2mat(
	const arma::field<arma::SpMat<std::complex<double> > > &fld, const bool sort){
	// walk over fields and count number of non-zeros
	arma::Mat<arma::uword> exist(fld.n_rows,fld.n_cols);
	arma::Mat<arma::uword> num_cols(fld.n_rows,fld.n_cols);
	arma::Mat<arma::uword> num_rows(fld.n_rows,fld.n_cols);
	arma::uword nnz = 0;

	// walk over field
	for(arma::uword i=0;i<fld.n_rows;i++){
		for(arma::uword j=0;j<fld.n_cols;j++){
			// check if matrix set
			if(!fld(i,j).is_empty()){
				// check existance
				exist(i,j) = true;

				// get number of columns and number of rows
				num_rows(i,j) = fld(i,j).n_rows;
				num_cols(i,j) = fld(i,j).n_cols;

				// get number of non-zeros
				nnz += num_nz(fld(i,j));
			}else{
				exist(i,j) = false;
			}
		}
	}

	// check if each row and each column has at least one matrix
	// assert(arma::as_scalar(arma::all(arma::any(exist,0),1)));
	// assert(arma::as_scalar(arma::all(arma::any(exist,1),0)));

	// check if columns are consistent
	arma::Row<arma::uword> num_cols_c(fld.n_cols);
	for(arma::uword i=0;i<fld.n_cols;i++){
		// find non-empty matrix in this column
		arma::Row<arma::uword> idx = arma::find(exist.col(i)).t();

		// check if any matrix in this column
		if(!idx.is_empty()){
			// get number of columns
			arma::Col<arma::uword> cnt = num_cols.submat(idx,arma::Row<arma::uword>{i});

			// check if all the same
			assert(arma::all(cnt==cnt(0)));

			// set number of columns
			num_cols_c(i) = cnt(0);
		}else{
			// no columns in this part of the matrix
			num_cols_c(i) = 0;
		}
	}
	
	// check if rows are consistent
	arma::Row<arma::uword> num_rows_c(fld.n_rows);
	for(arma::uword i=0;i<fld.n_rows;i++){
		// find non-empty matrix in this column
		arma::Row<arma::uword> idx = arma::find(exist.row(i)).t();

		// check if any matrix in this row
		if(!idx.is_empty()){
			// get number of columns
			arma::Row<arma::uword> cnt = num_rows.submat(arma::Row<arma::uword>{i},idx);

			// check if all the same
			assert(arma::all(cnt==cnt(0)));

			// set number of columns
			num_rows_c(i) = cnt(0);
		}else{
			// no columns in this part of the matrix
			num_rows_c(i) = 0;
		}
	}
	
	// calculate start indexes for columns
	arma::Row<arma::uword> idx_cols(fld.n_cols+1,arma::fill::zeros);
	idx_cols.cols(1,fld.n_cols) = arma::cumsum(num_cols_c);
	
	// calculate start indexes for rows
	arma::Row<arma::uword> idx_rows(fld.n_rows+1,arma::fill::zeros);
	std::cout<<fld.n_rows<<std::endl;
	std::cout<<num_rows_c<<std::endl;
	idx_rows.cols(1,fld.n_rows) = arma::cumsum(num_rows_c);
	
	// allocate new matrix entries
	arma::Row<arma::uword> col(nnz);
	arma::Row<arma::uword> row(nnz);
	arma::Row<std::complex<double> > val(nnz);
	
	// fill matrix
	arma::uword pos_idx = 0;
	for(arma::uword i=0;i<fld.n_rows;i++){
		for(arma::uword j=0;j<fld.n_cols;j++){
			if(!fld(i,j).is_empty()){
				// get indexes and values from matrix
				arma::Row<arma::uword> subrow, subcol;
				arma::Row<std::complex<double> > subval;
				extract_elements(subrow,subcol,subval,fld(i,j));

				// insert into matrix at correct location
				row.cols(pos_idx,pos_idx+subrow.n_elem-1) = subrow + idx_rows(i);
				col.cols(pos_idx,pos_idx+subrow.n_elem-1) = subcol + idx_cols(j);
				val.cols(pos_idx,pos_idx+subrow.n_elem-1) = subval;

				// increment index
				pos_idx += subrow.n_elem;
			}
		}
	}

	// construct output matrix from entries
	arma::SpMat<std::complex<double> > M(arma::join_vert(row,col),
		val,arma::sum(num_rows_c),arma::sum(num_cols_c),sort,true);

	// return matrix
	return M;
}

// field to mat for sparse complex matrix
arma::SpMat<double> Sparse::field2mat(
	const arma::field<arma::SpMat<double> > &fld, const bool sort){
	// walk over fields and count number of non-zeros
	arma::Mat<arma::uword> exist(fld.n_rows,fld.n_cols);
	arma::Mat<arma::uword> num_cols(fld.n_rows,fld.n_cols);
	arma::Mat<arma::uword> num_rows(fld.n_rows,fld.n_cols);
	arma::uword nnz = 0;

	// walk over field
	for(arma::uword i=0;i<fld.n_rows;i++){
		for(arma::uword j=0;j<fld.n_cols;j++){
			// check if matrix set
			if(!fld(i,j).is_empty()){
				// check existance
				exist(i,j) = true;

				// get number of columns and number of rows
				num_rows(i,j) = fld(i,j).n_rows;
				num_cols(i,j) = fld(i,j).n_cols;

				// get number of non-zeros
				nnz += num_nz(fld(i,j));
			}else{
				exist(i,j) = false;
			}
		}
	}

	// check if each row and each column has at least one matrix
	// assert(arma::as_scalar(arma::all(arma::any(exist,0),1)));
	// assert(arma::as_scalar(arma::all(arma::any(exist,1),0)));

	// check if columns are consistent
	arma::Row<arma::uword> num_cols_c(fld.n_cols);
	for(arma::uword i=0;i<fld.n_cols;i++){
		// find non-empty matrix in this column
		arma::Row<arma::uword> idx = arma::find(exist.col(i)).t();

		// check if any matrix in this column
		if(!idx.is_empty()){
			// get number of columns
			arma::Col<arma::uword> cnt = num_cols.submat(idx,arma::Row<arma::uword>{i});

			// check if all the same
			assert(arma::all(cnt==cnt(0)));

			// set number of columns
			num_cols_c(i) = cnt(0);
		}else{
			// no columns in this part of the matrix
			num_cols_c(i) = 0;
		}
	}
	
	// check if rows are consistent
	arma::Row<arma::uword> num_rows_c(fld.n_rows);
	for(arma::uword i=0;i<fld.n_rows;i++){
		// find non-empty matrix in this column
		arma::Row<arma::uword> idx = arma::find(exist.row(i)).t();

		// check if any matrix in this row
		if(!idx.is_empty()){
			// get number of columns
			arma::Row<arma::uword> cnt = num_rows.submat(arma::Row<arma::uword>{i},idx);

			// check if all the same
			assert(arma::all(cnt==cnt(0)));

			// set number of columns
			num_rows_c(i) = cnt(0);
		}else{
			// no columns in this part of the matrix
			num_rows_c(i) = 0;
		}
	}
	
	// calculate start indexes for columns
	arma::Row<arma::uword> idx_cols(fld.n_cols+1,arma::fill::zeros);
	idx_cols.cols(1,fld.n_cols) = arma::cumsum(num_cols_c);
	
	// calculate start indexes for rows
	arma::Row<arma::uword> idx_rows(fld.n_rows+1,arma::fill::zeros);
	idx_rows.cols(1,fld.n_rows) = arma::cumsum(num_rows_c);
	
	// allocate new matrix entries
	arma::Row<arma::uword> col(nnz);
	arma::Row<arma::uword> row(nnz);
	arma::Row<double> val(nnz);
	
	// fill matrix
	arma::uword pos_idx = 0;
	for(arma::uword i=0;i<fld.n_rows;i++){
		for(arma::uword j=0;j<fld.n_cols;j++){
			if(!fld(i,j).is_empty()){
				// get indexes and values from matrix
				arma::Row<arma::uword> subrow, subcol;
				arma::Row<double> subval;
				extract_elements(subrow,subcol,subval,fld(i,j));

				// insert into matrix at correct location
				row.cols(pos_idx,pos_idx+subrow.n_elem-1) = subrow + idx_rows(i);
				col.cols(pos_idx,pos_idx+subrow.n_elem-1) = subcol + idx_cols(j);
				val.cols(pos_idx,pos_idx+subrow.n_elem-1) = subval;

				// std::cout<<i<<" "<<j<<" "<<subrow.n_elem<<std::endl;

				// increment index
				pos_idx += subrow.n_elem;
			}
		}
	}

	// sanity check
	assert(pos_idx==nnz);

	// construct output matrix from entries
	arma::SpMat<double> M(arma::join_vert(row,col),
		val,arma::sum(num_rows_c),arma::sum(num_cols_c),sort,true);

	// return matrix
	return M;
}



// field to mat for sparse complex matrix
arma::SpMat<std::complex<double> > Sparse::add_field(
	const arma::field<arma::SpMat<std::complex<double> > > &fld, 
	const bool sort){

	// count number of non-zeros
	arma::uword nnz = 0;
	for(arma::uword i=0;i<fld.n_elem;i++){
		nnz += num_nz(fld(i));
	}

	// allocate
	arma::Row<arma::uword> row(nnz);
	arma::Row<arma::uword> col(nnz);
	arma::Row<std::complex<double> > val(nnz);

	// copy
	arma::uword idx = 0;
	for(arma::uword i=0;i<fld.n_elem;i++){
		// get indexes and values from matrix
		arma::Row<arma::uword> subrow, subcol;
		arma::Row<std::complex<double> > subval;
		extract_elements(subrow,subcol,subval,fld(i));

		// store elements
		row.cols(idx,idx+subcol.n_elem-1) = subrow;
		col.cols(idx,idx+subcol.n_elem-1) = subcol;
		val.cols(idx,idx+subcol.n_elem-1) = subval;		
	}

	// construct output matrix from entries
	arma::SpMat<std::complex<double> > M(arma::join_vert(row,col),
		val,fld(0).n_rows,fld(0).n_cols,sort,false);

	// return matrix
	return M;
}
 
// expand matrix (like expand indexes)
// in for example: Mout = [M,..,..;..,M,..;..,..,M]
arma::SpMat<std::complex<double> > Sparse::expand(
	const arma::SpMat<std::complex<double> > &M, const arma::uword num_dim){

	// create field
	arma::field<arma::SpMat<std::complex<double> > > fld(num_dim,num_dim);

	// insert matrix
	for(arma::uword i=0;i<num_dim;i++)
		fld(i,i) = M;

	// combine and return
	return field2mat(fld,false);
}

// place matrix into new (larger) empty sparse matrix
arma::SpMat<std::complex<double> > Sparse::migrate(
	const arma::SpMat<std::complex<double> > &M, 
	const arma::uword n_cols, const arma::uword n_rows, 
	const arma::uword idx_col, const arma::uword idx_row){

	// get elements
	arma::Row<arma::uword> row, col; 
	arma::Row<std::complex<double> > val; 
	extract_elements(row,col,val,M);

	// create output matrix
	arma::SpMat<std::complex<double> > Mout(
		arma::join_vert(row+idx_row,col+idx_col),
		val,n_rows,n_cols,false,false);

	// return new matrix
	return Mout;
}


// number of nonzeros in the matrix
arma::uword Sparse::num_nz(const arma::SpMat<double> &M){
	// get number of non zeros
	return M.n_nonzero;
}

// number of nonzeros in the matrix
arma::uword Sparse::num_nz(const arma::SpMat<std::complex<double> > &M){
	// get number of non zeros
	return M.n_nonzero;
}

// extract non-zeros from complex sparse matrix
void Sparse::extract_elements(
	arma::Row<arma::uword> &row, 
	arma::Row<arma::uword> &col, 
	arma::Row<std::complex<double> > &val, 
	const arma::SpMat<std::complex<double> > &M){

	// // get iterators
	arma::SpMat<std::complex<double> >::const_iterator it = M.begin();
	// arma::SpMat<std::complex<double> >::const_iterator end = M.end();

	// // get number of non zeros
	arma::uword nnz = M.n_nonzero;

	// allocate
	row.set_size(nnz); col.set_size(nnz); val.set_size(nnz);

	// walk over all values
	for(arma::uword i=0;i<nnz;++it,i++){
		row(i) = it.row(); col(i) = it.col(); val(i) = (*it);
	}
}

// extract non-zeros from real sparse matrix
void Sparse::extract_elements(
	arma::Row<arma::uword> &row, 
	arma::Row<arma::uword> &col, 
	arma::Row<double> &val, 
	const arma::SpMat<double> &M){

	// get iterators
	arma::SpMat<double>::const_iterator it = M.begin();
	// arma::SpMat<double>::const_iterator end = M.end();

	// get number of non zeros
	arma::uword nnz = M.n_nonzero;

	// allocate
	row.set_size(nnz); col.set_size(nnz); val.set_size(nnz);

	// walk over all values
	for(arma::uword i=0;i<nnz;++it,i++){
		row(i) = it.row(); col(i) = it.col(); val(i) = (*it);
	}
}


// row multiplication since .each_row() does not work yet for armadillo
void Sparse::multiply_each_row(arma::SpMat<double> &M, const arma::Row<double> &v){
	// check sizes
	assert(M.n_cols==v.n_elem);

	// get iterators
	arma::SpMat<double>::iterator it = M.begin();
	arma::SpMat<double>::iterator end = M.end();

	// walk over all values
	arma::uword idx = 0;
	for(;it!=end;++it,idx++)(*it)*=v(it.col());
	assert(idx==M.n_nonzero);
}

// row multiplication since .each_row() does not work yet for armadillo
void Sparse::multiply_each_col(arma::SpMat<double> &M, const arma::Col<double> &v){
	// check sizes
	assert(M.n_rows==v.n_elem);

	// get iterators
	arma::SpMat<double>::iterator it = M.begin();
	arma::SpMat<double>::iterator end = M.end();

	// walk over all values
	arma::uword idx = 0;
	for(;it!=end;++it,idx++)(*it)*=v(it.row());
	assert(idx==M.n_nonzero);
}

// diagonal matrix from vector
arma::SpMat<double> Sparse::diagmat(const arma::Col<double> &v){
	// create indices
	arma::Mat<arma::uword> indices(2,v.n_elem);
	indices.each_row() = arma::regspace<arma::Row<arma::uword> >(0,v.n_elem-1);

	// assemble matrix
	return arma::SpMat<double>(indices,v,v.n_elem,v.n_elem,true,false);
}




// compress matrix
arma::Col<arma::uword> Sparse::compress(
	const arma::Col<arma::uword> &idx, const arma::uword num_size){

	// check sorting
	assert(idx.is_sorted("ascend"));

	// number of non-zeros
	arma::uword num_nz = idx.n_elem;

	// count number of entries in each column
	arma::Col<arma::uword> ncol(num_size,arma::fill::zeros);
	for(arma::uword i=0;i<num_nz;i++)ncol(idx(i))++;

	// create compression array
	const arma::Col<arma::uword> idx_c = arma::join_vert(
		arma::Col<arma::uword>{0},arma::cumsum(ncol));

 	// return indexes
	return idx_c;
}

// sort sparse matrix
void Sparse::sort_columnwise(
	arma::Col<arma::uword> &row,
	arma::Col<arma::uword> &col,
	arma::Mat<double> &val){

	// this sorting is slow and should be replaced by better assembly!
	arma::Col<arma::uword> sort_index = arma::sort_index(row);
	col = col.rows(sort_index); 
	row = row.rows(sort_index); 
	val = val.rows(sort_index);

	// sort columns
	sort_index = arma::stable_sort_index(col);
	col = col.rows(sort_index); 
	row = row.rows(sort_index); 
	val = val.rows(sort_index);
}


// sort sparse matrix
void Sparse::sort_rowwise(
	arma::Col<arma::uword> &row,
	arma::Col<arma::uword> &col,
	arma::Mat<double> &val){

	// this sorting is slow and should be replaced by better assembly!
	arma::Col<arma::uword> sort_index = arma::sort_index(col);
	col = col.rows(sort_index); 
	row = row.rows(sort_index); 
	val = val.rows(sort_index);

	// sort columns
	sort_index = arma::stable_sort_index(row);
	col = col.rows(sort_index); 
	row = row.rows(sort_index); 
	val = val.rows(sort_index);
}