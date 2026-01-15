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

// this file was evolved from:
// http://math.nist.gov/iml++/

// include header file
#include "gmshfile.hh"

// common headers
#include "error.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// constructor
	GmshFile::GmshFile(const boost::filesystem::path &fname){
		// open file for writing
		fid_.open(fname.string());

		// write header 
		fid_ << "$MeshFormat\n";
		fid_ << "2.0 0 8\n";
		fid_ << "$EndMeshFormat\n";
	}

	// destructor
	GmshFile::~GmshFile(){
		fid_.close();
	}

	// factory
	ShGmshFilePr GmshFile::create(const boost::filesystem::path &fname){
		return std::make_shared<GmshFile>(fname);
	}

	// write nodes
	void GmshFile::write_nodes(
		const arma::Mat<fltp> &Rn){

		// get number of nodes
		const arma::uword num_nodes = Rn.n_cols;

		// write header
		fid_ << "$Nodes\n";

		// write node data
		fid_ << num_nodes<<"\n";
		if(Rn.n_rows==3)
			for(arma::uword i=0;i<num_nodes;i++)
				fid_ << i+1 << " " << Rn(0,i) << " " << Rn(1,i) << " " << Rn(2,i) << "\n";
		else if(Rn.n_rows==2)
			for(arma::uword i=0;i<num_nodes;i++)
				fid_ << i+1 << " " << Rn(0,i) << " " << Rn(1,i) << " " << RAT_CONST(0.0) << "\n";
		else rat_throw_line("node coordinate matrix must have 2 or 3 rows");

		// footer
		fid_ << "$EndNodes\n";
	}

	// write elements without ID
	void GmshFile::write_elements(
		const arma::Mat<arma::uword> &n){
		// create list of id's all zero
		const arma::Row<arma::uword> id(n.n_cols, arma::fill::ones);

		// use general method
		write_elements(n,id);
	}

	// write elements
	void GmshFile::write_elements(
		const arma::Mat<arma::uword> &n, 
		const arma::Row<arma::uword> &id){

		// check input 
		assert(n.n_cols==id.n_cols);

		// number of elements
		const arma::uword num_elements = n.n_cols;

		// write header
		fid_ << "$Elements" << "\n";

		// write elements 
		// (elm-number elm-type number-of-tags < tag > ... node-number-list)
		fid_ << num_elements << "\n";
		for(arma::uword i=0;i<num_elements;i++){
			fid_ << i+1 << " ";
			if(n.n_rows==2)fid_ << 1;
			else if(n.n_rows==4)fid_ << 3; 
			else if(n.n_rows==8)fid_ << 5; 
			else if(n.n_rows==3)fid_ << 2;
			fid_ << " " << 2 << " " << 99 << " " << id(i) << " ";
			for(arma::uword j=0;j<n.n_rows;j++){
				fid_ << n(j,i)+1 << " ";
			}
			fid_ << "\n";
		}

		// write footer
		fid_ << "$EndElements\n";
	}

	// write elements
	void GmshFile::write_elements(
		const arma::Mat<arma::uword> &n, 
		const arma::Row<arma::uword> &id, 
		const arma::Mat<arma::uword> &s, 
		const arma::Row<arma::uword> &sid){

		// check input 
		assert(n.n_cols==id.n_cols);
		assert(s.n_cols==sid.n_cols);

		// number of elements
		arma::uword num_volume_elements = n.n_cols;
		arma::uword num_surface_elements = s.n_cols;

		// write header
		fid_ << "$Elements" << "\n";

		// number of elements
		fid_ << num_volume_elements + num_surface_elements << "\n";

		// write elements 
		// (elm-number elm-type number-of-tags < tag > ... node-number-list)
		for(arma::uword i=0;i<num_volume_elements;i++){
			fid_ << i+1 << " ";
			if(n.n_rows==2)fid_ << 1;
			else if(n.n_rows==4)fid_ << 3; 
			else if(n.n_rows==8)fid_ << 5; 
			else if(n.n_rows==3)fid_ << 2;
			fid_ << " " << 2 << " " << 99 << " " << id(i) << " ";
			for(arma::uword j=0;j<n.n_rows;j++){
				fid_ << n(j,i)+1 << " ";
			}
			fid_ << "\n";
		}

		// write surface elements 
		// (elm-number elm-type number-of-tags < tag > ... node-number-list)
		for(arma::uword i=0;i<num_surface_elements;i++){
			fid_ << num_volume_elements+i+1 << " ";
			if(s.n_rows==2)fid_ << 1;
			else if(s.n_rows==4)fid_ << 3; 
			else if(s.n_rows==8)fid_ << 5; 
			else if(s.n_rows==3)fid_ << 2;
			fid_ << " " << 2 << " " << 99 << " " << sid(i) << " ";
			for(arma::uword j=0;j<s.n_rows;j++){
				fid_ << s(j,i)+1 << " ";
			}
			fid_ << "\n";
		}

		// write footer
		fid_ << "$EndElements\n";
	}

	// write scalar-data at nodes
	void GmshFile::write_nodedata(
		const arma::Mat<fltp> &v,
		const std::string &datname){

		// number of nodes
		const arma::uword num_nodes = v.n_cols;

		// get dimensionality of data
		const arma::uword num_dim = v.n_rows;

		// header
		fid_ << "$NodeData" << std::endl;
		fid_ << 1 << std::endl;
		fid_ << "\"" << datname << "\"" << std::endl;
		fid_ << 1 << std::endl;
		fid_ << 0.0 << std::endl;
		fid_ << 3 << std::endl;
		fid_ << 0 << std::endl;

		// write data
		fid_ << num_dim << "\n";
		fid_ << num_nodes << std::endl;
		for(arma::uword i=0;i<num_nodes;i++){
			fid_ << i+1;
			for(arma::uword j=0;j<num_dim;j++)
				fid_ << " " << v(j,i);
			fid_ << "\n";
		}

		// footer
		fid_ << "$EndNodeData" << std::endl;
	}

	// write vector-data at elements
	void GmshFile::write_elementdata(
		const arma::Mat<fltp> &v,
		const std::string &datname){

		// number of nodes
		const arma::uword num_elements = v.n_cols;

		// get dimensionality of data
		const arma::uword num_dim = v.n_rows;

		// header
		fid_ << "$ElementData\n";
		fid_ << 1 << "\n";
		fid_ << "\"" << datname << "\"" << std::endl;
		fid_ << 1 << "\n";
		fid_ << 0.0 << "\n";
		fid_ << 3 << "\n";
		fid_ << 0 << "\n";

		// write data
		fid_ << num_dim << "\n";
		fid_ << num_elements << "\n";
		for(arma::uword i=0;i<num_elements;i++){
			fid_ << i+1;
			for(arma::uword j=0;j<num_dim;j++)
				fid_ << " " << v(j,i);
			fid_ << "\n";
		}

		// footer
		fid_ << "$EndElementData\n";
	}

}}