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

// constructor
GmshFile::GmshFile(const std::string &fname){
	// open file for writing
	fid_.open(fname);

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
ShGmshFilePr GmshFile::create(const std::string &fname){
	return std::make_shared<GmshFile>(fname);
}

// write nodes
void GmshFile::write_nodes(
	const arma::Mat<double> &Rn){

	// get number of nodes
	arma::uword num_nodes = Rn.n_cols;

	// write header
	fid_ << "$Nodes\n";

	// write node data
	fid_ << num_nodes<<"\n";
	for(arma::uword i=0;i<num_nodes;i++){
		fid_ << i+1 << " " << Rn(0,i) << " " << Rn(1,i) << " " << Rn(2,i) << "\n";
	}

	// footer
	fid_ << "$EndNodes\n";
}

// write elements
void GmshFile::write_elements(const arma::Mat<arma::uword> &n){

	// number of elements
	arma::uword num_elements = n.n_cols;

	// write header
	fid_ << "$Elements" << "\n";

	// write elements (elm-number elm-type number-of-tags < tag > ... node-number-list)
	fid_ << num_elements << "\n";
	for(arma::uword i=0;i<num_elements;i++){
		fid_ << i+1 << " ";
		if(n.n_rows==2)fid_ << 1; // line
		else if(n.n_rows==4)fid_ << 3; // quad
		else if(n.n_rows==8)fid_ << 5; // hex
		else if(n.n_rows==3)fid_ << 2; // triangle
		fid_ << " " << 2 << " " << 99 << " " << 1 << " ";
		for(arma::uword j=0;j<n.n_rows;j++){
			fid_ << n(j,i)+1 << " ";
		}
		fid_ << "\n";
	}

	// write footer
	fid_ << "$EndElements\n";
}

// write scalar-data at nodes
void GmshFile::write_nodedata(
	const arma::Mat<double> &v,
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
	const arma::Mat<double> &v,
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