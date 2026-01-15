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
#include "stlfile.hh"
#include "extra.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// constructor
	STLFile::STLFile(const boost::filesystem::path &fname){
		fname_ = fname;
	}

	// destructor
	STLFile::~STLFile(){
		write();
	}

	// factory
	ShSTLFilePr STLFile::create(const boost::filesystem::path &fname){
		return std::make_shared<STLFile>(fname);
	}

	// add triangles
	void STLFile::add_triangles(const MeshData &data){
		assert(data.t.n_rows==3); assert(data.R.n_rows==3); assert(data.t.max()<data.R.n_cols);
		meshes_.push_back(data);
	}

	// write nodes
	void STLFile::write(){
		// binary file header
		std::string header_info = "RAT Mesh, UNITS=mm";
		char head[80];
		std::strncpy(head,header_info.c_str(),sizeof(head)-1);
		char attribute[2] = "0";

		// walk over objects and count number of triangles
		unsigned int num_tri_long = 0;
		for(auto it=meshes_.begin();it!=meshes_.end();it++)
			num_tri_long+=unsigned((*it).t.n_cols);

		// open file for writing
		std::ofstream fid;
		fid.open(fname_.c_str(),  std::ios::out | std::ios::binary);

		// write header
		fid.write(head,sizeof(head)); // 80 bytes
		fid.write(reinterpret_cast<char*>(&num_tri_long),sizeof(unsigned int)); // 4 bytes

		// check sizes
		assert(sizeof(unsigned int)==4);
		assert(sizeof(float)==4);

		// walk over objects
		for(auto it=meshes_.begin();it!=meshes_.end();it++){
			// get reference
			MeshData& data = (*it);

			// get triangles
			const arma::Mat<arma::uword> &t = data.t;

			// create normal vector
			const arma::Mat<fltp> N = Extra::cross(
				data.R.cols(t.row(1)) - data.R.cols(t.row(0)),
				data.R.cols(t.row(2)) - data.R.cols(t.row(0)));

			// convert to float and mm
			const arma::Mat<float> Rf = arma::conv_to<arma::Mat<float> >::from(1000*data.R);
			const arma::Mat<float> Nf = arma::conv_to<arma::Mat<float> >::from(1000*N);

			//write down every triangle
			for (arma::uword i=0;i<t.n_cols;i++){
				// write facet normal vector
				fid.write(reinterpret_cast<const char*>(&Nf(0,i)), 3*sizeof(float)); // 12 bytes

				// write vertices
				for(arma::uword k=0;k<3;k++)
					fid.write(reinterpret_cast<const char*>(&Rf(0,t(k,i))), 3*sizeof(float)); // 12 bytes (x3)

				// write attribute
				fid.write(attribute,sizeof(attribute)); // 2 bytes
			}

		}

		// close file
		fid.close();
	}

}}