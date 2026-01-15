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

#ifndef GMSH_FILE_HH
#define GMSH_FILE_HH

#include <armadillo> 
#include <cassert>
#include <memory>

// shared pointer definition for log
typedef std::shared_ptr<class GmshFile> ShGmshFilePr;

// logging to the terminal
class GmshFile{
	// properties
	private:
		// settings
		std::ofstream fid_;

	// methods 
	public:
		// constructor
		GmshFile(const std::string &fname);

		// destructor
		~GmshFile();

		// factory
		static ShGmshFilePr create(const std::string &fname);

		// write nodes
		void write_nodes(const arma::Mat<double> &Rn);

		// write elements
		void write_elements(const arma::Mat<arma::uword> &n);

		// write scalar-data at nodes
		void write_nodedata(const arma::Mat<double> &v,const std::string &datname);

		// write vector-data at elements
		void write_elementdata(const arma::Mat<double> &v,const std::string &datname);
};


#endif