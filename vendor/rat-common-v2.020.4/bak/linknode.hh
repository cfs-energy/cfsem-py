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

#ifndef CMN_LINK_NODE_HH
#define CMN_LINK_NODE_HH

#include <armadillo> 
#include <cassert>
#include <memory>
#include <list>

#include "defines.hh"
#include "node.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// shared pointer definition
	typedef std::shared_ptr<class LinkNode> ShLinkNodePr;

	// circle distance function for distmesh
	class LinkNode: public Node{
		// properties
		private:
			arma::uword idx_;

		public:
			// regular constructors
			LinkNode(const arma::uword idx);
			static ShLinkNodePr create(const arma::uword idx);

			// serialization
			Json::Value serialize() const;
			static std::string get_type();
			arma::uword get_index() const;
			LinkNode(const Json::Value &js);
			static ShNodePr create(const Json::Value &js);

			// list operations
			static void add_node(Json::Value &js, std::list<ShNodePr> &list, ShNodePr node);
			static ShNodePr get_node(const Json::Value &js, std::list<ShNodePr> &list); 

	};

}}

#endif
