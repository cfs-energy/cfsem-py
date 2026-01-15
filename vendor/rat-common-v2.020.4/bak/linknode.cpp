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

// include header
#include "linknode.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// constructor
	LinkNode::LinkNode(const arma::uword idx){
		idx_ = idx;
	}
	
	// factory function
	ShLinkNodePr LinkNode::create(const arma::uword idx){
		return std::make_shared<LinkNode>(idx);
	}

	// get index
	arma::uword LinkNode::get_index() const{
		return idx_;
	}

	// type
	std::string LinkNode::get_type(){
		return "linknode";
	}

	// link node
	Json::Value LinkNode::serialize() const{
// create json node
		Json::Value js;	
		
		// store type ID
		js["type"] = get_type();
		js["index"] = (unsigned int)idx_;
		
		// return
		return js;
	}

	// constructor
	LinkNode::LinkNode(const Json::Value &js){
		idx_ = js["index"].asUInt64();
	}

	// factory function
	ShNodePr LinkNode::create(const Json::Value &js){
		return std::make_shared<LinkNode>(js);
	}


	// add a node to the list if it is already on the list it will become a link to that node
	void LinkNode::add_node(Json::Value &js, std::list<ShNodePr> &list, ShNodePr node){
		// find if the node is in the list
		std::list<ShNodePr>::iterator it = 
			std::find(list.begin(), list.end(), node);

		// add supplied node to list
		if(it==list.end()){
			 node->serialize_nodes(list); list.push_back(node);
		}

		// create a link
		else{
			list.push_back(LinkNode::create(std::distance(list.begin(), it)));
		}
	}
	
	// retreive a node from the list, if it is a link it will be taken from its index
	ShNodePr LinkNode::get_node(const Json::Value &js, std::list<ShNodePr> &list){
		// allocate output node pointer
		ShNodePr node;

		// try to downcast node to link
		ShLinkNodePr possible_link = std::dynamic_pointer_cast<LinkNode>(list.back());

		// in case of a regular node
		if(possible_link==NULL){
			node = list.back(); list.pop_back(); node->deserialize_nodes(list);
		}

		// in case of a link
		else{
			// check length
			const arma::uword idx = possible_link->get_index();

			// check length of list
			if(idx>=list.size())rat_throw_line("linking failed element not on list");

			// Create iterator pointing to first element
			std::list<ShNodePr>::iterator it = list.begin(); std::advance(it, idx);
		
			// take node from iterator
			node = (*it);

			// remove link from list
			list.pop_back();
		}


		// return node
		return node;
	}

}}

