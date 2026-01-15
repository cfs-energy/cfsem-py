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
#include "nodegroup.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// default constructor
	NodeGroup::NodeGroup(){
		
	}
	
	// factory
	ShNodeGroupPr NodeGroup::create(){
		//return ShPathCirclePr(new PathCircle);
		return std::make_shared<NodeGroup>();
	}


	// get node with specific index
	ShNodePr NodeGroup::get_tree_node(const arma::uword index) const{
		auto it = nodes_.find(index);
		if(it==nodes_.end())rat_throw_line("node id does not exist");
		return (*it).second;
	}

	// add node
	arma::uword NodeGroup::add_tree_node(const cmn::ShNodePr &node){
		const arma::uword index = nodes_.size() + 1;
		nodes_.insert({index, node}); return index;
	}

	// delete child node
	bool NodeGroup::delete_tree_node(const arma::uword index){
		auto it = nodes_.find(index);
		if(it==nodes_.end())return false;
		nodes_.erase(it); return true;
	}

	// get number of child nodes
	arma::uword NodeGroup::get_num_tree_nodes() const{
		return nodes_.size();
	}

	// reindex
	void NodeGroup::reindex(){
		std::map<arma::uword, ShNodePr> new_map; arma::uword idx = 1;
		for(auto it=nodes_.begin();it!=nodes_.end();it++,idx++)
			new_map.insert({idx, (*it).second});
		nodes_ = new_map;
	}

	// get type
	std::string NodeGroup::get_type(){
		return "rat::cmn::nodegroup";
	}

	// method for serialization into json
	void NodeGroup::serialize(
		Json::Value &js, cmn::SList &list) const{
		// type
		js["type"] = get_type();

		// serialize nodes
		for(auto it = nodes_.begin();it!=nodes_.end();it++)
			js["nodes"].append(cmn::Node::serialize_node((*it).second, list));
	}

	// method for deserialisation from json
	void NodeGroup::deserialize(
		const Json::Value &js, cmn::DSList &list, 
		const cmn::NodeFactoryMap &factory_list, 
		const boost::filesystem::path &pth){
		// deserialize node list
		for(auto it = js["nodes"].begin();it!=js["nodes"].end();it++)
			add_tree_node(cmn::Node::deserialize_node<Node>(
				(*it), list, factory_list, pth));
	}

}}
