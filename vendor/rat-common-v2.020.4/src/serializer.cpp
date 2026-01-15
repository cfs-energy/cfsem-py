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

// for time
#include <iomanip>

// include header
#include "serializer.hh"
#include "nodegroup.hh"
#include "version.hh"
#include "extra.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// constructor
	Serializer::Serializer(){
		register_constructors();
	}

	// constructor
	Serializer::Serializer(const ShNodePr &root_node){
		register_constructors(); flatten_tree(root_node);
	}

	// constructor
	Serializer::Serializer(const boost::filesystem::path &fname){
		register_constructors(); import_json(fname);
	}

	// factory
	ShSerializerPr Serializer::create(){
		return std::make_shared<Serializer>();
	}

	// factory
	ShSerializerPr Serializer::create(ShNodePr root_node){
		return std::make_shared<Serializer>(root_node);
	}

	// factory
	ShSerializerPr Serializer::create(const boost::filesystem::path &fname){
		return std::make_shared<Serializer>(fname);
	}

	// get json object
	bool Serializer::has_valid_json_root() const{
		return root_json_.isObject();
	}

	// flatten tree for raw pointer
	void Serializer::flatten_tree(const Node* root_node){
		// cheat hacks (shallow copy)
		flatten_tree(std::make_shared<Node>(*root_node));
	}

	// serialize
	void Serializer::flatten_tree(const ShNodePr& root_node){
		// reset root node
		root_json_ = Json::Value();
		// root_json_.clear();

		// create all nodes
		SList node_list;
		root_json_["tree"] = Node::serialize_node(root_node, node_list);

		// get date and time
		auto timdata = std::time(nullptr);
		auto localtime = *std::localtime(&timdata);
		std::ostringstream date_stream,time_stream;
		date_stream << std::put_time(&localtime, "%d-%m-%Y");
		time_stream << std::put_time(&localtime, "%H:%M:%S");
		const std::string date_str = date_stream.str();
		const std::string time_str = time_stream.str();

		// add version number
		root_json_["version"] = Extra::create_version_tag(
			CMN_PROJECT_VER_MAJOR,CMN_PROJECT_VER_MINOR,CMN_PROJECT_VER_PATCH);
		root_json_["major_version"] = CMN_PROJECT_VER_MAJOR;
		root_json_["minor_version"] = CMN_PROJECT_VER_MINOR;
		root_json_["patch_version"] = CMN_PROJECT_VER_PATCH;
		root_json_["date"] = date_str;
		root_json_["time"] = time_str;
		root_json_["num_nodes"] = static_cast<int>(node_list.size());
	}

	// clear tree
	void Serializer::clear_json(){
		root_json_.clear();
	}

	// allow editing of the root node
	Json::Value& Serializer::root_json(){
		return root_json_;
	}

	// deserialize
	ShNodePr Serializer::construct_tree_core(){
		// check registry
		if(factory_list_.size()==0)rat_throw_line("no classes registered for construction");

		// allocate list of nodes
		DSList node_list;

		// create directory
		boost::filesystem::path pth = pth_;

		// create root node
		ShNodePr root_node = Node::deserialize_node<Node>(root_json_["tree"],node_list,factory_list_,pth);

		// check if number of nodes set
		if(!root_json_["num_nodes"].isNull()){
			// check remaining length
			const arma::uword num_nodes = root_json_["num_nodes"].asUInt64();
			if(node_list.size()!=num_nodes)
				std::cout<<"warning: number of nodes incorrect: "<<node_list.size()<<" / "<<num_nodes<<std::endl;
		}

		// return the root node
		return root_node;
	}

	// loading input files
	void Serializer::import_json(const boost::filesystem::path &fname){
		// load json file
		if(fname.extension().string().compare(".json")==0)
			root_json_ = Node::parse_json(fname);
		else if(fname.extension().string().compare(".json.gz")==0)
			root_json_ = Node::parse_json_compressed(fname); 
		else if(fname.extension().string().compare(".gz")==0) // for backwards compatibility
			root_json_ = Node::parse_json_compressed(fname); 
		else rat_throw_line("extension not recognized");
		
		// set root
		if(root_json_["tree"].isNull()){
			Json::Value new_root;
			new_root["tree"] = root_json_;
			root_json_ = new_root;
		}

		// set path of the root file
		pth_ = boost::filesystem::path(fname).parent_path();
	}

	// writing input files
	void Serializer::export_json(const boost::filesystem::path &fname){
		// create directory
		if(fname.has_parent_path())
			boost::filesystem::create_directories(fname.parent_path());

		// call write function on node
		if(fname.extension().string().compare(".json")==0)
			Node::write_json(fname, root_json_);
		else if(fname.extension().string().compare(".json.gz")==0)
			Node::write_json_compressed(fname, root_json_);
		else if(fname.extension().string().compare(".gz")==0) // for backwards compatibility
			Node::write_json_compressed(fname, root_json_);
		else rat_throw_line("extension not recognized");
	}
	
	// registration
	void Serializer::register_factory(
		const std::string &str,NodeFactory factory){
		// check if already on the list
		// NodeFactoryMap::iterator it = factory_list_.find(str);
		// if(it!=factory_list_.end())rat_throw_line("duplicate nodefactory being registered: " + str);

		// check if already on the list
		if(factory_list_.count(str)>0)rat_throw_line("duplicate nodefactory being registered: " + str);

		// add to list
		factory_list_[str] = factory;
	}

	// registration
	void Serializer::override_factory(
		const std::string &str,NodeFactory factory){
		// check if already on the list
		// NodeFactoryMap::iterator it = factory_list_.find(str);
		// if(it!=factory_list_.end())rat_throw_line("duplicate nodefactory being registered: " + str);

		// check if already on the list
		if(factory_list_.count(str)==0)rat_throw_line("factory does not exist: " + str);

		// add to list
		factory_list_[str] = factory;
	}

	// print registered types
	void Serializer::list_factories(ShLogPr lg) const{
		lg->msg(2,"%sserializer factory list%s\n",KBLU,KNRM);
		lg->msg("number of entries: %llu\n",factory_list_.size());
		for(auto it = factory_list_.begin();it!=factory_list_.end();it++)
			lg->msg("%s\n",(*it).first.c_str());
		lg->msg(-2,"\n");
	}

	// get factory list
	NodeFactoryMap Serializer::get_factory_list() const{
		return factory_list_;
	}

	// export to string
	std::string Serializer::export_string() const{
		Json::StreamWriterBuilder builder;
		builder["useSpecialFloats"] = true;
		builder["indentation"] = ""; // If you want whitespace-less output
		return Json::writeString(builder, root_json_);
	}

	// import from string
	bool Serializer::import_string(const std::string &str){
		Json::CharReaderBuilder builder;
		builder["allowSpecialFloats"] = true;
		std::string errs;
		std::stringstream sstr(str);
		return Json::parseFromStream(builder, sstr, &root_json_, &errs);
	}

	// register constructors
	void Serializer::register_constructors(){
		register_factory<NodeGroup>();
	}

}}