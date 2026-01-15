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
#include "node.hh"

// boost streams
#include <boost/iostreams/filter/counter.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/regex.hpp>

// code specific to Rat
namespace rat{namespace cmn{

	// serialize nodes
	Json::Value Node::serialize_node(const ShNodePr &node, SList &list){
		// allocate output json
		Json::Value js; 

		// in case of Null
		if(node==NULL){
			js["type"] = "cmn::null"; return js;
		}

		// if it is not on the list add the node to the list and serialize it
		if(list.count(node)==0){
			node->serialize(js,list); 
			arma::uword id = list.size();
			list[node] = id;
			js["id"] = static_cast<int>(id);
		}

		// create a link
		else{
			js["type"] = "cmn::link";
			js["target_index"] = static_cast<int>(list.at(node));
		}

		// return json
		return js;
	}


	// deserialize nodes
	ShNodePr Node::deserialize_node_core(
		const Json::Value &js, DSList &list, 
		const NodeFactoryMap &factory_list,
		const boost::filesystem::path &pth){

		// check if it is an array of objects
		if(js.isArray()){
			rat_throw_line("input json node is array");
		}

		// check if json input is NULL
		if(js.isNull()){
			return NULL;
		}

		// check if the json input is a file instead of an object
		Json::Value js_object = js;
		boost::filesystem::path next_path = pth;
		if(js.isString()){
			boost::filesystem::path filepath(js.asString());
			if(filepath.empty())rat_throw_line("object is a serialized as string but is not a filename");
			if(filepath.is_relative())filepath = boost::filesystem::absolute(pth/filepath);
			next_path = filepath.parent_path();
			js_object = parse_json(filepath);
		}

		//.check 
		if(!js_object.isObject())rat_throw_line("json is not an object");

		// get type string
		const std::string type = js_object["type"].asString();

		// allocate output node pointer
		ShNodePr node;

		// check if string exists
		if(type.length()==0){
			rat_throw_line("type is empty string");
		}
		
		// check for node this means that the type was not specified in class
		else if(type.compare("cmn::node")==0){
			rat_throw_line("type not specified");
		}
			
		// check for null node
		else if(type.compare("cmn::null")==0){
			node = NULL;
		}

		// in case of a link
		else if(type.compare("cmn::link")==0){
			// check length
			const arma::uword idx = js_object["target_index"].asUInt64();

			// find linked index in the list
			auto it = list.find(idx);

			// check when triggered usually the order of serialization is off
			if(it==list.end()){
				// rat_throw_line("could not find linked object: " + std::to_string(idx)); 
				std::cout<<"warning: could not find linked object: "<<idx<<std::endl;
				node = NULL;
			}

			// point to linked node
			else node = (*it).second;
		}

		// in case of a regular node
		else{
			// find the factory based on the type string
			NodeFactoryMap::const_iterator it = factory_list.find(type);
			
			// check if the factory was found and return
			if(it!=factory_list.end())node = it->second();
			else rat_throw_line("type not registered: " + type);

			// now deserialize it
			node->deserialize(js_object,list,factory_list,next_path);

			// add node to list
			list[js_object["id"].asUInt64()] = node;
		}

		// return node
		return node;
	}

	// fix inf and nan
	void Node::fix_file(const boost::filesystem::path &fname){
		// read file
		std::ifstream infile(fname.string(),std::ios_base::in|std::ios_base::binary);
		std::stringstream buffer;
		buffer << infile.rdbuf();
		std::string str = buffer.str();
		infile.close();

		// apply regex filter
		boost::regex pattern("1e\\+9999");
		str = boost::regex_replace(str, pattern, "Infinity");

		// write file
		std::ofstream outfile(fname.string(), std::ios_base::out|std::ios_base::binary|std::ios_base::trunc);
		outfile << str;
		outfile.close();
	}

	// static json parse function
	Json::Value Node::parse_json(const boost::filesystem::path &fname, const bool is_compressed){
		// check if path exists
		boost::system::error_code ec;
		if(!boost::filesystem::exists(fname,ec))rat_throw_line("json file '" + fname.string() + "' does not exist: " + ec.message());

		// create reader
		Json::CharReaderBuilder rbuilder;
		rbuilder["allowSpecialFloats"] = true;

		// open a boost stream
		boost::iostreams::filtering_istream in;
		if(is_compressed)in.push(boost::iostreams::gzip_decompressor());
		in.push(boost::iostreams::file_source(fname.string(),std::ios_base::in|std::ios_base::binary));

		// allocate json
		Json::Value js;

		// read json
		std::string errs;
		bool ok = Json::parseFromStream(rbuilder, in, &js, &errs);

		// check for errors
		if(ok==false){
			// try to fix the file
			// this is because previously when not using 
			// useSpecialFloats jsoncpp serialized infinity 
			// to 1e+9999 which it can not load back in
			// this allows the user to load back in broken
			// json files from before v2.017.3
			if(!is_compressed){
				std::cout<<"encountered: "<<errs<<std::endl;
				std::cout<<"try fixing file"<<std::endl;
				
				// run the fix
				fix_file(fname);
				
				// read json again
				in.reset(); in.push(boost::iostreams::file_source(fname.string())); errs.clear(); js.clear();
				ok = Json::parseFromStream(rbuilder, in, &js, &errs);
			}

			// could not parse json
			if(ok==false){rat_throw_line("json parsing error: " + fname.string() + errs);}

			// the file was parsed correctly this time
			else{std::cout<<"file seems to be fixed now"<<std::endl;}
		}

		// return json
		return js;
	}

	// static json parse function
	Json::Value Node::parse_json_compressed(const boost::filesystem::path &fname){
		return parse_json(fname,true);
	}

	// static json write function
	void Node::write_json(const boost::filesystem::path &fname, const Json::Value &js){
		// create a streamwriter
		Json::StreamWriterBuilder builder;
		builder["useSpecialFloats"] = true;

		// create writer
		const std::unique_ptr<Json::StreamWriter> 
			writer(builder.newStreamWriter());

		// open file
		std::ofstream of_id(fname.string(),std::ios_base::out|std::ios_base::binary);

		// write to file
		writer->write(js, &of_id);

		// close file stream
		of_id.close();
	}

	// static json write function
	void Node::write_json_compressed(const boost::filesystem::path &fname, const Json::Value &js){
		// create a streamwriter
		Json::StreamWriterBuilder builder;
		builder["useSpecialFloats"] = true;

		// create writer
		const std::unique_ptr<Json::StreamWriter> 
			writer(builder.newStreamWriter());

		// create stream with compressor
		boost::iostreams::filtering_ostream out;
		out.push(boost::iostreams::gzip_compressor());
		out.push(boost::iostreams::file_sink(fname.string(), std::ios_base::out|std::ios_base::binary));

		// write to file
		writer->write(js, &out);

		// close file stream
		boost::iostreams::close(out); // Needed for flushing the data from compressor
	}

	// get name function
	const std::string& Node::get_name() const{
		return myname_;
	}

	// set name function
	void Node::set_name(const std::string &name){
		myname_ = name;
	}

	// append name
	void Node::append_name(const std::string &ext, const bool append_back){
		if(append_back==false)
			myname_ = ext + "_" + myname_;
		else myname_ = myname_ + "_" + ext;
	}

	// menu status (for gui)
	bool Node::get_tree_open()const{
		return tree_open_;
	}

	// set menu status
	void Node::set_tree_open(const bool tree_open){
		tree_open_ = tree_open;
	}

	// toggle tree menu status
	void Node::toggle_tree_open(){
		tree_open_ = !tree_open_;
	}

	// setters
	void Node::set_enable(const bool enable){
		enable_ = enable;
	}

	// getters
	bool Node::get_enable()const{
		return enable_;
	}

	// toggle tree menu status
	void Node::toggle_enable(){
		enable_ = !enable_;
	}

	// check validity 
	// by default nodes are always valid
	bool Node::is_valid(const bool /*enable_throws*/) const{
		return true;
	}

	// get node with specific index
	ShNodePr Node::get_tree_node(const arma::uword /*index*/) const{
		rat_throw_line("can not get because node is a leaf: " + get_name()); return NULL;
	}
	
	// add node
	arma::uword Node::add_tree_node(const cmn::ShNodePr &/*node*/){
		return 0;
	}

	// delete child node
	bool Node::delete_tree_node(const arma::uword /*index*/){
		return false;
	}

	// get number of child nodes
	arma::uword Node::get_num_tree_nodes() const{
		return 0;
	}

	// get multiple nodes
	std::list<ShNodePr> Node::get_tree_nodes() const{
		std::list<ShNodePr> node_list;
		const arma::uword num_nodes = get_num_tree_nodes();
		for(arma::uword i=1;i<=num_nodes;i++)
			node_list.push_back(get_tree_node(i));
		return node_list;
	}

	// add multiple nodes
	arma::Row<arma::uword> Node::add_tree_nodes(const std::list<ShNodePr> &nodes){
		// create an index map
		std::map<ShNodePr, arma::uword> index_map; arma::uword idx=0;

		// add nodes
		for(auto it=nodes.begin();it!=nodes.end();it++){
			add_tree_node(*it);
			index_map.insert({*it,idx++});
		}

		// check where the nodes ended up
		arma::Row<arma::uword> indices(nodes.size());
		for(arma::uword i=1;i<=get_num_tree_nodes();i++){
			const ShNodePr& node = get_tree_node(i);
			if(node==NULL)continue;
			auto it = index_map.find(node);
			if(it==index_map.end())continue;
			indices((*it).second) = i;
		}

		// return the indices
		return indices;
	}

	// try to add node
	bool Node::is_addable(const cmn::ShNodePr &node){
		arma::uword idx = add_tree_node(node);
		if(idx==0)return false;
		if(delete_tree_node(idx)!=true)
			rat_throw_line("can not delete added node");
		reindex(); // always reindex after delete 
		return true;
	}

	// clear nodes
	void Node::clear_tree_nodes(){
		const arma::uword num_nodes = get_num_tree_nodes();
		for(arma::uword i=1;i<=num_nodes;i++)delete_tree_node(i);
		reindex();
	}

	// get type
	std::string Node::get_type(){
		return "rat::cmn::node";
	}

	// method for serialization into json
	void Node::serialize(
		Json::Value &js, cmn::SList &/*list*/) const{
		js["type"] = get_type();
		
		// properties
		js["tree_open"] = tree_open_;
		js["name"] = myname_;
		js["enable"] = enable_;
	}

	// method for deserialisation from json
	void Node::deserialize(
		const Json::Value &js, cmn::DSList &/*list*/, 
		const cmn::NodeFactoryMap &/*factory_list*/, 
		const boost::filesystem::path &/*pth*/){
		if(js.isMember("name"))set_name(js["name"].asString());
		if(js.isMember("tree_open"))set_tree_open(js["tree_open"].asBool());
		if(js.isMember("enable"))set_enable(js["enable"].asBool());
	}

	// serialization of armadillo row vectors
	Json::Value Node::serialize_matrix(const arma::Mat<fltp> &V){
		Json::Value js; 
		js["num_rows"] = static_cast<unsigned int>(V.n_rows);
		js["num_cols"] = static_cast<unsigned int>(V.n_cols);
		for(arma::uword i=0;i<V.n_elem;i++)
			js["data"].append(V(i));
		return js;
	}

	// deserialization of armadillo row vectors
	arma::Mat<fltp> Node::deserialize_matrix(const Json::Value &js){
		// get matrix dimensions
		const arma::uword n = js["num_rows"].asUInt64();
		const arma::uword m = js["num_cols"].asUInt64();

		// allocate
		arma::Mat<fltp> V(n,m);
		
		// fill matrix column wise
		arma::uword idx = 0;
		for(auto it = js["data"].begin();it!=js["data"].end();it++,idx++)
			V(idx) = (*it).ASFLTP();

		// check 
		if(idx!=n*m)rat_throw_line("matrix dimensions do not match number of elements");

		// return matrix
		return V;
	}

	// serialization of armadillo cube
	Json::Value Node::serialize_cube(const arma::Cube<fltp>& V) {
		Json::Value js;
		js["num_rows"] = static_cast<unsigned int>(V.n_rows);
		js["num_cols"] = static_cast<unsigned int>(V.n_cols);
		js["num_slices"] = static_cast<unsigned int>(V.n_slices);
		for(arma::uword i = 0; i < V.n_elem; i++)
			js["data"].append(V(i));
		return js;
	}

	// deserialization of armadillo cube
	arma::Cube<fltp> Node::deserialize_cube(const Json::Value& js) {
		// get cube dimensions
		const arma::uword n = js["num_rows"].asUInt64();
		const arma::uword m = js["num_cols"].asUInt64();
		const arma::uword s = js["num_slices"].asUInt64();

		// allocate
		arma::Cube<fltp> V(n, m, s);

		// fill matrix column wise
		arma::uword idx = 0;
		for(auto it = js["data"].begin(); it != js["data"].end(); it++, idx++)
			V(idx) = (*it).ASFLTP();

		// check
		if(idx != n * m * s) rat_throw_line("cube dimensions do not match number of elements");

		// return matrix
		return V;
	}

	// serialization of armadillo row vectors
	Json::Value Node::serialize_uword_matrix(const arma::Mat<arma::uword> &V){
		Json::Value js; 
		js["num_rows"] = static_cast<unsigned int>(V.n_rows);
		js["num_cols"] = static_cast<unsigned int>(V.n_cols);
		for(arma::uword i=0;i<V.n_elem;i++)
			js["data"].append(static_cast<unsigned int>(V(i)));
		return js;
	}

	// deserialization of armadillo row vectors
	arma::Mat<arma::uword> Node::deserialize_uword_matrix(const Json::Value &js){
		// get matrix dimensions
		const arma::uword n = js["num_rows"].asUInt64();
		const arma::uword m = js["num_cols"].asUInt64();

		// allocate
		arma::Mat<arma::uword> V(n,m);
		
		// fill matrix column wise
		arma::uword idx = 0;
		for(auto it = js["data"].begin();it!=js["data"].end();it++,idx++)
			V(idx) = (*it).asUInt64();

		// check 
		if(idx!=n*m)rat_throw_line("matrix dimensions do not match number of elements");

		// return matrix
		return V;
	}

	// set fileseparators for any path to the
	// correct direction based on the operating system
	std::string Node::fix_filesep(const std::string& path){
		std::string fixed_path = path;
		#if defined(__linux__) || defined(__APPLE__) || defined(__MACH__)
		std::replace(fixed_path.begin(), fixed_path.end(), '\\', '/');
		#elif defined(_WIN32)
		std::replace(fixed_path.begin(), fixed_path.end(), '/', '\\');
		#endif
		return fixed_path;
	}

}}
