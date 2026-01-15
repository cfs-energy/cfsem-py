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
#include "nameable.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// get name function
	std::string Nameable::get_name() const{
		return myname_;
	}

	// set name function
	void Nameable::set_name(const std::string &name){
		myname_ = name;
	}

	// append name
	void Nameable::append_name(const std::string &append){
		myname_ = append + myname_;
	}

	// get type
	std::string Nameable::get_type(){
		return "rat::mdl::nameable";
	}

	// method for serialization into json
	void Nameable::serialize(Json::Value &js, cmn::SList &) const{
		js["name"] = myname_;
	}

	// method for deserialisation from json
	void Nameable::deserialize(const Json::Value &js, cmn::DSList &/*list*/, 
		const cmn::NodeFactoryMap &/*node_list*/, const boost::filesystem::path &/*pth*/){
		myname_ = js["name"].asString();
	}
	
}}