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

// general headers
#include <limits>
#include <cmath>

// include error
#include "error.hh"

// json header
#include <json/json.h>

// this tests of the json cpp library can correctly serialize infinity
// if this test fails likely your jsoncpp version is out of date!

// string to json
Json::Value string2json(const std::string& str){
	// create reader
	Json::CharReaderBuilder builder;
	builder["allowSpecialFloats"] = true;

	// create stream
	std::string errs;
	std::stringstream sstr(str);
	
	// parse
	Json::Value js;
	bool ok = Json::parseFromStream(builder, sstr, &js, &errs);

	// check for errors
	if(ok==false)rat_throw_line("json parsing error: " + errs);

	// return json
	return js;
}

// json to string
std::string json2string(const Json::Value& js){
	// write to string
	Json::StreamWriterBuilder builder;
	builder["useSpecialFloats"] = true;
	builder["indentation"] = ""; // If you want whitespace-less output

	// parse to string
	const std::string str = Json::writeString(builder, js);

	// output string
	return str;
}

// main
int main(){
	// make a test json value
	Json::Value js;
	js["thisisone"] = 1.0;
	js["thisisinf"] = std::numeric_limits<double>::infinity();
	
	// start testing
	std::string str = json2string(js);
	
	std::cout<<str<<std::endl;

	Json::Value js2 = string2json(str);

	std::cout<<js2<<std::endl;

	// check output
	if(!std::isinf(js2["thisisinf"].asDouble()))
		rat_throw_line("is not inf");
}