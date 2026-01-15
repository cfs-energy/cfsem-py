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

// General headers
#include <algorithm>

// command line parser
#include <tclap/CmdLine.h>

// headers for common
#include "log.hh"
#include "version.hh"
#include "extra.hh"

// main
int main(int argc, char * argv[]){
	// create tclap object
	TCLAP::CmdLine cmd("Common Library",' ',rat::cmn::Extra::create_version_tag(
		CMN_PROJECT_VER_MAJOR,CMN_PROJECT_VER_MINOR,CMN_PROJECT_VER_PATCH));

	// filename input argument
	TCLAP::UnlabeledValueArg<std::string> input_filename_argument(
		"if","Input filename",true, "json/geometry/moon.json", "string", cmd);

	const rat::cmn::ShLogPr lg = rat::cmn::Log::create();
	lg->msg("%s\n",input_filename_argument.getValue().c_str());

	// done
	return 0;
}