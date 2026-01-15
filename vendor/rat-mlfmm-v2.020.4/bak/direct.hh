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

#ifndef DIRECT_HH
#define DIRECT_HH

#include <armadillo> 
#include <complex>
#include <cassert>
#include <cmath> // contains M_PI
#include <algorithm>
#include <cassert>
#include <memory>

#include "common/extra.hh"
#include "sources.hh"
#include "targets.hh"
#include "tarsrc.hh"
#include "settings.hh"

// shared pointer definition
class Direct;
typedef std::shared_ptr<Direct> ShDirectPr;

// direct calculation routines
class Direct: public TarSrc{
	// properties
	private:
		// pointer to settings
		ShSettingsPr stngs_;
		
	// methods
	public:
		// constructor
		Direct();
		static ShDirectPr create();
		void set_settings(ShSettingsPr stngs);
		void calculate();

};

#endif
