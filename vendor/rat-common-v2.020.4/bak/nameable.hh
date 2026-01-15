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

#ifndef CMN_NAMEABLE_HH
#define CMN_NAMEABLE_HH

#include <string>

#include "node.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// template for coil
	class Nameable: virtual public Node{
		// properties
		protected:
			// update lock
			std::string myname_ = "";

		// methods
		public:
			// get name
			std::string get_name()const;

			// get name
			void set_name(const std::string &name);

			// append name
			void append_name(const std::string &append);

			// serialization
			static std::string get_type();
			virtual void serialize(Json::Value &js, cmn::SList &list) const override;
			virtual void deserialize(const Json::Value &js, cmn::DSList &list, const cmn::NodeFactoryMap &factory_list, const boost::filesystem::path &pth) override;
	};

}}

#endif
