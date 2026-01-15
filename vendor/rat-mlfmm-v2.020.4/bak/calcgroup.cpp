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
#include "calcgroup.hh"

// code specific to Rat
namespace rat{namespace mdl{

	// factory
	ShCalcGroupPr CalcGroup::create(){
		return std::make_shared<CalcGroup>();
	}


	// add to list
	void CalcGroup::add(ShCalcPr calc){
		// check input
		if(calc==NULL)rat_throw_line("calculation points to zero");

		// get number of models
		const arma::uword num_calc = calc_.n_elem;

		// allocate new source list
		ShCalcPrList new_calc(num_calc + 1);

		// set old and new sources
		for(arma::uword i=0;i<num_calc;i++)new_calc(i) = calc_(i);
		new_calc(num_calc) = calc;
		
		// set new source list
		calc_ = new_calc;
	}

	// must have setup function
	void CalcGroup::calculate(cmn::ShLogPr lg){
		for(arma::uword i=0;i<calc_.n_elem;i++)
			calc_(i)->calculate(lg);
	}

	// get type
	std::string CalcGroup::get_type(){
		return "mdl::calcgroup";
	}

	// method for serialization into json
	void CalcGroup::serialize(Json::Value &js, std::list<cmn::ShNodePr> &list) const{
		js["type"] = get_type();
		for(arma::uword i=0;i<calc_.n_elem;i++)js["calc"].append(cmn::Node::serialize_node(calc_(i), list));
	}

	// method for deserialisation from json
	void CalcGroup::deserialize(const Json::Value &js, std::list<cmn::ShNodePr> &list, const cmn::NodeFactoryMap &factory_list){
		// settings
		const arma::uword num_calc = js["calc"].size(); calc_.set_size(num_calc);
		for(arma::uword i=0;i<num_calc;i++)calc_(i) = cmn::Node::deserialize_node<Calc>(js["calc"].get(i,0), list, factory_list);
	}


}}

