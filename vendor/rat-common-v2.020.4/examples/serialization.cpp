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
#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>

// specific headers
#include "node.hh"
#include "serializer.hh"

// shared pointer definition
typedef std::shared_ptr<class Animal> ShAnimalPr;
typedef std::shared_ptr<class Bear> ShBearPr;
typedef std::shared_ptr<class Duck> ShDuckPr;
typedef std::shared_ptr<class Farmer> ShFarmerPr;


class Animal: public rat::cmn::Node{
	public:
		virtual ~Animal(){};
};

class Bear: public Animal{
	public:
		Bear(){};
		static std::string get_type(){return "cmn::bear";};
		void serialize(Json::Value &js, rat::cmn::SList &) const override{
			js["type"] = get_type();
			js["says"] = "growl";
		}
		static ShBearPr create(){return std::make_shared<Bear>();};
};

class Duck: public Animal{
	public:
		Duck(){};
		static std::string get_type(){return "cmn::duck";};
		void serialize(Json::Value &js, rat::cmn::SList &) const override{
			js["type"] = get_type();
			js["says"] = "quack";
		}
		static ShDuckPr create(){return std::make_shared<Duck>();};
};

class Farmer: public Animal{
	private:
		std::list<ShAnimalPr> animals_;

	public:
		Farmer(){};
		static std::string get_type(){return "cmn::farmer";}
		void give(ShAnimalPr animal){animals_.push_back(animal);}
		void serialize(Json::Value &js, rat::cmn::SList &list) const override{
			js["type"] = get_type();
			js["says"] = "hi";
			for(std::list<ShAnimalPr>::const_iterator it = animals_.begin();it!=animals_.end();it++){
				js["animals"].append(Node::serialize_node((*it),list));
			}
		}
		void deserialize(const Json::Value &js, rat::cmn::DSList &list, const rat::cmn::NodeFactoryMap &factory_list, const boost::filesystem::path &pth) override{
			for(auto it = js["animals"].begin();it!=js["animals"].end();it++)			
				animals_.push_back(Node::deserialize_node<Animal>(*it, list, factory_list, pth));
		}
		static ShFarmerPr create(){return std::make_shared<Farmer>();}
};

// main
int main(){
	// create a duck
	ShDuckPr duck1 = Duck::create();
	ShDuckPr duck2 = Duck::create();
	ShFarmerPr farmer1 = Farmer::create();
	ShFarmerPr farmer2 = Farmer::create();
	ShFarmerPr farmer3 = Farmer::create();
	farmer1->give(farmer2);
	farmer1->give(farmer3);
	farmer2->give(duck1);
	farmer3->give(duck1);
	farmer1->give(duck2);
	farmer3->give(duck2);


	// write to output file
	rat::cmn::ShSerializerPr slzr = rat::cmn::Serializer::create();
	slzr->flatten_tree(farmer1); slzr->export_json("./test.gzip");

	slzr->register_factory(Bear::get_type(), []()->rat::cmn::ShNodePr{return Bear::create();} );
	slzr->register_factory(Farmer::get_type(), []()->rat::cmn::ShNodePr{return Farmer::create();} );
	slzr->register_factory(Duck::get_type(), []()->rat::cmn::ShNodePr{return Duck::create();} );

	// go back 
	slzr->import_json("./test.gzip");
	ShFarmerPr tree = slzr->construct_tree<Farmer>();

	// write again
	slzr->flatten_tree(tree); slzr->export_json("test2.json");
}