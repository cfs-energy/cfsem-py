/* MagRat - A heavilly vectorised and parallelised 
implementation of the Multi-Level Fast Multipole Method (MLFMM).
Copyright (C) 2017  Jeroen van Nugteren

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.*/

// include guard
#ifndef TAR_SRC_HH
#define TAR_SRC_HH

// general headers
#include <armadillo> 
#include <cassert>

// specific headers
#include "common/defines.hh"
#include "sources.hh"
#include "targets.hh"

// superclass for objects that store targets and sources

// code specific to Rat
namespace Rat{
	// class template
	class TarSrc{
		// properties
		protected:
			// pointers to targets and sources
			arma::field<ShSourcesPr> src_;
			arma::field<ShTargetsPr> tar_;

		// methods
		public:
			// set pointers
			void set_sources(ShSourcesPr src);
			void set_targets(ShTargetsPr tar);
			void add_sources(ShSourcesPr src);
			void add_targets(ShTargetsPr tar);
			void set_sources(arma::field<ShSourcesPr> src);
			void set_targets(arma::field<ShTargetsPr> tar);
			void add_sources(arma::field<ShSourcesPr> src);
			void add_targets(arma::field<ShTargetsPr> tar);

			// getting pointers
			arma::field<ShSourcesPr> get_sources() const;
			arma::field<ShTargetsPr> get_targets() const;

			// counters
			arma::uword num_source_objects() const;
			arma::uword num_target_objects() const;
	};
}

#endif