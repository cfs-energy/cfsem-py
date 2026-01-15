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

// system header
#include <armadillo>
#include <cmath>

// multipole method header
#include "mlfmm.hh"
#include "softcurrentsources.hh"
#include "magnetictargets.hh"
#include "settings.hh"

// main
int main(){
    // create armadillo objects from memory pointers to mex matrices this avoids copying of data.
    arma::Mat<double> Rt; Rt.load("../meshes/dipole_test/target_coordinates.txt",arma::csv_ascii); Rt = Rt.t();
    arma::Mat<double> Rs; Rs.load("../meshes/dipole_test/source_coordinates.txt",arma::csv_ascii); Rs = Rs.t();
    arma::Mat<double> dRs; dRs.load("../meshes/dipole_test/source_direction.txt",arma::csv_ascii); dRs = dRs.t();
    arma::Mat<double> Is; Is.load("../meshes/dipole_test/source_current.txt",arma::csv_ascii); Is = Is.t();
    arma::Mat<double> epss; epss.load("../meshes/dipole_test/source_softness.txt",arma::csv_ascii); epss = epss.t();
    unsigned int num_exp = 5;

    // create armadillo timer
    arma::wall_clock timer;

    // create sources
    ShSoftCurrentSourcesPr mysources = SoftCurrentSources::create();
    mysources->set_coords(Rs); 
    mysources->set_currents(dRs.each_row()%Is);
    mysources->set_softening(epss);

    // create targets
    ShMagneticTargetsPr mytargets = MagneticTargets::create();
    mytargets->set_coords(Rt);
    mytargets->set_field_type("AHM",arma::Row<arma::uword>{3,3,3});

    // create settings
    ShSettingsPr settings = Settings::create();
    settings->set_num_exp(num_exp);
    settings->set_num_refine(240);
    settings->set_num_refine_min(0);

    // create multipole method object
    ShMlfmmPr myfmm = Mlfmm::create(settings);
    myfmm->set_sources(mysources);
    myfmm->set_targets(mytargets);

    // run calculation
    timer.tic();
    myfmm->setup();
    double tsetup = timer.toc();
    timer.tic();
    myfmm->calculate(); 
    double tfmm = timer.toc();

    // provide times
    std::printf("setup time: %2.2f\n", tsetup);
    std::printf("mlfmm time: %2.2f\n", tfmm);

    // return
    return 0;
}