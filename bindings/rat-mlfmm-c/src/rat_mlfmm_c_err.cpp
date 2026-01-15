#include "rat_mlfmm_c.h"

#include <string>

namespace {
thread_local std::string g_last_error;
}

extern "C" void rat_mlfmm_set_last_error(const char *msg) {
    if (msg) {
        g_last_error = msg;
    } else {
        g_last_error.clear();
    }
}

extern "C" const char *rat_mlfmm_last_error(void) {
    return g_last_error.c_str();
}
