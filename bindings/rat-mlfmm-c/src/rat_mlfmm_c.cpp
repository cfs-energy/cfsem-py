#include "rat_mlfmm_c.h"

#include <armadillo>
#include <exception>
#include <new>
#include <vector>

#include "currentsources.hh"
#include "mgntargets.hh"
#include "mlfmm.hh"
#include "settings.hh"

extern "C" void rat_mlfmm_set_last_error(const char *msg);

namespace {
struct Context {
    rat::fmm::ShCurrentSourcesPr sources;
    rat::fmm::ShMgnTargetsPr targets;
    rat::fmm::ShSettingsPr settings;
    rat::fmm::ShMlfmmPr mlfmm;
    bool use_van_lanen = true;
};

static arma::Mat<rat::fltp> copy_mat_3xn(const double *data, size_t n_cols) {
    arma::Mat<rat::fltp> out(3, n_cols);
    for (size_t j = 0; j < n_cols; ++j) {
        const size_t base = 3 * j;
        out(0, j) = static_cast<rat::fltp>(data[base]);
        out(1, j) = static_cast<rat::fltp>(data[base + 1]);
        out(2, j) = static_cast<rat::fltp>(data[base + 2]);
    }
    return out;
}

static arma::Row<rat::fltp> copy_row(const double *data, size_t n) {
    arma::Row<rat::fltp> out(n);
    for (size_t i = 0; i < n; ++i) {
        out(i) = static_cast<rat::fltp>(data[i]);
    }
    return out;
}

static int set_error_and_return(const char *msg) {
    rat_mlfmm_set_last_error(msg);
    return 0;
}

static int handle_exception(const std::exception &ex) {
    return set_error_and_return(ex.what());
}
} // namespace

extern "C" rat_mlfmm_context *rat_mlfmm_context_create(void) {
    try {
        auto *ctx = new Context();
        ctx->settings = rat::fmm::Settings::create();
        rat_mlfmm_set_last_error(nullptr);
        return reinterpret_cast<rat_mlfmm_context *>(ctx);
    } catch (const std::exception &ex) {
        handle_exception(ex);
    } catch (...) {
        set_error_and_return("unknown error in rat_mlfmm_context_create");
    }
    return nullptr;
}

extern "C" void rat_mlfmm_context_destroy(rat_mlfmm_context *ctx) {
    if (!ctx) {
        return;
    }
    auto *raw = reinterpret_cast<Context *>(ctx);
    delete raw;
}

extern "C" int rat_mlfmm_context_set_sources_linear(
    rat_mlfmm_context *ctx,
    const double *rs_xyz,
    const double *drs_xyz,
    const double *currents,
    const double *eps,
    size_t num_sources) {

    if (!ctx || !rs_xyz || !drs_xyz || !currents || !eps) {
        return set_error_and_return("null pointer in set_sources_linear");
    }
    if (num_sources == 0) {
        return set_error_and_return("num_sources must be positive");
    }

    try {
        auto *raw = reinterpret_cast<Context *>(ctx);
        arma::Mat<rat::fltp> Rs = copy_mat_3xn(rs_xyz, num_sources);
        arma::Mat<rat::fltp> dRs = copy_mat_3xn(drs_xyz, num_sources);
        arma::Row<rat::fltp> Is = copy_row(currents, num_sources);
        arma::Row<rat::fltp> epss = copy_row(eps, num_sources);

        raw->sources = rat::fmm::CurrentSources::create(Rs, dRs, Is, epss);
        raw->sources->set_van_Lanen(raw->use_van_lanen);
        rat_mlfmm_set_last_error(nullptr);
        return 1;
    } catch (const std::exception &ex) {
        return handle_exception(ex);
    } catch (...) {
        return set_error_and_return("unknown error in set_sources_linear");
    }
}

extern "C" int rat_mlfmm_context_set_targets(
    rat_mlfmm_context *ctx,
    const double *rt_xyz,
    size_t num_targets) {

    if (!ctx || !rt_xyz) {
        return set_error_and_return("null pointer in set_targets");
    }
    if (num_targets == 0) {
        return set_error_and_return("num_targets must be positive");
    }

    try {
        auto *raw = reinterpret_cast<Context *>(ctx);
        arma::Mat<rat::fltp> Rt = copy_mat_3xn(rt_xyz, num_targets);
        raw->targets = rat::fmm::MgnTargets::create(Rt);
        raw->targets->set_field_type('B', 3);
        rat_mlfmm_set_last_error(nullptr);
        return 1;
    } catch (const std::exception &ex) {
        return handle_exception(ex);
    } catch (...) {
        return set_error_and_return("unknown error in set_targets");
    }
}

extern "C" int rat_mlfmm_context_set_van_lanen(
    rat_mlfmm_context *ctx,
    int use_van_lanen) {

    if (!ctx) {
        return set_error_and_return("null pointer in set_van_lanen");
    }

    try {
        auto *raw = reinterpret_cast<Context *>(ctx);
        raw->use_van_lanen = (use_van_lanen != 0);
        if (raw->sources) {
            raw->sources->set_van_Lanen(raw->use_van_lanen);
        }
        rat_mlfmm_set_last_error(nullptr);
        return 1;
    } catch (const std::exception &ex) {
        return handle_exception(ex);
    } catch (...) {
        return set_error_and_return("unknown error in set_van_lanen");
    }
}

extern "C" int rat_mlfmm_context_set_num_exp(
    rat_mlfmm_context *ctx,
    int num_exp) {

    if (!ctx) {
        return set_error_and_return("null pointer in set_num_exp");
    }
    if (num_exp <= 0) {
        return set_error_and_return("num_exp must be positive");
    }

    try {
        auto *raw = reinterpret_cast<Context *>(ctx);
        raw->settings->set_num_exp(num_exp);
        rat_mlfmm_set_last_error(nullptr);
        return 1;
    } catch (const std::exception &ex) {
        return handle_exception(ex);
    } catch (...) {
        return set_error_and_return("unknown error in set_num_exp");
    }
}

extern "C" int rat_mlfmm_context_set_direct_mode(
    rat_mlfmm_context *ctx,
    rat_mlfmm_direct_mode mode) {

    if (!ctx) {
        return set_error_and_return("null pointer in set_direct_mode");
    }

    try {
        auto *raw = reinterpret_cast<Context *>(ctx);
        rat::fmm::DirectMode dm = rat::fmm::DirectMode::NEVER;
        switch (mode) {
            case RAT_MLFMM_DIRECT_ALWAYS:
                dm = rat::fmm::DirectMode::ALWAYS;
                break;
            case RAT_MLFMM_DIRECT_THRESHOLD:
                dm = rat::fmm::DirectMode::TRESHOLD;
                break;
            case RAT_MLFMM_DIRECT_NEVER:
                dm = rat::fmm::DirectMode::NEVER;
                break;
            default:
                return set_error_and_return("invalid direct mode");
        }
        raw->settings->set_direct(dm);
        rat_mlfmm_set_last_error(nullptr);
        return 1;
    } catch (const std::exception &ex) {
        return handle_exception(ex);
    } catch (...) {
        return set_error_and_return("unknown error in set_direct_mode");
    }
}

extern "C" int rat_mlfmm_context_set_direct_threshold(
    rat_mlfmm_context *ctx,
    double threshold) {

    if (!ctx) {
        return set_error_and_return("null pointer in set_direct_threshold");
    }
    if (threshold <= 0.0) {
        return set_error_and_return("direct threshold must be positive");
    }

    try {
        auto *raw = reinterpret_cast<Context *>(ctx);
        raw->settings->set_direct_tresh(static_cast<rat::fltp>(threshold));
        rat_mlfmm_set_last_error(nullptr);
        return 1;
    } catch (const std::exception &ex) {
        return handle_exception(ex);
    } catch (...) {
        return set_error_and_return("unknown error in set_direct_threshold");
    }
}

extern "C" int rat_mlfmm_context_compute_b(
    rat_mlfmm_context *ctx,
    double *out_b_xyz,
    size_t out_len) {

    if (!ctx || !out_b_xyz) {
        return set_error_and_return("null pointer in compute_b");
    }

    try {
        auto *raw = reinterpret_cast<Context *>(ctx);
        if (!raw->sources) {
            return set_error_and_return("sources not set");
        }
        if (!raw->targets) {
            return set_error_and_return("targets not set");
        }

        const size_t num_targets = raw->targets->num_targets();
        const size_t needed = 3 * num_targets;
        if (out_len < needed) {
            return set_error_and_return("output buffer too small");
        }

        raw->mlfmm = rat::fmm::Mlfmm::create(raw->sources, raw->targets, raw->settings);
        raw->mlfmm->setup();
        raw->mlfmm->calculate();

        const arma::Mat<rat::fltp> B = raw->targets->get_field('B');
        if (B.n_rows != 3 || B.n_cols != num_targets) {
            return set_error_and_return("unexpected B-field shape");
        }

        for (size_t j = 0; j < num_targets; ++j) {
            const size_t base = 3 * j;
            out_b_xyz[base] = static_cast<double>(B(0, j));
            out_b_xyz[base + 1] = static_cast<double>(B(1, j));
            out_b_xyz[base + 2] = static_cast<double>(B(2, j));
        }

        rat_mlfmm_set_last_error(nullptr);
        return 1;
    } catch (const std::exception &ex) {
        return handle_exception(ex);
    } catch (...) {
        return set_error_and_return("unknown error in compute_b");
    }
}
