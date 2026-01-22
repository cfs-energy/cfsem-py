#ifndef RAT_MLFMM_C_H
#define RAT_MLFMM_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #if defined(RAT_MLFMM_C_BUILD)
    #define RAT_MLFMM_C_API __declspec(dllexport)
  #else
    #define RAT_MLFMM_C_API __declspec(dllimport)
  #endif
#else
  #define RAT_MLFMM_C_API __attribute__((visibility("default")))
#endif

typedef struct rat_mlfmm_context rat_mlfmm_context;

typedef enum rat_mlfmm_direct_mode {
    RAT_MLFMM_DIRECT_ALWAYS = 0,
    RAT_MLFMM_DIRECT_THRESHOLD = 1,
    RAT_MLFMM_DIRECT_NEVER = 2
} rat_mlfmm_direct_mode;

RAT_MLFMM_C_API rat_mlfmm_context *rat_mlfmm_context_create(void);
RAT_MLFMM_C_API void rat_mlfmm_context_destroy(rat_mlfmm_context *ctx);

RAT_MLFMM_C_API int rat_mlfmm_context_set_sources_linear(
    rat_mlfmm_context *ctx,
    const double *rs_x,
    const double *rs_y,
    const double *rs_z,
    const double *drs_x,
    const double *drs_y,
    const double *drs_z,
    const double *currents,
    const double *eps,
    size_t num_sources);

RAT_MLFMM_C_API int rat_mlfmm_context_set_targets(
    rat_mlfmm_context *ctx,
    const double *rt_x,
    const double *rt_y,
    const double *rt_z,
    size_t num_targets);

RAT_MLFMM_C_API int rat_mlfmm_context_set_van_lanen(
    rat_mlfmm_context *ctx,
    int use_van_lanen);

RAT_MLFMM_C_API int rat_mlfmm_context_set_num_exp(
    rat_mlfmm_context *ctx,
    int num_exp);

RAT_MLFMM_C_API int rat_mlfmm_context_set_direct_mode(
    rat_mlfmm_context *ctx,
    rat_mlfmm_direct_mode mode);

RAT_MLFMM_C_API int rat_mlfmm_context_set_direct_threshold(
    rat_mlfmm_context *ctx,
    double threshold);

RAT_MLFMM_C_API int rat_mlfmm_context_compute_ba(
    rat_mlfmm_context *ctx,
    double *out_bx,
    double *out_by,
    double *out_bz,
    size_t out_b_len,
    double *out_ax,
    double *out_ay,
    double *out_az,
    size_t out_a_len);

RAT_MLFMM_C_API const char *rat_mlfmm_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
