use std::env;
use std::path::{Path, PathBuf};

fn main() {
    if env::var("CARGO_FEATURE_RAT_MLFMM").is_err() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let wrapper_dir = manifest_dir.join("bindings").join("rat-mlfmm-c");
    let rat_mlfmm_dir = manifest_dir.join("vendor").join("rat-mlfmm-v2.020.4");
    let rat_common_dir = manifest_dir.join("vendor").join("rat-common-v2.020.4");
    let jsoncpp_dir = manifest_dir.join("vendor").join("jsoncpp-1.9.6");
    let armadillo_dir = manifest_dir.join("vendor").join("armadillo-15.2.3");
    let tclap_dir = manifest_dir.join("vendor").join("tclap-1.2.5");

    if !wrapper_dir.join("CMakeLists.txt").exists() {
        panic!("rat-mlfmm C wrapper missing: {}", wrapper_dir.display());
    }
    if !rat_mlfmm_dir.join("CMakeLists.txt").exists() {
        panic!("rat-mlfmm vendor dir missing: {}", rat_mlfmm_dir.display());
    }
    if !rat_common_dir.join("CMakeLists.txt").exists() {
        panic!("rat-common vendor dir missing: {}", rat_common_dir.display());
    }
    if !jsoncpp_dir.join("CMakeLists.txt").exists() {
        panic!("jsoncpp vendor dir missing: {}", jsoncpp_dir.display());
    }
    if !armadillo_dir.join("CMakeLists.txt").exists() {
        panic!("armadillo vendor dir missing: {}", armadillo_dir.display());
    }
    if !tclap_dir.join("CMakeLists.txt").exists() {
        panic!("tclap vendor dir missing: {}", tclap_dir.display());
    }

    let mut cfg = cmake::Config::new(&wrapper_dir);
    cfg.define("RAT_MLFMM_DIR", rat_mlfmm_dir.to_str().unwrap());
    cfg.define("RAT_COMMON_DIR", rat_common_dir.to_str().unwrap());
    cfg.define("JSONCPP_SRC_DIR", jsoncpp_dir.to_str().unwrap());
    cfg.define("ARMADILLO_SRC_DIR", armadillo_dir.to_str().unwrap());
    cfg.define("TCLAP_SRC_DIR", tclap_dir.to_str().unwrap());
    if let Ok(prefix) = env::var("CMAKE_PREFIX_PATH") {
        cfg.define("CMAKE_PREFIX_PATH", &prefix);
    }
    if let Ok(jsoncpp_dir) = env::var("JSONCPP_DIR") {
        cfg.define("JsonCPP_DIR", &jsoncpp_dir);
    }
    if let Ok(tclap_dir) = env::var("TCLAP_DIR") {
        cfg.define("TCLAP_DIR", &tclap_dir);
    }
    if let Ok(armadillo_dir) = env::var("ARMADILLO_DIR") {
        cfg.define("Armadillo_DIR", &armadillo_dir);
    }

    let dst = cfg.build();

    let lib_dir = dst.join("lib");
    let bin_dir = dst.join("bin");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    if bin_dir.exists() {
        println!("cargo:rustc-link-search=native={}", bin_dir.display());
    }
    println!("cargo:rustc-link-lib=rat_mlfmm_c");

    rerun_if_changed(&wrapper_dir.join("CMakeLists.txt"));
    rerun_if_changed(&wrapper_dir.join("src/rat_mlfmm_c.cpp"));
    rerun_if_changed(&wrapper_dir.join("src/rat_mlfmm_c_err.cpp"));
    rerun_if_changed(&wrapper_dir.join("include/rat_mlfmm_c.h"));
    rerun_if_changed(&wrapper_dir.join("cmake/FindJsonCPP.cmake"));
    rerun_if_changed(&wrapper_dir.join("cmake/FindTCLAP.cmake"));
    rerun_if_changed(&wrapper_dir.join("cmake/ArmadilloConfig.cmake"));
    rerun_if_changed(&rat_common_dir.join("CMakeLists.txt"));
    rerun_if_changed(&rat_mlfmm_dir.join("CMakeLists.txt"));
    rerun_if_changed(&jsoncpp_dir.join("CMakeLists.txt"));
    rerun_if_changed(&armadillo_dir.join("CMakeLists.txt"));
    rerun_if_changed(&tclap_dir.join("CMakeLists.txt"));
}

fn rerun_if_changed(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
}
