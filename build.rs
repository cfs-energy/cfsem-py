use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    let boost_dir = manifest_dir.join("vendor").join("boost-boost-1.90.0");

    ensure_submodules(
        &manifest_dir,
        &[
            wrapper_dir.join("CMakeLists.txt"),
            rat_mlfmm_dir.join("CMakeLists.txt"),
            rat_common_dir.join("CMakeLists.txt"),
            jsoncpp_dir.join("CMakeLists.txt"),
            armadillo_dir.join("CMakeLists.txt"),
            tclap_dir.join("CMakeLists.txt"),
            boost_dir.join("tools/build/src/engine/build.sh"),
        ],
    );

    let mut cfg = cmake::Config::new(&wrapper_dir);
    cfg.define("RAT_MLFMM_DIR", rat_mlfmm_dir.to_str().unwrap());
    cfg.define("RAT_COMMON_DIR", rat_common_dir.to_str().unwrap());
    cfg.define("JSONCPP_SRC_DIR", jsoncpp_dir.to_str().unwrap());
    cfg.define("ARMADILLO_SRC_DIR", armadillo_dir.to_str().unwrap());
    cfg.define("TCLAP_SRC_DIR", tclap_dir.to_str().unwrap());
    cfg.define("BOOST_SRC_DIR", boost_dir.to_str().unwrap());
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

    let lib_dir = dst.join("build").join("lib");
    let bin_dir = dst.join("build").join("bin");
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
    rerun_if_changed(&boost_dir.join("CMakeLists.txt"));
    rerun_if_changed(&wrapper_dir.join("cmake/BoostConfig.cmake.in"));
    rerun_if_changed(&wrapper_dir.join("cmake/ArmadilloConfig.cmake"));
}

fn rerun_if_changed(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
}

fn ensure_submodules(manifest_dir: &Path, required_paths: &[PathBuf]) {
    let missing: Vec<_> = required_paths
        .iter()
        .filter(|path| !path.exists())
        .collect();

    if missing.is_empty() {
        return;
    }

    let status = Command::new("git")
        .args(["submodule", "update", "--init", "--recursive"])
        .current_dir(manifest_dir)
        .status()
        .expect("failed to run git submodule update");

    if !status.success() {
        panic!("git submodule update failed with status {status}");
    }

    for path in required_paths {
        if !path.exists() {
            panic!("required path missing after submodule update: {}", path.display());
        }
    }
}
