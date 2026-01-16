use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    if env::var("CARGO_FEATURE_RAT_MLFMM").is_err() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let wrapper_dir = manifest_dir.join("bindings").join("rat-mlfmm-c");
    let rat_mlfmm_dir = manifest_dir.join("vendor").join("rat-mlfmm");
    let rat_common_dir = manifest_dir.join("vendor").join("rat-common");
    let jsoncpp_dir = manifest_dir.join("vendor").join("jsoncpp");
    let armadillo_dir = manifest_dir.join("vendor").join("armadillo-15.2.3");
    let armadillo_zip = manifest_dir.join("vendor").join("armadillo-15.2.3.zip");
    let tclap_dir = manifest_dir.join("vendor").join("tclap");
    let boost_dir = manifest_dir.join("vendor").join("boost-boost-1.90.0");

    ensure_submodules(
        &manifest_dir,
        &[
            wrapper_dir.join("CMakeLists.txt"),
            rat_mlfmm_dir.join("CMakeLists.txt"),
            rat_common_dir.join("CMakeLists.txt"),
            jsoncpp_dir.join("CMakeLists.txt"),
            tclap_dir.join("CMakeLists.txt"),
            boost_dir.join("tools/build/src/engine/build.sh"),
        ],
    );
    ensure_armadillo_extracted(&armadillo_dir, &armadillo_zip);

    let mut cfg = cmake::Config::new(&wrapper_dir);
    let profile = env::var("PROFILE").unwrap_or_else(|_| "release".to_string());
    let build_type = if profile == "release" {
        "Release"
    } else {
        "Debug"
    };
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let cpu_flag = resolve_cpu_flag(&target_arch);
    let (c_flags_release, c_flags_debug) = compose_c_flags(&cpu_flag);
    let (cxx_flags_release, cxx_flags_debug) = compose_cxx_flags(&cpu_flag);
    let boost_cxxflags = compose_boost_cxxflags(&cpu_flag);
    cfg.profile(build_type);
    cfg.define("CMAKE_BUILD_TYPE", build_type);
    cfg.define("CMAKE_C_FLAGS_RELEASE", &c_flags_release);
    cfg.define("CMAKE_CXX_FLAGS_RELEASE", &cxx_flags_release);
    cfg.define("CMAKE_C_FLAGS_DEBUG", &c_flags_debug);
    cfg.define("CMAKE_CXX_FLAGS_DEBUG", &cxx_flags_debug);
    cfg.define("CFSEM_BOOST_CXXFLAGS", &boost_cxxflags);
    cfg.define(
        "CFSEM_EXPECT_RELEASE",
        if build_type == "Release" { "ON" } else { "OFF" },
    );
    if build_type == "Release" {
        cfg.define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "ON");
    }
    if target_os == "macos" {
        let deploy = env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "11.0".to_string());
        cfg.define("CMAKE_OSX_DEPLOYMENT_TARGET", &deploy);
    } else {
        cfg.define("BLA_STATIC", "ON");
        if target_os == "windows" {
            cfg.define("CMAKE_FIND_LIBRARY_SUFFIXES", ".lib;.a");
        } else if target_os == "linux" {
            cfg.define("CMAKE_FIND_LIBRARY_SUFFIXES", ".a");
        }
    }
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
    let _ = lib_dir;

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
    rerun_if_changed(&armadillo_zip);
    rerun_if_changed(&tclap_dir.join("CMakeLists.txt"));
    rerun_if_changed(&boost_dir.join("CMakeLists.txt"));
    rerun_if_changed(&wrapper_dir.join("cmake/BoostConfig.cmake.in"));
    rerun_if_changed(&wrapper_dir.join("cmake/ArmadilloConfig.cmake"));
}

fn rerun_if_changed(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
}

fn resolve_cpu_flag(target_arch: &str) -> Option<String> {
    let mut target_cpu = extract_target_cpu_from_rustflags();
    if target_cpu.is_none() && matches!(target_arch, "x86_64" | "x64") {
        target_cpu = Some("x86-64-v3".to_string());
    }

    target_cpu.map(|cpu| {
        if matches!(target_arch, "x86_64" | "x64") {
            format!("-march={cpu}")
        } else if matches!(target_arch, "aarch64") {
            format!("-mcpu={cpu}")
        } else {
            format!("-march={cpu}")
        }
    })
}

fn extract_target_cpu_from_rustflags() -> Option<String> {
    let encoded = env::var("CARGO_ENCODED_RUSTFLAGS").ok()?;
    let parts: Vec<&str> = encoded.split('\u{1f}').collect();
    for i in 0..parts.len() {
        let part = parts[i];
        if let Some(value) = part.strip_prefix("-Ctarget-cpu=") {
            return Some(value.to_string());
        }
        if part == "-C" {
            if let Some(next) = parts.get(i + 1) {
                if let Some(value) = next.strip_prefix("target-cpu=") {
                    return Some(value.to_string());
                }
            }
        }
    }
    None
}

fn compose_c_flags(cpu_flag: &Option<String>) -> (String, String) {
    let mut release = vec!["-O3"];
    let mut debug = vec!["-O3", "-g"];
    if let Some(flag) = cpu_flag {
        release.push(flag);
        debug.push(flag);
    }
    (release.join(" "), debug.join(" "))
}

fn compose_cxx_flags(cpu_flag: &Option<String>) -> (String, String) {
    compose_c_flags(cpu_flag)
}

fn compose_boost_cxxflags(cpu_flag: &Option<String>) -> String {
    let mut flags = vec!["-O3"];
    if let Some(flag) = cpu_flag {
        flags.push(flag);
    }
    flags.join(" ")
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
            panic!(
                "required path missing after submodule update: {}",
                path.display()
            );
        }
    }
}

fn ensure_armadillo_extracted(armadillo_dir: &Path, armadillo_zip: &Path) {
    if armadillo_dir.join("CMakeLists.txt").exists() {
        return;
    }
    if !armadillo_zip.exists() {
        panic!("armadillo zip missing: {}", armadillo_zip.display());
    }

    std::fs::create_dir_all(armadillo_dir).expect("failed to create armadillo directory");

    let file = std::fs::File::open(armadillo_zip)
        .unwrap_or_else(|err| panic!("failed to open armadillo zip: {err}"));
    let mut archive = zip::ZipArchive::new(file)
        .unwrap_or_else(|err| panic!("failed to read armadillo zip: {err}"));

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .unwrap_or_else(|err| panic!("failed to read armadillo zip entry: {err}"));
        let name = entry.name().to_string();
        let stripped = name.splitn(2, '/').nth(1).unwrap_or("");
        if stripped.is_empty() {
            continue;
        }
        let outpath = armadillo_dir.join(stripped);
        if entry.is_dir() {
            std::fs::create_dir_all(&outpath)
                .unwrap_or_else(|err| panic!("failed to create dir {outpath:?}: {err}"));
            continue;
        }

        if let Some(parent) = outpath.parent() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|err| panic!("failed to create dir {parent:?}: {err}"));
        }

        let mut outfile = std::fs::File::create(&outpath)
            .unwrap_or_else(|err| panic!("failed to create file {outpath:?}: {err}"));
        io::copy(&mut entry, &mut outfile)
            .unwrap_or_else(|err| panic!("failed to extract {outpath:?}: {err}"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Some(mode) = entry.unix_mode() {
                let mut perms = outfile
                    .metadata()
                    .unwrap_or_else(|err| panic!("failed to stat {outpath:?}: {err}"))
                    .permissions();
                perms.set_mode(mode);
                std::fs::set_permissions(&outpath, perms)
                    .unwrap_or_else(|err| panic!("failed to set perms {outpath:?}: {err}"));
            }
        }
    }
}
