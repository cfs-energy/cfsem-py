use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    if env::var("CARGO_FEATURE_RAT_MLFMM").is_err() {
        return;
    }

    let python_feature = env::var_os("CARGO_FEATURE_PYTHON").is_some();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let patches_dir = manifest_dir.join("bindings").join("vendor-patches");
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
    ensure_boost_headers(&boost_dir);
    ensure_armadillo_extracted(&armadillo_dir, &armadillo_zip);
    apply_vendor_patches(&manifest_dir, &patches_dir, &rat_common_dir, &rat_mlfmm_dir);

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let cpu_flag = resolve_cpu_flag(&target_arch);
    let (mut c_flags_release, mut c_flags_debug) = compose_c_flags(&cpu_flag);
    let (mut cxx_flags_release, mut cxx_flags_debug) = compose_cxx_flags(&cpu_flag);
    let build_profile = env::var("PROFILE").unwrap_or_else(|_| "release".to_string());
    let is_release = build_profile == "release";
    if target_os == "linux" {
        ensure_tool("clang");
        ensure_tool("clang++");
        ensure_tool("ld.lld");
    }

    let rat_mlfmm_c_src = wrapper_dir.join("src").join("rat_mlfmm_c.cpp");
    let rat_mlfmm_c_include = wrapper_dir.join("include");
    let rat_mlfmm_include = rat_mlfmm_dir.join("include");
    let rat_common_include = rat_common_dir.join("include");
    let boost_include = boost_dir.clone();
    let rat_common_shim = out_dir.join("rat-common-include");
    let rat_common_shim_dest = rat_common_shim.join("rat").join("common");
    std::fs::create_dir_all(&rat_common_shim_dest)
        .unwrap_or_else(|err| panic!("failed to create {rat_common_shim_dest:?}: {err}"));
    if let Ok(entries) = std::fs::read_dir(&rat_common_include) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("hh") {
                continue;
            }
            if let Some(name) = path.file_name() {
                let dest = rat_common_shim_dest.join(name);
                let _ = std::fs::copy(&path, dest);
            }
        }
    }
    let armadillo_include = armadillo_dir.join("include");
    let jsoncpp_include = jsoncpp_dir.join("include");
    let mut cc_build = cc::Build::new();
    cc_build.cargo_metadata(false); // We emit rat_mlfmm_c link directives ourselves to avoid duplicate archives.
    cc_build.cpp(true);
    if target_os == "linux" {
        cc_build.compiler("clang++");
    }
    cc_build.file(&rat_mlfmm_c_src);
    cc_build.include(&rat_mlfmm_c_include);
    cc_build.include(&rat_mlfmm_include);
    cc_build.include(&rat_common_shim);
    cc_build.include(&armadillo_include);
    cc_build.include(&jsoncpp_include);
    cc_build.include(&boost_include);
    cc_build.define("RAT_MLFMM_C_BUILD", None);
    cc_build.define("RAT_DOUBLE_PRECISION", None);
    cc_build.flag_if_supported("-std=c++14");
    cc_build.flag_if_supported("-fPIC");
    if is_release {
        for flag in cxx_flags_release.split_whitespace() {
            cc_build.flag(flag);
        }
    } else {
        for flag in cxx_flags_debug.split_whitespace() {
            cc_build.flag(flag);
        }
    }
    cc_build.out_dir(&out_dir);
    cc_build.compile("rat_mlfmm_c");

    let mut cfg = cmake::Config::new(&wrapper_dir);
    let profile = env::var("PROFILE").unwrap_or_else(|_| "release".to_string());
    let build_type = if profile == "release" {
        "Release"
    } else {
        "Debug"
    };
    let boost_cxxflags = compose_boost_cxxflags(&cpu_flag);
    cfg.profile(build_type);
    cfg.define("CMAKE_BUILD_TYPE", build_type);
    cfg.define("CMAKE_C_FLAGS_RELEASE", &c_flags_release);
    cfg.define("CMAKE_CXX_FLAGS_RELEASE", &cxx_flags_release);
    cfg.define("CMAKE_C_FLAGS_DEBUG", &c_flags_debug);
    cfg.define("CMAKE_CXX_FLAGS_DEBUG", &cxx_flags_debug);
    if target_os == "linux" {
        cfg.define("CMAKE_C_COMPILER", "clang");
        cfg.define("CMAKE_CXX_COMPILER", "clang++");
        cfg.define("CMAKE_EXE_LINKER_FLAGS", "-fuse-ld=lld");
        cfg.define("CMAKE_SHARED_LINKER_FLAGS", "-fuse-ld=lld");
        cfg.define("CMAKE_MODULE_LINKER_FLAGS", "-fuse-ld=lld");
    }
    cfg.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
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
    let rat_common_lib_dir = dst.join("build").join("rat-common-build").join("lib");
    let rat_mlfmm_lib_dir = dst.join("build").join("rat-mlfmm-build").join("lib");
    let rat_common_lib = rat_common_lib_dir.join("libratcmn.a");
    let rat_mlfmm_lib = rat_mlfmm_lib_dir.join("libratmlfmm.a");
    let rat_mlfmm_c_lib = out_dir.join("librat_mlfmm_c.a");
    let boost_lib_dir = dst.join("build").join("boost-install").join("lib");
    let armadillo_lib_dir = dst.join("build").join("armadillo-build");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        rat_common_lib_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        rat_mlfmm_lib_dir.display()
    );
    println!("cargo:rustc-link-search=native={}", boost_lib_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        armadillo_lib_dir.display()
    );
    if bin_dir.exists() {
        println!("cargo:rustc-link-search=native={}", bin_dir.display());
    }
    let mut static_libs = vec![
        "rat_mlfmm_c",
        "boost_filesystem",
        "boost_iostreams",
        "boost_thread",
        "boost_chrono",
        "jsoncpp",
        "armadillo",
    ];
    if python_feature && target_os == "linux" {
        static_libs.retain(|lib| *lib != "rat_mlfmm_c");
    }
    if target_os == "linux" && !python_feature {
        static_libs.insert(1, "ratmlfmm");
        static_libs.insert(2, "ratcmn");
    }
    if target_os == "linux" && !python_feature {
        emit_link_group(&static_libs);
    } else {
        for lib in &static_libs {
            println!("cargo:rustc-link-lib=static={lib}");
        }
    }
    if !python_feature {
        println!("cargo:rustc-link-lib=static=ratmlfmm");
        println!("cargo:rustc-link-lib=static=ratcmn");
    } else if target_os == "linux" {
        println!("cargo:rustc-link-arg-cdylib=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg-cdylib=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg-cdylib={}", rat_mlfmm_c_lib.display());
        println!("cargo:rustc-link-arg-cdylib={}", rat_mlfmm_lib.display());
        println!("cargo:rustc-link-arg-cdylib={}", rat_common_lib.display());
        println!("cargo:rustc-link-arg-cdylib=-Wl,--no-whole-archive");
    }
    println!("cargo:rustc-link-lib=z");
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=openblas");
    }
    if target_os == "macos" {
        println!(
            "cargo:rustc-link-arg-cdylib=-Wl,-force_load,{}",
            rat_mlfmm_c_lib.display()
        );
        println!(
            "cargo:rustc-link-arg-cdylib=-Wl,-force_load,{}",
            rat_mlfmm_lib.display()
        );
        println!(
            "cargo:rustc-link-arg-cdylib=-Wl,-force_load,{}",
            rat_common_lib.display()
        );
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=c++abi");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    let _ = lib_dir;

    rerun_if_changed(&wrapper_dir.join("CMakeLists.txt"));
    rerun_if_changed(&wrapper_dir.join("src/rat_mlfmm_c.cpp"));
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
    rerun_if_changed(&patches_dir.join("rat-common-static.patch"));
    rerun_if_changed(&patches_dir.join("rat-mlfmm-static.patch"));
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
    if cfg!(not(target_os = "windows")) {
        release.push("-fPIC");
        debug.push("-fPIC");
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
    if cfg!(not(target_os = "windows")) {
        flags.push("-fPIC");
    }
    flags.join(" ")
}

fn emit_link_group(static_libs: &[&str]) {
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    for lib in static_libs {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    println!("cargo:rustc-link-arg=-Wl,--end-group");
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
        .args([
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--depth",
            "1",
        ])
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

fn apply_vendor_patches(
    manifest_dir: &Path,
    patches_dir: &Path,
    rat_common_dir: &Path,
    rat_mlfmm_dir: &Path,
) {
    let patches = [
        ("rat-common-static.patch", rat_common_dir),
        ("rat-mlfmm-static.patch", rat_mlfmm_dir),
    ];

    for (patch_name, repo_dir) in patches {
        let patch = patches_dir.join(patch_name);
        if !patch.exists() {
            panic!("required patch missing: {}", patch.display());
        }
        if !repo_dir.exists() {
            panic!("required vendor dir missing: {}", repo_dir.display());
        }

        let check = Command::new("git")
            .args(["apply", "--check", patch.to_str().unwrap()])
            .current_dir(repo_dir)
            .status()
            .expect("failed to run git apply --check");

        if check.success() {
            let status = Command::new("git")
                .args(["apply", patch.to_str().unwrap()])
                .current_dir(repo_dir)
                .status()
                .expect("failed to run git apply");
            if !status.success() {
                panic!(
                    "failed to apply patch {} in {}",
                    patch.display(),
                    repo_dir.display()
                );
            }
            continue;
        }

        let reverse_check = Command::new("git")
            .args(["apply", "--reverse", "--check", patch.to_str().unwrap()])
            .current_dir(repo_dir)
            .status()
            .expect("failed to run git apply --reverse --check");

        if !reverse_check.success() {
            panic!(
                "patch {} does not apply cleanly in {}",
                patch.display(),
                repo_dir.display()
            );
        }
    }

    let _ = manifest_dir;
}

fn ensure_boost_headers(boost_dir: &Path) {
    let boost_headers = boost_dir.join("boost");
    if boost_headers.is_dir() {
        return;
    }
    // Boost's git superproject does not include the generated `boost/` header tree.
    // Running `bootstrap.sh` + `b2 headers` creates it so includes like <boost/assert.hpp> work.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // Windows uses bootstrap.bat and b2.exe; Unix uses bootstrap.sh and b2.
    let bootstrap = if target_os == "windows" {
        boost_dir.join("bootstrap.bat")
    } else {
        boost_dir.join("bootstrap.sh")
    };
    if !bootstrap.exists() {
        panic!(
            "Boost headers missing and bootstrap.sh not found at {}",
            bootstrap.display()
        );
    }
    let status = if target_os == "windows" {
        Command::new("cmd")
            .current_dir(boost_dir)
            .args(["/C", bootstrap.to_str().unwrap()])
            .status()
    } else {
        Command::new(&bootstrap).current_dir(boost_dir).status()
    }
    .unwrap_or_else(|err| panic!("failed to run {}: {err}", bootstrap.display()));
    if !status.success() {
        panic!("Boost bootstrap failed with status {}", status);
    }
    // b2 headers generates the consolidated boost/ headers directory from the repo layout.
    let b2 = if target_os == "windows" {
        boost_dir.join("b2.exe")
    } else {
        boost_dir.join("b2")
    };
    // Generate the consolidated headers into `boost/` for downstream compilers.
    let status = if target_os == "windows" {
        Command::new("cmd")
            .current_dir(boost_dir)
            .args(["/C", b2.to_str().unwrap(), "headers"])
            .status()
    } else {
        Command::new(&b2)
            .current_dir(boost_dir)
            .arg("headers")
            .status()
    }
    .unwrap_or_else(|err| panic!("failed to run {}: {err}", b2.display()));
    if !status.success() {
        panic!("Boost header generation failed with status {}", status);
    }
    if !boost_headers.is_dir() {
        panic!(
            "Boost header generation completed but {} is still missing",
            boost_headers.display()
        );
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

fn ensure_tool(tool: &str) {
    if Command::new(tool).arg("--version").output().is_err() {
        panic!("{tool} not found in PATH");
    }
}
