use std::env;
use std::path::PathBuf;

fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo::rerun-if-changed=TWMNET/nn_file.c");
    println!("cargo::rerun-if-changed=TWMNET/nn.c");
    println!("cargo::rerun-if-changed=TWMNET/nn.h");

    // Compile your C source files.
    cc::Build::new()
        .file("TWMNet/nn.c")
        .file("TWMNet/nn_file.c")
        .include("TWMNet")
        .compile("TWMNet");

    // Generate bindings from all headers.
    let bindings = bindgen::Builder::default()
        .header("TWMNet/nn.h")
        .clang_arg("-ITWMNet")
        .allowlist_item(r"^TWMNet.*$")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Failed to generate bindings");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_dir.join("TWMNet.rs"))
        .expect("Failed to write TWMNet bindings");
}
