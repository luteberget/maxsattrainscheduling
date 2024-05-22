fn main() {
    // We are linking to libipamir.a from the current directory.
    println!("cargo:rustc-link-search=.");
    println!("cargo:rerun-if-changed=libipamir.a");
    println!("cargo:rerun-if-env-changed=IPAMIRSOLVER");

    // Linking to STD-C++ seems required for many maxsat solvers.
    println!("cargo:rustc-link-lib=stdc++");

    // Linking to Z is required for EvalMaxSAT2022
    println!("cargo:rustc-link-lib=z");

    // // Linking to GMP and the IPASIR version of cominisatps is required for UWrMaxSAT14
    // println!("cargo:rustc-link-lib=gmp");
    // println!("cargo:rustc-link-lib=ipasircominisatps");
}
