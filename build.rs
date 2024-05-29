fn main() {
    let libs = std::env::var("IPAMIR_LINKER_ARGS").unwrap();
    for link_arg in libs.split_whitespace() {
        println!("cargo:rustc-link-arg={}", link_arg);
    }
}
