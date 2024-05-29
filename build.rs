fn main() {
    let libs = std::env::var("LIBS").unwrap();
    for link_arg in libs.split_whitespace() {
        println!("cargo:rustc-link-arg={}", link_arg);
    }
}
