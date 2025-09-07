fn main() {
    // The build script stands alone; ignore other changes (e.g. edits to
    // benchmarks in the benches subfolder).
    println!("cargo:rerun-if-changed=build.rs");

    // Check CPU feature support and error out if we don't have the appropriate
    // features. This isn't a fool-proof – someone could build on a machine with
    // AVX2 support, then try running those binaries elsewhere – but is a good
    // first line of defense.
    if std::env::var("CARGO_FEATURE_JIT").is_ok() {
        #[cfg(target_arch = "x86_64")]
        if !std::arch::is_x86_feature_detected!("avx2") {
            eprintln!(
                "`x86_64` build with `jit` enabled requires AVX2 instructions"
            );
            std::process::exit(1);
        }

        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!(
                "`aarch64` build with `jit` enabled requires NEON instructions"
            );
            std::process::exit(1);
        }
    }
}
