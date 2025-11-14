use std::env;

fn main() {
    // Définir le chemin de la bibliothèque libvosk
    let libvosk_path = env::var("LIBVOSK_PATH")
        .unwrap_or_else(|_| "../../vosk-linux-x86-0.3.42".to_string());

    // Indiquer au compilateur où trouver libvosk.so
    println!("cargo:rustc-link-search=native={}", libvosk_path);
    println!("cargo:rustc-link-lib=dylib=vosk");

    // S'assurer que le chemin est surveillé pour les changements
    println!("cargo:rerun-if-changed={}", libvosk_path);
}
