set -ex

export CARGO_INCREMENTAL=0

if [ "$DOCS" = "yes" ]; then
    cargo clean
    cargo doc --all --no-deps
    travis-cargo doc-upload || true
elif [ "$RUSTFMT" = "yes" ]; then
    cargo fmt --all -- --check
elif [ "$CLIPPY" = "yes" ]; then
      cargo clippy --all -- -D warnings
else
    cargo build
    cargo build --tests
    cargo build --examples

    cd rustacuda_core
    cargo build
    cargo build --tests
    cargo build --examples
    cd ..

    cd rustacuda_derive
    cargo build
    cargo build --tests
    cargo build --examples
    cd ..

fi