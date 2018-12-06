//! RustaCUDA-core is a minimal subset of RustaCUDA which is intended to be used in device-side
//! crates.
//!
//! It includes a small number of types needed when sharing types between the host and device.
//! This is not intended to be used in a standalone way - see RustaCUDA for full documentation.

#![no_std]
#![warn(
    missing_docs,
    missing_debug_implementations,
    unused_import_braces,
    unused_results,
    unused_qualifications
)]
#![allow(unknown_lints)]

mod memory;
pub use crate::memory::*;
