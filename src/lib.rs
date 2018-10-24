#![warn(
    missing_docs,
    missing_debug_implementations,
    unused_import_braces,
    unused_results,
    unused_qualifications
)]
// TODO: Add the missing_doc_code_examples warning, switch these to Deny later.

// Allow clippy lints
#![allow(unknown_lints)]

extern crate cuda_sys;

pub mod error;
pub mod memory;
pub(crate) mod private;
