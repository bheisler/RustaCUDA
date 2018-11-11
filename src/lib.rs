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

#[macro_use]
extern crate bitflags;
extern crate cuda_sys;

pub mod error;
pub mod memory;
pub(crate) mod private;

mod derive_compile_fail;
mod device;

use cuda_sys::cuda::{cuDriverGetVersion, cuInit};
pub use device::{Device, DeviceAttribute, Devices};
use error::{CudaResult, ToResult};

/*
TODO:
- Implement context management, basic module management
- It may be useful to have a quick-init function that initialized the driver and binds a context to
  the first device found. That's what most people will want, and if it's a well-documented helper
  method that specifies what it actually does, it's probably fine.
*/

bitflags! {
    /// Bit flags for initializing the CUDA driver. Currently, no flags are defined,
    /// so ZERO is the only valid value.
    pub struct CudaFlags: u32 {
        /// No flags set. As there are currently no flags defined, this is the only accepted value.
        const ZERO = 0;
    }
}

/// Initialize the CUDA Driver API.
///
/// This must be called before any other RustaCUDA (or CUDA) function is called. Typically, this
/// should be at the start of your program. All other functions will fail unless the API is
/// initialized first.
///
/// The `flags` parameter is used to configure the CUDA API. Currently no flags are defined, so
/// it must be `CudaFlags::ZERO`.
pub fn init(flags: CudaFlags) -> CudaResult<()> {
    unsafe { cuInit(flags.bits()).toResult() }
}

/// Struct representing the CUDA API version number.
#[derive(Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Copy, Clone)]
pub struct CudaApiVersion {
    version: i32,
}
impl CudaApiVersion {
    /// Returns the latest CUDA version supported by the CUDA driver.
    pub fn get() -> CudaResult<CudaApiVersion> {
        unsafe {
            let mut version: i32 = 0;
            cuDriverGetVersion(&mut version as *mut i32).toResult()?;
            Ok(CudaApiVersion { version })
        }
    }

    /// Return the major version number - eg. the 9 in version 9.2
    #[inline]
    pub fn major(self) -> i32 {
        self.version / 1000
    }

    /// Return the minor version number - eg. the 2 in version 9.2
    #[inline]
    pub fn minor(self) -> i32 {
        (self.version % 1000) / 10
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_api_version() {
        let version = CudaApiVersion { version: 9020 };
        assert_eq!(version.major(), 9);
        assert_eq!(version.minor(), 2);
    }

    #[test]
    fn test_init_twice() {
        init(CudaFlags::ZERO).unwrap();
        init(CudaFlags::ZERO).unwrap();
    }
}
