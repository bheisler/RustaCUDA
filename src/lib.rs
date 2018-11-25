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

pub mod context;
pub mod device;
pub mod error;
pub mod memory;
pub mod module;
pub(crate) mod private;
pub mod stream;

#[macro_export]
pub mod function;

mod derive_compile_fail;

use context::{Context, ContextFlags};
use cuda_sys::cuda::{cuDriverGetVersion, cuInit};
use device::Device;
use error::{CudaResult, ToResult};

/*
TODO before announcement:
- Document this module. Just do a pass or two over the documentation and examples in general.
- Write the user guide
    - Basic example
    - Using nvcc to compile kernels
    - Using ptx-builder to compile kernels
- Set up CI to generate docs and (if possible) compile (but not test)
- Rework path tracer to use RustaCUDA
- Write up the announcement post
- Add a prelude? What should be in it?
    - CopyDestination should be. 
    - Probably DeviceBuffer/DeviceBox or UnifiedBuffer/UnifiedBox or all of those as well.

- Write contributor docs and such
Help wanted:
- Perhaps somebody smarter than I am can think of a way to make the context management truly safe.
  I haven't been able to manage it.
- Which types should implement Send/Sync?
- What should be #[inline]'d?
- Implement the rest of the driver API:
    - Asynchronous memcpy
    - Events
    - Primary contexts
    - JIT linking
    - CUDA arrays
    - textures
    - surfaces (what even is this stuff?)
    - Unified memory prefetching, advising, attributes
    - More
*/

bitflags! {
    /// Bit flags for initializing the CUDA driver. Currently, no flags are defined,
    /// so `CudaFlags::empty()` is the only valid value.
    pub struct CudaFlags: u32 {
        // We need to give bitflags at least one constant.
        #[doc(hidden)]
        const _ZERO = 0;
    }
}

/// Initialize the CUDA Driver API.
///
/// This must be called before any other RustaCUDA (or CUDA) function is called. Typically, this
/// should be at the start of your program. All other functions will fail unless the API is
/// initialized first.
///
/// The `flags` parameter is used to configure the CUDA API. Currently no flags are defined, so
/// it must be `CudaFlags::empty()`.
pub fn init(flags: CudaFlags) -> CudaResult<()> {
    unsafe { cuInit(flags.bits()).to_result() }
}

/// Shortcut for initializing the CUDA Driver API and creating a CUDA context with default settings
/// for the first device.
///
/// This is useful for testing or just setting up a basic CUDA context quickly. Users with more
/// complex needs (multiple devices, custom flags, etc.) should use `init` and create their own
/// context.
pub fn quick_init() -> CudaResult<Context> {
    init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
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
            cuDriverGetVersion(&mut version as *mut i32).to_result()?;
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
        init(CudaFlags::empty()).unwrap();
        init(CudaFlags::empty()).unwrap();
    }
}
