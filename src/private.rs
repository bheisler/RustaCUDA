//! This module contains private sealed traits that should not be used or implemented outside of
//! RustaCUDA. These traits are public because they are used as bounds in certain functions.
//! These traits may change in any way at any time with no warning, and this will not be considered
//! a breaking change.

use memory::{DeviceCopy, DevicePointer, UnifiedPointer};
use std::os::raw::c_void;

/// Trait implemented by the types which can be passed to [`cuda_free`](../memory/fn.cuda_free.html).
/// This is not intended to be used or implemented outside of RustaCUDA. See the
/// [`private module`](../private/index.html) for details.
pub trait CudaFreeable: private_2::SealedCudaFreeable {
    #[doc(hidden)]
    fn __to_raw(&mut self) -> *mut c_void;
}

impl<T: DeviceCopy> CudaFreeable for DevicePointer<T> {
    fn __to_raw(&mut self) -> *mut c_void {
        self.as_raw_mut() as *mut c_void
    }
}
impl<T: DeviceCopy> CudaFreeable for UnifiedPointer<T> {
    fn __to_raw(&mut self) -> *mut c_void {
        self.as_raw_mut() as *mut c_void
    }
}

mod private_2 {
    use super::*;

    pub trait SealedCudaFreeable {}

    impl<T: DeviceCopy> SealedCudaFreeable for DevicePointer<T> {}
    impl<T: DeviceCopy> SealedCudaFreeable for UnifiedPointer<T> {}
}
