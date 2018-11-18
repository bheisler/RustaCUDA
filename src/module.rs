//! This module provides functions for loading and working with CUDA modules.

// TODO: Write better documentation.

use cuda_sys::cuda;
use error::{CudaResult, ToResult};
use std::ffi::{c_void, CStr};
use std::mem;
use std::ptr;

/// A compiled CUDA module, loaded into a context.
#[derive(Debug)]
pub struct Module {
    inner: cuda::CUmodule,
}
impl Module {
    /// Load a module from the given file name into the current context.
    ///
    /// The given file should be either a cubin file, a ptx file, or a fatbin file such as
    /// those produced by `nvcc`.
    ///
    /// # Example:
    ///
    /// ```
    /// # use rustacuda::*;
    /// # let _ctx = quick_init().unwrap();
    /// use rustacuda::module::Module;
    /// use std::ffi::CString;
    ///
    /// let filename = CString::new("./resources/add.ptx").unwrap();
    /// let module = Module::load(&filename).unwrap();
    /// ```
    pub fn load(filename: &CStr) -> CudaResult<Module> {
        unsafe {
            let mut module = Module {
                inner: ptr::null_mut(),
            };
            cuda::cuModuleLoad(&mut module.inner as *mut cuda::CUmodule, filename.as_ptr())
                .toResult()?;
            Ok(module)
        }
    }

    /// Load a module from a CStr.
    ///
    /// This is useful in combination with `include_str!`, to include the device code into the
    /// compiled executable.
    ///
    /// The given CStr must contain the bytes of a cubin file, a ptx file or a fatbin file such as
    /// those produced by `nvcc`.
    ///
    /// # Example:
    ///
    /// ```
    /// # use rustacuda::*;
    /// # let _ctx = quick_init().unwrap();
    /// use rustacuda::module::Module;
    /// use std::ffi::CString;
    ///
    /// let image = CString::new(include_str!("../resources/add.ptx")).unwrap();
    /// let module = Module::load_data(&image).unwrap();
    /// ```
    pub fn load_data(image: &CStr) -> CudaResult<Module> {
        unsafe {
            let mut module = Module {
                inner: ptr::null_mut(),
            };
            cuda::cuModuleLoadData(
                &mut module.inner as *mut cuda::CUmodule,
                image.as_ptr() as *const c_void,
            ).toResult()?;
            Ok(module)
        }
    }
}
impl Drop for Module {
    fn drop(&mut self) {
        if self.inner.is_null() {
            return;
        }
        unsafe {
            // No choice but to panic if this fails...
            let module = mem::replace(&mut self.inner, ptr::null_mut());
            cuda::cuModuleUnload(module).toResult().unwrap();
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use quick_init;
    use std::ffi::CString;

    #[test]
    fn test_load_from_file() {
        let _context = quick_init();

        let filename = CString::new("./resources/add.ptx").unwrap();
        let module = Module::load(&filename).unwrap();
        drop(module)
    }

    #[test]
    fn test_load_from_memory() {
        let _context = quick_init();
        let ptx_text = CString::new(include_str!("../resources/add.ptx")).unwrap();
        let module = Module::load_data(&ptx_text).unwrap();
        drop(module)
    }
}
