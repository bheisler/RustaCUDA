//! This module provides functions for loading and working with CUDA modules.

// TODO: Write better documentation.

use cuda_sys::cuda;
use error::{CudaResult, ToResult};
use memory::{CopyDestination, DeviceCopy, DevicePointer};
use std::ffi::{c_void, CStr};
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

// TODO: Add symbol and function structs

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

    /// Get a reference to a global symbol, which can then be copied to/from.
    ///
    /// # Panics:
    ///
    /// This function panics if the size of the symbol is not the same as the `mem::sizeof<T>()`.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use rustacuda::memory::CopyDestination;
    /// # let _ctx = quick_init().unwrap();
    /// use rustacuda::module::Module;
    /// use std::ffi::CString;
    ///
    /// let filename = CString::new("./resources/add.ptx").unwrap();
    /// let module = Module::load(&filename).unwrap();
    /// let name = CString::new("my_constant").unwrap();
    /// let symbol = module.get_global::<u32>(&name).unwrap();
    /// let mut host_const = 0;
    /// symbol.copy_to(&mut host_const).unwrap();
    /// assert_eq!(314, host_const);
    /// ```
    pub fn get_global<'a, T: DeviceCopy>(&'a self, name: &CStr) -> CudaResult<Symbol<'a, T>> {
        unsafe {
            let mut ptr: DevicePointer<T> = DevicePointer::null();
            let mut size: usize = 0;

            cuda::cuModuleGetGlobal_v2(
                &mut ptr as *mut DevicePointer<T> as *mut cuda::CUdeviceptr,
                &mut size as *mut usize,
                self.inner,
                name.as_ptr(),
            ).toResult()?;
            assert_eq!(size, mem::size_of::<T>());
            Ok(Symbol {
                ptr: ptr,
                module: PhantomData,
            })
        }
    }

    /// Get a reference to a kernel function which can then be launched.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use rustacuda::*;
    /// # let _ctx = quick_init().unwrap();
    /// use rustacuda::module::Module;
    /// use std::ffi::CString;
    ///
    /// let filename = CString::new("./resources/add.ptx").unwrap();
    /// let module = Module::load(&filename).unwrap();
    /// let name = CString::new("sum").unwrap();
    /// let function = module.get_function(&name).unwrap();
    /// ```
    pub fn get_function<'a>(&'a self, name: &CStr) -> CudaResult<Function<'a>> {
        unsafe {
            let mut func: cuda::CUfunction = ptr::null_mut();

            cuda::cuModuleGetFunction(
                &mut func as *mut cuda::CUfunction,
                self.inner,
                name.as_ptr(),
            ).toResult()?;
            Ok(Function {
                func: func,
                module: PhantomData,
            })
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

/// Handle to a symbol defined within a CUDA module.
#[derive(Debug)]
pub struct Symbol<'a, T: DeviceCopy> {
    ptr: DevicePointer<T>,
    module: PhantomData<&'a Module>,
}
impl<'a, T: DeviceCopy> ::private::Sealed for Symbol<'a, T> {}
impl<'a, T: DeviceCopy> fmt::Pointer for Symbol<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}
impl<'a, T: DeviceCopy> CopyDestination<T> for Symbol<'a, T> {
    fn copy_from(&mut self, val: &T) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyHtoD_v2(
                    self.ptr.as_raw_mut() as u64,
                    val as *const T as *const c_void,
                    size,
                ).toResult()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut T) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cuda::cuMemcpyDtoH_v2(
                    val as *const T as *mut c_void,
                    self.ptr.as_raw() as u64,
                    size,
                ).toResult()?
            }
        }
        Ok(())
    }
}

/// Handle to a global kernel function.
#[derive(Debug)]
pub struct Function<'a> {
    func: cuda::CUfunction,
    module: PhantomData<&'a Module>,
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

    #[test]
    fn test_copy_from_module() {
        let _context = quick_init();

        let filename = CString::new("./resources/add.ptx").unwrap();
        let module = Module::load(&filename).unwrap();

        let constant_name = CString::new("my_constant").unwrap();
        let symbol = module.get_global::<u32>(&constant_name).unwrap();

        let mut constant_copy = 0u32;
        symbol.copy_to(&mut constant_copy).unwrap();
        assert_eq!(314, constant_copy);
    }

    #[test]
    fn test_copy_to_module() {
        let _context = quick_init();

        let filename = CString::new("./resources/add.ptx").unwrap();
        let module = Module::load(&filename).unwrap();

        let constant_name = CString::new("my_constant").unwrap();
        let mut symbol = module.get_global::<u32>(&constant_name).unwrap();

        symbol.copy_from(&100).unwrap();

        let mut constant_copy = 0u32;
        symbol.copy_to(&mut constant_copy).unwrap();
        assert_eq!(100, constant_copy);
    }
}
