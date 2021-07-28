//! Functions and types for working with CUDA modules.

use crate::error::{CudaResult, DropResult, ToResult};
use crate::function::Function;
use crate::memory::{CopyDestination, DeviceCopy, DevicePointer};
use std::ffi::{c_void, CStr};
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

/// A compiled CUDA module, loaded into a context.
#[derive(Debug)]
pub struct Module {
    inner: cuda_driver_sys::CUmodule,
}
impl Module {
    /// Load a module from the given file name into the current context.
    ///
    /// The given file should be either a cubin file, a ptx file, or a fatbin file such as
    /// those produced by `nvcc`.
    ///
    /// # Example
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use rustacuda::module::Module;
    /// use std::ffi::CString;
    ///
    /// let filename = CString::new("./resources/add.ptx")?;
    /// let module = Module::load_from_file(&filename)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_from_file(filename: &CStr) -> CudaResult<Module> {
        unsafe {
            let mut module = Module {
                inner: ptr::null_mut(),
            };
            cuda_driver_sys::cuModuleLoad(
                &mut module.inner as *mut cuda_driver_sys::CUmodule,
                filename.as_ptr(),
            )
            .to_result()?;
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
    /// # Example
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use rustacuda::module::Module;
    /// use std::ffi::CString;
    ///
    /// let image = CString::new(include_str!("../resources/add.ptx"))?;
    /// let module = Module::load_from_string(&image)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_from_string(image: &CStr) -> CudaResult<Module> {
        unsafe {
            let mut module = Module {
                inner: ptr::null_mut(),
            };
            cuda_driver_sys::cuModuleLoadData(
                &mut module.inner as *mut cuda_driver_sys::CUmodule,
                image.as_ptr() as *const c_void,
            )
            .to_result()?;
            Ok(module)
        }
    }

    /// Get a reference to a global symbol, which can then be copied to/from.
    ///
    /// # Panics:
    ///
    /// This function panics if the size of the symbol is not the same as the `mem::sizeof<T>()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use rustacuda::memory::CopyDestination;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use rustacuda::module::Module;
    /// use std::ffi::CString;
    ///
    /// let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    /// let module = Module::load_from_string(&ptx)?;
    /// let name = CString::new("my_constant")?;
    /// let symbol = module.get_global::<u32>(&name)?;
    /// let mut host_const = 0;
    /// symbol.copy_to(&mut host_const)?;
    /// assert_eq!(314, host_const);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_global<'a, T: DeviceCopy>(&'a self, name: &CStr) -> CudaResult<Symbol<'a, T>> {
        unsafe {
            let mut ptr: DevicePointer<T> = DevicePointer::null();
            let mut size: usize = 0;

            cuda_driver_sys::cuModuleGetGlobal_v2(
                &mut ptr as *mut DevicePointer<T> as *mut cuda_driver_sys::CUdeviceptr,
                &mut size as *mut usize,
                self.inner,
                name.as_ptr(),
            )
            .to_result()?;
            assert_eq!(size, mem::size_of::<T>());
            Ok(Symbol {
                ptr,
                module: PhantomData,
            })
        }
    }

    /// Get a reference to a kernel function which can then be launched.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use rustacuda::module::Module;
    /// use std::ffi::CString;
    ///
    /// let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    /// let module = Module::load_from_string(&ptx)?;
    /// let name = CString::new("sum")?;
    /// let function = module.get_function(&name)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_function<'a>(&'a self, name: &CStr) -> CudaResult<Function<'a>> {
        unsafe {
            let mut func: cuda_driver_sys::CUfunction = ptr::null_mut();

            cuda_driver_sys::cuModuleGetFunction(
                &mut func as *mut cuda_driver_sys::CUfunction,
                self.inner,
                name.as_ptr(),
            )
            .to_result()?;
            Ok(Function::new(func, self))
        }
    }

    /// Destroy a `Module`, returning an error.
    ///
    /// Destroying a module can return errors from previous asynchronous work. This function
    /// destroys the given module and returns the error and the un-destroyed module on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use rustacuda::module::Module;
    /// use std::ffi::CString;
    ///
    /// let ptx = CString::new(include_str!("../resources/add.ptx"))?;
    /// let module = Module::load_from_string(&ptx)?;
    /// match Module::drop(module) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, module)) => {
    ///         println!("Failed to destroy module: {:?}", e);
    ///         // Do something with module
    ///     },
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn drop(mut module: Module) -> DropResult<Module> {
        if module.inner.is_null() {
            return Ok(());
        }

        unsafe {
            let inner = mem::replace(&mut module.inner, ptr::null_mut());
            match cuda_driver_sys::cuModuleUnload(inner).to_result() {
                Ok(()) => {
                    mem::forget(module);
                    Ok(())
                }
                Err(e) => Err((e, Module { inner })),
            }
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
            cuda_driver_sys::cuModuleUnload(module)
                .to_result()
                .expect("Failed to unload CUDA module");
        }
    }
}

/// Handle to a symbol defined within a CUDA module.
#[derive(Debug)]
pub struct Symbol<'a, T: DeviceCopy> {
    ptr: DevicePointer<T>,
    module: PhantomData<&'a Module>,
}
impl<'a, T: DeviceCopy> crate::private::Sealed for Symbol<'a, T> {}
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
                cuda_driver_sys::cuMemcpyHtoD_v2(
                    self.ptr.as_raw_mut() as u64,
                    val as *const T as *const c_void,
                    size,
                )
                .to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut T) -> CudaResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                cuda_driver_sys::cuMemcpyDtoH_v2(
                    val as *const T as *mut c_void,
                    self.ptr.as_raw() as u64,
                    size,
                )
                .to_result()?
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::quick_init;
    use std::error::Error;
    use std::ffi::CString;

    #[test]
    fn test_load_from_file() -> Result<(), Box<dyn Error>> {
        let _context = quick_init();

        let filename = CString::new("./resources/add.ptx")?;
        let module = Module::load_from_file(&filename)?;
        drop(module);
        Ok(())
    }

    #[test]
    fn test_load_from_memory() -> Result<(), Box<dyn Error>> {
        let _context = quick_init();
        let ptx_text = CString::new(include_str!("../resources/add.ptx"))?;
        let module = Module::load_from_string(&ptx_text)?;
        drop(module);
        Ok(())
    }

    #[test]
    fn test_copy_from_module() -> Result<(), Box<dyn Error>> {
        let _context = quick_init();

        let ptx = CString::new(include_str!("../resources/add.ptx"))?;
        let module = Module::load_from_string(&ptx)?;

        let constant_name = CString::new("my_constant")?;
        let symbol = module.get_global::<u32>(&constant_name)?;

        let mut constant_copy = 0u32;
        symbol.copy_to(&mut constant_copy)?;
        assert_eq!(314, constant_copy);
        Ok(())
    }

    #[test]
    fn test_copy_to_module() -> Result<(), Box<dyn Error>> {
        let _context = quick_init();

        let ptx = CString::new(include_str!("../resources/add.ptx"))?;
        let module = Module::load_from_string(&ptx)?;

        let constant_name = CString::new("my_constant")?;
        let mut symbol = module.get_global::<u32>(&constant_name)?;

        symbol.copy_from(&100)?;

        let mut constant_copy = 0u32;
        symbol.copy_to(&mut constant_copy)?;
        assert_eq!(100, constant_copy);
        Ok(())
    }
}
