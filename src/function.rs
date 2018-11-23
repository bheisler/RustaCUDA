use context::{CacheConfig, SharedMemoryConfig};
use cuda_sys::cuda::{self, CUfunction};
use error::{CudaResult, ToResult};
use module::Module;
use std::marker::PhantomData;
use std::mem::transmute;

/// All supported function attributes for [Function::get_attribute](struct.Function.html#method.get_attribute)
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FunctionAttribute {
    /// The maximum number of threads per block, beyond which a launch would fail. This depends on
    /// both the function and the device.
    MaxThreadsPerBlock = 0,

    /// The size in bytes of the statically-allocated shared memory required by this function.
    SharedMemorySizeBytes = 1,

    /// The size in bytes of the constant memory required by this function
    ConstSizeBytes = 2,

    /// The size in bytes of local memory used by each thread of this function
    LocalSizeBytes = 3,

    /// The number of registers used by each thread of this function
    NumRegisters = 4,

    /// The PTX virtual architecture version for which the function was compiled. This value is the
    /// major PTX version * 10 + the minor PTX version, so version 1.3 would return the value 13.
    PtxVersion = 5,

    /// The binary architecture version for which the function was compiled. Encoded the same way as
    /// PtxVersion.
    BinaryVersion = 6,

    /// The attribute to indicate whether the function has been compiled with user specified
    /// option "-Xptxas --dlcm=ca" set.
    CacheModeCa = 7,

    #[doc(hidden)]
    __Nonexhaustive = 8,
}

/// Handle to a global kernel function.
#[derive(Debug)]
pub struct Function<'a> {
    inner: CUfunction,
    module: PhantomData<&'a Module>,
}
impl<'a> Function<'a> {
    pub(crate) fn new(inner: CUfunction, _module: &Module) -> Function {
        Function {
            inner,
            module: PhantomData,
        }
    }

    /// Returns information about a function.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use rustacuda::*;
    /// # let _ctx = quick_init().unwrap();
    /// # use rustacuda::module::Module;
    /// # use std::ffi::CString;
    /// # let filename = CString::new("./resources/add.ptx").unwrap();
    /// # let module = Module::load(&filename).unwrap();
    /// # let name = CString::new("sum").unwrap();
    /// use rustacuda::function::FunctionAttribute;
    /// let function = module.get_function(&name).unwrap();
    /// let shared_memory = function.get_attribute(FunctionAttribute::SharedMemorySizeBytes).unwrap();
    /// println!("This function uses {} bytes of shared memory", shared_memory);
    /// ```
    pub fn get_attribute(&self, attr: FunctionAttribute) -> CudaResult<i32> {
        unsafe {
            let mut val = 0i32;
            cuda::cuFuncGetAttribute(
                &mut val as *mut i32,
                // This should be safe, as the repr and values of FunctionAttribute should match.
                ::std::mem::transmute(attr),
                self.inner,
            ).toResult()?;
            Ok(val)
        }
    }

    /// Sets the preferred cache configuration for this function.
    ///
    /// On devices where L1 cache and shared memory use the same hardware resources, this sets the
    /// preferred cache configuration for this function. This is only a preference. The
    /// driver will use the requested configuration if possible, but is free to choose a different
    /// configuration if required to execute the function. This setting will override the
    /// context-wide setting.
    ///
    /// This setting does nothing on devices where the size of the L1 cache and shared memory are
    /// fixed.
    ///
    /// # Example:
    ///
    /// ```
    /// # use rustacuda::*;
    /// # let _ctx = quick_init().unwrap();
    /// # use rustacuda::module::Module;
    /// # use std::ffi::CString;
    /// # let filename = CString::new("./resources/add.ptx").unwrap();
    /// # let module = Module::load(&filename).unwrap();
    /// # let name = CString::new("sum").unwrap();
    /// use rustacuda::context::CacheConfig;
    /// let mut function = module.get_function(&name).unwrap();
    /// function.set_cache_config(CacheConfig::PreferL1).unwrap();
    /// ```
    pub fn set_cache_config(&mut self, config: CacheConfig) -> CudaResult<()> {
        unsafe { cuda::cuFuncSetCacheConfig(self.inner, transmute(config)).toResult() }
    }

    /// Sets the preferred shared memory configuration for this function.
    ///
    /// On devices with configurable shared memory banks, this function will set this function's
    /// shared memory bank size which is used for subsequent launches of this function. If not set,
    /// the context-wide setting will be used instead.
    ///
    /// # Example:
    ///
    /// ```
    /// # use rustacuda::*;
    /// # let _ctx = quick_init().unwrap();
    /// # use rustacuda::module::Module;
    /// # use std::ffi::CString;
    /// # let filename = CString::new("./resources/add.ptx").unwrap();
    /// # let module = Module::load(&filename).unwrap();
    /// # let name = CString::new("sum").unwrap();
    /// use rustacuda::context::SharedMemoryConfig;
    /// let mut function = module.get_function(&name).unwrap();
    /// function.set_shared_memory_config(SharedMemoryConfig::EightByteBankSize).unwrap();
    /// ```
    pub fn set_shared_memory_config(&mut self, cfg: SharedMemoryConfig) -> CudaResult<()> {
        unsafe { cuda::cuFuncSetSharedMemConfig(self.inner, transmute(cfg)).toResult() }
    }

    pub(crate) fn to_inner(&self) -> CUfunction {
        self.inner
    }
}
