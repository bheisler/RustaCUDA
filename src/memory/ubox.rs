use error::*;
use cuda_sys::cudart::*;
use std::mem;
use std::ptr;
use std::os::raw::c_void;
use std::convert::{AsMut, AsRef};
use std::borrow::{Borrow, BorrowMut};
use std::fmt::{self, Display, Pointer};
use std::ops::{Deref, DerefMut};
use std::hash::{Hash, Hasher};
use super::DeviceCopy;

/// A pointer type for heap-allocation in CUDA Unified Memory. See the module-level-documentation
/// for more information on unified memory. Should behave equivalently to std::boxed::Box, except
/// that the allocated memory can be seamlessly shared between host and device.
#[derive(Eq, Ord, Clone, Debug, PartialEq, PartialOrd)]
pub struct UBox<T: DeviceCopy> {
    ptr: *mut T,
}
impl<T: DeviceCopy> UBox<T> {
    /// Allocate unified memory and place val into it. 
    pub fn new(val: T) -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(UBox{ ptr: ptr::null_mut() })
        }
        else {
            let mut ptr: *mut c_void = ptr::null_mut();
            unsafe {
                cudaMallocManaged(&mut ptr as *mut *mut c_void, mem::size_of::<T>(), cudaMemAttachGlobal).toResult()?;
                let ptr = ptr as *mut T;
                *ptr = val;
                Ok(UBox{ ptr : ptr as *mut T })
            }
        }
    }

    /// Constructs a UBox from a raw pointer.
    /// 
    /// After calling this function, the raw pointer and the memory it points to is owned by the
    /// UBox. The UBox destructor will call the destructor of T and free the allocated memory.
    /// This function may accept any pointer produced by the `cudaMallocManaged` CUDA API call,
    /// such as one taken from `UBox::into_raw`.
    /// 
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        UBox{ ptr }
    }

    /// Consumes the UBox, returning the wrapped unified-memory pointer.
    /// 
    /// After calling this function, the caller is responsible for the memory previously managed by
    /// the UBox. In particular, the caller should properly destroy T and deallocate the memory.
    /// The easiest way to do so is to create a new UBox using the UBox::from_raw function.
    /// 
    /// Note: This is an associated function, which means that you have to all it as 
    /// `UBox::into_raw(b)` instead of `b.into_raw()` This is so that there is no conflict with
    /// a method on the inner type.
    pub fn into_raw(b: UBox<T>) -> *mut T {
        b.ptr
    }

    /// Consumes and leaks the UBox, returning a mutable reference, &'a mut T. Note that the type T
    /// must outlive the chosen lifetime 'a. If the type has only static references, or none at all,
    /// this may be chosen to be 'static.
    /// 
    /// This is mainly useful for data that lives for the remainder of the program's life. Dropping
    /// the returned reference will cause a memory leak. If this is not acceptable, the reference
    /// should be wrapped with the UBox::from_raw function to produce a new UBox. This UBox can then
    /// be dropped, which will properly destroy T and release the allocated memory.
    /// 
    /// Note: This is an associated function, which means that you have to all it as 
    /// `UBox::leak(b)` instead of `b.leak()` This is so that there is no conflict with
    /// a method on the inner type.
    pub fn leak<'a>(b: UBox<T>) -> &'a mut T 
        where T: 'a
    {
        unsafe { &mut *UBox::into_raw(b) }
    }
}
impl<T: DeviceCopy> Drop for UBox<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // No choice but to panic if this fails. It will probably abort the process as well, but
            // it can only happen if the user calls from_raw with an invalid pointer.
            unsafe {
                cudaFree(self.ptr as *mut c_void).toResult().unwrap();
            }
        }
        self.ptr = ptr::null_mut();
    }
}

impl<T: DeviceCopy> Borrow<T> for UBox<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}
impl<T: DeviceCopy> BorrowMut<T> for UBox<T> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}
impl<T: DeviceCopy> AsRef<T> for UBox<T> {
    fn as_ref(& self) -> &T {
        &**self
    }
}
impl<T: DeviceCopy> AsMut<T> for UBox<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}
impl <T: DeviceCopy> Deref for UBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            &*self.ptr
        }
    }
}
impl <T: DeviceCopy> DerefMut for UBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe {
            &mut *self.ptr
        }
    }
}
impl<T: Hash + DeviceCopy> Hash for UBox<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}
impl<T: Display + DeviceCopy> Display for UBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}
impl<T: DeviceCopy> Pointer for UBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}