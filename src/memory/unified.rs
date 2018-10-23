use super::DeviceCopy;
use error::*;
use memory::UnifiedPointer;
use memory::{cuda_free, cuda_malloc_unified};
use std::borrow::{Borrow, BorrowMut};
use std::convert::{AsMut, AsRef};
use std::fmt::{self, Display, Pointer};
use std::mem;
use std::ops::{Deref, DerefMut};

/*
You should be able to:
- Prefetch unified data to/from the device

*/

/// A pointer type for heap-allocation in CUDA Unified Memory. See the module-level-documentation
/// for more information on unified memory. Should behave equivalently to std::boxed::Box, except
/// that the allocated memory can be seamlessly shared between host and device.
#[derive(Eq, Ord, Clone, Debug, PartialEq, PartialOrd, Hash)]
pub struct UBox<T: DeviceCopy> {
    ptr: UnifiedPointer<T>,
}
impl<T: DeviceCopy> UBox<T> {
    /// Allocate unified memory and place val into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let five = UBox::new(5).unwrap();
    /// ```
    pub fn new(val: T) -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(UBox {
                ptr: UnifiedPointer::null(),
            })
        } else {
            unsafe {
                let mut ptr = cuda_malloc_unified(1)?;
                *ptr.as_raw_mut() = val;
                Ok(UBox { ptr })
            }
        }
    }

    /// Constructs a UBox from a raw pointer.
    ///
    /// After calling this function, the raw pointer and the memory it points to is owned by the
    /// UBox. The UBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cudaMallocManaged` CUDA API
    /// call, such as one taken from `UBox::into_raw`.
    ///
    /// # Safety:
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    ///
    /// # Examples:
    /// ```
    /// use rustacuda::memory::*;
    /// let x = UBox::new(5).unwrap();
    /// let ptr = UBox::into_raw(x);
    /// let x = unsafe { UBox::from_raw(ptr) };
    /// ```
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        UBox {
            ptr: UnifiedPointer::wrap(ptr),
        }
    }

    /// Consumes the UBox, returning the wrapped unified-memory pointer.
    ///
    /// After calling this function, the caller is responsible for the memory previously managed by
    /// the UBox. In particular, the caller should properly destroy T and deallocate the memory.
    /// The easiest way to do so is to create a new UBox using the UBox::from_raw function.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `UBox::as_raw(b)` instead of `b.as_raw()` This is so that there is no conflict with
    /// a method on the inner type.
    ///
    /// # Examples:
    /// ```
    /// use rustacuda::memory::*;
    /// let x = UBox::new(5).unwrap();
    /// let ptr = UBox::into_raw(x);
    /// # unsafe { UBox::from_raw(ptr) };
    /// ```
    #[allow(wrong_self_convention)]
    pub fn into_raw(mut b: UBox<T>) -> *mut T {
        let ptr = b.ptr.as_raw_mut();
        mem::forget(b);
        ptr
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
    where
        T: 'a,
    {
        unsafe { &mut *UBox::into_raw(b) }
    }
}
impl<T: DeviceCopy> Drop for UBox<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let ptr = ::std::mem::replace(&mut self.ptr, UnifiedPointer::null());
            // No choice but to panic if this fails.
            unsafe {
                cuda_free(ptr).expect("Failed to deallocate CUDA memory.");
            }
        }
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
    fn as_ref(&self) -> &T {
        &**self
    }
}
impl<T: DeviceCopy> AsMut<T> for UBox<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}
impl<T: DeviceCopy> Deref for UBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.ptr.as_raw() }
    }
}
impl<T: DeviceCopy> DerefMut for UBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr.as_raw_mut() }
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

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Clone, Debug)]
    struct ZeroSizedType;
    unsafe impl ::memory::DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_allocate_and_free_ubox() {
        let mut x = UBox::new(5u64).unwrap();
        *x = 10;
        assert_eq!(10, *x);
        drop(x);
    }

    #[test]
    fn test_ubox_allocates_for_non_zst() {
        let x = UBox::new(5u64).unwrap();
        let ptr = UBox::into_raw(x);
        assert!(!ptr.is_null());
        let _ = unsafe { UBox::from_raw(ptr) };
    }

    #[test]
    fn test_ubox_doesnt_allocate_for_zero_sized_type() {
        let x = UBox::new(ZeroSizedType).unwrap();
        let ptr = UBox::into_raw(x);
        assert!(ptr.is_null());
        let _ = unsafe { UBox::from_raw(ptr) };
    }
}
