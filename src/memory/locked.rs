use super::DeviceCopy;
use crate::error::*;
use crate::memory::malloc::{cuda_free_locked, cuda_malloc_locked};
use core::fmt;
use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::fmt::{Display, Pointer};
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::slice;

/// A pointer type for heap-allocation in CUDA page-locked memory.
///
/// See the [`module-level documentation`](../memory/index.html) for more information on page-locked memory.
#[derive(Debug)]
pub struct LockedBox<T> {
    pub(crate) ptr: *mut T,
}
impl<T: Copy> LockedBox<T> {
    /// Allocate page-locked memory and place val into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Errors
    ///
    /// If a CUDA error occurs, return the error.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let five = LockedBox::new(5).unwrap();
    /// ```
    pub fn new(val: T) -> CudaResult<Self> {
        let locked_box = unsafe { LockedBox::uninitialized()? };
        unsafe { core::ptr::write(locked_box.ptr, val) };
        Ok(locked_box)
    }
}
impl<T> LockedBox<T> {
    /// Allocate page-locked memory, but do not initialize it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Safety
    ///
    /// Since the backing memory is not initialized, this function is not safe. The caller must
    /// ensure that the backing memory is set to a valid value before it is read, else undefined
    /// behavior may occur.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let mut five = unsafe { LockedBox::uninitialized().unwrap() };
    /// *five = 5u64;
    /// ```
    pub unsafe fn uninitialized() -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(LockedBox {
                ptr: core::ptr::null_mut(),
            })
        } else {
            let ptr = cuda_malloc_locked(1)?;
            Ok(LockedBox { ptr })
        }
    }

    /// Constructs a LockedBox from a **page-locked** raw pointer.
    ///
    /// After calling this function, the raw pointer and the memory it points to is owned by the
    /// LockedBox. The LockedBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cuMemAllocManaged` CUDA API
    /// call.
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    ///
    /// Additionally, this function has the additional requirement that the pointer must be page-locked.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let x = LockedBox::new(5).unwrap();
    /// let ptr = LockedBox::into_raw(x);
    /// let x = unsafe { LockedBox::from_raw(ptr) };
    /// ```
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        LockedBox { ptr }
    }

    /// Consumes the LockedBox, returning a pointer to the underlying data.
    ///
    /// After calling this function, the caller is responsible for the memory previously managed by
    /// the LockedBox. In particular, the caller should properly destroy T and deallocate the memory.
    /// The easiest way to do so is to create a new LockedBox using the `LockedBox::from_raw` function.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `LockedBox::into_raw(b)` instead of `b.into_raw()` This is so that there is no conflict with
    /// a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let x = LockedBox::new(5).unwrap();
    /// let ptr = LockedBox::into_raw(x);
    /// # unsafe { LockedBox::from_raw(ptr) };
    /// ```
    #[allow(clippy::wrong_self_convention)]
    pub fn into_raw(mut b: LockedBox<T>) -> *mut T {
        let ptr = mem::replace(&mut b.ptr, core::ptr::null_mut());
        mem::forget(b);
        ptr
    }

    /// Consumes and leaks the LockedBox, returning a mutable reference, &'a mut T. Note that the type T
    /// must outlive the chosen lifetime 'a. If the type has only static references, or none at all,
    /// this may be chosen to be 'static.
    ///
    /// This is mainly useful for data that lives for the remainder of the program's life. Dropping
    /// the returned reference will cause a memory leak. If this is not acceptable, the reference
    /// should be wrapped with the LockedBox::from_raw function to produce a new LockedBox. This LockedBox can then
    /// be dropped, which will properly destroy T and release the allocated memory.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `LockedBox::leak(b)` instead of `b.leak()` This is so that there is no conflict with
    /// a method on the inner type.
    pub fn leak<'a>(b: LockedBox<T>) -> &'a mut T
    where
        T: 'a,
    {
        unsafe { &mut *LockedBox::into_raw(b) }
    }

    /// Returns the contained pointer without consuming the box.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let mut x = LockedBox::new(5).unwrap();
    /// let ptr = x.as_ptr();
    /// println!("{:p}", ptr);
    /// ```
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the contained mutable pointer without consuming the box.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let mut x = LockedBox::new(5).unwrap();
    /// let ptr = x.as_mut_ptr();
    /// println!("{:p}", ptr);
    /// ```
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Destroy a `LockedBox`, returning an error.
    ///
    /// Deallocating locked memory can return errors from previous asynchronous work. This function
    /// destroys the given box and returns the error and the un-destroyed box on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let x = LockedBox::new(5).unwrap();
    /// match LockedBox::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, locked_box)) => {
    ///         println!("Failed to destroy box: {:?}", e);
    ///         // Do something with locked_box
    ///     },
    /// }
    /// ```
    pub fn drop(mut locked_box: LockedBox<T>) -> DropResult<LockedBox<T>> {
        if locked_box.ptr.is_null() {
            return Ok(());
        }

        let ptr = mem::replace(&mut locked_box.ptr, core::ptr::null_mut());
        unsafe {
            match cuda_free_locked(ptr) {
                Ok(()) => {
                    mem::forget(locked_box);
                    Ok(())
                }
                Err(e) => Err((e, LockedBox { ptr })),
            }
        }
    }
}
impl<T> Drop for LockedBox<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        let ptr = mem::replace(&mut self.ptr, core::ptr::null_mut());
        // No choice but to panic if this fails.
        unsafe {
            cuda_free_locked(ptr).expect("Failed to deallocate CUDA memory.");
        }
    }
}
impl<T> crate::private::Sealed for LockedBox<T> {}

impl<T: DeviceCopy> Borrow<T> for LockedBox<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}
impl<T: DeviceCopy> BorrowMut<T> for LockedBox<T> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}
impl<T: DeviceCopy> AsRef<T> for LockedBox<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}
impl<T: DeviceCopy> AsMut<T> for LockedBox<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}
impl<T: DeviceCopy> Deref for LockedBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.ptr }
    }
}
impl<T: DeviceCopy> DerefMut for LockedBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr }
    }
}
impl<T: Display + DeviceCopy> Display for LockedBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}
impl<T: DeviceCopy> Pointer for LockedBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}
impl<T: DeviceCopy + PartialEq> PartialEq for LockedBox<T> {
    fn eq(&self, other: &LockedBox<T>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}
impl<T: DeviceCopy + Eq> Eq for LockedBox<T> {}
impl<T: DeviceCopy + PartialOrd> PartialOrd for LockedBox<T> {
    fn partial_cmp(&self, other: &LockedBox<T>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
    fn lt(&self, other: &LockedBox<T>) -> bool {
        PartialOrd::lt(&**self, &**other)
    }
    fn le(&self, other: &LockedBox<T>) -> bool {
        PartialOrd::le(&**self, &**other)
    }
    fn ge(&self, other: &LockedBox<T>) -> bool {
        PartialOrd::ge(&**self, &**other)
    }
    fn gt(&self, other: &LockedBox<T>) -> bool {
        PartialOrd::gt(&**self, &**other)
    }
}
impl<T: DeviceCopy + Ord> Ord for LockedBox<T> {
    fn cmp(&self, other: &LockedBox<T>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}
impl<T: DeviceCopy + Hash> Hash for LockedBox<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

/// Fixed-size host-side buffer in page-locked memory.
///
/// See the [`module-level documentation`](../memory/index.html) for more details on page-locked
/// memory.
#[derive(Debug)]
pub struct LockedBuffer<T: DeviceCopy> {
    buf: *mut T,
    capacity: usize,
}
impl<T: DeviceCopy + Clone> LockedBuffer<T> {
    /// Allocate a new page-locked buffer large enough to hold `size` `T`'s and initialized with
    /// clones of `value`.
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA. If `size` is large enough that
    /// `size * mem::sizeof::<T>()` overflows usize, then returns InvalidMemoryAllocation.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let mut buffer = LockedBuffer::new(&0u64, 5).unwrap();
    /// buffer[0] = 1;
    /// ```
    pub fn new(value: &T, size: usize) -> CudaResult<Self> {
        unsafe {
            let mut uninit = LockedBuffer::uninitialized(size)?;
            for x in 0..size {
                *uninit.get_unchecked_mut(x) = value.clone();
            }
            Ok(uninit)
        }
    }

    /// Allocate a new page-locked buffer of the same size as `slice`, initialized with a clone of
    /// the data in `slice`.
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let values = [0u64; 5];
    /// let mut buffer = LockedBuffer::from_slice(&values).unwrap();
    /// buffer[0] = 1;
    /// ```
    pub fn from_slice(slice: &[T]) -> CudaResult<Self> {
        unsafe {
            let mut uninit = LockedBuffer::uninitialized(slice.len())?;
            for (i, x) in slice.iter().enumerate() {
                *uninit.get_unchecked_mut(i) = x.clone();
            }
            Ok(uninit)
        }
    }
}
impl<T: DeviceCopy> LockedBuffer<T> {
    /// Allocate a new page-locked buffer large enough to hold `size` `T`'s, but without
    /// initializing the contents.
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA. If `size` is large enough that
    /// `size * mem::sizeof::<T>()` overflows usize, then returns InvalidMemoryAllocation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the contents of the buffer are initialized before reading from
    /// the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let mut buffer = unsafe { LockedBuffer::uninitialized(5).unwrap() };
    /// for i in buffer.iter_mut() {
    ///     *i = 0u64;
    /// }
    /// ```
    pub unsafe fn uninitialized(size: usize) -> CudaResult<Self> {
        let ptr: *mut T = if size > 0 && mem::size_of::<T>() > 0 {
            cuda_malloc_locked(size)?
        } else {
            ptr::NonNull::dangling().as_ptr()
        };
        Ok(LockedBuffer {
            buf: ptr as *mut T,
            capacity: size,
        })
    }

    /// Extracts a slice containing the entire buffer.
    ///
    /// Equivalent to `&s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let buffer = LockedBuffer::new(&0u64, 5).unwrap();
    /// let sum : u64 = buffer.as_slice().iter().sum();
    /// ```
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Extracts a mutable slice of the entire buffer.
    ///
    /// Equivalent to `&mut s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let mut buffer = LockedBuffer::new(&0u64, 5).unwrap();
    /// for i in buffer.as_mut_slice() {
    ///     *i = 12u64;
    /// }
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Creates a `LockedBuffer<T>` directly from the raw components of another locked buffer.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `ptr` needs to have been previously allocated via `LockedBuffer` or
    /// [`cuda_malloc_locked`](fn.cuda_malloc_locked.html).
    /// * `ptr`'s `T` needs to have the same size and alignment as it was allocated with.
    /// * `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the CUDA driver's
    /// internal data structures.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `LockedBuffer<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    ///
    /// # Examples
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use std::mem;
    /// use rustacuda::memory::*;
    ///
    /// let mut buffer = LockedBuffer::new(&0u64, 5).unwrap();
    /// let ptr = buffer.as_mut_ptr();
    /// let size = buffer.len();
    ///
    /// mem::forget(buffer);
    ///
    /// let buffer = unsafe { LockedBuffer::from_raw_parts(ptr, size) };
    /// ```
    pub unsafe fn from_raw_parts(ptr: *mut T, size: usize) -> LockedBuffer<T> {
        LockedBuffer {
            buf: ptr,
            capacity: size,
        }
    }

    /// Destroy a `LockedBuffer`, returning an error.
    ///
    /// Deallocating page-locked memory can return errors from previous asynchronous work. This function
    /// destroys the given buffer and returns the error and the un-destroyed buffer on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let x = LockedBuffer::new(&0u64, 5).unwrap();
    /// match LockedBuffer::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, buf)) => {
    ///         println!("Failed to destroy buffer: {:?}", e);
    ///         // Do something with buf
    ///     },
    /// }
    /// ```
    pub fn drop(mut buf: LockedBuffer<T>) -> DropResult<LockedBuffer<T>> {
        if buf.buf.is_null() {
            return Ok(());
        }

        if buf.capacity > 0 && mem::size_of::<T>() > 0 {
            let capacity = buf.capacity;
            let ptr = mem::replace(&mut buf.buf, ptr::null_mut());
            unsafe {
                match cuda_free_locked(ptr) {
                    Ok(()) => {
                        mem::forget(buf);
                        Ok(())
                    }
                    Err(e) => Err((e, LockedBuffer::from_raw_parts(ptr, capacity))),
                }
            }
        } else {
            Ok(())
        }
    }
}

impl<T: DeviceCopy> AsRef<[T]> for LockedBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}
impl<T: DeviceCopy> AsMut<[T]> for LockedBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}
impl<T: DeviceCopy> Deref for LockedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            let p = self.buf;
            slice::from_raw_parts(p, self.capacity)
        }
    }
}
impl<T: DeviceCopy> DerefMut for LockedBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            let ptr = self.buf;
            slice::from_raw_parts_mut(ptr, self.capacity)
        }
    }
}
impl<T: DeviceCopy> Drop for LockedBuffer<T> {
    fn drop(&mut self) {
        if self.buf.is_null() {
            return;
        }

        if self.capacity > 0 && mem::size_of::<T>() > 0 {
            // No choice but to panic if this fails.
            unsafe {
                cuda_free_locked(self.buf).expect("Failed to deallocate CUDA page-locked memory.");
            }
        }
        self.capacity = 0;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::mem;

    #[derive(Clone, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_new() {
        let _context = crate::quick_init().unwrap();
        let val = 0u64;
        let mut buffer = LockedBuffer::new(&val, 5).unwrap();
        buffer[0] = 1;
    }

    #[test]
    fn test_from_slice() {
        let _context = crate::quick_init().unwrap();
        let values = [0u64; 10];
        let mut buffer = LockedBuffer::from_slice(&values).unwrap();
        for i in buffer[0..3].iter_mut() {
            *i = 10;
        }
    }

    #[test]
    fn from_raw_parts() {
        let _context = crate::quick_init().unwrap();
        let mut buffer = LockedBuffer::new(&0u64, 5).unwrap();
        buffer[2] = 1;
        let ptr = buffer.as_mut_ptr();
        let len = buffer.len();
        mem::forget(buffer);

        let buffer = unsafe { LockedBuffer::from_raw_parts(ptr, len) };
        assert_eq!(&[0u64, 0, 1, 0, 0], buffer.as_slice());
        drop(buffer);
    }

    #[test]
    fn zero_length_buffer() {
        let _context = crate::quick_init().unwrap();
        let buffer = LockedBuffer::new(&0u64, 0).unwrap();
        drop(buffer);
    }

    #[test]
    fn zero_size_type() {
        let _context = crate::quick_init().unwrap();
        let buffer = LockedBuffer::new(&ZeroSizedType, 10).unwrap();
        drop(buffer);
    }

    #[test]
    fn overflows_usize() {
        let _context = crate::quick_init().unwrap();
        let err = LockedBuffer::new(&0u64, ::std::usize::MAX - 1).unwrap_err();
        assert_eq!(CudaError::InvalidMemoryAllocation, err);
    }

    #[test]
    fn test_allocate_correct_size() {
        let _context = crate::quick_init().unwrap();

        // Placeholder - read out available system memory here
        let allocation_size = 1;
        unsafe {
            // Test if allocation fails with an out-of-memory error
            let _buffer = LockedBuffer::<u64>::uninitialized(allocation_size).unwrap();
        }
    }
}
