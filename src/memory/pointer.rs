use memory::DeviceCopy;
use std::ptr;

/// A pointer to device memory.
///
/// DevicePointer cannot be dereferenced by the CPU, as it is a pointer to a memory allocation in
/// the device. It can be safely copied to the device (eg. as part of a kernel launch) and either
/// unwrapped or transmuted to an appropriate pointer.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct DevicePointer<T: DeviceCopy>(*mut T);
unsafe impl<T: DeviceCopy> DeviceCopy for DevicePointer<T> {}
impl<T: DeviceCopy> DevicePointer<T> {
    /// Returns a null device pointer.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let ptr : DevicePointer<u64> = DevicePointer::null();
    /// assert!(ptr.is_null());
    /// ```
    pub fn null() -> Self {
        unsafe { Self::wrap(ptr::null_mut()) }
    }

    /// Wrap the given raw pointer in a DevicePointer. The given pointer is assumed to be a valid,
    /// device pointer or null.
    ///
    /// # Safety
    ///
    /// The given pointer must have been allocated with [`cuda_malloc`](fn.cuda_malloc.html) or
    /// be null.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// use std::ptr;
    /// unsafe {
    ///     let null : *mut u64 = ptr::null_mut();
    ///     assert!(DevicePointer::wrap(null).is_null());
    /// }
    /// ```
    pub unsafe fn wrap(ptr: *mut T) -> Self {
        DevicePointer(ptr)
    }

    /// Returns the contained pointer as a raw pointer. The returned pointer is not valid on the CPU
    /// and must not be dereferenced.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let dev_ptr = cuda_malloc::<u64>(1).unwrap();
    ///     let ptr: *const u64 = dev_ptr.as_raw();
    ///     cuda_free(dev_ptr);
    /// }
    /// ```
    pub fn as_raw(&self) -> *const T {
        self.0
    }

    /// Returns the contained pointer as a mutable raw pointer. The returned pointer is not valid on the CPU
    /// and must not be dereferenced.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(1).unwrap();
    ///     let ptr: *mut u64 = dev_ptr.as_raw_mut();
    ///     cuda_free(dev_ptr);
    /// }
    /// ```
    pub fn as_raw_mut(&mut self) -> *mut T {
        self.0
    }

    /// Returns true if the pointer is null.
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// use std::ptr;
    /// unsafe {
    ///     let null : *mut u64 = ptr::null_mut();
    ///     assert!(DevicePointer::wrap(null).is_null());
    /// }
    /// ```
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Calculates the offset from a device pointer.
    ///
    /// `count` is in units of T; eg. a `count` of 3 represents a pointer offset of
    /// `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of *the same* allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum, **in bytes** must fit in a usize.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.offset(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub unsafe fn offset(self, count: isize) -> Self {
        Self::wrap(self.0.offset(count))
    }

    /// Calculates the offset from a device pointer using wrapping arithmetic.
    ///
    /// `count` is in units of T; eg. a `count` of 3 represents a pointer offset of
    /// `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    /// In particular, the resulting pointer may *not* be used to access a
    /// different allocated object than the one `self` points to. In other
    /// words, `x.wrapping_offset(y.wrapping_offset_from(x))` is
    /// *not* the same as `y`, and dereferencing it is undefined behavior
    /// unless `x` and `y` point into the same allocated object.
    ///
    /// Always use `.offset(count)` instead when possible, because `offset`
    /// allows the compiler to optimize better.  If you need to cross object
    /// boundaries, cast the pointer to an integer and do the arithmetic there.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.wrapping_offset(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub fn wrapping_offset(self, count: isize) -> Self {
        unsafe { Self::wrap(self.0.wrapping_offset(count)) }
    }

    /// Calculates the offset from a pointer (convenience for `.offset(count as isize)`).
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of an allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum must fit in a `usize`.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.add(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    #[allow(should_implement_trait)]
    pub unsafe fn add(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.offset(count as isize)
    }

    /// Calculates the offset from a pointer (convenience for
    /// `.offset((count as isize).wrapping_neg())`).
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of an allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum must fit in a `usize`.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.add(4).sub(3); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    #[allow(should_implement_trait)]
    pub unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.offset((count as isize).wrapping_neg())
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset(count as isize)`)
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference.
    ///
    /// Always use `.add(count)` instead when possible, because `add`
    /// allows the compiler to optimize better.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.wrapping_add(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub fn wrapping_add(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.wrapping_offset(count as isize)
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset((count as isize).wrapping_sub())`)
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    ///
    /// Always use `.sub(count)` instead when possible, because `sub`
    /// allows the compiler to optimize better.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // Iterate using a raw pointer in increments of two elements (backwards)
    /// let data = [1u8, 2, 3, 4, 5];
    /// let mut ptr: *const u8 = data.as_ptr();
    /// let start_rounded_down = ptr.wrapping_sub(2);
    /// ptr = ptr.wrapping_add(4);
    /// let step = 2;
    /// // This loop prints "5, 3, 1, "
    /// while ptr != start_rounded_down {
    ///     unsafe {
    ///         print!("{}, ", *ptr);
    ///     }
    ///     ptr = ptr.wrapping_sub(step);
    /// }
    /// ```
    pub fn wrapping_sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.wrapping_offset((count as isize).wrapping_neg())
    }
}
impl<T: DeviceCopy> ::std::fmt::Pointer for DevicePointer<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::std::fmt::Pointer::fmt(&self.0, f)
    }
}

/// A pointer to unified memory.
///
/// UnifiedPointer can be safely dereferenced by the CPU, as the memory allocation it points to is
/// shared between the CPU and the GPU. It can also be safely copied to the device (eg. as part of
/// a kernel launch).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct UnifiedPointer<T: DeviceCopy>(*mut T);
unsafe impl<T: DeviceCopy> DeviceCopy for UnifiedPointer<T> {}
impl<T: DeviceCopy> UnifiedPointer<T> {
    /// Returns a null unified pointer.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// let ptr : UnifiedPointer<u64> = UnifiedPointer::null();
    /// assert!(ptr.is_null());
    /// ```
    pub fn null() -> Self {
        unsafe { Self::wrap(ptr::null_mut()) }
    }

    /// Wrap the given raw pointer in a UnifiedPointer. The given pointer is assumed to be a valid,
    /// unified-memory pointer or null.
    ///
    /// # Safety
    ///
    /// The given pointer must have been allocated with
    /// [`cuda_malloc_unified`](fn.cuda_malloc_unified.html) or be null.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// use std::ptr;
    /// unsafe {
    ///     let null : *mut u64 = ptr::null_mut();
    ///     assert!(UnifiedPointer::wrap(null).is_null());
    /// }
    /// ```
    pub unsafe fn wrap(ptr: *mut T) -> Self {
        UnifiedPointer(ptr)
    }

    /// Returns the contained pointer as a raw pointer.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let unified_ptr = cuda_malloc_unified::<u64>(1).unwrap();
    ///     let ptr: *const u64 = unified_ptr.as_raw();
    ///     cuda_free(unified_ptr);
    /// }
    /// ```
    pub fn as_raw(&self) -> *const T {
        self.0
    }

    /// Returns the contained pointer as a mutable raw pointer.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let mut unified_ptr = cuda_malloc_unified::<u64>(1).unwrap();
    ///     let ptr: *mut u64 = unified_ptr.as_raw_mut();
    ///     *ptr = 5u64;
    ///     cuda_free(unified_ptr);
    /// }
    /// ```
    pub fn as_raw_mut(&mut self) -> *mut T {
        self.0
    }

    /// Returns true if the pointer is null.
    ///
    /// # Examples:
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// use std::ptr;
    /// unsafe {
    ///     let null : *mut u64 = ptr::null_mut();
    ///     assert!(UnifiedPointer::wrap(null).is_null());
    /// }
    /// ```
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Calculates the offset from a unified pointer.
    ///
    /// `count` is in units of T; eg. a `count` of 3 represents a pointer offset of
    /// `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of *the same* allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum, **in bytes** must fit in a usize.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let mut unified_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = unified_ptr.offset(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free(unified_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub unsafe fn offset(self, count: isize) -> Self {
        Self::wrap(self.0.offset(count))
    }

    /// Calculates the offset from a unified pointer using wrapping arithmetic.
    ///
    /// `count` is in units of T; eg. a `count` of 3 represents a pointer offset of
    /// `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    /// In particular, the resulting pointer may *not* be used to access a
    /// different allocated object than the one `self` points to. In other
    /// words, `x.wrapping_offset(y.wrapping_offset_from(x))` is
    /// *not* the same as `y`, and dereferencing it is undefined behavior
    /// unless `x` and `y` point into the same allocated object.
    ///
    /// Always use `.offset(count)` instead when possible, because `offset`
    /// allows the compiler to optimize better.  If you need to cross object
    /// boundaries, cast the pointer to an integer and do the arithmetic there.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.wrapping_offset(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub fn wrapping_offset(self, count: isize) -> Self {
        unsafe { Self::wrap(self.0.wrapping_offset(count)) }
    }

    /// Calculates the offset from a pointer (convenience for `.offset(count as isize)`).
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of an allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum must fit in a `usize`.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.add(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    #[allow(should_implement_trait)]
    pub unsafe fn add(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.offset(count as isize)
    }

    /// Calculates the offset from a pointer (convenience for
    /// `.offset((count as isize).wrapping_neg())`).
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of an allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum must fit in a `usize`.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.add(4).sub(3); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    #[allow(should_implement_trait)]
    pub unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.offset((count as isize).wrapping_neg())
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset(count as isize)`)
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference.
    ///
    /// Always use `.add(count)` instead when possible, because `add`
    /// allows the compiler to optimize better.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustacuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = cuda_malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.wrapping_add(1); // Points to the 2nd u64 in the buffer
    ///     cuda_free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    pub fn wrapping_add(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.wrapping_offset(count as isize)
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset((count as isize).wrapping_sub())`)
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    ///
    /// Always use `.sub(count)` instead when possible, because `sub`
    /// allows the compiler to optimize better.
    ///
    /// # Examples
    ///
    /// ```
    /// // Iterate using a raw pointer in increments of two elements (backwards)
    /// let data = [1u8, 2, 3, 4, 5];
    /// let mut ptr: *const u8 = data.as_ptr();
    /// let start_rounded_down = ptr.wrapping_sub(2);
    /// ptr = ptr.wrapping_add(4);
    /// let step = 2;
    /// // This loop prints "5, 3, 1, "
    /// while ptr != start_rounded_down {
    ///     unsafe {
    ///         print!("{}, ", *ptr);
    ///     }
    ///     ptr = ptr.wrapping_sub(step);
    /// }
    /// ```
    pub fn wrapping_sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.wrapping_offset((count as isize).wrapping_neg())
    }
}
impl<T: DeviceCopy> ::std::fmt::Pointer for UnifiedPointer<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::std::fmt::Pointer::fmt(&self.0, f)
    }
}
