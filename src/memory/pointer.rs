use memory::DeviceCopy;

/// A struct representing a pointer to device memory.
///
/// DevicePointer cannot be dereferenced by the CPU, as it is a pointer to a memory allocation in
/// the device. It can be safely copied to the device (eg. as part of a kernel launch) and either
/// unwrapped or transmuted to an appropriate pointer.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct DevicePointer<T: DeviceCopy>(*mut T);
unsafe impl<T: DeviceCopy> DeviceCopy for DevicePointer<T> {}
impl<T: DeviceCopy> DevicePointer<T> {
    /// Wrap the given raw pointer in a DevicePointer. The given pointer is assumed to be a valid,
    /// device pointer or null.
    pub fn wrap(ptr: *mut T) -> DevicePointer<T> {
        DevicePointer(ptr)
    }

    /// Unwrap the contained pointer.
    pub fn into_raw(self) -> *mut T {
        self.0
    }

    /// Returns true if the pointer is null.
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Calculates the offset from a device pointer.
    ///
    /// `count` is in units of T; eg. a `count` of 3 represents a pointer offset of
    /// `3 * size_of::<T>()` bytes.
    pub unsafe fn offset(self, count: isize) -> DevicePointer<T> {
        Self::wrap(self.0.offset(count))
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    ///
    /// `count` is in units of T; eg. a `count` of 3 represents a pointer offset of
    /// `3 * size_of::<T>()` bytes.
    pub unsafe fn wrapping_offset(self, count: isize) -> DevicePointer<T> {
        Self::wrap(self.0.wrapping_offset(count))
    }

    /// Calculates the offset from a pointer (convenience for `.offset(count as isize)`).
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
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
    #[allow(should_implement_trait)]
    pub unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        self.offset((count as isize).wrapping_neg())
    }
}
impl<T: DeviceCopy> ::std::fmt::Pointer for DevicePointer<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::std::fmt::Pointer::fmt(&self.0, f)
    }
}
impl<T: DeviceCopy> ::std::convert::From<UnifiedPointer<T>> for DevicePointer<T> {
    fn from(ptr: UnifiedPointer<T>) -> DevicePointer<T> {
        DevicePointer::wrap(ptr.into_raw())
    }
}

/// A struct representing a pointer to unified memory.
///
/// UnifiedPointer can be safely dereferenced by the CPU, as the memory allocation it points to is
/// shared between the CPU and the GPU. It can also be safely copied to the device (eg. as part of
/// a kernel launch).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct UnifiedPointer<T: DeviceCopy>(*mut T);
unsafe impl<T: DeviceCopy> DeviceCopy for UnifiedPointer<T> {}
impl<T: DeviceCopy> UnifiedPointer<T> {
    /// Wrap the given raw pointer in a DevicePointer. The given pointer is assumed to be a valid,
    /// device pointer or null.
    pub fn wrap(ptr: *mut T) -> UnifiedPointer<T> {
        UnifiedPointer(ptr)
    }

    /// Unwrap the contained pointer.
    pub fn into_raw(self) -> *mut T {
        self.0
    }
}
impl<T: DeviceCopy> ::std::fmt::Pointer for UnifiedPointer<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::std::fmt::Pointer::fmt(&self.0, f)
    }
}
impl<T: DeviceCopy> ::std::ops::Deref for UnifiedPointer<T> {
    type Target = *mut T;

    fn deref(&self) -> &*mut T {
        &self.0
    }
}
impl<T: DeviceCopy> ::std::ops::DerefMut for UnifiedPointer<T> {
    fn deref_mut(&mut self) -> &mut *mut T {
        &mut self.0
    }
}
