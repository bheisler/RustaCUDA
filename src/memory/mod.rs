//! Access to CUDA's memory allocation and transfer functions.
//!
//! The memory module provides a safe wrapper around CUDA's memory allocation and transfer functions.
//! This includes access to device memory, unified memory, and page-locked host memory.
//!
//! # Device Memory
//!
//! Device memory is just what it sounds like - memory allocated on the device. Device memory
//! cannot be accessed from the host directly, but data can be copied to and from the device.
//! RustaCUDA exposes device memory through the [`DeviceBox`](struct.Devicebox.html) and
//! [`DeviceBuffer`](struct.DeviceBuffer.html) structures, and pointers to device memory are
//! represented by [`DevicePointer`](struct.DevicePointer.html).
//!
//! # Unified Memory
//!
//! Unified memory is a memory allocation which can be read from and written to by both the host
//! and the device. When the host (or device) attempts to access a page of unified memory, it is
//! seamlessly transferred from CPU RAM to GPU RAM or vice versa. The programmer may also choose to
//! explicitly prefetch data to one side or another. RustaCUDA exposes unified memory through the
//! [`UnifiedBox`](struct.UnifiedBox.html) and [`UBuffer`](struct.UBuffer.html) structures, and pointers to
//! unified memory are represented by [`UnifiedPointer`](struct.UnifiedPointer.html).
//!
//! Unified memory is generally easier to use than device memory, but there are drawbacks. It is
//! possible to allocate more memory than is available on the card, and this can result in very slow
//! paging behavior. Additionally, it can require careful use of prefetching to achieve optimum
//! performance. Finally, unified memory is not supported on some older systems.
//!
//! # Page-locked Host Memory
//!
//! Page-locked memory is memory that the operating system has locked into physical RAM, and will
//! not page out to disk. When copying data from the process' memory space to the device, the CUDA
//! driver needs to first copy the data to a page-locked region of host memory, then initiate a DMA
//! transfer to copy the data to the device itself. Likewise, when transferring from device to host,
//! the driver copies the data into page-locked host memory then into the normal memory space. This
//! extra copy can be eliminated if the data is loaded or generated directly into page-locked
//! memory. RustaCUDA exposes page-locked memory through the
//! [`LockedBuffer`](struct.LockedBuffer.html) struct.
//!
//! For example, if the programmer needs to read an array of bytes from disk and transfer it to the
//! device, it would be best to create a `LockedBuffer`, load the bytes directly into the
//! `LockedBuffer`, and then copy them to a `DeviceBuffer`. If the bytes are in a `Vec<u8>`, there
//! would be no advantage to using a `LockedBuffer`.
//!
//! However, since the OS cannot page out page-locked memory, excessive use can slow down the entire
//! system (including other processes) as physical RAM is tied up.  Therefore, page-locked memory
//! should be used sparingly.

mod locked;
mod malloc;
mod pointer;
mod unified;
//mod device;

pub use self::locked::*;
pub use self::malloc::*;
pub use self::pointer::*;
pub use self::unified::*;
//pub use self::device::*;

use std::marker::PhantomData;
use std::num::*;

/// Marker trait for types which can safely be copied to or from a CUDA device.
///
/// A type can be safely copied if its value can be duplicated simply by copying bits and if it does
/// not contain a reference to memory which is not accessible to the device. Additionally, the
/// DeviceCopy trait does not imply copy semantics as the Copy trait does.
///
/// ## How can I implement DeviceCopy?
///
/// There are two ways to implement DeviceCopy on your type. The simplest is to use `derive`:
///
/// ```
/// // TODO: Implement this.
/// ```
///
/// You can also implement `DeviceCopy` manually:
///
/// ```
/// use rustacuda::memory::DeviceCopy;
///
/// #[derive(Clone)]
/// struct MyStruct{
///     x: u8
/// }
///
/// unsafe impl DeviceCopy for MyStruct { }
/// ```
///
/// ## What is the difference between `DeviceCopy` and `Copy`?
///
/// `DeviceCopy` is stricter than `Copy`. `DeviceCopy` must only be implemented for types which
/// do not contain references or raw pointers to non-device-accessible memory. `DeviceCopy` also
/// does not imply copy semantics - that is, `DeviceCopy` values are not implicitly copied on
/// assignment the way that `Copy` values are. This is helpful, as it may be desirable to implement
/// `DeviceCopy` for large structures that would be inefficient to copy for every assignment.
///
/// ## When can't my type be `DeviceCopy`?
///
/// Some types cannot be safely copied to the device. For example, copying `&T` would create an
/// invalid reference on the device which would segfault if dereferenced. Generalizing this, any
/// type implementing `Drop` cannot be `DeviceCopy` since it is responsible for some resource that
/// would not be available on the device.
pub unsafe trait DeviceCopy: Clone {
    // Empty
}

macro_rules! impl_device_copy {
    ($($t:ty)*) => {
        $(
            unsafe impl DeviceCopy for $t {}
        )*
    }
}

impl_device_copy!(
    usize u8 u16 u32 u64 u128
    isize i8 i16 i32 i64 i128
    f32 f64
    bool char

    NonZeroU8 NonZeroU16 NonZeroU32 NonZeroU64 NonZeroU128
);
unsafe impl<T: DeviceCopy> DeviceCopy for Option<T> {}
unsafe impl<L: DeviceCopy, R: DeviceCopy> DeviceCopy for Result<L, R> {}
unsafe impl<T: ?Sized + DeviceCopy> DeviceCopy for PhantomData<T> {}
unsafe impl<T: DeviceCopy> DeviceCopy for Wrapping<T> {}

macro_rules! impl_device_copy_array {
    ($($n:expr)*) => {
        $(
            unsafe impl<T: DeviceCopy> DeviceCopy for [T;$ n] {}
        )*
    }
}

impl_device_copy_array!{
    1 2 3 4 5 6 7 8 9 10
    11 12 13 14 15 16 17 18 19 20
    21 22 23 24 25 26 27 28 29 30
    31 32
}
unsafe impl DeviceCopy for () {}
unsafe impl<A: DeviceCopy, B: DeviceCopy> DeviceCopy for (A, B) {}
unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy> DeviceCopy for (A, B, C) {}
unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy, D: DeviceCopy> DeviceCopy
    for (A, B, C, D)
{}
unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy, D: DeviceCopy, E: DeviceCopy> DeviceCopy
    for (A, B, C, D, E)
{}
unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy, D: DeviceCopy, E: DeviceCopy, F: DeviceCopy>
    DeviceCopy for (A, B, C, D, E, F)
{}
unsafe impl<
        A: DeviceCopy,
        B: DeviceCopy,
        C: DeviceCopy,
        D: DeviceCopy,
        E: DeviceCopy,
        F: DeviceCopy,
        G: DeviceCopy,
    > DeviceCopy for (A, B, C, D, E, F, G)
{}
unsafe impl<
        A: DeviceCopy,
        B: DeviceCopy,
        C: DeviceCopy,
        D: DeviceCopy,
        E: DeviceCopy,
        F: DeviceCopy,
        G: DeviceCopy,
        H: DeviceCopy,
    > DeviceCopy for (A, B, C, D, E, F, G, H)
{}
