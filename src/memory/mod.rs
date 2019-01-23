//! Access to CUDA's memory allocation and transfer functions.
//!
//! The memory module provides a safe wrapper around CUDA's memory allocation and transfer functions.
//! This includes access to device memory, unified memory, and page-locked host memory.
//!
//! # Device Memory
//!
//! Device memory is just what it sounds like - memory allocated on the device. Device memory
//! cannot be accessed from the host directly, but data can be copied to and from the device.
//! RustaCUDA exposes device memory through the [`DeviceBox`](struct.DeviceBox.html) and
//! [`DeviceBuffer`](struct.DeviceBuffer.html) structures. Pointers to device memory are
//! represented by [`DevicePointer`](struct.DevicePointer.html), while slices in device memory are
//! represented by [`DeviceSlice`](struct.DeviceSlice.html).
//!
//! # Unified Memory
//!
//! Unified memory is a memory allocation which can be read from and written to by both the host
//! and the device. When the host (or device) attempts to access a page of unified memory, it is
//! seamlessly transferred from host RAM to device RAM or vice versa. The programmer may also
//! choose to explicitly prefetch data to one side or another (though this is not currently exposed
//! through RustaCUDA). RustaCUDA exposes unified memory through the
//! [`UnifiedBox`](struct.UnifiedBox.html) and [`UnifiedBuffer`](struct.UnifiedBuffer.html)
//! structures, and pointers to unified memory are represented by
//! [`UnifiedPointer`](struct.UnifiedPointer.html). Since unified memory is accessible to the host,
//! slices in unified memory are represented by normal Rust slices.
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
//!
//! # FFI Information
//!
//! The internal representations of `DevicePointer<T>` and `UnifiedPointer<T>` are guaranteed to be
//! the same as `*mut T` and they can be safely passed through an FFI boundary to code expecting
//! raw pointers (though keep in mind that device-only pointers cannot be dereferenced on the CPU).
//! This is important when launching kernels written in C.
//!
//! As with regular Rust, all other types (eg. `DeviceBuffer` or `UnifiedBox`) are not FFI-safe.
//! Their internal representations are not guaranteed to be anything in particular, and are not
//! guaranteed to be the same in different versions of RustaCUDA. If you need to pass them through
//! an FFI boundary, you must convert them to FFI-safe primitives yourself. For example, with
//! `UnifiedBuffer`, use the `as_unified_ptr()` and `len()` functions to get the primitives, and
//! `mem::forget()` the Buffer so that it isn't dropped. Again, as with regular Rust, the caller is
//! responsible for reconstructing the `UnifiedBuffer` using `from_raw_parts()` and dropping it to
//! ensure that the memory allocation is safely cleaned up.

pub mod array;

mod device;
mod locked;
mod malloc;
mod unified;

pub use self::device::*;
pub use self::locked::*;
pub use self::malloc::*;
pub use self::unified::*;
pub use rustacuda_core::{DeviceCopy, DevicePointer, UnifiedPointer};
