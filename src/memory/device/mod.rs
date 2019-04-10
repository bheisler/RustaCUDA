use crate::error::CudaResult;
use crate::stream::Stream;

mod device_box;
mod device_buffer;
mod device_slice;

pub use self::device_box::*;
pub use self::device_buffer::*;
pub use self::device_slice::*;

/// Sealed trait implemented by types which can be the source or destination when copying data
/// to/from the device or from one device allocation to another.
pub trait CopyDestination<O: ?Sized>: crate::private::Sealed {
    /// Copy data from `source`. `source` must be the same size as `self`.
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, return the error.
    fn copy_from(&mut self, source: &O) -> CudaResult<()>;

    /// Copy data to `dest`. `dest` must be the same size as `self`.
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, return the error.
    fn copy_to(&self, dest: &mut O) -> CudaResult<()>;
}

/// Sealed trait implemented by types which can be the source or destination when copying data
/// asynchronously to/from the device or from one device allocation to another.
///
/// ## Safety:
///
/// The fuctions of this trait are unsafe, as these functions return while the copying
/// from source to destination is likely still taking place in the background, allowing
/// the source and destination arguments to be potentially used in unsafe ways.
///
/// Thus to enforce safety, the following invariants should be enforced:
/// * The source and destination are not deallocated
/// * The source is read-only
/// * The destination is not written or read by any other operation
///
pub trait AsyncCopyDestination<O: ?Sized>: crate::private::Sealed {
    /// Asynchronously copy data from `source`. `source` must be the same size as `self`.
    ///
    /// Host memory used as a source or destination must be page-locked.
    ///
    /// For why this function is unsafe, see [AsyncCopyDestination](trait.AsyncCopyDestination.html)
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, return the error.
    unsafe fn async_copy_from(&mut self, source: &O, stream: &Stream) -> CudaResult<()>;

    /// Asynchronously copy data to `dest`. `dest` must be the same size as `self`.
    ///
    /// Host memory used as a source or destination must be page-locked.
    ///
    /// For why this function is unsafe, see [AsyncCopyDestination](trait.AsyncCopyDestination.html)
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, return the error.
    unsafe fn async_copy_to(&self, dest: &mut O, stream: &Stream) -> CudaResult<()>;
}
