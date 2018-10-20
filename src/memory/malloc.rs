use error::*;
use cuda_sys::cudart::{
    cudaMallocManaged as cudaMallocManaged_raw, 
    cudaMalloc as cudaMalloc_raw,
    cudaFree as cudaFree_raw,
    cudaMemAttachGlobal,
};
use std::mem;
use std::ptr;
use std::os::raw::c_void;
use super::DeviceCopy;
use memory::DevicePointer;
use memory::UnifiedPointer;

/*
You should be able to:
- Allocate and free unified memory
- Allocate and free Device memory
- Allocate and free locked memory
*/

/// Unsafe wrapper around the cudaMalloc function, which allocates some device memory and
/// returns a DevicePointer pointing to it.
/// 
/// Note that `count` is in units of T; thus a `count` of 3 will allocate `3 * size_of::<T>()` bytes
/// of memory.
/// 
/// Memory buffers allocated using cuda_malloc must be freed using cuda_free.
pub unsafe fn cuda_malloc<T: DeviceCopy>(count: usize) -> CudaResult<DevicePointer<T>> {
    let mut ptr: *mut c_void = ptr::null_mut();
    cudaMalloc_raw(&mut ptr as *mut *mut c_void, count * mem::size_of::<T>()).toResult()?;
    let ptr = ptr as *mut T;
    Ok(DevicePointer::wrap( ptr as *mut T ))
}

/// Unsafe wrapper around the cudaMallocManaged function, which allocates some unified memory and
/// returns a UnifiedPointer pointing to it.
/// 
/// Note that `count` is in units of T; thus a `count` of 3 will allocate `3 * size_of::<T>()` bytes
/// of memory.
/// 
/// Memory buffers allocated using cuda_malloc_unified must be freed using cuda_free.
pub unsafe fn cuda_malloc_unified<T: DeviceCopy>(count: usize) -> CudaResult<UnifiedPointer<T>> {
    let mut ptr: *mut c_void = ptr::null_mut();
    cudaMallocManaged_raw(&mut ptr as *mut *mut c_void, count * mem::size_of::<T>(), cudaMemAttachGlobal).toResult()?;
    let ptr = ptr as *mut T;
    Ok(UnifiedPointer::wrap( ptr as *mut T ))
}

/// Free memory allocated with cuda_malloc or cuda_malloc_unified.
pub unsafe fn cuda_free<T: DeviceCopy, P: Into<DevicePointer<T>>>(p: P) -> CudaResult<()> {
    let ptr = p.into().unwrap();
    cudaFree_raw(ptr as *mut c_void).toResult()?;
    Ok(())
}