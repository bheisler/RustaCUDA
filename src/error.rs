use cuda_sys::cuda::cudaError_t;
use cuda_sys::cudart::cudaError_t as cudaRtError_t;
use std::result::Result;

// TODO: Implement debug properly
// TODO: implement display
// TODO: Wrapping cuda-sys errors makes it hard to match on errors. Is that a problem?
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaError {
    CudaError(cudaError_t),
    CudaRtError(cudaRtError_t),
    InvalidMemoryAllocation,
    __Nonexhaustive,
}

pub type CudaResult<T> = Result<T, CudaError>;
pub type DropResult<T> = Result<(), (CudaError, T)>;
pub(crate) trait ToResult {
    fn toResult(self) -> CudaResult<()>;
}
impl ToResult for cudaError_t {
    fn toResult(self) -> CudaResult<()> {
        if self == cudaError_t::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(CudaError::CudaError(self))
        }
    }
}
impl ToResult for cudaRtError_t {
    fn toResult(self) -> CudaResult<()> {
        if self == cudaRtError_t::Success {
            Ok(())
        } else {
            Err(CudaError::CudaRtError(self))
        }
    }
}
