use std::os::raw::c_uint;

use cuda_sys::cuda::{CUarray, CUarray_format, CUarray_format_enum};

use crate::error::*;

/// Describes the format used for a CUDA Array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayFormat {
    /// Unsigned 8-bit integer
    UnsignedInt8,
    /// Unsigned 16-bit integer
    UnsignedInt16,
    /// Unsigned 32-bit integer
    UnsignedInt32,
    /// Signed 8-bit integer
    SignedInt8,
    /// Signed 16-bit integer
    SignedInt16,
    /// Signed 32-bit integer
    SignedInt32,
    /// Half-precision floating point number
    Half,
    /// Single-precision floating point number
    Float,
}

impl ArrayFormat {
    /// Creates ArrayFormat from the CUDA Driver API enum
    pub fn from_raw(raw: CUarray_format) -> Self {
        match raw {
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT8 => ArrayFormat::UnsignedInt8,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16 => ArrayFormat::UnsignedInt16,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32 => ArrayFormat::UnsignedInt32,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT8 => ArrayFormat::SignedInt8,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16 => ArrayFormat::SignedInt16,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32 => ArrayFormat::SignedInt32,
            CUarray_format_enum::CU_AD_FORMAT_HALF => ArrayFormat::Half,
            CUarray_format_enum::CU_AD_FORMAT_FLOAT => ArrayFormat::Float,
        }
    }

    /// Converts ArrayFormat to the CUDA Driver API enum
    pub fn to_raw(self) -> CUarray_format {
        match self {
            ArrayFormat::UnsignedInt8 => CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT8,
            ArrayFormat::UnsignedInt16 => CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16,
            ArrayFormat::UnsignedInt32 => CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32,
            ArrayFormat::SignedInt8 => CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT8,
            ArrayFormat::SignedInt16 => CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16,
            ArrayFormat::SignedInt32 => CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32,
            ArrayFormat::Half => CUarray_format_enum::CU_AD_FORMAT_HALF,
            ArrayFormat::Float => CUarray_format_enum::CU_AD_FORMAT_FLOAT,
        }
    }
}

bitflags! {
    /// Flags which modify the behavior of CUDA array creation.
    #[derive(Default)]
    pub struct ArrayObjectFlags: c_uint {
        /// Enables creation of layered CUDA array.s When this flag is set, depth specifies the
        /// number of layers, not the depth of a 3D array.
        const LAYERED = cuda_sys::cuda::CUDA_ARRAY3D_LAYERED;

        /// Enables surface references to be bound to the CUDA array.
        const SURFACE_LDST = cuda_sys::cuda::CUDA_ARRAY3D_SURFACE_LDST;

        /// Enables creation of cubemaps. If this flag is set, Width must be equal to Height, and
        /// Depth must be six. If the `LAYERED` flag is also set, then Depth must be a multiple of
        /// six.
        const CUBEMAP = cuda_sys::cuda::CUDA_ARRAY3D_CUBEMAP;

        /// Indicates that the CUDA array will be used for texture gather. Texture gather can only
        /// be performed on 2D CUDA arrays.
        const TEXTURE_GATHER = cuda_sys::cuda::CUDA_ARRAY3D_TEXTURE_GATHER;
    }
}

impl ArrayObjectFlags {
    /// Creates a default flags object with no flags set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Checks if LAYERED flag is set
    pub fn layered(self) -> bool {
        self.contains(ArrayObjectFlags::LAYERED)
    }

    /// Returns a copy of `self` with the LAYERED flag set
    pub fn mark_layered(self) -> ArrayObjectFlags {
        self | ArrayObjectFlags::LAYERED
    }

    /// Checks if SURFACE_LDST flag is set
    pub fn surface_ldst(self) -> bool {
        self.contains(ArrayObjectFlags::SURFACE_LDST)
    }

    /// Returns a copy of `self` with the SURFACE_LDST flag set
    pub fn mark_surface_ldst(self) -> ArrayObjectFlags {
        self | ArrayObjectFlags::SURFACE_LDST
    }

    /// Checks if CUBEMAP flag is set
    pub fn cubemap(self) -> bool {
        self.contains(ArrayObjectFlags::CUBEMAP)
    }

    /// Returns a copy of `self` with the CUBEMAP flag set
    pub fn mark_cubemap(self) -> ArrayObjectFlags {
        self | ArrayObjectFlags::CUBEMAP
    }

    /// Checks if TEXTURE_GATHER flag is set
    pub fn texture_gather(self) -> bool {
        self.contains(ArrayObjectFlags::TEXTURE_GATHER)
    }

    /// Returns a copy of `self` with the TEXTURE_GATHER flag set
    pub fn mark_texture_gather(self) -> ArrayObjectFlags {
        self | ArrayObjectFlags::TEXTURE_GATHER
    }
}

/// Describes a CUDA Array
#[derive(Clone, Copy, Debug)]
pub struct ArrayDescriptor {
    desc: cuda_sys::cuda::CUDA_ARRAY3D_DESCRIPTOR,
}

impl ArrayDescriptor {
    /// Constructs an ArrayDescriptor from a CUDA Driver API Array Descriptor.
    pub fn from_raw(desc: cuda_sys::cuda::CUDA_ARRAY3D_DESCRIPTOR) -> Self {
        Self { desc }
    }

    /// Creates a new ArrayDescriptor from a set of dimensions and format.
    pub fn from_dims_format(dims: [usize; 3], format: ArrayFormat) -> Self {
        Self {
            desc: cuda_sys::cuda::CUDA_ARRAY3D_DESCRIPTOR {
                Width: dims[0],
                Height: dims[1],
                Depth: dims[2],
                Format: format.to_raw(),
                NumChannels: 1,
                Flags: ArrayObjectFlags::default().bits(),
            },
        }
    }

    /// Returns the dimensions of the ArrayDescriptor
    pub fn dims(&self) -> [usize; 3] {
        [self.desc.Width, self.desc.Height, self.desc.Depth]
    }

    /// Sets the dimensions of the ArrayDescriptor
    pub fn set_dims(&mut self, dims: [usize; 3]) {
        self.desc.Width = dims[0];
        self.desc.Height = dims[1];
        self.desc.Depth = dims[2];
    }

    /// Sets the dimensions of the ArrayDescriptor. Composes with other with_ routines.
    pub fn with_dims(mut self, dims: [usize; 3]) -> Self {
        self.set_dims(dims);
        self
    }

    /// Returns the width of the ArrayDescripor
    pub fn width(&self) -> usize {
        self.desc.Width
    }

    /// Sets the width of the ArrayDescriptor
    pub fn set_width(&mut self, width: usize) {
        self.desc.Width = width;
    }

    /// Sets the width of the ArrayDescriptor. Composes with other with_ routines.
    pub fn mark_width(mut self, width: usize) -> Self {
        self.set_width(width);
        self
    }

    /// Returns the height of the ArrayDescripor
    pub fn height(&self) -> usize {
        self.desc.Height
    }

    /// Sets the height of the ArrayDescriptor
    pub fn set_height(&mut self, height: usize) {
        self.desc.Height = height;
    }

    /// Sets the height of the ArrayDescriptor. Composes with other with_ routines.
    pub fn with_height(mut self, height: usize) -> Self {
        self.set_height(height);
        self
    }

    /// Returns the depth of the ArrayDescripor
    pub fn depth(&self) -> usize {
        self.desc.Depth
    }

    /// Sets the depth of the ArrayDescriptor
    pub fn set_depth(&mut self, depth: usize) {
        self.desc.Depth = depth;
    }

    /// Sets the depth of the ArrayDescriptor. Composes with other with_ routines.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.set_depth(depth);
        self
    }

    /// Returns the format of the ArrayDescripor
    pub fn format(&self) -> ArrayFormat {
        ArrayFormat::from_raw(self.desc.Format)
    }

    /// Sets the format of the ArrayDescriptor
    pub fn set_format(&mut self, format: ArrayFormat) {
        self.desc.Format = format.to_raw();
    }

    /// Sets the format of the ArrayDescriptor. Composes with other with_ routines.
    pub fn with_format(mut self, format: ArrayFormat) -> Self {
        self.set_format(format);
        self
    }

    /// Returns the number of channels in the ArrayDescriptor
    pub fn num_channels(&self) -> c_uint {
        self.desc.NumChannels
    }

    /// Sets the number of channels in the ArrayDescriptor
    pub fn set_num_channels(&mut self, num_channels: c_uint) {
        self.desc.NumChannels = num_channels;
    }

    /// Sets the number of channels in the ArrayDescriptor. Composes with other with_ routinese.
    pub fn with_num_channels(mut self, num_channels: c_uint) -> Self {
        self.set_num_channels(num_channels);
        self
    }

    /// Returns the flags of the ArrayDescriptor
    pub fn flags(&self) -> ArrayObjectFlags {
        ArrayObjectFlags::from_bits_truncate(self.desc.Flags)
    }

    /// Sets the flags of the ArrayDescriptor.
    pub fn set_flags(&mut self, flags: ArrayObjectFlags) {
        self.desc.Flags = flags.bits();
    }

    /// Sets the flags of the ArrayDescriptor. Composes with other with_ routines.
    pub fn with_flags(mut self, flags: ArrayObjectFlags) -> Self {
        self.set_flags(flags);
        self
    }
}

/// A CUDA Array. Can be bound to a texture or surface.
pub struct ArrayObject {
    handle: CUarray,
}

impl ArrayObject {
    /// Constructs a generic ArrayObject
    pub fn from_descriptor(descriptor: ArrayDescriptor) -> CudaResult<Self> {
        // We validate the descriptor up front. This provides a good error message to the user.
        assert_ne!(
            0,
            descriptor.width(),
            "Cannot allocate an array with 0 Width"
        );

        if !descriptor.flags().layered() && descriptor.depth() > 0 {
            assert_ne!(
                0,
                descriptor.height(),
                "If Depth is non-zero and the descriptor is not LAYERED, then Height must also be \
                 non-zero."
            );
        }

        if descriptor.flags().cubemap() {
            assert_eq!(
                descriptor.height(),
                descriptor.width(),
                "Height and Width must be equal for CUBEMAP arrays."
            );

            if descriptor.flags().layered() {
                assert_eq!(
                    0,
                    descriptor.depth() % 6,
                    "Depth must be a multiple of 6 when the array descriptor is for a LAYERED \
                     CUBEMAP."
                );
            } else {
                assert_eq!(
                    6,
                    descriptor.depth(),
                    "Depth must be equal to 6 when the array descriptor is for a CUBEMAP."
                );
            }
        }

        if descriptor.num_channels() != 1
            && descriptor.num_channels() != 2
            && descriptor.num_channels() != 4
        {
            panic!(
                "NumChannels was set to {}. It must be 1, 2, or 4.",
                descriptor.num_channels()
            );
        }

        let mut handle = unsafe { std::mem::uninitialized() };
        unsafe { cuda_sys::cuda::cuArray3DCreate_v2(&mut handle, &descriptor.desc) }.to_result()?;
        Ok(Self { handle })
    }

    /// Gets the descriptor associated with this array.
    pub fn descriptor(&self) -> CudaResult<ArrayDescriptor> {
        let mut raw_descriptor = unsafe { std::mem::uninitialized() };
        unsafe { cuda_sys::cuda::cuArray3DGetDescriptor_v2(&mut raw_descriptor, self.handle) }
            .to_result()?;

        Ok(ArrayDescriptor::from_raw(raw_descriptor))
    }

    /// Try to destroy an `ArrayObject`. Can fail - if it does, returns the CUDA error and the
    /// un-destroyed array object
    pub fn drop(array: ArrayObject) -> DropResult<ArrayObject> {
        match unsafe { cuda_sys::cuda::cuArrayDestroy(array.handle) }.to_result() {
            Ok(()) => Ok(()),
            Err(e) => Err((e, array)),
        }
    }
}

impl std::fmt::Debug for ArrayObject {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.descriptor().fmt(f)
    }
}

impl Drop for ArrayObject {
    fn drop(&mut self) {
        unsafe { cuda_sys::cuda::cuArrayDestroy(self.handle) }
            .to_result()
            .expect("Failed to destroy CUDA Array")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn descriptor_round_trip() {
        let _context = crate::quick_init().unwrap();

        let obj = ArrayObject::from_descriptor(
            ArrayDescriptor::from_dims_format([1, 2, 3], ArrayFormat::Float).with_num_channels(2),
        )
        .unwrap();

        let descriptor = obj.descriptor().unwrap();
        assert_eq!([1, 2, 3], descriptor.dims());
        assert_eq!(ArrayFormat::Float, descriptor.format());
        assert_eq!(2, descriptor.num_channels());
        assert_eq!(ArrayObjectFlags::default(), descriptor.flags());
    }

    #[test]
    fn allow_1d_arrays() {
        let _context = crate::quick_init().unwrap();

        let obj = ArrayObject::from_descriptor(ArrayDescriptor::from_dims_format(
            [10, 0, 0],
            ArrayFormat::Float,
        ))
        .unwrap();

        let descriptor = obj.descriptor().unwrap();
        assert_eq!([10, 0, 0], descriptor.dims());
    }

    #[test]
    fn allow_2d_arrays() {
        let _context = crate::quick_init().unwrap();

        let obj = ArrayObject::from_descriptor(ArrayDescriptor::from_dims_format(
            [10, 20, 0],
            ArrayFormat::Float,
        ))
        .unwrap();

        let descriptor = obj.descriptor().unwrap();
        assert_eq!([10, 20, 0], descriptor.dims());
    }

    #[test]
    fn allow_1d_layered_arrays() {
        let _context = crate::quick_init().unwrap();

        let obj = ArrayObject::from_descriptor(
            ArrayDescriptor::from_dims_format([10, 0, 20], ArrayFormat::Float)
                .with_flags(ArrayObjectFlags::LAYERED),
        )
        .unwrap();

        let descriptor = obj.descriptor().unwrap();
        assert_eq!([10, 0, 20], descriptor.dims());
        assert_eq!(ArrayObjectFlags::LAYERED, descriptor.flags());
    }

    #[test]
    fn allow_cubemaps() {
        let _context = crate::quick_init().unwrap();

        let obj = ArrayObject::from_descriptor(
            ArrayDescriptor::from_dims_format([4, 4, 6], ArrayFormat::Float)
                .with_flags(ArrayObjectFlags::CUBEMAP),
        )
        .unwrap();

        let descriptor = obj.descriptor().unwrap();
        assert_eq!([4, 4, 6], descriptor.dims());
        assert_eq!(ArrayObjectFlags::CUBEMAP, descriptor.flags());
    }

    #[test]
    fn allow_layered_cubemaps() {
        let _context = crate::quick_init().unwrap();

        let obj = ArrayObject::from_descriptor(
            ArrayDescriptor::from_dims_format([4, 4, 24], ArrayFormat::Float)
                .with_flags(ArrayObjectFlags::new().mark_layered().mark_cubemap()),
        )
        .unwrap();

        let descriptor = obj.descriptor().unwrap();
        assert_eq!([4, 4, 24], descriptor.dims());
        assert_eq!(
            ArrayObjectFlags::CUBEMAP | ArrayObjectFlags::LAYERED,
            descriptor.flags()
        );
    }

    #[test]
    #[should_panic(expected = "\
assertion failed: `(left != right)`
  left: `0`,
 right: `0`: Cannot allocate an array with 0 Width")]
    fn fail_on_zero_size_widths() {
        let _context = crate::quick_init().unwrap();

        let _ = ArrayObject::from_descriptor(ArrayDescriptor::from_dims_format(
            [0, 10, 20],
            ArrayFormat::Float,
        ))
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "\
assertion failed: `(left == right)`
  left: `3`,
 right: `2`: Height and Width must be equal for CUBEMAP arrays.")]
    fn fail_cubemaps_with_unmatching_width_height() {
        let _context = crate::quick_init().unwrap();

        let _ = ArrayObject::from_descriptor(
            ArrayDescriptor::from_dims_format([2, 3, 6], ArrayFormat::Float)
                .with_flags(ArrayObjectFlags::CUBEMAP),
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "\
assertion failed: `(left == right)`
  left: `6`,
 right: `5`: Depth must be equal to 6 when the array descriptor is for a CUBEMAP.")]
    fn fail_cubemaps_with_non_six_depth() {
        let _context = crate::quick_init().unwrap();

        let _ = ArrayObject::from_descriptor(
            ArrayDescriptor::from_dims_format([4, 4, 5], ArrayFormat::Float)
                .with_flags(ArrayObjectFlags::CUBEMAP),
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "\
assertion failed: `(left == right)`
  left: `0`,
 right: `4`: Depth must be a multiple of 6 when the array descriptor is for a LAYERED CUBEMAP.")]
    fn fail_cubemaps_with_non_six_multiple_depth() {
        let _context = crate::quick_init().unwrap();

        let _ = ArrayObject::from_descriptor(
            ArrayDescriptor::from_dims_format([4, 4, 10], ArrayFormat::Float)
                .with_flags(ArrayObjectFlags::LAYERED | ArrayObjectFlags::CUBEMAP),
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "\
assertion failed: `(left != right)`
  left: `0`,
 right: `0`: If Depth is non-zero and the descriptor is not LAYERED, then Height must also be \
             non-zero.")]
    fn fail_with_depth_without_height() {
        let _context = crate::quick_init().unwrap();

        let _ = ArrayObject::from_descriptor(ArrayDescriptor::from_dims_format(
            [10, 0, 20],
            ArrayFormat::Float,
        ))
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "NumChannels was set to 3. It must be 1, 2, or 4.")]
    fn fails_on_invalid_num_channels() {
        let _context = crate::quick_init().unwrap();

        let _ = ArrayObject::from_descriptor(
            ArrayDescriptor::from_dims_format([1, 2, 3], ArrayFormat::Float).with_num_channels(3),
        )
        .unwrap();
    }
}
