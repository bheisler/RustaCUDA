//! Routines for allocating and using CUDA Array Objects.
//!
//! Detailed documentation about allocating CUDA Arrays can be found in the
//! [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7)

use std::mem::MaybeUninit;
use std::os::raw::c_uint;

use cuda_sys::cuda::{CUarray, CUarray_format, CUarray_format_enum};

use crate::context::CurrentContext;
use crate::device::DeviceAttribute;
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
        /// Enables creation of layered CUDA arrays. When this flag is set, depth specifies the
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

    /// Constructs an ArrayDescriptor from dimensions, format, num_channels, and flags.
    pub fn new(
        dims: [usize; 3],
        format: ArrayFormat,
        num_channels: c_uint,
        flags: ArrayObjectFlags,
    ) -> Self {
        Self {
            desc: cuda_sys::cuda::CUDA_ARRAY3D_DESCRIPTOR {
                Width: dims[0],
                Height: dims[1],
                Depth: dims[2],
                Format: format.to_raw(),
                NumChannels: num_channels,
                Flags: flags.bits(),
            },
        }
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

    /// Returns the width of the ArrayDescripor
    pub fn width(&self) -> usize {
        self.desc.Width
    }

    /// Sets the width of the ArrayDescriptor
    pub fn set_width(&mut self, width: usize) {
        self.desc.Width = width;
    }

    /// Returns the height of the ArrayDescripor
    pub fn height(&self) -> usize {
        self.desc.Height
    }

    /// Sets the height of the ArrayDescriptor
    pub fn set_height(&mut self, height: usize) {
        self.desc.Height = height;
    }

    /// Returns the depth of the ArrayDescripor
    pub fn depth(&self) -> usize {
        self.desc.Depth
    }

    /// Sets the depth of the ArrayDescriptor
    pub fn set_depth(&mut self, depth: usize) {
        self.desc.Depth = depth;
    }

    /// Returns the format of the ArrayDescripor
    pub fn format(&self) -> ArrayFormat {
        ArrayFormat::from_raw(self.desc.Format)
    }

    /// Sets the format of the ArrayDescriptor
    pub fn set_format(&mut self, format: ArrayFormat) {
        self.desc.Format = format.to_raw();
    }

    /// Returns the number of channels in the ArrayDescriptor
    pub fn num_channels(&self) -> c_uint {
        self.desc.NumChannels
    }

    /// Sets the number of channels in the ArrayDescriptor
    pub fn set_num_channels(&mut self, num_channels: c_uint) {
        self.desc.NumChannels = num_channels;
    }

    /// Returns the flags of the ArrayDescriptor
    pub fn flags(&self) -> ArrayObjectFlags {
        ArrayObjectFlags::from_bits_truncate(self.desc.Flags)
    }

    /// Sets the flags of the ArrayDescriptor.
    pub fn set_flags(&mut self, flags: ArrayObjectFlags) {
        self.desc.Flags = flags.bits();
    }
}

/// A CUDA Array. Can be bound to a texture or surface.
pub struct ArrayObject {
    handle: CUarray,
}

impl ArrayObject {
    /// Constructs a generic ArrayObject from an `ArrayDescriptor`.
    pub fn from_descriptor(descriptor: &ArrayDescriptor) -> CudaResult<Self> {
        // We validate the descriptor up front in debug mode. This provides a good error message to
        // the user when they get something wrong, but doesn't re-validate in release mode.
        if cfg!(debug_assertions) {
            assert_ne!(
                0,
                descriptor.width(),
                "Cannot allocate an array with 0 Width"
            );

            if !descriptor.flags().contains(ArrayObjectFlags::LAYERED) && descriptor.depth() > 0 {
                assert_ne!(
                    0,
                    descriptor.height(),
                    "If Depth is non-zero and the descriptor is not LAYERED, then Height must also \
                    be non-zero."
                );
            }

            if descriptor.flags().contains(ArrayObjectFlags::CUBEMAP) {
                assert_eq!(
                    descriptor.height(),
                    descriptor.width(),
                    "Height and Width must be equal for CUBEMAP arrays."
                );

                if descriptor.flags().contains(ArrayObjectFlags::LAYERED) {
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

            assert!(
                descriptor.num_channels() == 1
                    || descriptor.num_channels() == 2
                    || descriptor.num_channels() == 4,
                "NumChannels was set to {}. It must be 1, 2, or 4.",
                descriptor.num_channels()
            );

            // Exhaustively check bounds of arrays
            let device = CurrentContext::get_device()?;

            let attr = |attr| Ok(1..=(device.get_attribute(attr)? as usize));

            let (description, bounds) = if descriptor.flags().contains(ArrayObjectFlags::CUBEMAP) {
                if descriptor.flags().contains(ArrayObjectFlags::LAYERED) {
                    (
                        "Layered Cubemap",
                        vec![[
                            attr(DeviceAttribute::MaximumTextureCubemapLayeredWidth)?,
                            attr(DeviceAttribute::MaximumTextureCubemapLayeredWidth)?,
                            attr(DeviceAttribute::MaximumTextureCubemapLayeredLayers)?,
                        ]],
                    )
                } else {
                    (
                        "Cubemap",
                        vec![[
                            attr(DeviceAttribute::MaximumTextureCubemapWidth)?,
                            attr(DeviceAttribute::MaximumTextureCubemapWidth)?,
                            6..=6,
                        ]],
                    )
                }
            } else if descriptor.flags().contains(ArrayObjectFlags::LAYERED) {
                if descriptor.height() > 0 {
                    (
                        "2D Layered",
                        vec![[
                            attr(DeviceAttribute::MaximumTexture2DLayeredWidth)?,
                            attr(DeviceAttribute::MaximumTexture2DLayeredHeight)?,
                            attr(DeviceAttribute::MaximumTexture2DLayeredLayers)?,
                        ]],
                    )
                } else {
                    (
                        "1D Layered",
                        vec![[
                            attr(DeviceAttribute::MaximumTexture1DLayeredWidth)?,
                            0..=0,
                            attr(DeviceAttribute::MaximumTexture1DLayeredLayers)?,
                        ]],
                    )
                }
            } else if descriptor.depth() > 0 {
                (
                    "3D",
                    vec![
                        [
                            attr(DeviceAttribute::MaximumTexture3DWidth)?,
                            attr(DeviceAttribute::MaximumTexture3DHeight)?,
                            attr(DeviceAttribute::MaximumTexture3DDepth)?,
                        ],
                        [
                            attr(DeviceAttribute::MaximumTexture3DWidthAlternate)?,
                            attr(DeviceAttribute::MaximumTexture3DHeightAlternate)?,
                            attr(DeviceAttribute::MaximumTexture3DDepthAlternate)?,
                        ],
                    ],
                )
            } else if descriptor.height() > 0 {
                if descriptor
                    .flags()
                    .contains(ArrayObjectFlags::TEXTURE_GATHER)
                {
                    (
                        "2D Texture Gather",
                        vec![[
                            attr(DeviceAttribute::MaximumTexture2DGatherWidth)?,
                            attr(DeviceAttribute::MaximumTexture2DGatherHeight)?,
                            0..=0,
                        ]],
                    )
                } else {
                    (
                        "2D",
                        vec![[
                            attr(DeviceAttribute::MaximumTexture2DWidth)?,
                            attr(DeviceAttribute::MaximumTexture2DHeight)?,
                            0..=0,
                        ]],
                    )
                }
            } else {
                assert!(descriptor.width() > 0);
                (
                    "1D",
                    vec![[attr(DeviceAttribute::MaximumTexture1DWidth)?, 0..=0, 0..=0]],
                )
            };

            let bounds_invalid = |x: &[::std::ops::RangeInclusive<usize>; 3]| {
                (descriptor.width() >= *x[0].start() && descriptor.width() <= *x[0].end())
                    && (descriptor.height() >= *x[1].start() && descriptor.height() <= *x[1].end())
                    && (descriptor.depth() >= *x[2].start() && descriptor.depth() <= *x[2].end())
            };

            if !bounds.iter().any(bounds_invalid) {
                panic!(
                    "The dimensions of the {} ArrayObject did not fall within the valid bounds for \
                     the array. descriptor = {:?}, dims = {:?}, valid bounds = {:?}",
                     description,
                     descriptor,
                     [descriptor.width(), descriptor.height(), descriptor.depth()],
                     bounds
                );
            }
        }

        let mut handle = MaybeUninit::uninit();
        unsafe { cuda_sys::cuda::cuArray3DCreate_v2(handle.as_mut_ptr(), &descriptor.desc) }
            .to_result()?;
        Ok(Self {
            handle: unsafe { handle.assume_init() },
        })
    }

    /// Allocates a new CUDA Array that is up to 3-dimensions.
    ///
    /// `dims` contains the extents of the array. `dims[0]` must be non-zero. `dims[1]` must be
    /// non-zero if `dims[2]` is non-zero. The rank of the array is equal to the number of non-zero
    /// `dims`.
    ///
    /// `format` determines the data-type of the array.
    ///
    /// `num_channels` determines the number of channels per array element (1, 2, or 4).
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use rustacuda::memory::array::{ArrayObject, ArrayFormat};
    ///
    /// let one_dim_array = ArrayObject::new([10, 0, 0], ArrayFormat::Float, 1)?;
    /// let two_dim_array = ArrayObject::new([10, 12, 0], ArrayFormat::Float, 1)?;
    /// let three_dim_array = ArrayObject::new([10, 12, 14], ArrayFormat::Float, 1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(dims: [usize; 3], format: ArrayFormat, num_channels: c_uint) -> CudaResult<Self> {
        Self::from_descriptor(&ArrayDescriptor::new(
            dims,
            format,
            num_channels,
            Default::default(),
        ))
    }

    /// Allocates a new 1D CUDA Array.
    ///
    /// `width` must be non-zero.
    ///
    /// `format` determines the data-type of the array.
    ///
    /// `num_channels` determines the number of channels per array element (1, 2, or 4).
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use rustacuda::memory::array::{ArrayObject, ArrayFormat};
    ///
    /// // Allocates a 1D array of 10 single-precision, single-channel floating point values.
    /// let one_dim_array = ArrayObject::new_1d(10, ArrayFormat::Float, 1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_1d(width: usize, format: ArrayFormat, num_channels: c_uint) -> CudaResult<Self> {
        Self::from_descriptor(&ArrayDescriptor::new(
            [width, 0, 0],
            format,
            num_channels,
            Default::default(),
        ))
    }

    /// Allocates a new CUDA Array that is up to 2-dimensions.
    ///
    /// `dims` contains the extents of the array. `dims[0]` must be non-zero. The rank of the array
    /// is equal to the number of non-zero `dims`.
    ///
    /// `format` determines the data-type of the array.
    ///
    /// `num_channels` determines the number of channels per array element (1, 2, or 4).
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use rustacuda::memory::array::{ArrayObject, ArrayFormat};
    ///
    /// // Allocates an 8x24 array of single-precision, single-channel floating point values.
    /// let one_dim_array = ArrayObject::new_2d([8, 24], ArrayFormat::Float, 1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_2d(dims: [usize; 2], format: ArrayFormat, num_channels: c_uint) -> CudaResult<Self> {
        Self::from_descriptor(&ArrayDescriptor::new(
            [dims[0], dims[1], 0],
            format,
            num_channels,
            Default::default(),
        ))
    }

    /// Creates a new Layered 1D or 2D CUDA Array.
    ///
    /// `dims` contains the extents of the array. `dims[0]` must be non-zero. The rank of the array
    /// is equivalent to the number of non-zero dimensions.
    ///
    /// `num_layers` determines the number of layers in the array.
    ///
    /// `format` determines the data-type of the array.
    ///
    /// `num_channels` determines the number of channels per array element (1, 2, or 4).
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use rustacuda::memory::array::{ArrayObject, ArrayFormat};
    ///
    /// // Allocates a 7x8 array with 10 layers of single-precision, single-channel floating
    /// // point values.
    /// let layered_array = ArrayObject::new_layered([7, 8], 10, ArrayFormat::Float, 1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_layered(
        dims: [usize; 2],
        num_layers: usize,
        format: ArrayFormat,
        num_channels: c_uint,
    ) -> CudaResult<Self> {
        Self::from_descriptor(&ArrayDescriptor::new(
            [dims[0], dims[1], num_layers],
            format,
            num_channels,
            ArrayObjectFlags::LAYERED,
        ))
    }

    /// Creates a new Layered 1D CUDA Array.
    ///
    /// `width` must be non-zero.
    ///
    /// `num_layers` determines the number of layers in the array.
    ///
    /// `format` determines the data-type of the array.
    ///
    /// `num_channels` determines the number of channels per array element (1, 2, or 4).
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use rustacuda::memory::array::{ArrayObject, ArrayFormat};
    ///
    /// // Allocates a 5-element array with 10 layers of single-precision, single-channel floating
    /// // point values.
    /// let layered_array = ArrayObject::new_layered_1d(5, 10, ArrayFormat::Float, 1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_layered_1d(
        width: usize,
        num_layers: usize,
        format: ArrayFormat,
        num_channels: c_uint,
    ) -> CudaResult<Self> {
        Self::from_descriptor(&ArrayDescriptor::new(
            [width, 0, num_layers],
            format,
            num_channels,
            ArrayObjectFlags::LAYERED,
        ))
    }

    /// Creates a new Cubemap CUDA Array. The array is represented as 6 side x side 2D arrays.
    ///
    /// `side` is the length of an edge of the cube.
    ///
    /// `format` determines the data-type of the array.
    ///
    /// `num_channels` determines the number of channels per array element (1, 2, or 4).
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use rustacuda::memory::array::{ArrayObject, ArrayFormat};
    ///
    /// // Allocates an 8x8 Cubemap array of single-precision, single-channel floating point
    /// // numbers.
    /// let layered_array = ArrayObject::new_cubemap(8, ArrayFormat::Float, 1)?;
    ///
    /// // All non-layered cubemap arrays have a depth of 6.
    /// assert_eq!(6, layered_array.descriptor()?.depth());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_cubemap(side: usize, format: ArrayFormat, num_channels: c_uint) -> CudaResult<Self> {
        Self::from_descriptor(&ArrayDescriptor::new(
            [side, side, 6],
            format,
            num_channels,
            ArrayObjectFlags::CUBEMAP,
        ))
    }

    /// Creates a new Layered Cubemap CUDA Array. The array is represented as multiple 6 side x side
    /// 2D arrays.
    ///
    /// `side` is the length of an edge of the cube.
    ///
    /// `num_layers` is the number of cubemaps in the array. The actual "depth" of the array is
    /// `num_layers * 6`.
    ///
    /// `format` determines the data-type of the array.
    ///
    /// `num_channels` determines the number of channels per array element (1, 2, or 4).
    ///
    /// ```
    /// # use rustacuda::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// # let _ctx = quick_init()?;
    /// use rustacuda::memory::array::{ArrayObject, ArrayFormat};
    ///
    /// // Allocates an 8x8 Layered Cubemap array of single-precision, single-channel floating point
    /// // values with 5 layers.
    /// let layered_array = ArrayObject::new_layered_cubemap(8, 5, ArrayFormat::Float, 1)?;
    ///
    /// // The depth of a layered cubemap array is equal to the number of layers * 6.
    /// assert_eq!(30, layered_array.descriptor()?.depth());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_layered_cubemap(
        side: usize,
        num_layers: usize,
        format: ArrayFormat,
        num_channels: c_uint,
    ) -> CudaResult<Self> {
        Self::from_descriptor(&ArrayDescriptor::new(
            [side, side, num_layers * 6],
            format,
            num_channels,
            ArrayObjectFlags::CUBEMAP | ArrayObjectFlags::LAYERED,
        ))
    }

    /// Gets the descriptor associated with this array.
    pub fn descriptor(&self) -> CudaResult<ArrayDescriptor> {
        // Use "zeroed" incase CUDA_ARRAY3D_DESCRIPTOR has uninitialized padding
        let mut raw_descriptor = MaybeUninit::zeroed();
        unsafe {
            cuda_sys::cuda::cuArray3DGetDescriptor_v2(raw_descriptor.as_mut_ptr(), self.handle)
        }
        .to_result()?;

        Ok(ArrayDescriptor::from_raw(unsafe {
            raw_descriptor.assume_init()
        }))
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

        let obj = ArrayObject::new([1, 2, 3], ArrayFormat::Float, 2).unwrap();

        let descriptor = obj.descriptor().unwrap();
        assert_eq!([1, 2, 3], descriptor.dims());
        assert_eq!(ArrayFormat::Float, descriptor.format());
        assert_eq!(2, descriptor.num_channels());
        assert_eq!(ArrayObjectFlags::default(), descriptor.flags());
    }

    #[test]
    fn allow_1d_arrays() {
        let _context = crate::quick_init().unwrap();

        let obj = ArrayObject::new([10, 0, 0], ArrayFormat::Float, 1).unwrap();

        let descriptor = obj.descriptor().unwrap();
        assert_eq!([10, 0, 0], descriptor.dims());
    }

    #[test]
    fn allow_2d_arrays() {
        let _context = crate::quick_init().unwrap();

        let obj = ArrayObject::new([10, 20, 0], ArrayFormat::Float, 1).unwrap();

        let descriptor = obj.descriptor().unwrap();
        assert_eq!([10, 20, 0], descriptor.dims());
    }

    #[test]
    fn allow_1d_layered_arrays() {
        let _context = crate::quick_init().unwrap();

        let obj = ArrayObject::new_layered([10, 0], 20, ArrayFormat::Float, 1).unwrap();

        let descriptor = obj.descriptor().unwrap();
        assert_eq!([10, 0, 20], descriptor.dims());
        assert_eq!(ArrayObjectFlags::LAYERED, descriptor.flags());
    }

    #[test]
    fn allow_cubemaps() {
        let _context = crate::quick_init().unwrap();

        let obj = ArrayObject::new_cubemap(4, ArrayFormat::Float, 1).unwrap();

        let descriptor = obj.descriptor().unwrap();
        assert_eq!([4, 4, 6], descriptor.dims());
        assert_eq!(ArrayObjectFlags::CUBEMAP, descriptor.flags());
    }

    #[test]
    fn allow_layered_cubemaps() {
        let _context = crate::quick_init().unwrap();

        let obj = ArrayObject::new_layered_cubemap(4, 4, ArrayFormat::Float, 1).unwrap();

        let descriptor = obj.descriptor().unwrap();
        assert_eq!([4, 4, 24], descriptor.dims());
        assert_eq!(
            ArrayObjectFlags::CUBEMAP | ArrayObjectFlags::LAYERED,
            descriptor.flags()
        );
    }

    #[test]
    #[should_panic]
    fn fail_on_zero_width_1d_array() {
        let _context = crate::quick_init().unwrap();

        let _ = ArrayObject::new_1d(0, ArrayFormat::Float, 1).unwrap();
    }

    #[test]
    #[should_panic]
    fn fail_on_zero_size_widths() {
        let _context = crate::quick_init().unwrap();

        let _ = ArrayObject::new([0, 10, 20], ArrayFormat::Float, 1).unwrap();
    }

    #[test]
    #[should_panic]
    fn fail_cubemaps_with_unmatching_width_height() {
        let _context = crate::quick_init().unwrap();

        let mut descriptor = ArrayDescriptor::from_dims_format([2, 3, 6], ArrayFormat::Float);
        descriptor.set_flags(ArrayObjectFlags::CUBEMAP);

        let _ = ArrayObject::from_descriptor(&descriptor).unwrap();
    }

    #[test]
    #[should_panic]
    fn fail_cubemaps_with_non_six_depth() {
        let _context = crate::quick_init().unwrap();

        let mut descriptor = ArrayDescriptor::from_dims_format([4, 4, 5], ArrayFormat::Float);
        descriptor.set_flags(ArrayObjectFlags::CUBEMAP);

        let _ = ArrayObject::from_descriptor(&descriptor).unwrap();
    }

    #[test]
    #[should_panic]
    fn fail_cubemaps_with_non_six_multiple_depth() {
        let _context = crate::quick_init().unwrap();

        let mut descriptor = ArrayDescriptor::from_dims_format([4, 4, 10], ArrayFormat::Float);
        descriptor.set_flags(ArrayObjectFlags::LAYERED | ArrayObjectFlags::CUBEMAP);

        let _ = ArrayObject::from_descriptor(&descriptor).unwrap();
    }

    #[test]
    #[should_panic]
    fn fail_with_depth_without_height() {
        let _context = crate::quick_init().unwrap();

        let _ = ArrayObject::new([10, 0, 20], ArrayFormat::Float, 1).unwrap();
    }

    #[test]
    #[should_panic]
    fn fails_on_invalid_num_channels() {
        let _context = crate::quick_init().unwrap();

        let _ = ArrayObject::new([1, 2, 3], ArrayFormat::Float, 3).unwrap();
    }
}
