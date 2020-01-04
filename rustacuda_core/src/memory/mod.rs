mod pointer;
pub use self::pointer::*;

use core::marker::PhantomData;
use core::num::*;

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
/// #[macro_use]
/// extern crate rustacuda;
///
/// #[derive(Clone, DeviceCopy)]
/// struct MyStruct(u64);
/// ```
///
/// This is safe because the `DeviceCopy` derive macro will check that all fields of the struct,
/// enum or union implement `DeviceCopy`. For example, this fails to compile, because `Vec` cannot
/// be copied to the device:
///
/// ```compile_fail
/// # #[macro_use]
/// # extern crate rustacuda;
/// #[derive(Clone, DeviceCopy)]
/// struct MyStruct(Vec<u64>);
/// ```
///
/// You can also implement `DeviceCopy` unsafely:
///
/// ```
/// use rustacuda::memory::DeviceCopy;
///
/// #[derive(Clone)]
/// struct MyStruct(u64);
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
pub unsafe trait DeviceCopy {
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

impl_device_copy_array! {
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
{
}
unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy, D: DeviceCopy, E: DeviceCopy> DeviceCopy
    for (A, B, C, D, E)
{
}
unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy, D: DeviceCopy, E: DeviceCopy, F: DeviceCopy>
    DeviceCopy for (A, B, C, D, E, F)
{
}
unsafe impl<
        A: DeviceCopy,
        B: DeviceCopy,
        C: DeviceCopy,
        D: DeviceCopy,
        E: DeviceCopy,
        F: DeviceCopy,
        G: DeviceCopy,
    > DeviceCopy for (A, B, C, D, E, F, G)
{
}
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
{
}
