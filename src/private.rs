//! This module contains private sealed traits that should not be used or implemented outside of
//! RustaCUDA. These traits are public because they are used as bounds in certain functions.
//! These traits may change in any way at any time with no warning, and this will not be considered
//! a breaking change.

use memory::{DeviceBox, DeviceBuffer, DeviceCopy, DevicePointer, UnifiedPointer};

pub trait Sealed {}

impl<T: DeviceCopy> Sealed for DevicePointer<T> {}
impl<T: DeviceCopy> Sealed for UnifiedPointer<T> {}
impl<T: DeviceCopy> Sealed for DeviceBox<T> {}
impl<T: DeviceCopy> Sealed for DeviceBuffer<T> {}
