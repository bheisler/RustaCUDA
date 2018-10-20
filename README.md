<h1 align="center">RustaCuda</h1>

<div align="center">High-level Interface to [NVIDIA® CUDA™](https://developer.nvidia.com/cuda-zone) in Rust</div>

RustaCuda helps you bring GPU-acceleration to your projects by providing a high-level, easy-to-use
interface to the CUDA GPU computing toolkit. Bring the power and safety of Rust to a whole new
level of performance!

## Table of Contents
- [Table of Contents](#table-of-contents)
    - [Goals](#goals)
    - [Quickstart](#quickstart)
    - [Contributing](#contributing)
    - [Maintenance](#maintenance)
    - [License](#license)
    - [Requirements](#requirements)
    - [Related Projects](#related-projects)

### Goals

 The primary design goals are:

 - __High-Level__: Using RustaCuda should feel familiar and intuitive for Rust programmers
 - __Easy-to-Use__: RustaCuda should be well-documented and well-designed enough to help novice GPU programmers get started, while not limiting more experienced folks too much.
 - __Safe__: Many aspects of GPU-accelerated computing are difficult to reconcile with Rust's safety guarantees, but RustaCuda should provide the safest interface that is reasonably practical.
 - __Fast__: RustaCuda should aim to be as fast as possible, where it doesn't conflict with the other goals.

RustaCuda is intended to provide a programmer-friendly library for working with the host-side CUDA
API. It is not intended to assist in compiling Rust code to CUDA kernels (though see
[rust-ptx-builder](https://github.com/denzp/rust-ptx-builder) for that) or to provide device-side
utilities to be used within the kernels themselves (though I plan to build a device-side helper
library later, if nobody else gets there first).

RustaCuda is deliberately agnostic about how the kernels work or how they were compiled. This makes
it possible to (for example) use C kernels compiled with `nvcc`.

### Quickstart

TODO: Write this when I have something working

### Contributing

Thanks for your interest! Right now RustaCuda is still under initial development, and is not
accepting contributions at this time. Rest assured that contributions will be welcome once there
is something more substantial to contribute to.

### Maintenance

RustaCuda is currently under initial development by Brook Heisler (@bheisler)

### License

RustaCuda is dual-licensed under the Apache 2.0 license and the MIT license.

### Requirements

RustaCuda requires at least CUDA version 8 to be installed.

### Related Projects

- [accel](https://github.com/rust-accel/accel) is a full CUDA computing framework. Thanks to accel for creating and maintaining the `cuda-sys` FFI wrapper library.
- [rust-ptx-builder](https://github.com/denzp/rust-ptx-builder) is a `build.rs` helper library which makes it easy to compile Rust crates into CUDA kernels.