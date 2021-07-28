# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

[0.1.3] - July 28, 2021
### Added
- `Device::uuid` function, which returns the UUID of a device.

### Fixed
- Upgraded `cuda-sys` 0.2 to `cuda-driver-sys` 0.3.

## [0.1.2] - February 29, 2019
### Fixed
- Loosen restrictions on traits implemented by UnifiedPointer & DevicePointer
- Fixed a bug where RustaCUDA allocated more device memory than necessary.
- Use `MaybeUninit` internally for uninitialized data.

## [0.1.1] - May 16, 2019
### Added
- `Stream::add_callback` function, which adds a host-side callback into a stream's queue
- Added basic support for allocating CUDA arrays.
- Add support for CUDA Events.
- Add unsafe interface for asynchronous data copies.

### Fixed
- Fixed compile error on PPC64 architecture.
- Fix bug where NUL bytes could be included in the device name string.

## [0.1.0] - December 1, 2018
- Initial Release


[Unreleased]: https://github.com/bheisler/RustaCUDA/compare/0.1.2...HEAD
[0.1.0]: https://github.com/bheisler/RustaCUDA/compare/5e6d7bd...0.1.0
[0.1.1]: https://github.com/bheisler/RustaCUDA/compare/0.1.0...0.1.1
[0.1.2]: https://github.com/bheisler/RustaCUDA/compare/0.1.1...0.1.2
[0.1.3]: https://github.com/bheisler/RustaCUDA/compare/0.1.2...0.1.3
