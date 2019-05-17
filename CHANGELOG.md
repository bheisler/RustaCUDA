# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
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


[Unreleased]: https://github.com/bheisler/RustaCUDA/compare/0.1.0...HEAD
[0.1.0]: https://github.com/bheisler/RustaCUDA/compare/5e6d7bd...0.1.0
