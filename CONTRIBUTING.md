# Contributing to RustaCUDA

## Ideas, Experiences and Questions

The easiest way to contribute to RustaCUDA is to use it and report your experiences, ask questions and contribute ideas. We'd love to hear your thoughts on how to make RustaCUDA better, or your comments on why you are or are not currently using it.

Issues, ideas, requests and questions should be posted on the issue tracker at:

https://github.com/bheisler/RustaCUDA/issues

## Code

Pull requests are welcome, though please raise an issue or post a comment for discussion first. We're happy to assist new contributors.

If you're not sure what to work on, try checking the [Beginner label](https://github.com/bheisler/RustaCUDA/labels/Beginner)

To make changes to the code, fork the repo and clone it:

`git clone git@github.com:your-username/RustaCUDA.git`

Then make your changes to the code. When you're done, run the tests:

```
cargo test --all
```

It's a good idea to run clippy and fix any warnings as well:

```
rustup component add clippy-preview
cargo clippy --all
```

Finally, run Rustfmt to maintain a common code style:

```
rustup component add rustfmt-preview
cargo fmt
```

Don't forget to update the CHANGELOG.md file and any appropriate documentation. Once you're finished, push to your fork and submit a pull request. We try to respond to new issues and pull requests quickly, so if there hasn't been any response for more than a few days feel free to ping @bheisler.

Some things that will increase the chance that your pull request is accepted:

* Write tests
* Clearly document public methods, with examples if possible
* Write a good commit message

Good documentation is one of the core goals of the RustaCUDA project, so new code in pull requests should have clear and complete documentation.

## Organization

The module structure and code layout of RustaCUDA loosely mirrors the modules described
by the [official CUDA Driver API documentation](https://docs.nvidia.com/cuda/archive/8.0/cuda-driver-api/).
Particularly complex modules like the memory management section are broken into multiple private
files which are re-exported as a single module, while especially simple modules like initialization
and version management are combined into the crate root. If you're implementing a new module,
use your judgement as to where it belongs.

The custom-derive macro for `DeviceCopy` is defined in `rustacuda_derive`.

## Github Labels

RustaCUDA uses a simple set of labels to track issues. Most important are the difficulty labels:

- Beginner - Suitable for people new to RustaCUDA or CUDA in general
- Intermediate - More challenging, likely involves some non-obvious design decisions or knowledge of CUDA
- Bigger Project - Large and/or complex project such as designing a safe, Rusty wrapper around a complex part of the CUDA API

Additionally, there are a few other noteworthy labels:

- Breaking Change - Fixing this will have to wait until the next breaking-change release
- New CUDA Feature - Issues for exposing more features of the CUDA driver API through RustaCUDA
- Enhancement - Enhancements to existing functionality or documentation
- Help Wanted - Input and ideas requested

## Code of Conduct

We follow the [Rust Code of Conduct](http://www.rust-lang.org/conduct.html).
