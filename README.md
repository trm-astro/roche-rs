[![PyPi version](https://badgen.net/pypi/v/roche/)](https://pypi.org/project/roche)
[![crates.io](https://badgen.net/crates/v/roche-rs)](https://crates.io/crates/roche-rs)

roche-rs is a translation of Tom Marsh's C++ [cpp-roche](https://github.com/trmrsh/cpp-roche) package for modelling Roche-distorted binary systems. It also has a few useful functions and types from [cpp-subs](https://github.com/trmrsh/cpp-subs) such as Vec3 and Point.


### Rust
The latest version of roche-rs can be viewed at [roche-rs](https://crates.io/crates/roche-rs) and can be added to a rust project with

```
cargo add roche-rs
```

### Python
This package has also been wrapped for python to replicate [trm-roche](https://github.com/trmrsh/trm-roche/tree/master) and can be installed with pip from PyPI

```
pip install roche
```

This is very much a first go at it. Most functions have been tested against their trm.roche counterparts however not all functions have been tested as of yet and some bugs may remain. Please add an issue on the Github repo if you find any.

Some functions from cpp-roche/trm-roche have not been translated yet so this is still a work-in-progress. Please add an issue on the Github repo if there's a specific function from cpp-roche/trm-roche that you'd like adding.

Functions still to add for Python are:
* astream
* pvstream
* qirbs

Functions still to add for Rust are:
* third_contact
* fourth_contact
* eprob
* flobe1
* hits
* irrad
* rdot
* rlobe_eggleton