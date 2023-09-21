[![](https://img.shields.io/crates/v/mkaudiolibrary.svg)](https://crates.io/crates/mkaudiolibrary)
[![](https://img.shields.io/crates/l/mkaudiolibrary.svg)](https://crates.io/crates/mkaudiolibrary)
[![](https://docs.rs/mkaudiolibrary/badge.svg)](https://docs.rs/mkaudiolibrary/)

# mkaudiolibrary
Modular audio processing library including MKAU plugin format based on Rust.

# Modules
buffer : includes push buffer and circular buffer.

simulation : includes convolution and saturation struct and function for audio processing.

processor : includes MKAU plugin format.

# Version
0.1.14 - Added from_raw function for Buffers.

0.1.13 - Buffers return LayoutError when error occured allocating buffer, added resize, into_slice, and into_slice_mut functions for Buffers.

0.1.12 - Changen I/O type of methonds of simulation and Processor trait into Buffer.

0.1.11 - Added Buffer for simple format of audio buffer. Inline-abled processing functions.

0.1.10 - Used boxed slice for Saturation for block processing. Always inlined processing functions.

0.1.9 - Used boxed slice instead of CircularBuffer for Processor.

0.1.8 - Used boxed slice instead of CircularBuffer for Convolution.

0.1.7 - Create Convolution struct. Dropped next and state reference for processor and convolution.

0.1.6 - Used raw pointer for buffers instead of Box<T>, and implied Drop trait. Minor fix to functions.

0.1.5 - Minor fix.

0.1.4 - Omitted unnecessary multithreading and optional for better performance.

0.1.1 - 0.1.3 - Documentation update.

0.1.0 - Initial version.

# License
The library is offered under GPLv3.0 license for open source usage.

If you want to use mkaudiolibrary for closed source project, please email to minjaekim@mkaudio.company for agreement and support.