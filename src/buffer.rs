//! Plain (non-atomic) audio sample containers.
//!
//! This module provides three buffer types used across the crate's audio
//! path:
//!
//! - `Buffer` - a resizable, owned block of samples (thin wrapper over `Box<[T]>`)
//! - `PushBuffer` - a sliding-window FIFO for convolution/FIR filtering
//! - `CircularBuffer` - a power-of-2 ring buffer for delay lines
//!
//! ## No internal locking
//!
//! None of these types use `Arc`/`RwLock`/`Mutex` internally - they're
//! plain owned containers accessed through normal `&self`/`&mut self`
//! borrowing, same as `Vec<T>`. Real-time audio processing in this crate is
//! single-owner per processing graph (a `&mut Processor`/`&mut dsp::*`
//! call chain), so there was never a genuine multi-writer scenario for
//! these containers to arbitrate; the audio thread owns its buffers
//! outright. If you need to share a buffer across threads (e.g. an audio
//! thread feeding a UI meter), wrap it yourself with whatever
//! synchronization fits that specific case (`Arc<Mutex<_>>`, a lock-free
//! ring buffer crate, etc.) rather than paying for locking on every sample
//! when nothing needs it.
//!
//! ## Example: FIR history with PushBuffer
//!
//! ```ignore
//! use mkaudiolibrary::buffer::PushBuffer;
//!
//! let mut history = PushBuffer::<f32>::new(4);
//! for x in [1.0, 2.0, 3.0] {
//!     history.push(x);
//! }
//! assert_eq!(&*history, &[0.0, 1.0, 2.0, 3.0]);
//! ```
//!
//! ## Example: Delay Line with CircularBuffer
//!
//! ```ignore
//! use mkaudiolibrary::buffer::CircularBuffer;
//!
//! let mut delay = CircularBuffer::<f32>::new(1024); // rounds up to a power of 2
//!
//! // Write samples
//! for i in 0..100 {
//!     delay.push(i as f32);
//! }
//!
//! // Read delayed samples
//! for _ in 0..100 {
//!     let sample = delay.next();
//! }
//! ```

// ==========================================
// Buffer - General Purpose
// ==========================================

/// General-purpose owned audio buffer.
///
/// A thin, resizable wrapper over `Box<[T]>` for cache-friendly contiguous
/// storage. Implements `Deref`/`DerefMut<Target = [T]>` for direct slice
/// access - no guard types or locking involved.
#[derive(Clone)]
pub struct Buffer<T: Clone + Default> {
    data: Box<[T]>,
}

impl<T: Clone + Default> Buffer<T> {
    /// Create a new buffer with the given length, initialized to default values.
    pub fn new(len: usize) -> Self {
        Self {
            data: vec![T::default(); len].into_boxed_slice(),
        }
    }

    /// Create a buffer from an existing slice (copies data).
    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            data: slice.to_vec().into_boxed_slice(),
        }
    }

    /// Resize the buffer, resetting all elements to their default value.
    pub fn resize(&mut self, len: usize) {
        self.data = vec![T::default(); len].into_boxed_slice();
    }

    /// Get the length of the buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrow the buffer contents as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutably borrow the buffer contents as a slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Clone + Default> std::ops::Deref for Buffer<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T: Clone + Default> std::ops::DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T: Clone + Default> Default for Buffer<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

// ==========================================
// PushBuffer - FIFO for Convolution
// ==========================================

/// FIFO sliding-window buffer for convolution and FIR filtering.
///
/// Maintains a sliding window of the most recent `len` pushed samples,
/// oldest first. Ideal for real-time FIR filter implementations where you
/// need contiguous access to the last N samples for a dot product.
///
/// # Push Behavior
/// - Until the buffer is full: samples are appended at the current index
/// - When full: samples shift left, newest sample goes at the end (FIFO)
#[derive(Clone)]
pub struct PushBuffer<T: Copy + Default> {
    buffer: Box<[T]>,
    index: usize,
}

impl<T: Copy + Default> PushBuffer<T> {
    /// Create a new push buffer with the given length.
    ///
    /// Buffer is initialized with default values and the write index at 0.
    pub fn new(len: usize) -> Self {
        Self {
            buffer: vec![T::default(); len].into_boxed_slice(),
            index: 0,
        }
    }

    /// Create a push buffer from an existing slice.
    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            buffer: slice.to_vec().into_boxed_slice(),
            index: 0,
        }
    }

    /// Resize the buffer, clearing all data and resetting the write index.
    pub fn resize(&mut self, len: usize) {
        self.buffer = vec![T::default(); len].into_boxed_slice();
        self.index = 0;
    }

    /// Push a new sample into the buffer.
    ///
    /// If the buffer is not yet full, appends at the current index.
    /// If the buffer is full, shifts all samples left and places the new
    /// sample at the end (FIFO behavior).
    #[inline]
    pub fn push(&mut self, value: T) {
        let len = self.buffer.len();
        if len == 0 {
            return;
        }

        if self.index < len {
            let idx = self.index;
            self.buffer[idx] = value;
            self.index += 1;
        } else {
            self.buffer.copy_within(1..len, 0);
            self.buffer[len - 1] = value;
        }
    }

    /// Get the current write index.
    pub fn get_index(&self) -> usize {
        self.index
    }

    /// Set the write index.
    pub fn set_index(&mut self, index: usize) {
        self.index = index.min(self.buffer.len());
    }

    /// Get the length of the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl<T: Copy + Default> std::ops::Deref for PushBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<T: Copy + Default> std::ops::DerefMut for PushBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

impl<T: Copy + Default> std::ops::Index<usize> for PushBuffer<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.buffer[index]
    }
}

impl<T: Copy + Default> std::ops::IndexMut<usize> for PushBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.buffer[index]
    }
}

// ==========================================
// CircularBuffer - Ring Buffer for Delay Lines
// ==========================================

/// Circular buffer (ring buffer) for delay lines.
///
/// A fixed-size buffer where the read and write positions wrap around,
/// creating an infinite stream effect. Ideal for delay effects, lookahead
/// buffers, and any application requiring a fixed-length sample history.
///
/// # Power-of-2 Sizing
/// The buffer length is automatically rounded up to the next power of 2.
/// This enables efficient index wrapping using bitwise AND instead of modulo,
/// which is significantly faster in tight audio processing loops.
///
/// # Read/Write Pointers
/// - `push()` - Write a sample at the write position and advance
/// - `next()` - Read a sample from the read position and advance
/// - `peek()` - Read without advancing
#[derive(Clone)]
pub struct CircularBuffer<T: Copy + Default> {
    buffer: Box<[T]>,
    read: usize,
    write: usize,
    mask: usize, // For efficient modulo: index & mask
}

impl<T: Copy + Default> CircularBuffer<T> {
    /// Create a new circular buffer with the given length.
    ///
    /// The actual length is rounded up to the next power of 2 for
    /// efficient index wrapping. For example, requesting 100 samples
    /// will allocate 128.
    pub fn new(len: usize) -> Self {
        let actual_len = len.next_power_of_two().max(1);
        Self {
            buffer: vec![T::default(); actual_len].into_boxed_slice(),
            read: 0,
            write: 0,
            mask: actual_len - 1,
        }
    }

    /// Create a circular buffer from an existing slice.
    pub fn from_slice(slice: &[T]) -> Self {
        let len = slice.len().next_power_of_two().max(1);
        let mut buffer = vec![T::default(); len];
        buffer[..slice.len()].copy_from_slice(slice);
        Self {
            buffer: buffer.into_boxed_slice(),
            read: 0,
            write: slice.len() & (len - 1),
            mask: len - 1,
        }
    }

    /// Resize the buffer, clearing all data and resetting the pointers.
    pub fn resize(&mut self, len: usize) {
        let actual_len = len.next_power_of_two().max(1);
        self.buffer = vec![T::default(); actual_len].into_boxed_slice();
        self.read = 0;
        self.write = 0;
        self.mask = actual_len - 1;
    }

    /// Write a sample at the write position and advance the pointer.
    ///
    /// The pointer wraps automatically when reaching the buffer end.
    #[inline]
    pub fn push(&mut self, value: T) {
        let idx = self.write;
        self.buffer[idx] = value;
        self.write = (self.write + 1) & self.mask;
    }

    /// Read a sample from the read position and advance the pointer.
    ///
    /// The pointer wraps automatically when reaching the buffer end.
    ///
    /// Named to match the ring buffer's `push`/`next` pair, not
    /// `Iterator::next` (this type isn't an iterator: it never terminates
    /// and doesn't return `Option`).
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> T {
        let value = self.buffer[self.read];
        self.read = (self.read + 1) & self.mask;
        value
    }

    /// Peek at the sample at the read position without advancing.
    pub fn peek(&self) -> T {
        self.buffer[self.read]
    }

    /// Read a sample at an offset from the read position.
    ///
    /// Index wraps automatically using the buffer mask.
    pub fn read_offset(&self, offset: usize) -> T {
        self.buffer[(self.read + offset) & self.mask]
    }

    /// Write a sample at an offset from the write position.
    ///
    /// Index wraps automatically using the buffer mask.
    pub fn write_offset(&mut self, offset: usize, value: T) {
        let idx = (self.write + offset) & self.mask;
        self.buffer[idx] = value;
    }

    /// Get the current read pointer position.
    pub fn get_read(&self) -> usize {
        self.read
    }

    /// Get the current write pointer position.
    pub fn get_write(&self) -> usize {
        self.write
    }

    /// Set the read pointer position (masked to valid range).
    pub fn set_read(&mut self, index: usize) {
        self.read = index & self.mask;
    }

    /// Set the write pointer position (masked to valid range).
    pub fn set_write(&mut self, index: usize) {
        self.write = index & self.mask;
    }

    /// Get capacity (power of 2).
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Get logical length (same as capacity for a ring buffer).
    pub fn len(&self) -> usize {
        self.mask + 1
    }

    /// Check if empty (zero capacity).
    pub fn is_empty(&self) -> bool {
        self.capacity() == 0
    }

    /// Clear the buffer to default values and reset pointers.
    pub fn clear(&mut self) {
        self.buffer.fill(T::default());
        self.read = 0;
        self.write = 0;
    }
}

impl<T: Copy + Default> std::ops::Deref for CircularBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<T: Copy + Default> std::ops::DerefMut for CircularBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

impl<T: Copy + Default> std::ops::Index<usize> for CircularBuffer<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.buffer[index & self.mask]
    }
}

impl<T: Copy + Default> std::ops::IndexMut<usize> for CircularBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let idx = index & self.mask;
        &mut self.buffer[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_basic() {
        let mut b = Buffer::<f32>::new(4);
        assert_eq!(b.len(), 4);
        b[0] = 1.0;
        assert_eq!(b[0], 1.0);
        b.resize(2);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn test_push_buffer_sliding_window() {
        let mut history = PushBuffer::<f32>::new(4);
        assert_eq!(&*history, &[0.0, 0.0, 0.0, 0.0]);
        for x in [1.0, 2.0, 3.0] {
            history.push(x);
        }
        assert_eq!(&*history, &[1.0, 2.0, 3.0, 0.0]);
        history.push(4.0);
        assert_eq!(&*history, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_circular_buffer_rounds_to_power_of_two() {
        let cb = CircularBuffer::<f32>::new(100);
        assert_eq!(cb.capacity(), 128);
    }

    #[test]
    fn test_circular_buffer_delay() {
        let mut cb = CircularBuffer::<f32>::new(4);
        for i in 0..4 {
            cb.push(i as f32);
        }
        // read/write started equal, so the ring has lapped exactly once:
        // next() now returns what was pushed 4 samples ago.
        assert_eq!(cb.next(), 0.0);
        assert_eq!(cb.next(), 1.0);
    }
}
