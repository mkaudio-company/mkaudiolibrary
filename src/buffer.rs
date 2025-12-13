//! Thread-safe audio buffers for real-time concurrent processing.
//!
//! This module provides three buffer types designed for multi-threaded audio applications:
//!
//! - [`Buffer`] - General-purpose buffer with read/write locking
//! - [`PushBuffer`] - FIFO buffer for convolution and filtering operations
//! - [`CircularBuffer`] - Ring buffer with power-of-2 sizing for delay lines
//!
//! ## Thread Safety Model
//!
//! All buffers use `Arc<RwLock<...>>` internally, providing:
//! - **Multiple concurrent readers** - Any number of threads can read simultaneously
//! - **Exclusive writer access** - Writers block all other access
//! - **Clone shares data** - Cloning creates a new handle to the same underlying data
//!
//! ## RAII Guards
//!
//! Access to buffer data is controlled through guard types that implement `Deref`/`DerefMut`:
//! - `BufferReadGuard` / `BufferWriteGuard`
//! - `PushBufferReadGuard` / `PushBufferWriteGuard`
//! - `CircularBufferReadGuard` / `CircularBufferWriteGuard`
//!
//! Guards automatically release locks when dropped.
//!
//! ## Example: Shared Buffer Between Threads
//!
//! ```ignore
//! use mkaudiolibrary::buffer::Buffer;
//! use std::thread;
//!
//! let buffer = Buffer::<f64>::new(1024);
//! let buffer_clone = buffer.clone();  // Shares underlying data
//!
//! // Writer thread
//! let writer = thread::spawn(move || {
//!     let mut guard = buffer_clone.write();
//!     guard[0] = 1.0;
//! });
//!
//! // Reader thread (after writer completes)
//! writer.join().unwrap();
//! let guard = buffer.read();
//! assert_eq!(guard[0], 1.0);
//! ```
//!
//! ## Example: Delay Line with CircularBuffer
//!
//! ```ignore
//! use mkaudiolibrary::buffer::CircularBuffer;
//!
//! let delay = CircularBuffer::<f64>::new(1024).unwrap();
//! let mut guard = delay.write();
//!
//! // Write samples
//! for i in 0..100 {
//!     guard.push(i as f64);
//! }
//!
//! // Read delayed samples
//! for _ in 0..100 {
//!     let sample = guard.next();
//! }
//! ```

use std::alloc::LayoutError;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

// ==========================================
// Buffer - General Purpose
// ==========================================

/// Thread-safe general-purpose audio buffer with read-write locking.
///
/// Provides concurrent access to a contiguous block of samples. Multiple readers
/// can access the buffer simultaneously, while writers get exclusive access.
///
/// # Memory Layout
/// Uses `Box<[T]>` internally for cache-friendly contiguous storage with
/// minimal allocation overhead.
///
/// # Cloning
/// Cloning a `Buffer` creates a new handle to the **same** underlying data
/// (similar to `Arc`). The reference count is tracked internally.
///
/// # Thread Safety
/// Implements `Send + Sync` for use across thread boundaries.
pub struct Buffer<T : Clone + Default + Send + Sync>
{
    inner : Arc<BufferInner<T>>
}

struct BufferInner<T : Clone + Default + Send + Sync>
{
    data : RwLock<Box<[T]>>,
    reference : AtomicUsize
}

impl<T : Clone + Default + Send + Sync> Buffer<T>
{
    /// Create a new buffer with the given length, initialized to default values.
    pub fn new(len : usize) -> Self
    {
        Self
        {
            inner : Arc::new(BufferInner
            {
                data : RwLock::new(vec![T::default(); len].into_boxed_slice()),
                reference : AtomicUsize::new(1)
            })
        }
    }

    /// Create a buffer from an existing slice (copies data).
    pub fn from_slice(slice : &[T]) -> Self
    {
        Self
        {
            inner : Arc::new(BufferInner
            {
                data : RwLock::new(slice.to_vec().into_boxed_slice()),
                reference : AtomicUsize::new(1)
            })
        }
    }

    /// Acquire a read lock for shared access.
    pub fn read(&self) -> BufferReadGuard<'_, T>
    {
        BufferReadGuard { guard : self.inner.data.read().unwrap() }
    }

    /// Try to acquire a read lock without blocking.
    pub fn try_read(&self) -> Option<BufferReadGuard<'_, T>>
    {
        self.inner.data.try_read().ok().map(|guard| BufferReadGuard { guard })
    }

    /// Acquire a write lock for exclusive access.
    pub fn write(&self) -> BufferWriteGuard<'_, T>
    {
        BufferWriteGuard { guard : self.inner.data.write().unwrap() }
    }

    /// Try to acquire a write lock without blocking.
    pub fn try_write(&self) -> Option<BufferWriteGuard<'_, T>>
    {
        self.inner.data.try_write().ok().map(|guard| BufferWriteGuard { guard })
    }

    /// Resize the buffer (acquires write lock internally).
    pub fn resize(&self, len : usize)
    {
        let mut guard = self.inner.data.write().unwrap();
        *guard = vec![T::default(); len].into_boxed_slice();
    }

    /// Get the length of the buffer.
    pub fn len(&self) -> usize
    {
        self.inner.data.read().unwrap().len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Get the current reference count.
    pub fn ref_count(&self) -> usize { self.inner.reference.load(Ordering::Acquire) }
}

/// RAII read guard for [`Buffer`].
///
/// Provides immutable access to buffer contents. Multiple read guards can
/// exist simultaneously. Implements `Deref<Target = [T]>` for slice access.
///
/// The lock is released when this guard is dropped.
pub struct BufferReadGuard<'a, T : Clone + Default + Send + Sync>
{
    guard : RwLockReadGuard<'a, Box<[T]>>
}

impl<'a, T : Clone + Default + Send + Sync> std::ops::Deref for BufferReadGuard<'a, T>
{
    type Target = [T];
    fn deref(&self) -> &Self::Target { &self.guard }
}

impl<'a, T : Clone + Default + Send + Sync> BufferReadGuard<'a, T>
{
    /// Get the number of elements in the buffer.
    pub fn len(&self) -> usize { self.guard.len() }
    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool { self.guard.is_empty() }
}

/// RAII write guard for [`Buffer`].
///
/// Provides exclusive mutable access to buffer contents. Only one write guard
/// can exist at a time. Implements `DerefMut<Target = [T]>` for slice access.
///
/// The lock is released when this guard is dropped.
pub struct BufferWriteGuard<'a, T : Clone + Default + Send + Sync>
{
    guard : RwLockWriteGuard<'a, Box<[T]>>
}

impl<'a, T : Clone + Default + Send + Sync> std::ops::Deref for BufferWriteGuard<'a, T>
{
    type Target = [T];
    fn deref(&self) -> &Self::Target { &self.guard }
}

impl<'a, T : Clone + Default + Send + Sync> std::ops::DerefMut for BufferWriteGuard<'a, T>
{
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.guard }
}

impl<'a, T : Clone + Default + Send + Sync> BufferWriteGuard<'a, T>
{
    /// Get the number of elements in the buffer.
    pub fn len(&self) -> usize { self.guard.len() }
    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool { self.guard.is_empty() }
}

impl<T : Clone + Default + Send + Sync> Clone for Buffer<T>
{
    fn clone(&self) -> Self
    {
        self.inner.reference.fetch_add(1, Ordering::AcqRel);
        Self { inner : Arc::clone(&self.inner) }
    }
}

impl<T : Clone + Default + Send + Sync> Drop for Buffer<T>
{
    fn drop(&mut self)
    {
        self.inner.reference.fetch_sub(1, Ordering::AcqRel);
    }
}

unsafe impl<T : Clone + Default + Send + Sync> Send for Buffer<T> {}
unsafe impl<T : Clone + Default + Send + Sync> Sync for Buffer<T> {}

impl<T : Clone + Default + Send + Sync> Default for Buffer<T>
{
    fn default() -> Self { Self::new(0) }
}

// ==========================================
// PushBuffer - FIFO for Convolution
// ==========================================

/// Thread-safe FIFO buffer for convolution and FIR filtering.
///
/// Maintains a sliding window of samples, automatically shifting old samples
/// out as new ones are pushed. Ideal for real-time FIR filter implementations
/// where you need access to the last N samples.
///
/// # Push Behavior
/// - Until the buffer is full: samples are appended at the current index
/// - When full: samples shift left, newest sample goes at the end (FIFO)
///
/// # Use Cases
/// - FIR filter implementations
/// - Convolution with impulse responses
/// - Any DSP requiring a sliding window of samples
///
/// # Thread Safety
/// Uses `RwLock` for concurrent access. Multiple readers can access
/// simultaneously; writers get exclusive access.
pub struct PushBuffer<T : Copy + Default + Send + Sync>
{
    inner : Arc<PushBufferInner<T>>
}

struct PushBufferInner<T : Copy + Default + Send + Sync>
{
    data : RwLock<PushBufferData<T>>
}

struct PushBufferData<T : Copy + Default>
{
    buffer : Box<[T]>,
    index : usize
}

impl<T : Copy + Default + Send + Sync> PushBuffer<T>
{
    /// Create a new push buffer with the given length.
    ///
    /// Buffer is initialized with default values and the write index at 0.
    pub fn new(len : usize) -> Result<Self, LayoutError>
    {
        Ok(Self
        {
            inner : Arc::new(PushBufferInner
            {
                data : RwLock::new(PushBufferData
                {
                    buffer : vec![T::default(); len].into_boxed_slice(),
                    index : 0
                })
            })
        })
    }

    /// Create a push buffer from an existing slice.
    pub fn from_slice(slice : &[T]) -> Self
    {
        Self
        {
            inner : Arc::new(PushBufferInner
            {
                data : RwLock::new(PushBufferData
                {
                    buffer : slice.to_vec().into_boxed_slice(),
                    index : 0
                })
            })
        }
    }

    /// Acquire a read lock.
    pub fn read(&self) -> PushBufferReadGuard<'_, T>
    {
        PushBufferReadGuard { guard : self.inner.data.read().unwrap() }
    }

    /// Try to acquire a read lock without blocking.
    pub fn try_read(&self) -> Option<PushBufferReadGuard<'_, T>>
    {
        self.inner.data.try_read().ok().map(|guard| PushBufferReadGuard { guard })
    }

    /// Acquire a write lock.
    pub fn write(&self) -> PushBufferWriteGuard<'_, T>
    {
        PushBufferWriteGuard { guard : self.inner.data.write().unwrap() }
    }

    /// Try to acquire a write lock without blocking.
    pub fn try_write(&self) -> Option<PushBufferWriteGuard<'_, T>>
    {
        self.inner.data.try_write().ok().map(|guard| PushBufferWriteGuard { guard })
    }

    /// Resize the buffer, clearing all data.
    pub fn resize(&self, len : usize) -> Result<(), LayoutError>
    {
        let mut guard = self.inner.data.write().unwrap();
        guard.buffer = vec![T::default(); len].into_boxed_slice();
        guard.index = 0;
        Ok(())
    }

    /// Push a new value (acquires write lock internally).
    pub fn push(&self, value : T)
    {
        let mut guard = self.inner.data.write().unwrap();
        let len = guard.buffer.len();
        if len == 0 { return; }

        if guard.index < len
        {
            let idx = guard.index;
            guard.buffer[idx] = value;
            guard.index += 1;
        }
        else
        {
            guard.buffer.copy_within(1..len, 0);
            guard.buffer[len - 1] = value;
        }
    }

    /// Get the current write index.
    pub fn get_index(&self) -> usize { self.inner.data.read().unwrap().index }

    /// Set the write index.
    pub fn set_index(&self, index : usize)
    {
        let mut guard = self.inner.data.write().unwrap();
        guard.index = index.min(guard.buffer.len());
    }

    /// Get the length of the buffer.
    pub fn len(&self) -> usize { self.inner.data.read().unwrap().buffer.len() }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

/// RAII read guard for [`PushBuffer`].
///
/// Provides immutable access to buffer contents and index state.
/// Implements `Deref<Target = [T]>` and `Index<usize>` for element access.
pub struct PushBufferReadGuard<'a, T : Copy + Default + Send + Sync>
{
    guard : RwLockReadGuard<'a, PushBufferData<T>>
}

impl<'a, T : Copy + Default + Send + Sync> PushBufferReadGuard<'a, T>
{
    /// Get the buffer length.
    pub fn len(&self) -> usize { self.guard.buffer.len() }
    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool { self.guard.buffer.is_empty() }
    /// Get the current write index.
    pub fn get_index(&self) -> usize { self.guard.index }
}

impl<'a, T : Copy + Default + Send + Sync> std::ops::Deref for PushBufferReadGuard<'a, T>
{
    type Target = [T];
    fn deref(&self) -> &Self::Target { &self.guard.buffer }
}

impl<'a, T : Copy + Default + Send + Sync> std::ops::Index<usize> for PushBufferReadGuard<'a, T>
{
    type Output = T;
    fn index(&self, index : usize) -> &Self::Output { &self.guard.buffer[index] }
}

/// RAII write guard for [`PushBuffer`].
///
/// Provides exclusive mutable access to buffer contents. Supports the `push()`
/// operation for adding new samples. Implements `DerefMut<Target = [T]>` and
/// `IndexMut<usize>` for element access.
pub struct PushBufferWriteGuard<'a, T : Copy + Default + Send + Sync>
{
    guard : RwLockWriteGuard<'a, PushBufferData<T>>
}

impl<'a, T : Copy + Default + Send + Sync> PushBufferWriteGuard<'a, T>
{
    /// Get the buffer length.
    pub fn len(&self) -> usize { self.guard.buffer.len() }
    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool { self.guard.buffer.is_empty() }
    /// Get the current write index.
    pub fn get_index(&self) -> usize { self.guard.index }

    /// Push a new sample into the buffer.
    ///
    /// If the buffer is not yet full, appends at the current index.
    /// If the buffer is full, shifts all samples left and places the new
    /// sample at the end (FIFO behavior).
    pub fn push(&mut self, value : T)
    {
        let len = self.guard.buffer.len();
        if len == 0 { return; }

        if self.guard.index < len
        {
            let idx = self.guard.index;
            self.guard.buffer[idx] = value;
            self.guard.index += 1;
        }
        else
        {
            self.guard.buffer.copy_within(1..len, 0);
            self.guard.buffer[len - 1] = value;
        }
    }
}

impl<'a, T : Copy + Default + Send + Sync> std::ops::Deref for PushBufferWriteGuard<'a, T>
{
    type Target = [T];
    fn deref(&self) -> &Self::Target { &self.guard.buffer }
}

impl<'a, T : Copy + Default + Send + Sync> std::ops::DerefMut for PushBufferWriteGuard<'a, T>
{
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.guard.buffer }
}

impl<'a, T : Copy + Default + Send + Sync> std::ops::Index<usize> for PushBufferWriteGuard<'a, T>
{
    type Output = T;
    fn index(&self, index : usize) -> &Self::Output { &self.guard.buffer[index] }
}

impl<'a, T : Copy + Default + Send + Sync> std::ops::IndexMut<usize> for PushBufferWriteGuard<'a, T>
{
    fn index_mut(&mut self, index : usize) -> &mut Self::Output { &mut self.guard.buffer[index] }
}

impl<T : Copy + Default + Send + Sync> Clone for PushBuffer<T>
{
    fn clone(&self) -> Self { Self { inner : Arc::clone(&self.inner) } }
}

unsafe impl<T : Copy + Default + Send + Sync> Send for PushBuffer<T> {}
unsafe impl<T : Copy + Default + Send + Sync> Sync for PushBuffer<T> {}

// ==========================================
// CircularBuffer - Ring Buffer for Delay Lines
// ==========================================

/// Thread-safe circular buffer (ring buffer) for delay lines.
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
///
/// # Thread Safety
/// Uses `RwLock` for concurrent access. Multiple readers can access
/// simultaneously; writers get exclusive access.
///
/// # Example
/// ```ignore
/// let delay = CircularBuffer::<f64>::new(256).unwrap();  // Rounds to 256 (power of 2)
/// let mut guard = delay.write();
/// guard.push(1.0);  // Write sample
/// let sample = guard.next();  // Read delayed sample
/// ```
pub struct CircularBuffer<T : Copy + Default + Send + Sync>
{
    inner : Arc<CircularBufferInner<T>>
}

struct CircularBufferInner<T : Copy + Default + Send + Sync>
{
    data : RwLock<CircularBufferData<T>>
}

struct CircularBufferData<T : Copy + Default>
{
    buffer : Box<[T]>,
    read : usize,
    write : usize,
    mask : usize  // For efficient modulo: index & mask
}

impl<T : Copy + Default + Send + Sync> CircularBuffer<T>
{
    /// Create a new circular buffer with the given length.
    ///
    /// The actual length is rounded up to the next power of 2 for
    /// efficient index wrapping. For example, requesting 100 samples
    /// will allocate 128.
    pub fn new(len : usize) -> Result<Self, LayoutError>
    {
        let actual_len = len.next_power_of_two().max(1);
        Ok(Self
        {
            inner : Arc::new(CircularBufferInner
            {
                data : RwLock::new(CircularBufferData
                {
                    buffer : vec![T::default(); actual_len].into_boxed_slice(),
                    read : 0,
                    write : 0,
                    mask : actual_len - 1
                })
            })
        })
    }

    /// Create a circular buffer from an existing slice.
    pub fn from_slice(slice : &[T]) -> Self
    {
        let len = slice.len().next_power_of_two().max(1);
        let mut buffer = vec![T::default(); len];
        buffer[..slice.len()].copy_from_slice(slice);
        Self
        {
            inner : Arc::new(CircularBufferInner
            {
                data : RwLock::new(CircularBufferData
                {
                    buffer : buffer.into_boxed_slice(),
                    read : 0,
                    write : slice.len() & (len - 1),
                    mask : len - 1
                })
            })
        }
    }

    /// Acquire a read lock.
    pub fn read(&self) -> CircularBufferReadGuard<'_, T>
    {
        CircularBufferReadGuard { guard : self.inner.data.read().unwrap() }
    }

    /// Try to acquire a read lock without blocking.
    pub fn try_read(&self) -> Option<CircularBufferReadGuard<'_, T>>
    {
        self.inner.data.try_read().ok().map(|guard| CircularBufferReadGuard { guard })
    }

    /// Acquire a write lock.
    pub fn write(&self) -> CircularBufferWriteGuard<'_, T>
    {
        CircularBufferWriteGuard { guard : self.inner.data.write().unwrap() }
    }

    /// Try to acquire a write lock without blocking.
    pub fn try_write(&self) -> Option<CircularBufferWriteGuard<'_, T>>
    {
        self.inner.data.try_write().ok().map(|guard| CircularBufferWriteGuard { guard })
    }

    /// Resize the buffer, clearing all data.
    pub fn resize(&self, len : usize) -> Result<(), LayoutError>
    {
        let actual_len = len.next_power_of_two().max(1);
        let mut guard = self.inner.data.write().unwrap();
        guard.buffer = vec![T::default(); actual_len].into_boxed_slice();
        guard.read = 0;
        guard.write = 0;
        guard.mask = actual_len - 1;
        Ok(())
    }

    /// Push a value and advance write pointer (acquires write lock).
    pub fn push(&self, value : T)
    {
        let mut guard = self.inner.data.write().unwrap();
        let idx = guard.write;
        guard.buffer[idx] = value;
        guard.write = (guard.write + 1) & guard.mask;
    }

    /// Read the next value and advance read pointer (acquires write lock).
    pub fn next(&self) -> T
    {
        let mut guard = self.inner.data.write().unwrap();
        let value = guard.buffer[guard.read];
        guard.read = (guard.read + 1) & guard.mask;
        value
    }

    /// Peek at value without advancing (acquires read lock).
    pub fn peek(&self) -> T
    {
        let guard = self.inner.data.read().unwrap();
        guard.buffer[guard.read]
    }

    /// Get capacity (power of 2).
    pub fn capacity(&self) -> usize { self.inner.data.read().unwrap().buffer.len() }

    /// Get logical length.
    pub fn len(&self) -> usize { self.inner.data.read().unwrap().mask + 1 }

    /// Check if empty.
    pub fn is_empty(&self) -> bool { self.capacity() == 0 }

    /// Clear buffer to default values.
    pub fn clear(&self)
    {
        let mut guard = self.inner.data.write().unwrap();
        guard.buffer.fill(T::default());
        guard.read = 0;
        guard.write = 0;
    }
}

/// RAII read guard for [`CircularBuffer`].
///
/// Provides immutable access to buffer contents and pointer state.
/// Implements `Deref<Target = [T]>` for raw buffer access and
/// `Index<usize>` with automatic index wrapping.
pub struct CircularBufferReadGuard<'a, T : Copy + Default + Send + Sync>
{
    guard : RwLockReadGuard<'a, CircularBufferData<T>>
}

impl<'a, T : Copy + Default + Send + Sync> CircularBufferReadGuard<'a, T>
{
    /// Get the logical buffer length (power of 2).
    pub fn len(&self) -> usize { self.guard.mask + 1 }
    /// Get the actual buffer capacity.
    pub fn capacity(&self) -> usize { self.guard.buffer.len() }
    /// Get the current read pointer position.
    pub fn get_read(&self) -> usize { self.guard.read }
    /// Get the current write pointer position.
    pub fn get_write(&self) -> usize { self.guard.write }
    /// Peek at the sample at the read position without advancing.
    pub fn peek(&self) -> T { self.guard.buffer[self.guard.read] }

    /// Read a sample at an offset from the read position.
    ///
    /// Index wraps automatically using the buffer mask.
    pub fn read_offset(&self, offset : usize) -> T
    {
        self.guard.buffer[(self.guard.read + offset) & self.guard.mask]
    }
}

impl<'a, T : Copy + Default + Send + Sync> std::ops::Deref for CircularBufferReadGuard<'a, T>
{
    type Target = [T];
    fn deref(&self) -> &Self::Target { &self.guard.buffer }
}

impl<'a, T : Copy + Default + Send + Sync> std::ops::Index<usize> for CircularBufferReadGuard<'a, T>
{
    type Output = T;
    fn index(&self, index : usize) -> &Self::Output { &self.guard.buffer[index & self.guard.mask] }
}

/// RAII write guard for [`CircularBuffer`].
///
/// Provides exclusive mutable access to buffer contents. Supports ring buffer
/// operations (`push`, `next`, `peek`) and direct element access via
/// `DerefMut<Target = [T]>` and `IndexMut<usize>` with automatic wrapping.
pub struct CircularBufferWriteGuard<'a, T : Copy + Default + Send + Sync>
{
    guard : RwLockWriteGuard<'a, CircularBufferData<T>>
}

impl<'a, T : Copy + Default + Send + Sync> CircularBufferWriteGuard<'a, T>
{
    /// Get the logical buffer length (power of 2).
    pub fn len(&self) -> usize { self.guard.mask + 1 }
    /// Get the actual buffer capacity.
    pub fn capacity(&self) -> usize { self.guard.buffer.len() }
    /// Get the current read pointer position.
    pub fn get_read(&self) -> usize { self.guard.read }
    /// Get the current write pointer position.
    pub fn get_write(&self) -> usize { self.guard.write }

    /// Write a sample at the write position and advance the pointer.
    ///
    /// The pointer wraps automatically when reaching the buffer end.
    #[inline]
    pub fn push(&mut self, value : T)
    {
        let idx = self.guard.write;
        self.guard.buffer[idx] = value;
        self.guard.write = (self.guard.write + 1) & self.guard.mask;
    }

    /// Read a sample from the read position and advance the pointer.
    ///
    /// The pointer wraps automatically when reaching the buffer end.
    #[inline]
    pub fn next(&mut self) -> T
    {
        let value = self.guard.buffer[self.guard.read];
        self.guard.read = (self.guard.read + 1) & self.guard.mask;
        value
    }

    /// Peek at the sample at the read position without advancing.
    pub fn peek(&self) -> T { self.guard.buffer[self.guard.read] }

    /// Read a sample at an offset from the read position.
    ///
    /// Index wraps automatically using the buffer mask.
    pub fn read_offset(&self, offset : usize) -> T
    {
        self.guard.buffer[(self.guard.read + offset) & self.guard.mask]
    }

    /// Write a sample at an offset from the write position.
    ///
    /// Index wraps automatically using the buffer mask.
    pub fn write_offset(&mut self, offset : usize, value : T)
    {
        let idx = (self.guard.write + offset) & self.guard.mask;
        self.guard.buffer[idx] = value;
    }

    /// Set the read pointer position (masked to valid range).
    pub fn set_read(&mut self, index : usize) { self.guard.read = index & self.guard.mask; }

    /// Set the write pointer position (masked to valid range).
    pub fn set_write(&mut self, index : usize) { self.guard.write = index & self.guard.mask; }

    /// Clear the buffer to default values and reset pointers.
    pub fn clear(&mut self)
    {
        self.guard.buffer.fill(T::default());
        self.guard.read = 0;
        self.guard.write = 0;
    }
}

impl<'a, T : Copy + Default + Send + Sync> std::ops::Deref for CircularBufferWriteGuard<'a, T>
{
    type Target = [T];
    fn deref(&self) -> &Self::Target { &self.guard.buffer }
}

impl<'a, T : Copy + Default + Send + Sync> std::ops::DerefMut for CircularBufferWriteGuard<'a, T>
{
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.guard.buffer }
}

impl<'a, T : Copy + Default + Send + Sync> std::ops::Index<usize> for CircularBufferWriteGuard<'a, T>
{
    type Output = T;
    fn index(&self, index : usize) -> &Self::Output { &self.guard.buffer[index & self.guard.mask] }
}

impl<'a, T : Copy + Default + Send + Sync> std::ops::IndexMut<usize> for CircularBufferWriteGuard<'a, T>
{
    fn index_mut(&mut self, index : usize) -> &mut Self::Output
    {
        let idx = index & self.guard.mask;
        &mut self.guard.buffer[idx]
    }
}

impl<T : Copy + Default + Send + Sync> Clone for CircularBuffer<T>
{
    fn clone(&self) -> Self { Self { inner : Arc::clone(&self.inner) } }
}

unsafe impl<T : Copy + Default + Send + Sync> Send for CircularBuffer<T> {}
unsafe impl<T : Copy + Default + Send + Sync> Sync for CircularBuffer<T> {}
