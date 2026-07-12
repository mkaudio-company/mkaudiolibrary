/// Unique identifier for a node in the circuit graph.
pub type NodeId = usize;

/// Maximum block size supported by the engine.
pub const MAX_BLOCK_SIZE: usize = 128;

/// Cache-line aligned buffer for audio signal data.
/// Preallocated to avoid runtime allocation on the audio thread.
#[repr(C, align(64))]
pub struct AlignedBuffer<const N: usize> {
    /// The raw sample storage.
    pub data: [f32; N],
}

impl<const N: usize> AlignedBuffer<N> {
    /// Create a zero-filled buffer.
    pub const fn new() -> Self {
        Self { data: [0.0; N] }
    }

    /// Borrow the full backing array as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutably borrow the full backing array as a slice.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

impl<const N: usize> Default for AlignedBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Preallocated signal buffer for passing audio between nodes.
pub struct SignalBuffer {
    buffer: AlignedBuffer<MAX_BLOCK_SIZE>,
    len: usize,
}

impl SignalBuffer {
    /// Create an empty (zero-length) signal buffer.
    pub fn new() -> Self {
        Self {
            buffer: AlignedBuffer::new(),
            len: 0,
        }
    }

    /// Set the logical length (must be `<= MAX_BLOCK_SIZE`); does not
    /// clear or otherwise touch the underlying samples.
    pub fn resize(&mut self, len: usize) {
        assert!(len <= MAX_BLOCK_SIZE);
        self.len = len;
    }

    /// Current logical length in samples.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the logical length is zero.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the valid (logical-length) portion of the buffer.
    pub fn as_slice(&self) -> &[f32] {
        &self.buffer.data[..self.len]
    }

    /// Mutably borrow the valid (logical-length) portion of the buffer.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.buffer.data[..self.len]
    }

    /// Zero out the valid (logical-length) portion of the buffer.
    pub fn clear(&mut self) {
        for i in 0..self.len {
            self.buffer.data[i] = 0.0;
        }
    }
}

impl Default for SignalBuffer {
    fn default() -> Self {
        Self::new()
    }
}
