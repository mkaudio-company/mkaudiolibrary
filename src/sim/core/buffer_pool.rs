use super::node::{MAX_BLOCK_SIZE, SignalBuffer};

/// Preallocated pool of signal buffers for the circuit graph.
///
/// No runtime allocation — all buffers created at graph compile time.
pub struct BufferPool {
    buffers: Vec<SignalBuffer>,
}

impl BufferPool {
    /// Create a pool with `count` buffers, each sized to `block_size`.
    pub fn new(count: usize, block_size: usize) -> Self {
        assert!(block_size <= MAX_BLOCK_SIZE);
        let mut buffers = Vec::with_capacity(count);
        for _ in 0..count {
            let mut buf = SignalBuffer::new();
            buf.resize(block_size);
            buffers.push(buf);
        }
        Self { buffers }
    }

    /// Get a reference to buffer at index.
    pub fn get(&self, index: usize) -> &SignalBuffer {
        &self.buffers[index]
    }

    /// Get a mutable reference to buffer at index.
    pub fn get_mut(&mut self, index: usize) -> &mut SignalBuffer {
        &mut self.buffers[index]
    }

    /// Number of buffers in the pool.
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Whether the pool has no buffers.
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Clear all buffers to zero.
    pub fn clear_all(&mut self) {
        for buf in &mut self.buffers {
            buf.clear();
        }
    }

    /// Resize all buffers to a new block size.
    pub fn resize_all(&mut self, block_size: usize) {
        for buf in &mut self.buffers {
            buf.resize(block_size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_creation() {
        let pool = BufferPool::new(4, 64);
        assert_eq!(pool.len(), 4);
        for i in 0..4 {
            assert_eq!(pool.get(i).len(), 64);
        }
    }

    #[test]
    fn test_buffer_pool_clear() {
        let mut pool = BufferPool::new(2, 32);
        // Write some data
        pool.get_mut(0).as_mut_slice()[0] = 1.0;
        pool.clear_all();
        assert_eq!(pool.get(0).as_slice()[0], 0.0);
    }
}
