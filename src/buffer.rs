use std::alloc::{alloc_zeroed, dealloc, Layout, LayoutError};

/// Sized Buffer that can be used in multi-threaded environment, with reference count. Be conscious, this is unsafe from data hazards.
pub struct Buffer<T>
{
    buffer : *mut T,
    len : usize,
    count : *mut usize
}
impl<T : Sized + Send + Sync> Buffer<T>
{
    /// New Buffer with length.
    pub fn new(len : usize) -> Result<Self,LayoutError>
    {
        let layout = std::alloc::Layout::array::<T>(len)?;
        unsafe { Ok(Self { buffer : std::alloc::alloc_zeroed(layout) as * mut T , len, count : std::alloc::alloc_zeroed(std::alloc::Layout::new::<usize>()) as *mut usize }) }
    }
    /// New Buffer from raw pointer.
    pub fn from_raw(ptr : * mut T, len : usize) -> Self
    {
        Self { buffer : ptr, len, count : unsafe { std::alloc::alloc_zeroed(std::alloc::Layout::new::<usize>()) } as *mut usize }
    }
    /// Resizes the buffer.
    pub fn resize(&mut self, len : usize) -> Result<(), LayoutError>
    {
        let dealloc_layout = std::alloc::Layout::array::<T>(self.len)?;
        let alloc_layout = std::alloc::Layout::array::<T>(len)?;
        unsafe
        {
            std::alloc::dealloc(self.buffer as * mut u8, dealloc_layout);
            self.buffer = std::alloc::alloc_zeroed(alloc_layout) as * mut T;
            self.len = len;
        }
        Ok(())
    }
    /// Converts internal data chunk as silce
    pub fn into_slice(&self) -> &[T] { unsafe { std::slice::from_raw_parts(self.buffer, self.len) } }
    /// Converts internal data chunk as mutable silce
    pub fn into_slice_mut(&self) -> &mut[T] { unsafe{ std::slice::from_raw_parts_mut(self.buffer, self.len) } }
    /// Returns the length of the buffer.
    pub fn len(&self) -> usize { return self.len; }
}
impl<T> std::ops::Index<usize> for Buffer<T>
{
    type Output = T;

    /// 
    fn index(&self, index: usize) -> &Self::Output
    {
        let real_index = if index > self.len
        {
            eprintln!("Index out of range. Indexing to remain of given index divided by size of buffer");
            index % self.len
        } else { index };
        let data = unsafe { self.buffer.offset(real_index as isize).as_ref() };
        match data
        {
            None => { panic!("Access to invalid memory"); }
            Some(reference) => { return reference; }
        }
    }
}
impl<T> std::ops::IndexMut<usize> for Buffer<T>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output
    {
        let real_index = if index > self.len
        {
            eprintln!("Index out of range. Indexing to remain of given index divided by size of buffer");
            index % self.len
        } else { index };
        let data = unsafe { self.buffer.offset(real_index as isize).as_mut() };
        match data
        {
            None => { panic!("Access to invalid memory"); }
            Some(reference) => { return reference; }
        }
    }
}
impl<T : Sized + Send + Sync> std::ops::Deref for Buffer<T>
{
    type Target = [T];
    fn deref(&self) -> &Self::Target { self.into_slice() }
}
impl<T : Sized + Send + Sync> std::ops::DerefMut for Buffer<T> { fn deref_mut(&mut self) -> &mut Self::Target { self.into_slice_mut() } }
unsafe impl<T> Send for Buffer<T> {}
unsafe impl<T> Sync for Buffer<T> {}
impl<T> Clone for Buffer<T>
{
    fn clone(&self) -> Self
    {
        unsafe { *self.count += 1 }
        Self
        {
            buffer: self.buffer,
            len: self.len,
            count: self.count
        }
    }
}
impl<T> Drop for Buffer<T>
{
    fn drop(&mut self)
    {
        unsafe
        {
            if *self.count > 1
            {
                *self.count -= 1;
                return
            }
        }
        let layout = std::alloc::Layout::array::<T>(self.len);
        match layout
        {
            Ok(layout) => { unsafe { std::alloc::dealloc(self.buffer as * mut u8, layout) }; },
            Err(error) => { eprintln!("drop failed : {}", error); }
        }
    }
}

///The buffer that pushes the whole buffer when index meets the size of the buffer. Generic T must be either f32 or f64.
#[derive(Clone)]
pub struct PushBuffer<T>
{
    buffer : * mut T,
    index : usize,
    len : usize
}
impl<T : Copy> PushBuffer<T>
{
    ///New PushBuffer with length.

    pub fn new(len : usize) -> Result<Self, LayoutError>
    {
        let layout = Layout::array::<T>(len)?;
        unsafe { Ok(PushBuffer { buffer : alloc_zeroed(layout) as * mut T , index : 0, len : len }) }
    }
    ///New PushBuffer from raw pointer.

    pub fn from_raw(ptr : * mut T, len : usize) -> Self
    {
        Self { buffer : ptr, index : 0, len }
    }
    ///Resizes the buffer.

    pub fn resize(&mut self, len : usize) -> Result<(), LayoutError>
    {
        let dealloc_layout = std::alloc::Layout::array::<T>(self.len)?;
        let alloc_layout = std::alloc::Layout::array::<T>(len)?;
        unsafe
        {
            std::alloc::dealloc(self.buffer as * mut u8, dealloc_layout);
            self.buffer = std::alloc::alloc_zeroed(alloc_layout) as * mut T;
        }
        Ok(())
    }
    ///Converts internal data chunk as silce

    pub fn into_slice(&self) -> &[T] { unsafe { std::slice::from_raw_parts(self.buffer, self.len) } }
    ///Converts internal data chunk as mutable silce

    pub fn into_slice_mut(&self) -> &mut[T] { unsafe{ std::slice::from_raw_parts_mut(self.buffer, self.len) } }
    ///Pushes data to buffer.

    pub fn push(& mut self, value : T)
    {
        if self.index <= self.len
        {
            unsafe { * self.buffer.offset(self.index as isize) = value; }
            self.index += 1;
        }
        else
        {
            unsafe
                {
                    (1..self.len).for_each(|x| * self.buffer.offset(x as isize - 1) = * self.buffer.offset(x as isize));
                    * self.buffer.offset(self.index as isize) = value;
                }
        }
    }
    ///Get index.
    pub fn get_index(&self) -> usize { self.index }
    ///Set index.
    pub fn set_index(&mut self, index : usize) { self.index = index; }
    ///Returns the length of the buffer.
    pub fn len(& self) -> usize { return self.len; }
}
impl<T> std::ops::Index<usize> for PushBuffer<T>
{
    type Output = T;

    fn index(& self, index : usize) -> & Self::Output
    {
        let real_index = if index > self.len
        {
            eprintln!("Index out of range. Indexing to remain of given index divided by size of buffer");
            index % self.len
        } else { index };
        let data = unsafe { self.buffer.offset(real_index as isize).as_ref() };
        match data
        {
            None => { panic!("Access to invalid memory!"); }
            Some(reference) => { return reference; }
        }
    }
}
impl<T> std::ops::IndexMut<usize> for PushBuffer<T>
{
    fn index_mut(& mut self, index : usize) -> & mut Self::Output
    {
        let real_index = if index > self.len
        {
            eprintln!("Index out of range. Indexing to remain of given index divided by size of buffer");
            index % self.len
        } else { index };
        let data = unsafe { self.buffer.offset(real_index as isize).as_mut() };
        match data
        {
            None => { panic!("Access to invalid memory!"); }
            Some(reference) => { return reference; }
        }
    }
}
impl<T : Copy> std::ops::Deref for PushBuffer<T>
{
    type Target = [T];
    fn deref(&self) -> &Self::Target { self.into_slice() }
}
impl<T : Copy> std::ops::DerefMut for PushBuffer<T>
{
    fn deref_mut(&mut self) -> &mut Self::Target { self.into_slice_mut() }
}
impl<T> Drop for PushBuffer<T>
{
    fn drop(&mut self)
    {
        let layout = Layout::array::<T>(self.len as usize);
        match layout
        {
            Ok(layout) => { unsafe { dealloc(self.buffer as * mut u8, layout); } }
            Err(error) => { eprintln!("drop failed : {}", error); }
        }
    }
}

///Circular buffer or ring buffer. When index returns to 0 when index meets the size of the buffer. Generic T must be either f32 or f64.
#[derive(Clone)]
pub struct CircularBuffer<T>
{
    buffer : * mut T,
    read : usize,
    write : usize,
    len : usize
}
impl<T : Copy> CircularBuffer<T>
{
    ///New CircularBuffer with length.

    pub fn new(len : usize) -> Result<Self, LayoutError>
    {
        let layout = Layout::array::<T>(len)?;
        unsafe { Ok(CircularBuffer { buffer : alloc_zeroed(layout) as * mut T, read : 0, write : 0, len : len }) }
    }
    ///New CircularBuffer from raw pointer.

    pub fn from_raw(ptr : * mut T, len : usize) -> Self { Self { buffer : ptr, read : 0, write : 0, len } }
    ///Resizes the buffer.

    pub fn resize(&mut self, len : usize) -> Result<(), LayoutError>
    {
        let dealloc_layout = std::alloc::Layout::array::<T>(self.len)?;
        let alloc_layout = std::alloc::Layout::array::<T>(len)?;
        unsafe
        {
            std::alloc::dealloc(self.buffer as * mut u8, dealloc_layout);
            self.buffer = std::alloc::alloc_zeroed(alloc_layout) as * mut T;
        }
        self.read = 0;
        self.write = 0;

        Ok(())
    }
    ///Converts internal data chunk as silce
    pub fn into_slice(&self) -> &[T] { unsafe { std::slice::from_raw_parts(self.buffer, self.len) } }
    ///Converts internal data chunk as mutable silce
    pub fn into_slice_mut(&self) -> &mut[T] { unsafe{ std::slice::from_raw_parts_mut(self.buffer, self.len) } }
    ///Pushes data to buffer.
    pub fn push(& mut self, value : T)
    {
        unsafe { * self.buffer.offset(self.write as isize) = value; }
        if self.write < self.len() { self.write += 1; } else { self.write = 0; }
    }
    ///Reads next data of the buffer.
    pub fn next(& mut self) -> T
    {
        let value = unsafe { *self.buffer.offset(self.read as isize) };
        if self.write < self.len() { self.read += 1; } else { self.read = 0; }
        value
    }
    ///Initializes write index.
    pub fn init_write(& mut self, index : usize) { self.write = index; }
    ///Initializes read index.
    pub fn init_read(& mut self, index : usize) { self.write = index; }
    ///Returns the length of the buffer.
    pub fn len(& self) -> usize { return self.len; }
}
impl<T: Copy> std::ops::Deref for CircularBuffer<T>
{
    type Target = [T];
    fn deref(&self) -> &Self::Target { self.into_slice() }
}
impl<T: Copy> std::ops::DerefMut for CircularBuffer<T> { fn deref_mut(&mut self) -> &mut Self::Target { self.into_slice_mut() } }
impl<T> Drop for CircularBuffer<T>
{
    fn drop(&mut self)
    {
        let layout = Layout::array::<T>(self.len as usize);
        match layout
        {
            Ok(layout) => { unsafe { dealloc(self.buffer as * mut u8, layout); } }
            Err(error) => { eprintln!("drop failed : {}", error); }
        }
    }
}