use std::alloc::{alloc_zeroed, dealloc, Layout, LayoutError};

///A simple impliment of sized buffer.
#[derive(Clone)]
pub struct Buffer<T>
{
    buffer : * mut T,
    len : usize
}
impl<T> Buffer<T>
{
    ///New Buffer with length.
    #[inline]
    pub fn new(len : usize) -> Result<Self,LayoutError>
    {
        let layout = std::alloc::Layout::array::<T>(len);
        match layout
        {
            Ok(layout) =>
            {
                let buffer = unsafe { std::alloc::alloc_zeroed(layout) as * mut T };
                return Ok(Self { buffer, len });
            }
            Err(error) =>
            {
                return Err(error);
            }
        }
    }
    ///Resizes the buffer.
    #[inline]
    pub fn resize(&mut self, len : usize) -> Result<(), LayoutError>
    {
        let dealloc_layout = std::alloc::Layout::array::<T>(self.len)?;
        let alloc_layout = std::alloc::Layout::array::<T>(len)?;
        unsafe
        {
            std::alloc::dealloc(self.buffer as * mut u8, dealloc_layout);
            self.buffer = std::alloc::alloc_zeroed(alloc_layout) as * mut T;
        }
        return Ok(());
    }
    ///Converts internal data chunk as silce
    #[inline]
    pub fn into_slice(&self) -> &[T]
    {
        unsafe { std::slice::from_raw_parts(self.buffer, self.len) }
    }
    ///Converts internal data chunk as mutable silce
    #[inline]
    pub fn into_slice_mut(&self) -> &mut[T]
    {
        unsafe{ std::slice::from_raw_parts_mut(self.buffer, self.len) }
    }
    ///Returns the length of the buffer.
    #[inline]
    pub fn len(&self) -> usize { return self.len; }
}
impl<T> std::ops::Index<usize> for Buffer<T>
{
    type Output = T;

    #[inline]
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
    #[inline]
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
impl<T> Drop for Buffer<T>
{
    #[inline]
    fn drop(&mut self)
    {
        let layout = std::alloc::Layout::array::<T>(self.len);
        match layout
        {
            Ok(layout) => { unsafe { std::alloc::dealloc(self.buffer as * mut u8, layout) }; },
            Err(_) => {},
        }
        
    }
}

///The buffer that pushes the whole buffer when index meets the size of the buffer. Generic T must be either f32 or f64.
#[derive(Clone)]
pub struct PushBuffer<T>
{
    buffer : * mut T,
    pub index : usize,
    len : usize
}
impl<T : Copy> PushBuffer<T>
{
    ///New PushBuffer with length.
    #[inline]
    pub fn new(len : usize) -> Result<Self, LayoutError>
    {
        let layout = Layout::array::<T>(len)?;
        unsafe { Ok(PushBuffer { buffer : alloc_zeroed(layout) as * mut T , index : 0, len : len }) }
    }
    ///Resizes the buffer.
    #[inline]
    pub fn resize(&mut self, len : usize) -> Result<(), LayoutError>
    {
        let dealloc_layout = std::alloc::Layout::array::<T>(self.len)?;
        let alloc_layout = std::alloc::Layout::array::<T>(len)?;
        unsafe
        {
            std::alloc::dealloc(self.buffer as * mut u8, dealloc_layout);
            self.buffer = std::alloc::alloc_zeroed(alloc_layout) as * mut T;
        }
        return Ok(());
    }
    ///Converts internal data chunk as silce
    #[inline]
    pub fn into_slice(&self) -> &[T]
    {
        unsafe { std::slice::from_raw_parts(self.buffer, self.len) }
    }
    ///Converts internal data chunk as mutable silce
    #[inline]
    pub fn into_slice_mut(&self) -> &mut[T]
    {
        unsafe{ std::slice::from_raw_parts_mut(self.buffer, self.len) }
    }
    ///Pushes data to buffer.
    #[inline]
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
    ///Initialize index.
    #[inline]
    pub fn init_index(&mut self, index : usize) { self.index = index; }
    ///Returns the length of the buffer.
    #[inline]
    pub fn len(& self) -> usize { return self.len; }
}
impl<T> std::ops::Index<usize> for PushBuffer<T>
{
    type Output = T;

    #[inline]
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
    #[inline]
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
impl<T> Drop for PushBuffer<T>
{
    #[inline]
    fn drop(&mut self)
    {
        let layout = Layout::array::<T>(self.len as usize);
        match layout
        {
            Ok(layout) => { unsafe { dealloc(self.buffer as * mut u8, layout); } }
            Err(_) => {}
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
    #[inline]
    pub fn new(len : usize) -> Result<Self, LayoutError>
    {
        let layout = Layout::array::<T>(len)?;
        let buffer = unsafe { alloc_zeroed(layout) as * mut T };
        return Ok(CircularBuffer { buffer, read : 0, write : 0, len : len });
        
    }
    ///Resizes the buffer.
    #[inline]
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

        return Ok(());
    }
    ///Converts internal data chunk as silce
    #[inline]
    pub fn into_slice(&self) -> &[T]
    {
        unsafe { std::slice::from_raw_parts(self.buffer, self.len) }
    }
    ///Converts internal data chunk as mutable silce
    #[inline]
    pub fn into_slice_mut(&self) -> &mut[T]
    {
        unsafe{ std::slice::from_raw_parts_mut(self.buffer, self.len) }
    }
    ///Pushes data to buffer.
    #[inline]
    pub fn push(& mut self, value : T)
    {
        unsafe { * self.buffer.offset(self.write as isize) = value; }
        if self.write < self.len() { self.write += 1; } else { self.write = 0; }
    }
    ///Reads next data of the buffer.
    #[inline]
    pub fn next(& mut self) -> T
    {
        let value = unsafe { *self.buffer.offset(self.read as isize) };
        if self.write < self.len() { self.read += 1; } else { self.read = 0; }
        return value;
    }
    ///Initializes write index.
    #[inline]
    pub fn init_write(& mut self, index : usize) { self.write = index; }
    ///Initializes read index.
    #[inline]
    pub fn init_read(& mut self, index : usize) { self.write = index; }
    ///Returns the length of the buffer.
    #[inline]
    pub fn len(& self) -> usize { return self.len; }
}
impl<T> Drop for CircularBuffer<T>
{
    #[inline]
    fn drop(&mut self)
    {
        let layout = Layout::array::<T>(self.len as usize);
        match layout
        {
            Ok(layout) => { unsafe { dealloc(self.buffer as * mut u8, layout); } }
            Err(_) => {}
        }
    }
}