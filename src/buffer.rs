use std::alloc::{alloc_zeroed, dealloc, Layout};

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
    pub fn new(len : usize) -> Result<Self,()>
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
                eprintln!("{}", error);
                return Err(());
            }
        }
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
        let real = if index > self.len { index % self.len } else { index };
        let data = unsafe { self.buffer.offset(real as isize).as_ref() };
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
        let real = if index > self.len { index % self.len } else { index };
        let data = unsafe { self.buffer.offset(real as isize).as_mut() };
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
    pub index : isize,
    len : isize
}
impl<T> PushBuffer<T>
{
    ///New PushBuffer with length.
    #[inline]
    pub fn new(len : usize) -> Result<Self,()>
    {
        let layout = Layout::array::<T>(len);

        match layout
        {
            Ok(layout) =>
                {
                    let buffer = unsafe { alloc_zeroed(layout) as * mut T };
                    return Ok(PushBuffer { buffer, index : 0, len : len as isize });
                }
            Err(error) =>
            {
                eprintln!("{}", error);
                return Err(());
            }
        }
    }
}
impl<T : Copy> PushBuffer<T>
{
    ///Pushes data to buffer.
    #[inline]
    pub fn push(& mut self, value : T)
    {
        if self.index <= self.len
        {
            unsafe { * self.buffer.offset(self.index) = value; }
            self.index += 1;
        }
        else
        {
            unsafe
                {
                    (1..self.len).for_each(|x| * self.buffer.offset(x - 1) = * self.buffer.offset(x));
                    * self.buffer.offset(self.index) = value;
                }
        }
    }
    ///Returns the length of the buffer.
    #[inline]
    pub fn len(& self) -> isize { return self.len; }
}
impl<T> std::ops::Index<usize> for PushBuffer<T>
{
    type Output = T;

    #[inline]
    fn index(& self, index : usize) -> & Self::Output
    {
        let data = unsafe { self.buffer.offset(index as isize).as_ref() };
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
        let data = unsafe { self.buffer.offset(index as isize).as_mut() };
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
    read : isize,
    write : isize,
    len : isize
}
impl<T> CircularBuffer<T>
{
    ///New CircularBuffer with length.
    #[inline]
    pub fn new(len : usize) -> Result<Self, ()>
    {
        let layout = Layout::array::<T>(len);
        match layout
        {
            Ok(layout) =>
            {
                let buffer = unsafe { alloc_zeroed(layout) as * mut T };
                return Ok(CircularBuffer { buffer, read : 0, write : 0, len : len as isize });
            },
            Err(error) =>
            {
                eprintln!("{}", error);
                return Err(());
            },
        }
        
    }
}
impl<T : Copy> CircularBuffer<T>
{
    ///Pushes data to buffer.
    #[inline]
    pub fn push(& mut self, value : T)
    {
        unsafe { * self.buffer.offset(self.write) = value; }
        if self.write < self.len() { self.write += 1; } else { self.write = 0; }
    }
    ///Reads next data of the buffer.
    #[inline]
    pub fn next(& mut self) -> T
    {
        let value = unsafe { *self.buffer.offset(self.read) };
        if self.write < self.len() { self.read += 1; } else { self.read = 0; }
        return value;
    }
    ///Initializes write index.
    #[inline]
    pub fn init_write(& mut self, index : isize) { self.write = index; }
    ///Initializes read index.
    #[inline]
    pub fn init_read(& mut self, index : isize) { self.write = index; }
    ///Returns the length of the buffer.
    #[inline]
    pub fn len(& self) -> isize { return self.len; }
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