use std::alloc::{alloc_zeroed, dealloc, Layout, LayoutError};

/// Sized Buffer that can be used in multi-threaded environment, with reference count and lock.
pub struct Buffer<T : Clone + Default + Send + Sync>
{
    element : * mut T,
    len : *mut usize,
    lock : *mut bool,
    locked_here : bool,
    count : *mut usize,
    default : T
}
impl<T : Clone + Default + Send + Sync> Buffer<T>
{
    /// New Buffer with length.
    pub fn new(len : usize) -> Self
    {
        unsafe
        {
            let array_layout = std::alloc::Layout::array::<T>(len).unwrap();
            Self
            {
                element : std::alloc::alloc_zeroed(array_layout) as * mut T,
                len : std::alloc::alloc_zeroed(std::alloc::Layout::new::<usize>()) as * mut usize,
                lock : std::alloc::alloc_zeroed(std::alloc::Layout::new::<bool>()) as * mut bool,
                locked_here : false,
                count : std::alloc::alloc_zeroed(std::alloc::Layout::new::<usize>()) as * mut usize,
                default : T::default()
            }
        }
    }
    /// New Buffer from raw pointer.
    pub fn from_raw(ptr : * mut T, len : usize) -> Self
    {
        let lock = std::alloc::alloc_zeroed(std::alloc::Layout::new::<bool>()) as *mut bool;
        unsafe
        {
            *lock = false;
            Self { buffer : ptr, len : &mut len, lock : lock, count : std::alloc::alloc_zeroed(std::alloc::Layout::new::<usize>()) as *mut usize, default : T::default() }
        }
    }
    /// Try to lock in time. True if success and false if failed.
    pub fn try_lock(&mut self) -> bool
    {
        unsafe
        {
            if !*self.lock
            {
                *self.lock = true;
                self.locked_here = true;
                return true
            }
            false
        }
    }
    /// Lock the buffer.
    pub fn lock(&mut self)
    {
        unsafe
        {
            while *self.lock { std::thread::yield_now(); }
            if !*self.lock
            {
                *self.lock = true;
                self.locked_here = true;
            }
        }    
    }
    /// unlock the buffer.
    pub fn unlock(&mut self)
    {
        if self.locked_here
        {
            unsafe { *self.lock = false; }
            self.locked_here = false;
        }
    }
}
impl<T : Clone + Default + Send + Sync> std::ops::Index<usize> for Buffer<T>
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output { unsafe { self.element.offset((index % *self.len) as isize).as_ref().expect("No value assigned.") } }
}
impl<T : Clone + Default + Send + Sync> std::ops::IndexMut<usize> for Buffer<T>
{
    fn index_mut(& mut self, index: usize) -> & mut Self::Output
    {
        unsafe
        {
            if *self.lock && self.locked_here { return self.element.offset((index % *self.len) as isize).as_mut().expect("No value assigned.") }
            else if !self.locked_here
            {
                eprintln!("Buffer was locked somewhere else.");
                return &mut self.default
            }
            eprintln!("Buffer was not locked.");
            &mut self.default
        }
    }
}
impl<T : Clone + Default + Send + Sync> AsRef<[T]> for Buffer<T> { fn as_ref(&self) -> &[T] { unsafe { std::slice::from_raw_parts(self.element, *self.len) } } }
impl<T : Clone + Default + Send + Sync> AsMut<[T]> for Buffer<T>
{
    fn as_mut(&mut self) -> &mut [T]
    {
        unsafe
        {
            if *self.lock && self.locked_here { return std::slice::from_raw_parts_mut(self.element, *self.len) }
            else if !self.locked_here
            {
                eprintln!("Buffer was locked somewhere else.");
                return std::slice::from_mut(&mut self.default)
            }
            eprintln!("Buffer was not locked.");
            std::slice::from_mut(&mut self.default)
        }
    }
}
impl<T : Clone + Default + Send + Sync> std::ops::Deref for Buffer<T>
{
    type Target = [T];
    fn deref(&self) -> &Self::Target { self.as_ref() }
}
impl<T : Clone + Default + Send + Sync> std::ops::DerefMut for Buffer<T> { fn deref_mut(&mut self) -> &mut Self::Target { self.as_mut() } }
impl<T : Clone + Default + Send + Sync> Clone for Buffer<T>
{
    fn clone(&self) -> Self
    {
        unsafe 
        {
            *self.count += 1;
            Self
            {
                element : self.element,
                len : self.len,
                lock : self.lock,
                locked_here : false,
                count : self.count,
                default : T::default()
            }
        }
    }
}
unsafe impl<T : Clone + Default + Send + Sync> Send for Buffer<T> {}
unsafe impl<T : Clone + Default + Send + Sync> Sync for Buffer<T> {}
impl<T : Clone + Default + Send + Sync> Drop for Buffer<T>
{
    fn drop(&mut self)
    {
        unsafe
        {
            if *self.count > 0
            {
                if *self.locked_here { *self.lock = false; }
                *self.count -= 1;
                return
            }
            let array_layout = std::alloc::Layout::array::<T>(*self.len).unwrap();
            std::alloc::dealloc(self.element as *mut u8, array_layout);
            std::alloc::dealloc(self.len as *mut u8, std::alloc::Layout::new::<usize>());
            std::alloc::dealloc(self.lock as *mut u8, std::alloc::Layout::new::<bool>());
            std::alloc::dealloc(self.count as *mut u8, std::alloc::Layout::new::<usize>());
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
