//! Audio file loading and saving for WAV and AIFF formats.
//!
//! This module provides functionality to read and write audio files in WAV and AIFF
//! formats. All audio data is stored internally as normalized `f64` samples in the
//! range -1.0 to 1.0.
//!
//! # Supported Formats
//!
//! - **WAV** - RIFF WAVE format with PCM encoding
//! - **AIFF** - Audio Interchange File Format (uncompressed and compressed/AIFC)
//!
//! # Supported Bit Depths
//!
//! Both reading and writing support:
//! - 8-bit (signed integer)
//! - 16-bit (signed integer)
//! - 24-bit (signed integer)
//! - 32-bit (signed integer or IEEE float for WAV)
//!
//! # Sample Representation
//!
//! Regardless of the source file's bit depth, samples are stored internally as
//! normalized `f64` values:
//! - Full scale positive: +1.0
//! - Full scale negative: -1.0
//! - Silence: 0.0
//!
//! When saving, samples are clamped to the -1.0 to 1.0 range before quantization.
//!
//! # Buffer Integration
//!
//! The `AudioFile` struct integrates with the thread-safe `Buffer<T>` type for
//! DSP processing pipelines:
//!
//! ```ignore
//! use mkaudiolibrary::audiofile::{AudioFile, FileFormat};
//! use mkaudiolibrary::dsp::Compression;
//!
//! // Load audio file
//! let mut audio = AudioFile::default();
//! audio.load("input.wav");
//!
//! // Convert to buffers for processing
//! let mut buffers = audio.to_buffers();
//!
//! // Process with DSP (e.g., compression)
//! let mut compressor = Compression::new(audio.sample_rate() as f64);
//! // ... apply processing ...
//!
//! // Copy processed data back
//! audio.from_buffers(&buffers);
//!
//! // Save result
//! audio.save("output.wav", FileFormat::Wav);
//! ```
//!
//! # Example Usage
//!
//! ## Loading and Inspecting
//!
//! ```ignore
//! use mkaudiolibrary::audiofile::AudioFile;
//!
//! let mut audio = AudioFile::default();
//! audio.load("song.wav");
//!
//! println!("Channels: {}", audio.num_channel());
//! println!("Samples: {}", audio.num_sample());
//! println!("Sample rate: {} Hz", audio.sample_rate());
//! println!("Bit depth: {}", audio.bit_depth());
//! println!("Duration: {:.2} seconds", audio.length());
//! ```
//!
//! ## Creating and Saving
//!
//! ```ignore
//! use mkaudiolibrary::audiofile::{AudioFile, FileFormat};
//!
//! // Create stereo file with 44100 samples (1 second at 44.1kHz)
//! let mut audio = AudioFile::new(2, 44100);
//! audio.set_sample_rate(44100);
//! audio.set_bit_depth(16);
//!
//! // Generate a 440Hz sine wave on left channel
//! for i in 0..44100 {
//!     let t = i as f64 / 44100.0;
//!     let sample = (2.0 * std::f64::consts::PI * 440.0 * t).sin();
//!     audio.set_sample(0, i, sample);
//! }
//!
//! audio.save("sine.wav", FileFormat::Wav);
//! ```
//!
//! ## Direct Channel Access
//!
//! ```ignore
//! use mkaudiolibrary::audiofile::AudioFile;
//!
//! let mut audio = AudioFile::default();
//! audio.load("input.wav");
//!
//! // Read-only access to channel data
//! if let Some(left) = audio.channel(0) {
//!     let peak = left.iter().fold(0.0_f64, |max, &s| max.max(s.abs()));
//!     println!("Peak level: {:.4}", peak);
//! }
//!
//! // Mutable access for in-place processing
//! if let Some(left) = audio.channel_mut(0) {
//!     for sample in left.iter_mut() {
//!         *sample *= 0.5; // Apply -6dB gain
//!     }
//! }
//! ```

// Lookup table for AIFF extended precision sample rate encoding.
// Maps common sample rates to their 80-bit IEEE 754 extended precision representation.
const AIFF_SAMPLE_RATE_TABLE : [(usize, [u8;10]); 19] =
[
    (8000, [64, 11, 250, 0, 0, 0, 0, 0, 0, 0]),
    (11025, [64, 12, 172, 68, 0, 0, 0, 0, 0, 0]),
    (16000, [64, 12, 250, 0, 0, 0, 0, 0, 0, 0]),
    (22050, [64, 13, 172, 68, 0, 0, 0, 0, 0, 0]),
    (32000, [64, 13, 250, 0, 0, 0, 0, 0, 0, 0]),
    (37800, [64, 14, 147, 168, 0, 0, 0, 0, 0, 0]),
    (44056, [64, 14, 172, 24, 0, 0, 0, 0, 0, 0]),
    (44100, [64, 14, 172, 68, 0, 0, 0, 0, 0, 0]),
    (47250, [64, 14, 184, 146, 0, 0, 0, 0, 0, 0]),
    (48000, [64, 14, 187, 128, 0, 0, 0, 0, 0, 0]),
    (50000, [64, 14, 195, 80, 0, 0, 0, 0, 0, 0]),
    (50400, [64, 14, 196, 224, 0, 0, 0, 0, 0, 0]),
    (88200, [64, 15, 172, 68, 0, 0, 0, 0, 0, 0]),
    (96000, [64, 15, 187, 128, 0, 0, 0, 0, 0, 0]),
    (176400, [64, 16, 172, 68, 0, 0, 0, 0, 0, 0]),
    (192000, [64, 16, 187, 128, 0, 0, 0, 0, 0, 0]),
    (352800, [64, 17, 172, 68, 0, 0, 0, 0, 0, 0]),
    (2822400, [64, 20, 172, 68, 0, 0, 0, 0, 0, 0]),
    (5644800, [64, 21, 172, 68, 0, 0, 0, 0, 0, 0])
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum WavAudioFormat
{
    PCM,
    IEEEFloat,
    ALaw,
    MULaw,
    Extensible
}
impl WavAudioFormat
{
    fn from_num(num : usize) -> Option<Self>
    {
        if num == 0x0001 { return Some(Self::PCM) }
        else if num == 0x0003 { return Some(Self::IEEEFloat) }
        else if num == 0x0006 { return Some(Self::ALaw) }
        else if num == 0x0007 { return Some(Self::MULaw) }
        else if num == 0xFFFE { return Some(Self::Extensible) }
        None
    }
    fn to_num(self) -> usize
    {
        match self
        {
            WavAudioFormat::PCM => 0x0001,
            WavAudioFormat::IEEEFloat => 0x0003,
            WavAudioFormat::ALaw => 0x0006,
            WavAudioFormat::MULaw => 0x0007,
            WavAudioFormat::Extensible => 0xFFFE,
        }
    } 
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AIFFAudioFormat
{
    Uncompressed,
    Compressed,
    Error
}

/// Audio file format identifier.
///
/// Used to specify the format when saving files and to identify the format
/// of loaded files.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FileFormat
{
    /// No valid format detected (parsing failed or unknown format).
    None,
    /// File has not been loaded yet (default state).
    NotLoaded,
    /// WAV format (RIFF WAVE).
    Wav,
    /// AIFF format (Audio Interchange File Format).
    Aiff
}
impl FileFormat
{
    fn determine(data : &[u8]) -> Self
    {
        if let Ok(header) = String::from_utf8(data[0..4].to_vec())
        {
            if header == "RIFF" { return Self::Wav }
            else if header == "Form" { return Self::Aiff }
        }
        eprintln!("ERROR: Failed to determine audio format.");
        Self::None
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Endianness
{
    Big,
    Little
}

use crate::buffer::Buffer;

/// Audio file container for loading, manipulating, and saving audio data.
///
/// `AudioFile` provides a unified interface for working with WAV and AIFF audio files.
/// Audio data is stored as normalized `f64` samples regardless of the source file's
/// bit depth, enabling consistent processing across different formats.
///
/// # Structure
///
/// Audio data is organized as a vector of channels, where each channel contains
/// a vector of samples. Channel 0 is typically left, channel 1 is right for stereo files.
///
/// # Thread Safety
///
/// `AudioFile` itself is not thread-safe. For concurrent processing, convert channels
/// to thread-safe `Buffer<f64>` using [`to_buffer`] or [`to_buffers`], process in
/// parallel, then copy results back with [`from_buffer`] or [`from_buffers`].
///
/// # iXML Metadata
///
/// The `xml_chunk` field provides access to iXML metadata embedded in audio files.
/// This is commonly used for broadcast metadata in professional audio workflows.
///
/// [`to_buffer`]: AudioFile::to_buffer
/// [`to_buffers`]: AudioFile::to_buffers
/// [`from_buffer`]: AudioFile::from_buffer
/// [`from_buffers`]: AudioFile::from_buffers
pub struct AudioFile
{
    audio_buffer : Vec<Vec<f64>>,
    /// iXML metadata chunk from the audio file.
    /// Empty string if no iXML chunk is present.
    pub xml_chunk : String,
    file_format : FileFormat,
    sample_rate : usize,
    bit_depth : usize
}
impl AudioFile
{
    /// Create a new empty audio file with the given channel count and sample count.
    ///
    /// All samples are initialized to 0.0 (silence). Default sample rate is 44100 Hz
    /// and default bit depth is 16-bit.
    ///
    /// # Arguments
    /// * `channels` - Number of audio channels (1 for mono, 2 for stereo, etc.)
    /// * `samples` - Number of samples per channel
    pub fn new(channels : usize, samples : usize) -> Self
    {
        Self
        {
            audio_buffer: vec![vec![0.0; samples]; channels],
            xml_chunk: String::new(),
            file_format: FileFormat::None,
            sample_rate: 44100,
            bit_depth: 16
        }
    }

    /// Load audio file from a file path.
    ///
    /// Automatically detects the file format (WAV or AIFF) based on the file header.
    /// On success, populates all audio data and metadata. On failure, prints an error
    /// message to stderr.
    ///
    /// # Arguments
    /// * `path` - Path to the audio file
    pub fn load(&mut self, path : &str)
    {
        if let Ok(mut file) = std::fs::File::open(path)
        {
            let mut buffer = vec![];
            if let Err(error) = std::io::Read::read_to_end(&mut file, &mut buffer) { eprintln!("{}", error); }
            else { self.load_bytes(&buffer); }
        }
        else { eprintln!("ERROR: Failed to open file: {}", path); }
    }

    /// Save audio file to the specified path in the given format.
    ///
    /// Samples are clamped to -1.0 to 1.0 before quantization. The file is written
    /// with the current bit depth and sample rate settings.
    ///
    /// # Arguments
    /// * `path` - Destination file path
    /// * `format` - Output format (`FileFormat::Wav` or `FileFormat::Aiff`)
    pub fn save(&self, path : &str, format : FileFormat)
    {
        match format
        {
            FileFormat::Wav => self.save_wav(path),
            FileFormat::Aiff => self.save_aiff(path),
            _ => {}
        }
    }

    /// Load audio file from a byte slice.
    ///
    /// Useful for loading audio from memory buffers or embedded resources.
    /// Automatically detects the file format based on the header bytes.
    ///
    /// # Arguments
    /// * `data` - Raw file bytes
    pub fn load_bytes(&mut self, data : &[u8])
    {
        self.file_format = FileFormat::determine(data);
        match self.file_format
        {
            FileFormat::Wav => self.read_wav(data),
            FileFormat::Aiff => self.read_aiff(data),
            _ => {}
        }
    }

    /// Get a read-only slice of samples for a specific channel.
    ///
    /// # Arguments
    /// * `index` - Channel index (0 = left, 1 = right for stereo)
    ///
    /// # Returns
    /// `Some(&[f64])` if the channel exists, `None` otherwise.
    pub fn channel(&self, index : usize) -> Option<&[f64]>
    {
        self.audio_buffer.get(index).map(|v| v.as_slice())
    }

    /// Get a mutable slice of samples for a specific channel.
    ///
    /// Allows direct modification of sample data for in-place processing.
    ///
    /// # Arguments
    /// * `index` - Channel index (0 = left, 1 = right for stereo)
    ///
    /// # Returns
    /// `Some(&mut [f64])` if the channel exists, `None` otherwise.
    pub fn channel_mut(&mut self, index : usize) -> Option<&mut [f64]>
    {
        self.audio_buffer.get_mut(index).map(|v| v.as_mut_slice())
    }

    /// Get a single sample value at the specified channel and sample index.
    ///
    /// # Arguments
    /// * `channel` - Channel index
    /// * `sample` - Sample index within the channel
    ///
    /// # Returns
    /// `Some(f64)` if the indices are valid, `None` otherwise.
    pub fn get_sample(&self, channel : usize, sample : usize) -> Option<f64>
    {
        self.audio_buffer.get(channel).and_then(|c| c.get(sample).copied())
    }

    /// Set a single sample value at the specified channel and sample index.
    ///
    /// Silently ignores invalid indices.
    ///
    /// # Arguments
    /// * `channel` - Channel index
    /// * `sample` - Sample index within the channel
    /// * `value` - Sample value (typically in range -1.0 to 1.0)
    pub fn set_sample(&mut self, channel : usize, sample : usize, value : f64)
    {
        if let Some(ch) = self.audio_buffer.get_mut(channel)
        {
            if let Some(s) = ch.get_mut(sample) { *s = value; }
        }
    }

    /// Convert a single channel to a thread-safe `Buffer` for DSP processing.
    ///
    /// Creates a new `Buffer` containing a copy of the channel data. The buffer
    /// can be safely shared across threads for parallel processing.
    ///
    /// # Arguments
    /// * `channel` - Channel index to convert
    ///
    /// # Returns
    /// `Some(Buffer<f64>)` if the channel exists, `None` otherwise.
    pub fn to_buffer(&self, channel : usize) -> Option<Buffer<f64>>
    {
        self.audio_buffer.get(channel).map(|c| Buffer::from_slice(c))
    }

    /// Copy processed data from a `Buffer` back to a channel.
    ///
    /// Copies up to the minimum of the channel length and buffer length.
    /// Use this after processing audio with DSP components.
    ///
    /// # Arguments
    /// * `channel` - Channel index to copy to
    /// * `buffer` - Source buffer containing processed samples
    pub fn from_buffer(&mut self, channel : usize, buffer : &Buffer<f64>)
    {
        if let Some(ch) = self.audio_buffer.get_mut(channel)
        {
            let guard = buffer.read();
            let len = ch.len().min(guard.len());
            ch[..len].copy_from_slice(&guard[..len]);
        }
    }

    /// Convert all channels to thread-safe `Buffer` instances.
    ///
    /// Useful for parallel processing of multi-channel audio. Each buffer
    /// is independent and can be processed in a separate thread.
    ///
    /// # Returns
    /// A vector of `Buffer<f64>`, one per channel.
    pub fn to_buffers(&self) -> Vec<Buffer<f64>>
    {
        self.audio_buffer.iter().map(|c| Buffer::from_slice(c)).collect()
    }

    /// Copy processed data from `Buffer` instances back to all channels.
    ///
    /// Pairs buffers with channels by index. If there are fewer buffers than
    /// channels, only the first N channels are updated.
    ///
    /// # Arguments
    /// * `buffers` - Slice of buffers containing processed samples
    pub fn from_buffers(&mut self, buffers : &[Buffer<f64>])
    {
        for (channel, buffer) in self.audio_buffer.iter_mut().zip(buffers.iter())
        {
            let guard = buffer.read();
            let len = channel.len().min(guard.len());
            channel[..len].copy_from_slice(&guard[..len]);
        }
    }

    /// Get the number of channels in the audio file.
    pub fn num_channel(&self) -> usize { self.audio_buffer.len() }

    /// Get the number of samples per channel.
    pub fn num_sample(&self) -> usize { if self.audio_buffer.is_empty() { 0 } else { self.audio_buffer[0].len() } }

    /// Check if the audio file is mono (single channel).
    pub fn is_mono(&self) -> bool { self.audio_buffer.len() == 1 }

    /// Check if the audio file is stereo (two channels).
    pub fn is_stereo(&self) -> bool { self.audio_buffer.len() == 2 }

    /// Get the bit depth of the audio file.
    pub fn bit_depth(&self) -> usize { self.bit_depth }

    /// Get the sample rate of the audio file in Hz.
    pub fn sample_rate(&self) -> usize { self.sample_rate }

    /// Get the duration of the audio file in seconds.
    pub fn length(&self) -> f64 { self.num_sample() as f64 / self.sample_rate as f64 }

    /// Get the file format of the loaded audio file.
    pub fn format(&self) -> FileFormat { self.file_format }

    /// Resize the audio buffer to the specified channel and sample count.
    ///
    /// New samples are initialized to 0.0. Existing samples are preserved
    /// if the new size is larger.
    ///
    /// # Arguments
    /// * `channel` - New number of channels
    /// * `sample` - New number of samples per channel
    pub fn set_buffer_size(&mut self, channel : usize, sample : usize)
    {
        self.audio_buffer.resize(channel, vec![0.0; sample]);
        for channel in &mut self.audio_buffer { channel.resize(sample, 0.0); }
    }

    /// Set the number of channels, preserving existing sample count.
    ///
    /// # Arguments
    /// * `count` - New number of channels
    pub fn set_channels(&mut self, count : usize) { self.audio_buffer.resize(count, vec![0.0; self.num_sample()]); }

    /// Set the number of samples per channel, preserving existing channel count.
    ///
    /// # Arguments
    /// * `count` - New number of samples per channel
    pub fn set_samples(&mut self, count : usize) { for buffer in &mut self.audio_buffer { buffer.resize(count, 0.0); } }

    /// Set the bit depth for saving (8, 16, 24, or 32).
    ///
    /// # Arguments
    /// * `bit_depth` - Bit depth value
    pub fn set_bit_depth(&mut self, bit_depth : usize) { self.bit_depth = bit_depth; }

    /// Set the sample rate in Hz.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate value (e.g., 44100, 48000, 96000)
    pub fn set_sample_rate(&mut self, sample_rate : usize) { self.sample_rate = sample_rate }
    fn read_wav(&mut self, buffer : &[u8])
    {
        if let Ok(header_chunk_id) = String::from_utf8(buffer[0..4].to_vec())
        {
            if header_chunk_id != "RIFF"
            {
                eprintln!("ERROR: Wrong header chunk id.");
                return
            }
        }
        if let Ok(format) = String::from_utf8(buffer[8..12].to_vec())
        {
            if format != "WAVE"
            {
                eprintln!("ERROR: Wrong format.");
                return
            }
        }
        let index_of_data_chunk = get_index_of_chunk(buffer, "data", 12, Endianness::Little);
        let index_of_format_chunk = get_index_of_chunk(buffer, "fmt ", 12, Endianness::Little);
        let index_of_xmlchunk = get_index_of_chunk(buffer, "iXML", 12, Endianness::Little);
        let _format_chunk_id = String::from_utf8(buffer[index_of_format_chunk..index_of_format_chunk + 4].to_vec());
        let _format_chunk_size = get_u32(buffer, index_of_format_chunk + 4, Endianness::Little) as usize;
        let audio_format = WavAudioFormat::from_num(get_u16(buffer, index_of_format_chunk + 8, Endianness::Little) as usize);
        let num_channels = get_u16(buffer, index_of_format_chunk + 10, Endianness::Little) as usize;
        self.sample_rate = get_u32(buffer, index_of_format_chunk + 12, Endianness::Little) as usize;
        let num_bytes_per_second = get_u32(buffer, index_of_format_chunk + 16, Endianness::Little) as usize;
        let num_bytes_per_block = get_u16(buffer, index_of_format_chunk + 20, Endianness::Little) as usize;
        self.bit_depth = get_u16(buffer, index_of_format_chunk + 22, Endianness::Little) as usize;
        
        if self.bit_depth > size_of::<f64>() * 8
        {
            eprintln!("ERROR: you are trying to read a {}-bit file using a {}-bit sample type", self.bit_depth, size_of::<f64>() * 8);
            return
        }
        if audio_format.is_none()
        {
            eprintln!("ERROR: this .WAV file is encoded in a format that this library does not support at present");
            return
        }
        if num_channels < 1 || num_channels > 128
        {
            eprintln!("ERROR: this WAV file seems to be an invalid number of channels (or corrupted?)");
            return
        }
        if num_bytes_per_second != num_channels * self.sample_rate * self.bit_depth / 8 || num_bytes_per_block != num_channels * num_bytes_per_second
        {
            eprintln!("ERROR: the header data in this WAV file seems to be inconsistent");
            return
        }
        if self.bit_depth != 8 && self.bit_depth != 16 && self.bit_depth != 24 && self.bit_depth != 32
        {
            eprintln!("ERROR: this file has a bit depth that is not 8, 16, 24 or 32 bits");
            return
        }
        let num_bytes_per_sample = self.bit_depth / 8;

        let _data_chunk_id = String::from_utf8(buffer[index_of_data_chunk..index_of_data_chunk+ 4].to_vec());
        let data_chunk_size = get_u32(buffer, index_of_data_chunk + 4, Endianness::Little) as usize;
        let num_samples = data_chunk_size / (num_channels * self.bit_depth / 8);
        let samples_start_index = index_of_data_chunk + 8;
        
        self.audio_buffer.clear();
        self.audio_buffer.resize(num_channels, vec![]);

        for index in 0..num_samples
        {
            for channel in 0..num_channels
            {
                let sample_index = samples_start_index + num_bytes_per_block * index + channel * num_bytes_per_sample;
            
                if sample_index + (self.bit_depth / 8) - 1 >= buffer.len()
                {
                    eprintln!("ERROR: read file error as the metadata indicates more samples than there are in the file data");
                    return
                }
                
                if self.bit_depth == 8 { self.audio_buffer[channel].push(buffer[sample_index].cast_signed() as f64 / i8::MAX as f64); }
                else if self.bit_depth == 16
                {
                    let sample = get_u16(buffer, sample_index, Endianness::Little).cast_signed();
                    let sample = sample as f64 / i16::MAX as f64;
                    self.audio_buffer[channel].push(sample);
                }
                else if self.bit_depth == 24
                {
                    let mut sample = (((buffer[sample_index + 2] as u32) << 16) | ((buffer[sample_index + 1] as u32) << 8) | buffer[sample_index] as u32).cast_signed();
                    if sample & 0x800000 == 0 { sample = sample | !0xFFFFFF };
                    self.audio_buffer[channel].push(sample as f64 / 8388607.0);
                }
                else if self.bit_depth == 32
                {
                    let sample = get_u32(buffer, sample_index, Endianness::Little);
                    if audio_format.unwrap() == WavAudioFormat::IEEEFloat { self.audio_buffer[channel].push(f32::from_bits(sample) as f64); }
                    else { self.audio_buffer[channel].push(sample.cast_signed() as f64 / i32::MAX as f64); }
                }
                else
                {
                    eprintln!("ERROR: Wrong bit depth detected.");
                    return;
                }
            }
        }
        let chunk_size = get_u32(buffer, index_of_xmlchunk + 4, Endianness::Little) as usize;
        match String::from_utf8(buffer[index_of_xmlchunk + 8..index_of_xmlchunk + 8 + chunk_size].to_vec())
        {
            Ok(chunk) => { self.xml_chunk = chunk }
            Err(error) => eprintln!("{}", error)
        }
    }
    fn read_aiff(&mut self, buffer : &[u8])
    {
        if let Ok(header_chunk_id) = String::from_utf8(buffer[0..4].to_vec())
        {
            if header_chunk_id != "FORM"
            {
                eprintln!("ERROR: Wrong header chunk id.");
                return
            }
        }
        let audio_format = if let Ok(format) = String::from_utf8(buffer[8..12].to_vec())
        {
            if format == "AIFF" { AIFFAudioFormat::Uncompressed } else if format == "AIFC" { AIFFAudioFormat::Compressed } else { AIFFAudioFormat::Error }
        }
        else
        {
            eprintln!("ERROR: Wrong format.");
            return
        };
        let index_of_comm_chunk = get_index_of_chunk(buffer, "COMM", 12, Endianness::Big);
        let index_of_sound_data_chunk = get_index_of_chunk(buffer, "SSND", 12, Endianness::Big);
        let index_of_xmlchunk = get_index_of_chunk(buffer, "iXML", 12, Endianness::Big);
        
        if index_of_sound_data_chunk == 0 || index_of_comm_chunk == 0 || audio_format == AIFFAudioFormat::Error
        {
            eprintln!("ERROR: this doesn't seem to be a valid AIFF file");
            return
        }

        let _comm_chunk_id  = String::from_utf8(buffer[index_of_comm_chunk..index_of_comm_chunk + 4].to_vec());
        let _comm_chunk_size = get_u32(buffer, index_of_comm_chunk + 4, Endianness::Big) as usize;
        let num_channels = get_u16(buffer, index_of_comm_chunk + 8, Endianness::Big) as usize;
        let num_samples_per_channel = get_u32(buffer, index_of_comm_chunk + 10, Endianness::Big) as usize;
        
        self.bit_depth = get_u16(buffer, index_of_comm_chunk + 14, Endianness::Big) as usize;
        self.sample_rate = get_aiff_sample_rate(buffer, index_of_comm_chunk + 16);
        
        if self.bit_depth > size_of::<f64>() * 8
        {
            eprintln!("ERROR: you are trying to read a {}-bit file using a {}-bit sample type", self.bit_depth, size_of::<f64>() * 8);
            return
        }
        if self.sample_rate == 0
        {
            eprintln!("ERROR: this AIFF file has an unsupported sample rate");
            return
        }
        if num_channels < 1 ||num_channels > 2
        {
            eprintln!("ERROR: this AIFF file seems to be neither mono nor stereo (perhaps multi-track, or corrupted?)");
            return
        }
        if self.bit_depth != 8 && self.bit_depth != 16 && self.bit_depth != 24 && self.bit_depth != 32
        {
            eprintln!("ERROR: this file has a bit depth that is not 8, 16, 24 or 32 bits");
            return
        }
        let _sound_data_chunk_id =  String::from_utf8(buffer[index_of_sound_data_chunk..index_of_sound_data_chunk + 4].to_vec());
        let sound_data_chunk_size = get_u32(buffer, index_of_sound_data_chunk + 4, Endianness::Big) as usize;
        let offset = get_u32(buffer, index_of_sound_data_chunk + 8, Endianness::Big) as usize;
        let _block_size = get_u32(buffer, index_of_sound_data_chunk + 12, Endianness::Big) as usize;
        let num_bytes_per_sample = self.bit_depth / 8;
        let num_bytes_per_frame = num_bytes_per_sample * num_channels;
        let total_num_audio_sample_bytes = num_samples_per_channel * num_bytes_per_frame;
        let samples_start_index = index_of_sound_data_chunk + 16 + offset;
            
        if sound_data_chunk_size - 8 != total_num_audio_sample_bytes || total_num_audio_sample_bytes > buffer.len() - samples_start_index
        {
            eprintln!("ERROR: the metadatafor this file doesn't seem right");
            return
        }
        self.audio_buffer.clear();
        self.audio_buffer.resize(num_channels, vec![]);

        for index in 0..num_samples_per_channel
        {
            for channel in 0..num_channels
            {
                let sample_index = samples_start_index + (num_bytes_per_frame * index) + channel * num_bytes_per_sample;
            
                if sample_index + self.bit_depth / 8 - 1 >= buffer.len()
                {
                    eprintln!("ERROR: read file error as the metadata indicates more samples than there are in the file data");
                    return
                }
                
                if self.bit_depth == 8 { self.audio_buffer[channel].push(buffer[sample_index].cast_signed() as f64 / i8::MAX as f64); }
                else if self.bit_depth == 16 { self.audio_buffer[channel].push(get_u16(buffer, sample_index, Endianness::Big) as f64 / u16::MAX as f64); }
                else if self.bit_depth == 24
                {
                    let mut sample = ((buffer[sample_index] as i32) << 16) | ((buffer[sample_index + 1] as i32) << 8) | buffer[sample_index + 2] as i32;
                    
                    if sample & 0x800000 == 0 { sample = sample | !0xFFFFFF; }
                    self.audio_buffer[channel].push(sample as f64 / 8388607.0);
                }
                else if self.bit_depth == 32
                {
                    let sample = get_u32(buffer, sample_index, Endianness::Big);
                    
                    if audio_format == AIFFAudioFormat::Compressed { self.audio_buffer[channel].push(f32::from_bits(sample) as f64); }
                    else { self.audio_buffer[channel].push(sample.cast_signed() as f64 / i32::MAX as f64) }
                }
                else
                {
                    eprintln!("ERROR: Wrong bit depth detected.");
                    return;
                }
            }
        }
        let chunk_size = get_u32(buffer, index_of_xmlchunk + 4, Endianness::Little) as usize;
        if let Ok(xml) = String::from_utf8(buffer[index_of_xmlchunk + 8..index_of_xmlchunk + 8 + chunk_size].to_vec()) { self.xml_chunk = xml; }
    }
    fn save_wav(&self, path : &str)
    {
        let mut buffer = vec![];

        let data_chunk_size = self.num_sample() * self.num_channel() * self.bit_depth / 8;
        let audio_format =  WavAudioFormat::PCM;
        let format_chunk_size = 16;
        let i_xmlchunk_size = self.xml_chunk.len();

        set_string(&mut buffer, "RIFF");
        let mut file_size_in_bytes = 4 + format_chunk_size + 8 + 8 + data_chunk_size;
        if i_xmlchunk_size > 0 { file_size_in_bytes += 8 + i_xmlchunk_size; }
        set_u32(&mut buffer, file_size_in_bytes as u32, Endianness::Little);
        set_string(&mut buffer, "WAVE");
        set_string(&mut buffer, "fmt ");
        set_u32(&mut buffer, format_chunk_size as u32, Endianness::Little);
        set_u16(&mut buffer, audio_format.to_num() as u16, Endianness::Little);
        set_u16(&mut buffer, self.num_channel() as u16, Endianness::Little);
        set_u32(&mut buffer, self.sample_rate as u32, Endianness::Little);
        set_u32(&mut buffer, (self.num_channel() * self.sample_rate * self.bit_depth / 8) as u32, Endianness::Little);
        set_u16(&mut buffer, (self.num_channel() * (self.bit_depth / 8)) as u16, Endianness::Little);
        set_u16(&mut buffer, self.bit_depth as u16, Endianness::Little);
        set_string (&mut buffer, "data");
        set_u32(&mut buffer, data_chunk_size as u32, Endianness::Little);

        for index in 0..self.num_sample()
        {
            for channel in 0..self.num_channel()
            {
                let sample = self.audio_buffer[channel][index].clamp(-1.0, 1.0);
                if self.bit_depth == 8 { buffer.push(((sample * i8::MAX as f64) as i8).cast_unsigned()); }
                else if self.bit_depth == 16
                {
                    set_u16(&mut buffer, ((sample * i16::MAX as f64) as i16).cast_unsigned(), Endianness::Little);
                }
                else if self.bit_depth == 24
                {
                    let mut bytes = [0;3];
                    let sample = (sample * 8388607.0) as i32;

                    bytes[2] = (sample >> 16 & 0xFF)  as u8;
                    bytes[1] = (sample >>  8 & 0xFF) as u8;
                    bytes[0] = (sample & 0xFF) as u8;
                    
                    buffer.extend_from_slice(&bytes);
                }
                else if self.bit_depth == 32
                {
                    set_u32(&mut buffer, ((sample * i32::MAX as f64) as i32).cast_unsigned(), Endianness::Little);
                }
                else
                {
                    eprintln!("ERROR: Trying to write a file with unsupported bit depth");
                    return;
                }
            }
        }
        if i_xmlchunk_size > 0
        {
            set_string(&mut buffer, "iXML");
            set_u32(&mut buffer, i_xmlchunk_size as u32, Endianness::Little);
            set_string(&mut buffer, &self.xml_chunk);
        }
        if file_size_in_bytes != buffer.len() - 8 || data_chunk_size != (self.num_sample() * self.num_channel() * self.bit_depth / 8)
        {
            eprintln!("ERROR: file size doesn't match");
            return
        }
        if let Ok(mut file) = std::fs::File::create(path)
        {
            if let Err(error) = std::io::Write::write(&mut file, &buffer)
            {
                eprintln!("ERROR: couldn't save file to {} from error : {}", path, error);
            }
        } else { eprintln!("ERROR: couldn't create file to {}", path); }
    }
    fn save_aiff(&self, path : &str)
    {
        let mut buffer = vec![];
    
        let num_bytes_per_sample = self.bit_depth / 8;
        let num_bytes_per_frame = num_bytes_per_sample * self.num_channel();
        let total_num_audio_sample_bytes = self.num_sample() * num_bytes_per_frame;
        let sound_data_chunk_size = total_num_audio_sample_bytes + 8;
        let i_xmlchunk_size = self.xml_chunk.len();
        
        set_string(&mut buffer, "FORM");
        let mut file_size_in_bytes = 4 + 26 + 16 + total_num_audio_sample_bytes;
        if i_xmlchunk_size > 0
        {
            file_size_in_bytes += 8 + i_xmlchunk_size;
        }
    
        set_u32(&mut buffer, file_size_in_bytes as u32, Endianness::Big);
    
        set_string(&mut buffer, "AIFF");
        set_string(&mut buffer, "COMM");
        set_u32(&mut buffer, 18, Endianness::Big);
        set_u16(&mut buffer, self.num_channel() as u16, Endianness::Big);
        set_u32(&mut buffer, self.num_sample() as u32, Endianness::Big);
        set_u16(&mut buffer, self.bit_depth as u16, Endianness::Big);
        set_aiff_sample_rate(&mut buffer, self.sample_rate);
        set_string(&mut buffer, "SSND");
        set_u32(&mut buffer, sound_data_chunk_size as u32, Endianness::Big);
        set_u32(&mut buffer, 0, Endianness::Big);
        set_u32(&mut buffer, 0, Endianness::Big);
        
        for index in 0..self.num_sample()
        {
            for channel in 0..self.num_channel()
            {
                let sample = self.audio_buffer[channel][index].clamp(-1.0, 1.0);
                if self.bit_depth == 8 { buffer.push(((sample * i8::MAX as f64) as i8).cast_unsigned()); }
                else if self.bit_depth == 16
                {
                    set_u16(&mut buffer, ((sample * i16::MAX as f64) as i16).cast_unsigned(), Endianness::Big);
                }
                else if self.bit_depth == 24
                {
                    let mut bytes = [0;3];
                    let sample = (sample * 8388607.0) as i32;

                    bytes[0] = (sample >> 16 & 0xFF) as u8;
                    bytes[1] = (sample >> 8 & 0xFF) as u8;
                    bytes[2] = (sample & 0xFF) as u8;
                    
                    buffer.extend(&bytes);
                }
                else if self.bit_depth == 32 { set_u32(&mut buffer, ((sample * i32::MAX as f64) as i32).cast_unsigned(), Endianness::Big); }
                else
                {
                    eprintln!("Trying to write a file with unsupported bit depth");
                    return
                }
            }
        }
        if i_xmlchunk_size > 0
        {
            set_string(&mut buffer, "iXML");
            set_u32(&mut buffer, i_xmlchunk_size as u32, Endianness::Big);
            set_string(&mut buffer, &self.xml_chunk);
        }
        if let Ok(mut file) = std::fs::File::create(path)
        {
            if let Err(error) = std::io::Write::write(&mut file, &buffer)
            {
                eprintln!("ERROR: couldn't save file to {} from error : {}", path, error);
            }
        } else { eprintln!("ERROR: couldn't create file to {}", path); }
    }
}
impl Default for AudioFile
{
    fn default() -> Self
    {
        Self
        {
            audio_buffer: vec![vec![]],
            xml_chunk: String::new(),
            file_format: FileFormat::NotLoaded,
            sample_rate: 44100,
            bit_depth: 16
        }
    }
}

#[inline]
fn ten_byte_match(buffer1 : &[u8], start1 : usize, buffer2 : &[u8], start2 : usize) -> bool
{
    for index in 0..10 { if buffer1[start1 + index] != buffer2[start2 + index] { return false } } 
    true
}

#[inline]
fn get_aiff_sample_rate(buffer : &[u8], start : usize) -> usize
{
    for table in &AIFF_SAMPLE_RATE_TABLE { if ten_byte_match(buffer, start, &table.1, 0) { return table.0 } }
    eprintln!("ERROR: Sample rate not detected.");
    0
}

#[inline]
fn set_aiff_sample_rate(buffer : &mut Vec<u8>, sample_rate : usize)
{
    for data in &AIFF_SAMPLE_RATE_TABLE
    {
        if data.0 == sample_rate
        {
            buffer.extend_from_slice(&data.1);
            return
        }
    }
    eprintln!("ERROR: Sample rate not matching.");
}

#[inline]
fn set_string(buffer : &mut Vec<u8>, string : &str) { buffer.extend_from_slice(string.as_bytes()); }

#[inline]
fn get_index_of_chunk(buffer : &[u8], chunk : &str, start : usize, endianness : Endianness) -> usize
{
    let datalen = 4;

    if chunk.len() != datalen
    {
        eprintln!("ERROR: Invalid chunk header ID string");
        return 0;
    }

    let mut index = start;
    while index < buffer.len() - datalen
    {
        if &buffer[index..index + datalen] == chunk.as_bytes() { return index }
        index += datalen;
        if (index + 4) >= buffer.len()
        {
            eprintln!("ERROR: Chunk header ID not found.");
            return 0;
        }
        let chunk_size = get_u32(buffer, index, endianness) as usize;
        index += datalen + chunk_size;
    }
    return 0;
}

#[inline]
fn get_u32(buffer : &[u8], start : usize, endianness : Endianness) -> u32
{
    if buffer.len() >= (start + 4)
    {
        return match endianness
        {
            Endianness::Big =>
            {
                ((buffer[start + 3] as u32) << 24) | ((buffer[start + 2] as u32) << 16) | ((buffer[start + 1] as u32) << 8) | buffer[start] as u32
            },
            Endianness::Little =>
            {
                ((buffer[start] as u32) << 24) | ((buffer[start + 1] as u32) << 16) | ((buffer[start + 2] as u32) << 8) | buffer[start + 3] as u32
            },
        }
    }
    eprintln!("ERROR: Insufficient buffer length.");
    0
}

#[inline]
fn set_u32(buffer : &mut Vec<u8>, data : u32, endianness : Endianness)
{
    let mut bytes = [0;4];

    match endianness
    {
        Endianness::Big =>
        {
            bytes[0] = ((data >> 24) & 0xFF) as u8;
            bytes[1] = ((data >> 16) & 0xFF) as u8;
            bytes[2] = ((data >> 8) & 0xFF) as u8;
            bytes[3] = (data & 0xFF) as u8;
        },
        Endianness::Little =>
        {
            bytes[3] = ((data >> 24) & 0xFF) as u8;
            bytes[2] = ((data >> 16) & 0xFF) as u8;
            bytes[1] = ((data >> 8) & 0xFF) as u8;
            bytes[0] = (data & 0xFF) as u8;
        },
    }
    buffer.extend_from_slice(&bytes);
}

#[inline]
fn get_u16(buffer : &[u8], start : usize, endianness : Endianness) -> u16
{
    if buffer.len() >= (start + 2)
    {
        return match endianness
        {
            Endianness::Big =>
            {
                ((buffer[start + 1] as u16) << 8) | buffer[start] as u16
            },
            Endianness::Little =>
            {
                ((buffer[start] as u16) << 8) | buffer[start + 1] as u16
            },
        }
    }
    eprintln!("ERROR: Insufficient buffer length.");
    0
}

#[inline]
fn set_u16(buffer : &mut Vec<u8>, data : u16, endianness : Endianness)
{
    let mut bytes = [0;2];

    match endianness
    {
        Endianness::Big =>
        {
            bytes[0] = ((data >> 8) & 0xFF) as u8;
            bytes[1] = (data & 0xFF) as u8;
        },
        Endianness::Little =>
        {
            bytes[1] = ((data >> 8) & 0xFF) as u8;
            bytes[0] = (data & 0xFF) as u8;
        },
    }
    buffer.extend_from_slice(&bytes);
}