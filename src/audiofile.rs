//! Audio file loading and saving for WAV, BWF, and AIFF formats.
//!
//! This module provides functionality to read and write audio files in WAV, BWF
//! (Broadcast Wave Format), and AIFF formats. All audio data is stored internally
//! as normalized `f64` samples in the range -1.0 to 1.0.
//!
//! # Supported Formats
//!
//! - **WAV** - RIFF WAVE format with PCM encoding
//! - **BWF** - Broadcast Wave Format (WAV with bext metadata chunk)
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
//!
//! # BWF (Broadcast Wave Format) Support
//!
//! BWF extends the standard WAV format with professional broadcast metadata:
//!
//! - **bext chunk** - Broadcast extension metadata (description, originator, date, time, timecode)
//! - **Markers** - Cue points with labels for edit points, regions, and sync references
//! - **Tempo** - Musical tempo information for DAW integration
//!
//! ```ignore
//! use mkaudiolibrary::audiofile::{AudioFile, FileFormat, Marker};
//!
//! let mut audio = AudioFile::default();
//! audio.load("broadcast.wav");
//!
//! // Access BWF metadata
//! if let Some(bext) = audio.bext() {
//!     println!("Description: {}", bext.description);
//!     println!("Originator: {}", bext.originator);
//!     println!("Timecode: {}", bext.time_reference);
//! }
//!
//! // Access markers
//! for marker in audio.markers() {
//!     println!("Marker '{}' at sample {}", marker.label, marker.position);
//! }
//!
//! // Add a marker
//! audio.add_marker(Marker::new(44100, "Verse 1"));
//!
//! // Set tempo
//! audio.set_tempo(120.0);
//!
//! // Save as BWF (includes bext chunk)
//! audio.save_bwf("output.wav");
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

// ==========================================
// BWF (Broadcast Wave Format) Types
// ==========================================

/// Broadcast Extension (bext) chunk data for BWF files.
///
/// Contains metadata defined by the EBU Tech 3285 standard for broadcast audio.
/// This chunk is used in professional broadcast workflows to embed production
/// information directly in the audio file.
#[derive(Clone, Debug)]
pub struct BextChunk
{
    /// Free-form description of the audio content (max 256 characters).
    pub description : String,
    /// Name of the originator/creator (max 32 characters).
    pub originator : String,
    /// Unique reference identifier (max 32 characters).
    /// Typically format: CountryCode + OrganizationCode + SerialNumber.
    pub originator_reference : String,
    /// Origination date in format "yyyy-mm-dd" (10 characters).
    pub origination_date : String,
    /// Origination time in format "hh:mm:ss" (8 characters).
    pub origination_time : String,
    /// Sample count since midnight for timecode sync.
    /// Combined with sample rate, this gives the precise start time.
    pub time_reference : u64,
    /// BWF version number (typically 1 or 2).
    pub version : u16,
    /// SMPTE UMID (Unique Material Identifier), 64 bytes.
    pub umid : [u8; 64],
    /// Integrated loudness value (EBU R 128), in LUFS * 100.
    pub loudness_value : i16,
    /// Loudness range (EBU R 128), in LU * 100.
    pub loudness_range : i16,
    /// Maximum true peak level, in dBTP * 100.
    pub max_true_peak_level : i16,
    /// Maximum momentary loudness, in LUFS * 100.
    pub max_momentary_loudness : i16,
    /// Maximum short-term loudness, in LUFS * 100.
    pub max_short_term_loudness : i16,
    /// Free-form coding history string.
    /// Documents the signal chain and encoding history.
    pub coding_history : String,
}

impl Default for BextChunk
{
    fn default() -> Self
    {
        Self
        {
            description: String::new(),
            originator: String::new(),
            originator_reference: String::new(),
            origination_date: String::new(),
            origination_time: String::new(),
            time_reference: 0,
            version: 2,
            umid: [0u8; 64],
            loudness_value: 0,
            loudness_range: 0,
            max_true_peak_level: 0,
            max_momentary_loudness: 0,
            max_short_term_loudness: 0,
            coding_history: String::new(),
        }
    }
}

impl BextChunk
{
    /// Create a new empty BextChunk.
    pub fn new() -> Self { Self::default() }

    /// Create a BextChunk with basic metadata.
    ///
    /// # Arguments
    /// * `description` - Description of the audio content
    /// * `originator` - Name of the creator/organization
    pub fn with_description(description : &str, originator : &str) -> Self
    {
        Self
        {
            description: description.chars().take(256).collect(),
            originator: originator.chars().take(32).collect(),
            version: 2,
            ..Default::default()
        }
    }

    /// Set the origination date and time.
    ///
    /// # Arguments
    /// * `date` - Date string in "yyyy-mm-dd" format
    /// * `time` - Time string in "hh:mm:ss" format
    pub fn set_datetime(&mut self, date : &str, time : &str)
    {
        self.origination_date = date.chars().take(10).collect();
        self.origination_time = time.chars().take(8).collect();
    }

    /// Set current date and time as origination timestamp.
    pub fn set_current_datetime(&mut self)
    {
        // Simple placeholder - in production, use chrono or time crate
        self.origination_date = String::from("2025-01-01");
        self.origination_time = String::from("00:00:00");
    }
}

/// A marker (cue point) in an audio file.
///
/// Markers identify specific positions in the audio for editing, synchronization,
/// or navigation purposes. They are stored in the `cue ` and `LIST` chunks.
#[derive(Clone, Debug)]
pub struct Marker
{
    /// Unique identifier for this marker.
    pub id : u32,
    /// Sample position of the marker (0-indexed).
    pub position : u64,
    /// Label/name for the marker.
    pub label : String,
}

impl Marker
{
    /// Create a new marker at the specified position.
    ///
    /// # Arguments
    /// * `position` - Sample position (0-indexed)
    /// * `label` - Descriptive label for the marker
    pub fn new(position : u64, label : &str) -> Self
    {
        Self
        {
            id: 0,
            position,
            label: label.to_string(),
        }
    }

    /// Create a marker with a specific ID.
    pub fn with_id(id : u32, position : u64, label : &str) -> Self
    {
        Self
        {
            id,
            position,
            label: label.to_string(),
        }
    }
}

impl Default for Marker
{
    fn default() -> Self
    {
        Self
        {
            id: 0,
            position: 0,
            label: String::new(),
        }
    }
}

/// Tempo information for audio files.
///
/// Used in DAW workflows to synchronize audio with musical time.
/// The position field allows tempo markers to be placed at specific sample positions,
/// enabling tempo changes throughout the audio file.
#[derive(Clone, Debug)]
pub struct TempoInfo
{
    /// Tempo in beats per minute.
    pub bpm : f64,
    /// Time signature numerator (e.g., 4 for 4/4).
    pub time_sig_numerator : u8,
    /// Time signature denominator (e.g., 4 for 4/4).
    pub time_sig_denominator : u8,
    /// Sample position where this tempo starts (0 for file start).
    pub position : u64,
}

impl Default for TempoInfo
{
    fn default() -> Self
    {
        Self
        {
            bpm: 120.0,
            time_sig_numerator: 4,
            time_sig_denominator: 4,
            position: 0,
        }
    }
}

impl TempoInfo
{
    /// Create tempo info with the specified BPM at position 0.
    pub fn new(bpm : f64) -> Self
    {
        Self { bpm, ..Default::default() }
    }

    /// Create tempo info with BPM at a specific sample position.
    pub fn at_position(bpm : f64, position : u64) -> Self
    {
        Self { bpm, position, ..Default::default() }
    }

    /// Create tempo info with BPM and time signature at position 0.
    pub fn with_time_signature(bpm : f64, numerator : u8, denominator : u8) -> Self
    {
        Self
        {
            bpm,
            time_sig_numerator: numerator,
            time_sig_denominator: denominator,
            position: 0,
        }
    }

    /// Create tempo info with BPM, time signature, and position.
    pub fn with_time_signature_at(bpm : f64, numerator : u8, denominator : u8, position : u64) -> Self
    {
        Self
        {
            bpm,
            time_sig_numerator: numerator,
            time_sig_denominator: denominator,
            position,
        }
    }
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
    bit_depth : usize,
    // BWF (Broadcast Wave Format) fields
    bext_chunk : Option<BextChunk>,
    markers : Vec<Marker>,
    tempo : Option<TempoInfo>,
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
            bit_depth: 16,
            bext_chunk: None,
            markers: Vec::new(),
            tempo: None,
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

    // ==========================================
    // BWF (Broadcast Wave Format) Methods
    // ==========================================

    /// Get the BWF broadcast extension (bext) chunk if present.
    ///
    /// # Returns
    /// `Some(&BextChunk)` if the file contains BWF metadata, `None` otherwise.
    pub fn bext(&self) -> Option<&BextChunk> { self.bext_chunk.as_ref() }

    /// Get a mutable reference to the BWF broadcast extension chunk.
    ///
    /// # Returns
    /// `Some(&mut BextChunk)` if the file contains BWF metadata, `None` otherwise.
    pub fn bext_mut(&mut self) -> Option<&mut BextChunk> { self.bext_chunk.as_mut() }

    /// Set the BWF broadcast extension chunk.
    ///
    /// # Arguments
    /// * `bext` - The BextChunk to set
    pub fn set_bext(&mut self, bext : BextChunk) { self.bext_chunk = Some(bext); }

    /// Remove the BWF broadcast extension chunk.
    pub fn clear_bext(&mut self) { self.bext_chunk = None; }

    /// Check if this file has BWF metadata.
    pub fn is_bwf(&self) -> bool { self.bext_chunk.is_some() }

    /// Get all markers in the audio file.
    ///
    /// # Returns
    /// A slice of all markers, sorted by position.
    pub fn markers(&self) -> &[Marker] { &self.markers }

    /// Get a mutable reference to the markers.
    pub fn markers_mut(&mut self) -> &mut Vec<Marker> { &mut self.markers }

    /// Add a marker at the specified position.
    ///
    /// # Arguments
    /// * `marker` - The marker to add
    pub fn add_marker(&mut self, mut marker : Marker)
    {
        // Assign a unique ID if not set
        if marker.id == 0
        {
            marker.id = self.markers.iter().map(|m| m.id).max().unwrap_or(0) + 1;
        }
        self.markers.push(marker);
        self.markers.sort_by_key(|m| m.position);
    }

    /// Remove a marker by ID.
    ///
    /// # Arguments
    /// * `id` - The marker ID to remove
    ///
    /// # Returns
    /// `true` if the marker was found and removed, `false` otherwise.
    pub fn remove_marker(&mut self, id : u32) -> bool
    {
        if let Some(pos) = self.markers.iter().position(|m| m.id == id)
        {
            self.markers.remove(pos);
            true
        }
        else { false }
    }

    /// Remove all markers.
    pub fn clear_markers(&mut self) { self.markers.clear(); }

    /// Get the tempo information if present.
    ///
    /// # Returns
    /// `Some(&TempoInfo)` if tempo is set, `None` otherwise.
    pub fn tempo(&self) -> Option<&TempoInfo> { self.tempo.as_ref() }

    /// Set the tempo in BPM at position 0.
    ///
    /// # Arguments
    /// * `bpm` - Beats per minute
    pub fn set_tempo(&mut self, bpm : f64) { self.tempo = Some(TempoInfo::new(bpm)); }

    /// Set the tempo in BPM at a specific sample position.
    ///
    /// # Arguments
    /// * `bpm` - Beats per minute
    /// * `position` - Sample position where this tempo starts
    pub fn set_tempo_at(&mut self, bpm : f64, position : u64)
    {
        self.tempo = Some(TempoInfo::at_position(bpm, position));
    }

    /// Set tempo with time signature at position 0.
    ///
    /// # Arguments
    /// * `bpm` - Beats per minute
    /// * `numerator` - Time signature numerator
    /// * `denominator` - Time signature denominator
    pub fn set_tempo_with_time_sig(&mut self, bpm : f64, numerator : u8, denominator : u8)
    {
        self.tempo = Some(TempoInfo::with_time_signature(bpm, numerator, denominator));
    }

    /// Set tempo with time signature at a specific sample position.
    ///
    /// # Arguments
    /// * `bpm` - Beats per minute
    /// * `numerator` - Time signature numerator
    /// * `denominator` - Time signature denominator
    /// * `position` - Sample position where this tempo starts
    pub fn set_tempo_with_time_sig_at(&mut self, bpm : f64, numerator : u8, denominator : u8, position : u64)
    {
        self.tempo = Some(TempoInfo::with_time_signature_at(bpm, numerator, denominator, position));
    }

    /// Clear the tempo information.
    pub fn clear_tempo(&mut self) { self.tempo = None; }

    /// Save audio file as BWF (Broadcast Wave Format).
    ///
    /// This saves the file as a WAV with the bext chunk included.
    /// If no bext chunk exists, one is created with default values.
    ///
    /// # Arguments
    /// * `path` - Destination file path
    pub fn save_bwf(&mut self, path : &str)
    {
        // Ensure we have a bext chunk
        if self.bext_chunk.is_none()
        {
            self.bext_chunk = Some(BextChunk::new());
        }
        self.save_wav_internal(path, true);
    }

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
        // Read iXML chunk
        if index_of_xmlchunk > 0
        {
            let chunk_size = get_u32(buffer, index_of_xmlchunk + 4, Endianness::Little) as usize;
            if let Ok(chunk) = String::from_utf8(buffer[index_of_xmlchunk + 8..index_of_xmlchunk + 8 + chunk_size].to_vec())
            {
                self.xml_chunk = chunk;
            }
        }

        // Read BWF bext chunk
        let index_of_bext = get_index_of_chunk(buffer, "bext", 12, Endianness::Little);
        if index_of_bext > 0
        {
            self.bext_chunk = Some(read_bext_chunk(buffer, index_of_bext));
        }

        // Read cue chunk (markers)
        let index_of_cue = get_index_of_chunk(buffer, "cue ", 12, Endianness::Little);
        if index_of_cue > 0
        {
            self.markers = read_cue_chunk(buffer, index_of_cue);

            // Try to read marker labels from LIST/adtl chunk
            let index_of_list = get_index_of_chunk(buffer, "LIST", 12, Endianness::Little);
            if index_of_list > 0
            {
                read_marker_labels(buffer, index_of_list, &mut self.markers);
            }
        }

        // Read tempo from acid chunk (used by many DAWs)
        let index_of_acid = get_index_of_chunk(buffer, "acid", 12, Endianness::Little);
        if index_of_acid > 0
        {
            self.tempo = read_acid_chunk(buffer, index_of_acid);
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
        self.save_wav_internal(path, false);
    }

    fn save_wav_internal(&self, path : &str, include_bwf : bool)
    {
        let mut buffer = vec![];

        let data_chunk_size = self.num_sample() * self.num_channel() * self.bit_depth / 8;
        let audio_format = WavAudioFormat::PCM;
        let format_chunk_size = 16;
        let i_xmlchunk_size = self.xml_chunk.len();

        // Calculate BWF chunk sizes
        let bext_chunk_size = if include_bwf && self.bext_chunk.is_some()
        {
            let bext = self.bext_chunk.as_ref().unwrap();
            602 + bext.coding_history.len()  // Fixed header + coding history
        }
        else { 0 };

        let (cue_chunk_size, list_chunk_size) = if !self.markers.is_empty()
        {
            let cue_size = 4 + self.markers.len() * 24;  // num_cues + cue entries
            let mut list_size = 4;  // "adtl"
            for marker in &self.markers
            {
                if !marker.label.is_empty()
                {
                    let label_len = marker.label.len() + 1;  // +1 for null terminator
                    let padded_len = if label_len % 2 == 1 { label_len + 1 } else { label_len };
                    list_size += 8 + 4 + padded_len;  // chunk header + cue id + label
                }
            }
            (cue_size, if list_size > 4 { list_size } else { 0 })
        }
        else { (0, 0) };

        let acid_chunk_size = if self.tempo.is_some() { 24 } else { 0 };

        set_string(&mut buffer, "RIFF");
        let mut file_size_in_bytes = 4 + format_chunk_size + 8 + 8 + data_chunk_size;
        if i_xmlchunk_size > 0 { file_size_in_bytes += 8 + i_xmlchunk_size; }
        if bext_chunk_size > 0 { file_size_in_bytes += 8 + bext_chunk_size; }
        if cue_chunk_size > 0 { file_size_in_bytes += 8 + cue_chunk_size; }
        if list_chunk_size > 0 { file_size_in_bytes += 8 + list_chunk_size; }
        if acid_chunk_size > 0 { file_size_in_bytes += 8 + acid_chunk_size; }

        set_u32(&mut buffer, file_size_in_bytes as u32, Endianness::Little);
        set_string(&mut buffer, "WAVE");

        // Write bext chunk (BWF) - should come early in the file
        if bext_chunk_size > 0
        {
            write_bext_chunk(&mut buffer, self.bext_chunk.as_ref().unwrap());
        }

        // Write fmt chunk
        set_string(&mut buffer, "fmt ");
        set_u32(&mut buffer, format_chunk_size as u32, Endianness::Little);
        set_u16(&mut buffer, audio_format.to_num() as u16, Endianness::Little);
        set_u16(&mut buffer, self.num_channel() as u16, Endianness::Little);
        set_u32(&mut buffer, self.sample_rate as u32, Endianness::Little);
        set_u32(&mut buffer, (self.num_channel() * self.sample_rate * self.bit_depth / 8) as u32, Endianness::Little);
        set_u16(&mut buffer, (self.num_channel() * (self.bit_depth / 8)) as u16, Endianness::Little);
        set_u16(&mut buffer, self.bit_depth as u16, Endianness::Little);

        // Write cue chunk (markers)
        if cue_chunk_size > 0
        {
            write_cue_chunk(&mut buffer, &self.markers);
        }

        // Write LIST/adtl chunk (marker labels)
        if list_chunk_size > 0
        {
            write_list_adtl_chunk(&mut buffer, &self.markers);
        }

        // Write acid chunk (tempo)
        if acid_chunk_size > 0
        {
            write_acid_chunk(&mut buffer, self.tempo.as_ref().unwrap(), self.num_sample(), self.sample_rate);
        }

        // Write data chunk
        set_string(&mut buffer, "data");
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

        // Write iXML chunk
        if i_xmlchunk_size > 0
        {
            set_string(&mut buffer, "iXML");
            set_u32(&mut buffer, i_xmlchunk_size as u32, Endianness::Little);
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
            bit_depth: 16,
            bext_chunk: None,
            markers: Vec::new(),
            tempo: None,
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

#[inline]
fn get_u64(buffer : &[u8], start : usize, endianness : Endianness) -> u64
{
    if buffer.len() >= (start + 8)
    {
        return match endianness
        {
            Endianness::Big =>
            {
                ((buffer[start + 7] as u64) << 56) | ((buffer[start + 6] as u64) << 48) |
                ((buffer[start + 5] as u64) << 40) | ((buffer[start + 4] as u64) << 32) |
                ((buffer[start + 3] as u64) << 24) | ((buffer[start + 2] as u64) << 16) |
                ((buffer[start + 1] as u64) << 8) | buffer[start] as u64
            },
            Endianness::Little =>
            {
                ((buffer[start] as u64) << 56) | ((buffer[start + 1] as u64) << 48) |
                ((buffer[start + 2] as u64) << 40) | ((buffer[start + 3] as u64) << 32) |
                ((buffer[start + 4] as u64) << 24) | ((buffer[start + 5] as u64) << 16) |
                ((buffer[start + 6] as u64) << 8) | buffer[start + 7] as u64
            },
        }
    }
    0
}

#[inline]
fn set_u64(buffer : &mut Vec<u8>, data : u64, endianness : Endianness)
{
    let mut bytes = [0u8; 8];

    match endianness
    {
        Endianness::Big =>
        {
            bytes[0] = ((data >> 56) & 0xFF) as u8;
            bytes[1] = ((data >> 48) & 0xFF) as u8;
            bytes[2] = ((data >> 40) & 0xFF) as u8;
            bytes[3] = ((data >> 32) & 0xFF) as u8;
            bytes[4] = ((data >> 24) & 0xFF) as u8;
            bytes[5] = ((data >> 16) & 0xFF) as u8;
            bytes[6] = ((data >> 8) & 0xFF) as u8;
            bytes[7] = (data & 0xFF) as u8;
        },
        Endianness::Little =>
        {
            bytes[7] = ((data >> 56) & 0xFF) as u8;
            bytes[6] = ((data >> 48) & 0xFF) as u8;
            bytes[5] = ((data >> 40) & 0xFF) as u8;
            bytes[4] = ((data >> 32) & 0xFF) as u8;
            bytes[3] = ((data >> 24) & 0xFF) as u8;
            bytes[2] = ((data >> 16) & 0xFF) as u8;
            bytes[1] = ((data >> 8) & 0xFF) as u8;
            bytes[0] = (data & 0xFF) as u8;
        },
    }
    buffer.extend_from_slice(&bytes);
}

// ==========================================
// BWF Reading Helper Functions
// ==========================================

/// Read a fixed-length string from buffer, trimming null bytes.
#[inline]
fn read_fixed_string(buffer : &[u8], start : usize, len : usize) -> String
{
    if start + len > buffer.len() { return String::new(); }
    String::from_utf8_lossy(&buffer[start..start + len])
        .trim_end_matches('\0')
        .to_string()
}

/// Write a fixed-length string to buffer, padding with null bytes.
#[inline]
fn write_fixed_string(buffer : &mut Vec<u8>, string : &str, len : usize)
{
    let bytes = string.as_bytes();
    let write_len = bytes.len().min(len);
    buffer.extend_from_slice(&bytes[..write_len]);
    // Pad with zeros
    for _ in write_len..len { buffer.push(0); }
}

/// Read BWF bext chunk from buffer.
fn read_bext_chunk(buffer : &[u8], index : usize) -> BextChunk
{
    let _chunk_size = get_u32(buffer, index + 4, Endianness::Little) as usize;
    let data_start = index + 8;

    let mut bext = BextChunk::new();

    // Fixed-size fields according to EBU Tech 3285
    bext.description = read_fixed_string(buffer, data_start, 256);
    bext.originator = read_fixed_string(buffer, data_start + 256, 32);
    bext.originator_reference = read_fixed_string(buffer, data_start + 288, 32);
    bext.origination_date = read_fixed_string(buffer, data_start + 320, 10);
    bext.origination_time = read_fixed_string(buffer, data_start + 330, 8);

    // Time reference (sample count since midnight) - 8 bytes, little-endian
    bext.time_reference = get_u64(buffer, data_start + 338, Endianness::Little);

    // Version - 2 bytes
    bext.version = get_u16(buffer, data_start + 346, Endianness::Little);

    // UMID - 64 bytes
    if data_start + 412 <= buffer.len()
    {
        bext.umid.copy_from_slice(&buffer[data_start + 348..data_start + 412]);
    }

    // Loudness values (BWF version 2) - 10 bytes total
    if bext.version >= 2 && data_start + 422 <= buffer.len()
    {
        bext.loudness_value = get_u16(buffer, data_start + 412, Endianness::Little) as i16;
        bext.loudness_range = get_u16(buffer, data_start + 414, Endianness::Little) as i16;
        bext.max_true_peak_level = get_u16(buffer, data_start + 416, Endianness::Little) as i16;
        bext.max_momentary_loudness = get_u16(buffer, data_start + 418, Endianness::Little) as i16;
        bext.max_short_term_loudness = get_u16(buffer, data_start + 420, Endianness::Little) as i16;
    }

    // Coding history starts at offset 602 (after 180 reserved bytes)
    let coding_history_start = data_start + 602;
    if coding_history_start < buffer.len()
    {
        let chunk_end = index + 8 + _chunk_size;
        if chunk_end <= buffer.len()
        {
            bext.coding_history = read_fixed_string(buffer, coding_history_start, chunk_end - coding_history_start);
        }
    }

    bext
}

/// Read cue chunk (markers) from buffer.
fn read_cue_chunk(buffer : &[u8], index : usize) -> Vec<Marker>
{
    let mut markers = Vec::new();
    let data_start = index + 8;

    // Number of cue points
    let num_cue_points = get_u32(buffer, data_start, Endianness::Little) as usize;

    // Each cue point is 24 bytes
    for i in 0..num_cue_points
    {
        let cue_start = data_start + 4 + i * 24;
        if cue_start + 24 > buffer.len() { break; }

        let id = get_u32(buffer, cue_start, Endianness::Little);
        let position = get_u32(buffer, cue_start + 4, Endianness::Little) as u64;
        // Bytes 8-11: data chunk ID (usually "data")
        // Bytes 12-15: chunk start
        // Bytes 16-19: block start
        let sample_offset = get_u32(buffer, cue_start + 20, Endianness::Little) as u64;

        markers.push(Marker
        {
            id,
            position: position + sample_offset,
            label: String::new(),
        });
    }

    markers
}

/// Read marker labels from LIST/adtl chunk.
fn read_marker_labels(buffer : &[u8], index : usize, markers : &mut [Marker])
{
    let chunk_size = get_u32(buffer, index + 4, Endianness::Little) as usize;
    let data_start = index + 8;

    // Check if this is an "adtl" list
    if data_start + 4 > buffer.len() { return; }
    let list_type = read_fixed_string(buffer, data_start, 4);
    if list_type != "adtl" { return; }

    let mut pos = data_start + 4;
    let chunk_end = index + 8 + chunk_size;

    while pos + 8 < chunk_end && pos + 8 < buffer.len()
    {
        let sub_chunk_id = read_fixed_string(buffer, pos, 4);
        let sub_chunk_size = get_u32(buffer, pos + 4, Endianness::Little) as usize;

        if sub_chunk_id == "labl" || sub_chunk_id == "note"
        {
            let cue_id = get_u32(buffer, pos + 8, Endianness::Little);
            let label_len = sub_chunk_size.saturating_sub(4);
            let label = read_fixed_string(buffer, pos + 12, label_len);

            // Find and update the matching marker
            if let Some(marker) = markers.iter_mut().find(|m| m.id == cue_id)
            {
                marker.label = label;
            }
        }

        pos += 8 + sub_chunk_size;
        // Word alignment
        if sub_chunk_size % 2 == 1 { pos += 1; }
    }
}

/// Read acid chunk for tempo information.
fn read_acid_chunk(buffer : &[u8], index : usize) -> Option<TempoInfo>
{
    let data_start = index + 8;

    // acid chunk structure:
    // 4 bytes: type flags
    // 2 bytes: root note
    // 2 bytes: unknown
    // 4 bytes: unknown
    // 4 bytes: num beats
    // 2 bytes: meter denominator
    // 2 bytes: meter numerator
    // 4 bytes: tempo (float)

    if data_start + 24 > buffer.len() { return None; }

    let tempo_bits = get_u32(buffer, data_start + 20, Endianness::Little);
    let tempo = f32::from_bits(tempo_bits) as f64;

    if tempo > 0.0 && tempo < 1000.0
    {
        let numerator = get_u16(buffer, data_start + 18, Endianness::Little) as u8;
        let denominator = get_u16(buffer, data_start + 16, Endianness::Little) as u8;

        Some(TempoInfo
        {
            bpm: tempo,
            time_sig_numerator: if numerator > 0 { numerator } else { 4 },
            time_sig_denominator: if denominator > 0 { denominator } else { 4 },
            position: 0,  // acid chunk doesn't store position, default to file start
        })
    }
    else { None }
}

// ==========================================
// BWF Writing Helper Functions
// ==========================================

/// Write BWF bext chunk to buffer.
fn write_bext_chunk(buffer : &mut Vec<u8>, bext : &BextChunk)
{
    let chunk_size = 602 + bext.coding_history.len();

    set_string(buffer, "bext");
    set_u32(buffer, chunk_size as u32, Endianness::Little);

    // Fixed-size fields according to EBU Tech 3285
    write_fixed_string(buffer, &bext.description, 256);
    write_fixed_string(buffer, &bext.originator, 32);
    write_fixed_string(buffer, &bext.originator_reference, 32);
    write_fixed_string(buffer, &bext.origination_date, 10);
    write_fixed_string(buffer, &bext.origination_time, 8);

    // Time reference (8 bytes)
    set_u64(buffer, bext.time_reference, Endianness::Little);

    // Version (2 bytes)
    set_u16(buffer, bext.version, Endianness::Little);

    // UMID (64 bytes)
    buffer.extend_from_slice(&bext.umid);

    // Loudness values (10 bytes)
    set_u16(buffer, bext.loudness_value as u16, Endianness::Little);
    set_u16(buffer, bext.loudness_range as u16, Endianness::Little);
    set_u16(buffer, bext.max_true_peak_level as u16, Endianness::Little);
    set_u16(buffer, bext.max_momentary_loudness as u16, Endianness::Little);
    set_u16(buffer, bext.max_short_term_loudness as u16, Endianness::Little);

    // Reserved (180 bytes)
    for _ in 0..180 { buffer.push(0); }

    // Coding history (variable length)
    set_string(buffer, &bext.coding_history);
}

/// Write cue chunk (markers) to buffer.
fn write_cue_chunk(buffer : &mut Vec<u8>, markers : &[Marker])
{
    let chunk_size = 4 + markers.len() * 24;

    set_string(buffer, "cue ");
    set_u32(buffer, chunk_size as u32, Endianness::Little);

    // Number of cue points
    set_u32(buffer, markers.len() as u32, Endianness::Little);

    // Cue points (24 bytes each)
    for marker in markers
    {
        set_u32(buffer, marker.id, Endianness::Little);           // ID
        set_u32(buffer, marker.position as u32, Endianness::Little);  // Position
        set_string(buffer, "data");                               // Data chunk ID
        set_u32(buffer, 0, Endianness::Little);                   // Chunk start
        set_u32(buffer, 0, Endianness::Little);                   // Block start
        set_u32(buffer, 0, Endianness::Little);                   // Sample offset
    }
}

/// Write LIST/adtl chunk (marker labels) to buffer.
fn write_list_adtl_chunk(buffer : &mut Vec<u8>, markers : &[Marker])
{
    // Calculate total size
    let mut list_size = 4;  // "adtl"
    for marker in markers
    {
        if !marker.label.is_empty()
        {
            let label_len = marker.label.len() + 1;  // +1 for null terminator
            let padded_len = if label_len % 2 == 1 { label_len + 1 } else { label_len };
            list_size += 8 + 4 + padded_len;  // chunk header + cue id + label
        }
    }

    if list_size <= 4 { return; }

    set_string(buffer, "LIST");
    set_u32(buffer, list_size as u32, Endianness::Little);
    set_string(buffer, "adtl");

    // Write label sub-chunks
    for marker in markers
    {
        if !marker.label.is_empty()
        {
            let label_len = marker.label.len() + 1;
            let padded_len = if label_len % 2 == 1 { label_len + 1 } else { label_len };

            set_string(buffer, "labl");
            set_u32(buffer, (4 + padded_len) as u32, Endianness::Little);
            set_u32(buffer, marker.id, Endianness::Little);
            set_string(buffer, &marker.label);
            buffer.push(0);  // Null terminator
            if label_len % 2 == 1 { buffer.push(0); }  // Padding byte
        }
    }
}

/// Write acid chunk (tempo) to buffer.
fn write_acid_chunk(buffer : &mut Vec<u8>, tempo : &TempoInfo, num_samples : usize, sample_rate : usize)
{
    set_string(buffer, "acid");
    set_u32(buffer, 24, Endianness::Little);  // Chunk size

    // Type flags (4 bytes) - 0x01 = one-shot, 0x02 = root note valid, etc.
    set_u32(buffer, 0, Endianness::Little);

    // Root note (2 bytes) - MIDI note number
    set_u16(buffer, 60, Endianness::Little);  // Middle C

    // Unknown (2 bytes)
    set_u16(buffer, 0, Endianness::Little);

    // Unknown (4 bytes)
    set_u32(buffer, 0, Endianness::Little);

    // Number of beats (4 bytes)
    let duration_seconds = num_samples as f64 / sample_rate as f64;
    let num_beats = (duration_seconds * tempo.bpm / 60.0) as u32;
    set_u32(buffer, num_beats, Endianness::Little);

    // Time signature (4 bytes)
    set_u16(buffer, tempo.time_sig_denominator as u16, Endianness::Little);
    set_u16(buffer, tempo.time_sig_numerator as u16, Endianness::Little);

    // Tempo as float (4 bytes)
    set_u32(buffer, (tempo.bpm as f32).to_bits(), Endianness::Little);
}