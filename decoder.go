package qoa

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"unsafe"
)

// DecodeHeader decodes the QOA header and initializes the QOA struct with header information.
func DecodeHeader(bytes []byte) (*QOA, error) {
	size := len(bytes)
	if size < QOAMinFilesize {
		return nil, errors.New("qoa: file too small")
	}

	// Read the file header, verify the magic number ('qoaf') and read the total number of samples.
	fileHeader := binary.BigEndian.Uint64(bytes)

	if (fileHeader >> 32) != QOAMagic {
		return nil, errors.New("qoa: invalid magic number")
	}

	Samples := uint32(fileHeader & 0xffffffff)
	if Samples == 0 {
		return nil, errors.New("qoa: no samples found")
	}

	// Peek into the first frame header to get the number of channels and the SampleRate.
	frameHeader := binary.BigEndian.Uint64(bytes[8:])
	Channels := uint32(frameHeader>>56) & 0xff
	SampleRate := uint32(frameHeader>>32) & 0xffffff

	if Channels == 0 || SampleRate == 0 {
		return nil, errors.New("qoa: first frame header is invalid")
	}

	return &QOA{
		Samples:    Samples,
		Channels:   Channels,
		SampleRate: SampleRate,
	}, nil
}

func (d *Decoder) decodeFrame() ([]int16, error) {
	// minimum size of the frame to read
	minSize := 8 + QOALMSLen*4*int(d.Header.Channels)

	buf := d.framebuf

	// try to read the full frame
	size, err := io.ReadFull(d.r, buf)
	switch {
	case errors.Is(err, io.EOF):
		// end of file reached, no more frames
		return nil, nil

	case errors.Is(err, io.ErrUnexpectedEOF) && size < minSize:
		return nil, fmt.Errorf("read frame: %w", err)

	case errors.Is(err, io.ErrUnexpectedEOF) && size >= minSize:
		// all good

	case err != nil:
		return nil, fmt.Errorf("read frame: %w", err)
	}

	// Decode and verify the frame header
	frameHeader := binary.BigEndian.Uint64(buf[:8])
	buf = buf[8:]

	channels := uint32((frameHeader >> 56) & 0x000000FF)
	sampleRate := uint32((frameHeader >> 32) & 0x00FFFFFF)
	sampleCount := uint32((frameHeader >> 16) & 0x0000FFFF)
	frameSize := uint(frameHeader & 0x0000FFFF)

	dataSize := int(frameSize) - 8 - QOALMSLen*4*int(channels)
	numSlices := dataSize / 8
	maxTotalSamples := numSlices * QOASliceLen

	if channels != d.Header.Channels ||
		sampleRate != d.Header.SampleRate ||
		frameSize > uint(size) ||
		int(sampleCount*channels) > maxTotalSamples {
		return nil, errors.New("invalid header")
	}

	samples := d.samples[:sampleCount*d.Header.Channels]

	lms := [8]qoaLMS{}

	// Read the LMS state: 4 x 2 bytes history and 4 x 2 bytes weights per channel
	for c := uint32(0); c < channels; c++ {
		history := binary.BigEndian.Uint64(buf[0:])
		buf = buf[8:]

		weights := binary.BigEndian.Uint64(buf[0:])
		buf = buf[8:]

		for i := 0; i < QOALMSLen; i++ {
			lms[c].History[i] = int16(history >> 48)
			history <<= 16
			lms[c].Weights[i] = int16(weights >> 48)
			weights <<= 16
		}
	}

	// Decode slices for all channels in this frame
	for sampleIndex := uint32(0); sampleIndex < sampleCount; sampleIndex += QOASliceLen {
		for c := uint32(0); c < channels; c++ {
			slice := binary.BigEndian.Uint64(buf[0:])
			buf = buf[8:]

			scaleFactor := (slice >> 60) & 0xF
			slice <<= 4
			sliceStart := sampleIndex*channels + c
			sliceEnd := uint32(clamp(int(sampleIndex)+QOASliceLen, 0, int(sampleCount)))*channels + c

			for si := sliceStart; si < sliceEnd; si += channels {
				predicted := lms[c].predict()
				quantized := int((slice >> 61) & 0x7)
				dequantized := qoaDequantTable[scaleFactor][quantized]
				reconstructed := clampS16(predicted + int(dequantized))

				samples[si] = reconstructed
				slice <<= 3

				lms[c].update(reconstructed, dequantized)
			}
		}
	}

	return samples, nil
}

// DecodeAll decodes the provided QOA encoded bytes and returns the QOA struct and the decoded audio sample data.
// This decodes a full QOA file directly into a slice of samples.
func DecodeAll(r io.Reader) (*QOA, []int16, error) {
	dec, err := NewDecoder(r)
	if err != nil {
		return nil, nil, fmt.Errorf("create decoder: %w", err)
	}

	q := dec.Header

	// Calculate the required size of the sample buffer and allocate
	totalSamples := q.Samples * q.Channels
	samples := make([]int16, 0, totalSamples)

	for {
		var more bool
		var err error

		samples, more, err = dec.AppendSamples(samples)
		if err != nil {
			return nil, nil, fmt.Errorf("read samples: %w", err)
		}

		if !more {
			break
		}
	}

	return q, samples, nil
}

// Decoder allows for streaming decode of a qoa encoded file from a io.Reader.
type Decoder struct {
	r      readerWithSeek
	Header *QOA

	// buffer large enough that it can hold one full frame
	framebuf []byte

	// sample buffer for one frame
	samples []int16
}

func NewDecoder(r io.Reader) (*Decoder, error) {
	bufread := readerWithSeek{
		Reader:   bufio.NewReaderSize(r, 32*1024),
		Delegate: r,
	}

	buf, err := bufread.Peek(16)
	if err != nil {
		return nil, fmt.Errorf("read qoa header: %w", err)
	}

	header, err := DecodeHeader(buf[:])
	if err != nil {
		return nil, fmt.Errorf("parse qua header: %w", err)
	}

	// discard the header bytes. no error handling needed, we peeked 16 bytes, so
	// we are guaranteed to be able to discard 8 bytes
	_, _ = bufread.Discard(8)

	// calculate size of one frame
	const sliceSize = 8
	const headerSize = 8
	const encoderState = 16
	frameSize := headerSize + (encoderState+sliceSize*QOASlicesPerFrame)*header.Channels

	// calculate number of samples per frame
	samples := QOASlicesPerFrame * QOASliceLen * header.Channels

	dec := &Decoder{
		r:        bufread,
		Header:   header,
		framebuf: make([]byte, frameSize),
		samples:  make([]int16, samples),
	}

	return dec, nil
}

func (d *Decoder) AppendSamples(samplesTarget []int16) (samples []int16, more bool, err error) {
	read, err := d.decodeFrame()
	if err != nil {
		return nil, false, err
	}

	samples = append(samplesTarget, read...)

	return samples, len(read) > 0, err
}

func (d *Decoder) frameOffset(n int) int64 {
	const FileHeader = 8

	return int64(FileHeader + len(d.framebuf)*n)
}

func (d *Decoder) frameContainingSample(n int) int {
	// number of samples in a frame
	sampleCount := QOAFrameLen * int(d.Header.Channels)
	return n / sampleCount
}

func (d *Decoder) SeekTo(frame int) error {
	frameOffset := d.frameOffset(frame)

	err := d.r.SeekStart(frameOffset)
	if err != nil {
		return fmt.Errorf("seek to frame %d at %d: %w", frame, frameOffset, err)
	}

	return nil
}

type Stream struct {
	decoder  *Decoder
	buffered []byte
	samples  []int16
}

func NewStream(r *Decoder) *Stream {
	return &Stream{decoder: r}
}

func (s *Stream) Read(buf []byte) (n int, err error) {
	for {
		// take some samples from the buffer if available
		if n := min(len(buf), len(s.buffered)); n > 0 {
			copy(buf[:n], s.buffered[:n])
			s.buffered = s.buffered[n:]

			return n, nil
		}

		var more bool

		// buffer is empty, decode some samples
		s.samples, more, err = s.decoder.AppendSamples(s.samples[:0])
		if err != nil {
			return 0, fmt.Errorf("decode: %w", err)
		}

		// copy samples to buffer
		s.buffered = append(s.buffered, bytesViewOf(s.samples)...)

		// reached end of file
		if len(s.buffered) == 0 && !more {
			return 0, io.EOF
		}
	}
}

func (s *Stream) Seek(offset int64, whence int) (int64, error) {
	s.buffered = s.buffered[:0]
	s.samples = s.samples[:0]

	q := s.decoder.Header

	// calculate the frame to seek to
	frameOffset := s.decoder.frameContainingSample(int(offset / 2))
	if err := s.decoder.SeekTo(frameOffset); err != nil {
		return 0, err
	}

	// the byte position of the first sample in the new frame
	offsetSample := int64(frameOffset) * int64(q.Channels) * QOAFrameLen * 2

	// number of bytes to discard to reach the requested offset
	bytesToDiscard := offset - offsetSample

	// reach the requested offset
	if _, err := io.CopyN(io.Discard, s, bytesToDiscard); err != nil {
		return 0, fmt.Errorf("seek to %d: %w", offset, err)
	}

	return offset, nil
}

func bytesViewOf(samples []int16) []byte {
	data := unsafe.Pointer(unsafe.SliceData(samples))
	return unsafe.Slice((*byte)(data), len(samples)*2)
}

type readerWithSeek struct {
	*bufio.Reader
	Delegate io.Reader
}

func (r readerWithSeek) SeekStart(offset int64) error {
	seeker, ok := r.Delegate.(io.Seeker)
	if !ok {
		return errors.ErrUnsupported
	}

	_, err := seeker.Seek(offset, io.SeekStart)
	if err != nil {
		return err
	}

	// reset the bufio.Reader to discard any buffered data
	r.Reset(r.Delegate)

	return nil
}
