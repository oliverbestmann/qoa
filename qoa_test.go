package qoa

import (
	"embed"
	"io"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/require"
)

//go:embed *.qoa
var fs embed.FS

func TestDecodeAll(t *testing.T) {
	fp, _ := fs.Open("test.qoa")
	q, samples, err := DecodeAll(fp)

	require.NoError(t, err)
	require.Equal(t, uint32(2), q.Channels)
	require.Equal(t, uint32(44100), q.SampleRate)
	require.Equal(t, len(samples), int(q.Samples*q.Channels))
}

func TestStream_Read(t *testing.T) {
	fp, _ := fs.Open("test.qoa")
	dec, _ := NewDecoder(fp)

	st := NewStream(dec)

	bytes, err := io.ReadAll(st)

	require.NoError(t, err)
	require.Equal(t, len(bytes), int(dec.Header.Samples*dec.Header.Channels*2))
}

func TestStream_Seek(t *testing.T) {
	fp, _ := fs.Open("test.qoa")
	dec, _ := NewDecoder(fp)

	st := NewStream(dec)

	bytes, err := io.ReadAll(st)

	require.NoError(t, err)
	require.Equal(t, len(bytes), int(dec.Header.Samples*dec.Header.Channels*2))

	// seek to a random byte and read it again
	for range 100_000 {
		idx := int64(rand.IntN(len(bytes)))
		n, err := st.Seek(idx, io.SeekStart)
		require.NoError(t, err)
		require.Equal(t, idx, n)
	}

}
