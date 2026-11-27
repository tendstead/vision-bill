// Package ids generates random identifiers.
package ids

import (
	"crypto/rand"
	"encoding/hex"
)

// UUID returns a random (version 4) UUID in canonical form.
func UUID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic("ids: no randomness available: " + err.Error())
	}
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	h := hex.EncodeToString(b[:])
	return h[0:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:]
}

// Hex returns n random bytes as lowercase hex.
func Hex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic("ids: no randomness available: " + err.Error())
	}
	return hex.EncodeToString(b)
}
