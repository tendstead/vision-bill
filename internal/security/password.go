// Package security reproduces the Python implementation's password hashing
// and session tokens byte-for-byte so existing hashes and cookies keep working.
package security

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"golang.org/x/crypto/argon2"
)

// Argon2id parameters used for newly created hashes (m=64MiB, t=3, p=4).
const (
	argonMemory      = 64 * 1024
	argonTime        = 3
	argonParallelism = 4
	argonKeyLen      = 32
	argonSaltLen     = 16
	// maxArgonMemory bounds parameters read from stored hashes.
	maxArgonMemory = 1 << 20
)

// EffectivePepper is the explicit pepper, falling back to the secret key.
func EffectivePepper(pepper, secretKey string) string {
	if pepper != "" {
		return pepper
	}
	return secretKey
}

// peppered pre-digests the password: hex(HMAC-SHA256(pepper, password)).
// With an empty pepper the password is hashed directly.
func peppered(pepper, password string) string {
	if pepper == "" {
		return password
	}
	mac := hmac.New(sha256.New, []byte(pepper))
	mac.Write([]byte(password))
	return hex.EncodeToString(mac.Sum(nil))
}

// HashPassword returns an Argon2id PHC string compatible with argon2-cffi.
func HashPassword(pepper, password string) (string, error) {
	salt := make([]byte, argonSaltLen)
	if _, err := rand.Read(salt); err != nil {
		return "", err
	}
	key := argon2.IDKey([]byte(peppered(pepper, password)), salt, argonTime, argonMemory, argonParallelism, argonKeyLen)
	enc := base64.RawStdEncoding
	return fmt.Sprintf("$argon2id$v=%d$m=%d,t=%d,p=%d$%s$%s",
		argon2.Version, argonMemory, argonTime, argonParallelism,
		enc.EncodeToString(salt), enc.EncodeToString(key)), nil
}

type phc struct {
	memory, time uint32
	threads      uint8
	salt, key    []byte
}

var errBadHash = errors.New("malformed argon2id hash")

func parsePHC(hash string) (phc, error) {
	var p phc
	parts := strings.Split(hash, "$")
	if len(parts) != 6 || parts[0] != "" || parts[1] != "argon2id" || !strings.HasPrefix(parts[2], "v=") {
		return p, errBadHash
	}
	if v, err := strconv.Atoi(parts[2][2:]); err != nil || v != argon2.Version {
		return p, errBadHash
	}
	for _, kv := range strings.Split(parts[3], ",") {
		k, v, ok := strings.Cut(kv, "=")
		n, err := strconv.ParseUint(v, 10, 32)
		if !ok || err != nil {
			return p, errBadHash
		}
		switch k {
		case "m":
			p.memory = uint32(n)
		case "t":
			p.time = uint32(n)
		case "p":
			if n == 0 || n > 255 {
				return p, errBadHash
			}
			p.threads = uint8(n)
		default:
			return p, errBadHash
		}
	}
	if p.memory == 0 || p.memory > maxArgonMemory || p.time == 0 || p.threads == 0 {
		return p, errBadHash
	}
	enc := base64.RawStdEncoding
	var err error
	if p.salt, err = enc.DecodeString(parts[4]); err != nil || len(p.salt) == 0 {
		return p, errBadHash
	}
	if p.key, err = enc.DecodeString(parts[5]); err != nil || len(p.key) == 0 {
		return p, errBadHash
	}
	return p, nil
}

// VerifyPassword fails closed: any malformed hash or mismatch returns false.
func VerifyPassword(pepper, hash, password string) bool {
	p, err := parsePHC(hash)
	if err != nil {
		return false
	}
	got := argon2.IDKey([]byte(peppered(pepper, password)), p.salt, p.time, p.memory, p.threads, uint32(len(p.key)))
	return subtle.ConstantTimeCompare(got, p.key) == 1
}
