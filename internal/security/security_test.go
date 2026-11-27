package security

import (
	"testing"
	"time"
)

// Fixtures of an already-issued password hash and session cookie. They must
// keep verifying so existing accounts and sessions keep working.
const (
	storedHashFallbackPepper = "$argon2id$v=19$m=65536,t=3,p=4$/UfgkIzQXNwx8OEyLhrZMw$MG8QzS836ZeRedt7X1xBseivlEXG6KyPcksjHnP2qBk"
	storedHashExplicitPepper = "$argon2id$v=19$m=65536,t=3,p=4$Yjq6zbvDFRF+tfUWL6yvRw$yIwx6Q7lMArshWjqDZ/wroeLmYlM47Zy8hSrUmoiBcw"
	issuedToken              = "11111111-2222-3333-4444-555555555555.2795789108.cbf9596157c76a680a4b3733d1a2acd97f8fbe8d017bbd926f35978004959968"
	secret                   = "test-secret-key"
	password                 = "pässword 1"
	userID                   = "11111111-2222-3333-4444-555555555555"
)

func TestStoredHashesVerify(t *testing.T) {
	if !VerifyPassword(EffectivePepper("", secret), storedHashFallbackPepper, password) {
		t.Fatal("hash made with the secret-key pepper should verify")
	}
	if !VerifyPassword(EffectivePepper("custom-pepper", secret), storedHashExplicitPepper, password) {
		t.Fatal("hash made with an explicit pepper should verify")
	}
	if VerifyPassword(secret, storedHashFallbackPepper, "wrong") || VerifyPassword("other", storedHashFallbackPepper, password) {
		t.Fatal("wrong password or pepper must fail")
	}
}

func TestMalformedHashFailsClosed(t *testing.T) {
	for _, h := range []string{"", "x", "$argon2i$v=19$m=1,t=1,p=1$YQ$YQ", "$argon2id$v=19$m=1,t=1,p=0$YQ$YQ", "$argon2id$v=19$m=99999999,t=1,p=1$YQ$YQ"} {
		if VerifyPassword(secret, h, password) {
			t.Fatalf("%q must not verify", h)
		}
	}
}

func TestHashRoundTrip(t *testing.T) {
	h, err := HashPassword(secret, password)
	if err != nil {
		t.Fatal(err)
	}
	if !VerifyPassword(secret, h, password) || VerifyPassword(secret, h, "nope") {
		t.Fatal("round trip failed")
	}
}

func TestSessionTokens(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	if id, ok := DecodeToken(issuedToken, secret, now); !ok || id != userID {
		t.Fatalf("issued token rejected: %v %v", id, ok)
	}
	if _, ok := DecodeToken(issuedToken, "other", now); ok {
		t.Fatal("wrong secret accepted")
	}
	if _, ok := DecodeToken(issuedToken, secret, time.Unix(2_795_789_109, 0)); ok {
		t.Fatal("expired token accepted")
	}
	// Tokens are deterministic for a given expiry.
	if got := CreateToken(userID, 1_095_789_108*time.Second, secret, now); got != issuedToken {
		t.Fatalf("token mismatch: %s", got)
	}
	for _, bad := range []string{"", "a.b", issuedToken + ".x", "nope.1.2"} {
		if _, ok := DecodeToken(bad, secret, now); ok {
			t.Fatalf("%q accepted", bad)
		}
	}
}
