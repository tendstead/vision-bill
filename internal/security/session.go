package security

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
	"time"
)

// Token is "{user_id}.{exp}.{sig}" with sig = hex(HMAC-SHA256(secret, "{user_id}.{exp}")).
// It matches the Python session format exactly.

func sign(secret, message string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(message))
	return hex.EncodeToString(mac.Sum(nil))
}

// CreateToken issues a token for userID valid for maxAge from now.
func CreateToken(userID string, maxAge time.Duration, secret string, now time.Time) string {
	exp := now.Unix() + int64(maxAge/time.Second)
	msg := userID + "." + strconv.FormatInt(exp, 10)
	return msg + "." + sign(secret, msg)
}

// DecodeToken returns the user id of a valid, unexpired token.
func DecodeToken(token, secret string, now time.Time) (string, bool) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return "", false
	}
	userID, expS, sig := parts[0], parts[1], parts[2]
	exp, err := strconv.ParseInt(expS, 10, 64)
	if err != nil || !validUUID(userID) {
		return "", false
	}
	// Python re-serialises the parsed values; require canonical forms so that
	// the signed message is identical.
	if strconv.FormatInt(exp, 10) != expS || strings.ToLower(userID) != userID {
		return "", false
	}
	if !hmac.Equal([]byte(sig), []byte(sign(secret, userID+"."+expS))) {
		return "", false
	}
	if now.Unix() > exp {
		return "", false
	}
	return userID, true
}

func validUUID(s string) bool {
	if len(s) != 36 {
		return false
	}
	for i, c := range s {
		switch {
		case i == 8 || i == 13 || i == 18 || i == 23:
			if c != '-' {
				return false
			}
		case !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F'):
			return false
		}
	}
	return true
}
