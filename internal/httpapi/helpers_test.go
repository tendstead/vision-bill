package httpapi

import (
	"time"

	"github.com/tendstead/vision-bill/internal/security"
)

func mustToken(id string) string {
	return security.CreateToken(id, time.Hour, testSecret, time.Now())
}
