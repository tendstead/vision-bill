package httpapi

import (
	"github.com/tendstead/vision-bill/internal/config"
	"time"

	"github.com/tendstead/vision-bill/internal/security"
)

func mustToken(id string) string {
	return security.CreateToken(id, time.Hour, testSecret, time.Now())
}

type configT = config.Config
