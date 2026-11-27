//go:build e2e

package suite

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/tendstead/vision-bill/internal/ids"
)

// 1x1 PNG.
const genericPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII="

func genericPNG(t *testing.T) []byte {
	b, err := base64.StdEncoding.DecodeString(genericPNGBase64)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// signedToken builds a correctly signed session token expiring in expiresIn seconds.
func signedToken(userID string, expiresIn int) string {
	msg := userID + "." + strconv.FormatInt(time.Now().Unix()+int64(expiresIn), 10)
	mac := hmac.New(sha256.New, []byte(secretKey))
	mac.Write([]byte(msg))
	return msg + "." + hex.EncodeToString(mac.Sum(nil))
}

func dbConn(t *testing.T) *pgx.Conn {
	t.Helper()
	conn, err := pgx.Connect(context.Background(), pgURL)
	if err != nil {
		t.Fatalf("database: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close(context.Background()) })
	return conn
}

func TestMalformedIDsNeverServerError(t *testing.T) {
	newEnv(t)
	anon := newClient(t)
	for _, p := range []string{"/receipts/not-a-uuid", "/images/not-a-uuid", "/collections/not-a-uuid"} {
		if r := anon.Get(p); r.Status != 401 && r.Status != 404 && r.Status != 422 {
			t.Errorf("%s: %s", p, r)
		}
	}
	a := admin(t)
	for _, p := range []string{"/benchmarks/not-a-uuid", "/receipts/not-a-uuid", "/images/x/file?part=-1"} {
		if r := a.Get(p); r.Status >= 500 {
			t.Errorf("%s: %s", p, r)
		}
	}
}

func TestSQLMetacharactersAreStoredVerbatim(t *testing.T) {
	newEnv(t)
	username := "weird';--name\"; DROP TABLE users;--" + ids.UUID()[:6]
	c := newClient(t)
	c.Post("/auth/register", map[string]string{"username": username, "password": testPass}).Is(201)
	c.Post("/auth/login", map[string]string{"username": username, "password": testPass}).Is(200)
	var stored string
	if err := dbConn(t).QueryRow(context.Background(), "SELECT username FROM users WHERE username = $1", username).Scan(&stored); err != nil || stored != username {
		t.Fatalf("stored %q: %v", stored, err)
	}
}

func TestHostileCollectionNameIsNormalised(t *testing.T) {
	newEnv(t)
	c, _ := register(t, "sec")
	raw := "Script <img src=x onerror=\"alert(1)\"> & \"quotes\"\nnewline \U0001F680"
	want := strings.Join(strings.Fields(raw), " ")
	c.Post("/collections", map[string]any{"name": raw, "color": "#112233"}).Is(201)
	found := false
	for _, v := range c.Get("/collections").Is(200).L() {
		if str(v, "name") == want {
			found = true
		}
	}
	if !found {
		t.Fatal("normalised collection name not listed")
	}
}

func TestTagLengthBoundary(t *testing.T) {
	newEnv(t)
	c, _ := register(t, "sec")
	c.Post("/tags", map[string]string{"name": strings.Repeat("t", 100)}).Is(200, 201)
	c.Post("/tags", map[string]string{"name": strings.Repeat("t", 101)}).Is(422)
}

func TestUploadKeepsClientFileNameOnlyAsMetadata(t *testing.T) {
	newEnv(t)
	c, _ := register(t, "sec")
	c.Upload("", File{Name: "../../evil.png", ContentType: "image/png", Data: genericPNG(t)}).Is(201)
	for _, v := range c.Get("/images?limit=5").Is(200).L() {
		if str(v, "original_filename") == "../../evil.png" {
			if strings.Contains(str(v, "image_path"), "..") {
				t.Fatalf("stored path contains traversal: %s", str(v, "image_path"))
			}
			return
		}
	}
	t.Fatal("uploaded image not listed")
}

func TestSettingsAreAdminOnlyAndValidated(t *testing.T) {
	newEnv(t)
	user, _ := register(t, "sec")
	body := map[string]any{"llm": map[string]any{"provider": "openai", "host": "x", "model_name": "y", "temperature": 0.1}, "allow_registration": true}
	user.Put("/system/settings", body).Is(403)

	a := admin(t)
	view := a.Get("/system/settings").Is(200).M()
	bad := map[string]any{"llm": map[string]any{
		"provider": str(view, "llm", "provider"), "host": str(view, "llm", "host"),
		"model_name": str(view, "llm", "model_name"), "temperature": 99}, "allow_registration": view["allow_registration"]}
	a.Put("/system/settings", bad).Is(422)
}

func TestImageDeleteAfterAnalysisConflicts(t *testing.T) {
	newEnv(t)
	c, _ := register(t, "sec")
	r := c.Upload("", File{Name: "x.png", ContentType: "image/png", Data: genericPNG(t)}).Is(201)
	c.Delete("/images/" + str(r.M(), "image_id")).Is(409)
}

func TestCollectionsAreScopedPerUser(t *testing.T) {
	newEnv(t)
	a, _ := register(t, "seca")
	b, _ := register(t, "secb")
	id := str(a.Post("/collections", map[string]string{"name": "A only"}).Is(201).M(), "id")
	for _, v := range b.Get("/collections").Is(200).L() {
		if str(v, "id") == id {
			t.Fatal("collection leaked to another user")
		}
	}
	b.Get("/collections/" + id).Is(404)
	b.Delete("/collections/" + id).Is(404)
	if got := fmt.Sprint(b.Post("/collections/"+id+"/activate", nil).Status); got != "404" {
		t.Fatalf("activate: %s", got)
	}
}
