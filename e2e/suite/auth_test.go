//go:build e2e

package suite

import (
	"strings"
	"testing"
)

func TestRegisterLoginMeLogout(t *testing.T) {
	newEnv(t)
	c, user := register(t, "auth")
	if !strings.HasPrefix(str(user, "username"), "e2e-auth-") || user["is_admin"] != false {
		t.Fatalf("user %v", user)
	}
	if c.cookie("vb_session") == "" {
		t.Fatal("no session cookie after register")
	}
	if me := c.Get("/auth/me").Is(200).M(); str(me, "username") != str(user, "username") {
		t.Fatalf("me %v", me)
	}
	if got := c.Post("/auth/logout", nil).Is(200).M(); got["ok"] != true {
		t.Fatalf("logout %v", got)
	}
	c.Get("/auth/me").Is(401)
}

func TestRegisterDuplicateConflict(t *testing.T) {
	newEnv(t)
	_, user := register(t, "dup")
	r := newClient(t).Post("/auth/register", map[string]string{"username": str(user, "username"), "password": "other-pw-1"}).Is(409)
	if !strings.Contains(str(r.M(), "detail"), "already taken") {
		t.Fatal(r)
	}
}

func TestLoginWrongPassword(t *testing.T) {
	newEnv(t)
	_, user := register(t, "badpw")
	c := newClient(t)
	c.Post("/auth/login", map[string]string{"username": str(user, "username"), "password": "wrong-password"}).Is(401)
	if c.cookie("vb_session") != "" {
		t.Fatal("failed login must not set a cookie")
	}
}

func TestBootstrapAdminCanLogin(t *testing.T) {
	newEnv(t)
	c := newClient(t)
	body := c.Post("/auth/login", map[string]string{"username": adminUser, "password": adminPass}).Is(200).M()
	if body["is_admin"] != true || str(body, "username") != "admin" {
		t.Fatalf("%v", body)
	}
}

func TestSessionForgeriesAreRejected(t *testing.T) {
	newEnv(t)
	c, user := register(t, "forge")
	token := c.cookie("vb_session")
	withCookie := func(value string) R {
		return newClient(t).Raw("GET", "/api/v1/auth/me", nil, map[string]string{"Cookie": "vb_session=" + value})
	}
	last := token[len(token)-1]
	flipped := byte('0')
	if last == '0' {
		flipped = '1'
	}
	withCookie(token).Is(200)
	withCookie(token[:len(token)-1] + string(flipped)).Is(401) // tampered signature
	withCookie("garbage").Is(401)
	withCookie(signedToken(str(user, "id"), -60)).Is(401) // expired but correctly signed
}

func TestAuthInputBoundaries(t *testing.T) {
	newEnv(t)
	c := newClient(t)
	c.Post("/auth/register", map[string]string{"username": strings.Repeat("u", 100), "password": testPass}).Is(201)
	c.Post("/auth/register", map[string]string{"username": strings.Repeat("u", 101), "password": testPass}).Is(422)
	c.Post("/auth/register", map[string]string{"username": "e2e-pw-boundary", "password": strings.Repeat("p", 201)}).Is(422)
	c.Post("/auth/register", map[string]string{"username": "", "password": testPass}).Is(422)
}
