package httpapi

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/tendstead/vision-bill/internal/security"
	"github.com/tendstead/vision-bill/internal/store"
)

// userView is the API representation of the authenticated principal.
type userView struct {
	ID        string `json:"id"`
	Username  string `json:"username"`
	IsAdmin   bool   `json:"is_admin"`
	CanSeeAll bool   `json:"can_see_all"`
}

type credentials struct {
	Username *string `json:"username"`
	Password *string `json:"password"`
}

func (c credentials) validate() validationErrors {
	var errs validationErrors
	checkString(&errs, "username", c.Username, 1, 100)
	checkString(&errs, "password", c.Password, 1, 200)
	return errs
}

func (s *Server) authRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /api/v1/auth/register", s.register)
	mux.HandleFunc("POST /api/v1/auth/login", s.login)
	mux.HandleFunc("POST /api/v1/auth/logout", s.logout)
	mux.HandleFunc("GET /api/v1/auth/me", s.authed(func(w http.ResponseWriter, _ *http.Request, u userView) {
		writeJSON(w, http.StatusOK, u)
	}))
}

func (s *Server) setSessionCookie(w http.ResponseWriter, userID string) {
	c := s.cfg()
	token := security.CreateToken(userID, time.Duration(c.Auth.SessionMaxAgeSeconds)*time.Second, c.Auth.SecretKey, s.opts.Now())
	http.SetCookie(w, &http.Cookie{
		Name: c.Auth.SessionCookieName, Value: token, Path: "/",
		MaxAge: c.Auth.SessionMaxAgeSeconds, HttpOnly: true, SameSite: http.SameSiteLaxMode, Secure: c.Auth.SessionSecure,
	})
}

// withHashSlot bounds concurrent password hashing/verification.
func (s *Server) withHashSlot(ctx context.Context, fn func()) bool {
	select {
	case s.hashSem <- struct{}{}:
		defer func() { <-s.hashSem }()
		fn()
		return true
	case <-ctx.Done():
		return false
	}
}

func (s *Server) register(w http.ResponseWriter, r *http.Request) {
	var body credentials
	if !decodeJSON(w, r, &body) {
		return
	}
	if errs := body.validate(); len(errs) > 0 {
		errs.write(w)
		return
	}
	st := s.db()
	if st == nil {
		writeError(w, http.StatusServiceUnavailable, "Database is unavailable")
		return
	}
	c := s.cfg()
	if !c.Auth.AllowRegistration {
		writeError(w, http.StatusForbidden, "Registration is disabled")
		return
	}
	existing, err := st.UserByUsername(r.Context(), *body.Username)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	if existing != nil {
		writeError(w, http.StatusConflict, "Username is already taken")
		return
	}
	var hashed string
	var hashErr error
	if !s.withHashSlot(r.Context(), func() { hashed, hashErr = security.HashPassword(c.Auth.EffectivePepper(), *body.Password) }) {
		return
	}
	if hashErr != nil {
		s.internalError(w, r, hashErr)
		return
	}
	u, err := st.CreateUser(r.Context(), *body.Username, hashed, false)
	if errors.Is(err, store.ErrUsernameTaken) {
		writeError(w, http.StatusConflict, "Username is already taken")
		return
	}
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	s.opts.Logger.Info("registered user", "username", u.Username, "id", u.ID)
	s.setSessionCookie(w, u.ID)
	writeJSON(w, http.StatusCreated, userView{ID: u.ID, Username: u.Username, IsAdmin: u.IsAdmin})
}

func (s *Server) login(w http.ResponseWriter, r *http.Request) {
	var body credentials
	if !decodeJSON(w, r, &body) {
		return
	}
	if errs := body.validate(); len(errs) > 0 {
		errs.write(w)
		return
	}
	st := s.db()
	if st == nil {
		writeError(w, http.StatusServiceUnavailable, "Database is unavailable")
		return
	}
	u, err := st.UserByUsername(r.Context(), *body.Username)
	if err != nil {
		s.internalError(w, r, err)
		return
	}
	c := s.cfg()
	ok := false
	if u != nil {
		if !s.withHashSlot(r.Context(), func() {
			ok = security.VerifyPassword(c.Auth.EffectivePepper(), u.HashedPassword, *body.Password)
		}) {
			return
		}
	}
	if !ok {
		writeError(w, http.StatusUnauthorized, "Invalid username or password")
		return
	}
	s.setSessionCookie(w, u.ID)
	writeJSON(w, http.StatusOK, userView{ID: u.ID, Username: u.Username, IsAdmin: u.IsAdmin})
}

func (s *Server) logout(w http.ResponseWriter, _ *http.Request) {
	http.SetCookie(w, &http.Cookie{
		Name: s.cfg().Auth.SessionCookieName, Value: "", Path: "/", MaxAge: -1,
		Expires: time.Unix(0, 0), SameSite: http.SameSiteLaxMode,
	})
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func (s *Server) internalError(w http.ResponseWriter, r *http.Request, err error) {
	s.opts.Logger.Error("request failed", "method", r.Method, "path", r.URL.Path, "err", err)
	writeError(w, http.StatusInternalServerError, "Internal Server Error")
}

// authed resolves the session cookie to a user (401, or 503 while the user store is down).
func (s *Server) authed(h func(http.ResponseWriter, *http.Request, userView)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		st := s.db()
		if st == nil {
			writeError(w, http.StatusServiceUnavailable, "Authentication is unavailable")
			return
		}
		c := s.cfg()
		cookie, err := r.Cookie(c.Auth.SessionCookieName)
		if err != nil || cookie.Value == "" {
			writeError(w, http.StatusUnauthorized, "Not authenticated")
			return
		}
		id, ok := security.DecodeToken(cookie.Value, c.Auth.SecretKey, s.opts.Now())
		if !ok {
			writeError(w, http.StatusUnauthorized, "Not authenticated")
			return
		}
		u, err := st.UserByID(r.Context(), id)
		if err != nil {
			s.internalError(w, r, err)
			return
		}
		if u == nil {
			writeError(w, http.StatusUnauthorized, "Not authenticated")
			return
		}
		h(w, r, userView{ID: u.ID, Username: u.Username, IsAdmin: u.IsAdmin, CanSeeAll: u.IsAdmin && c.Auth.AdminCanSeeAll})
	}
}

// admin additionally requires administrator privileges (403 otherwise).
func (s *Server) admin(h func(http.ResponseWriter, *http.Request, userView)) http.HandlerFunc {
	return s.authed(func(w http.ResponseWriter, r *http.Request, u userView) {
		if !u.IsAdmin {
			writeError(w, http.StatusForbidden, "Administrator privileges required")
			return
		}
		h(w, r, u)
	})
}
