package store

import (
	"context"
	"errors"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// User is the authenticated principal row. CanSeeAll is derived by the API
// layer from configuration, never stored.
type User struct {
	ID       string `json:"id"`
	Username string `json:"username"`
	IsAdmin  bool   `json:"is_admin"`
}

// UserWithHash carries the stored password hash for credential checks.
type UserWithHash struct {
	User
	HashedPassword string
}

// ErrUsernameTaken is returned when the username unique constraint fires.
var ErrUsernameTaken = errors.New("username already taken")

const uniqueViolation = "23505"

func isUnique(err error) bool {
	var pg *pgconn.PgError
	return errors.As(err, &pg) && pg.Code == uniqueViolation
}

// CreateUser inserts a user.
func (s *Store) CreateUser(ctx context.Context, username, hashed string, admin bool) (User, error) {
	var u User
	err := s.Pool.QueryRow(ctx,
		`INSERT INTO users (username, hashed_password, is_admin) VALUES ($1, $2, $3)
		 RETURNING id::text, username, is_admin`, username, hashed, admin).Scan(&u.ID, &u.Username, &u.IsAdmin)
	if isUnique(err) {
		return User{}, ErrUsernameTaken
	}
	return u, err
}

// UserByUsername returns the user with its hash, or (nil, nil) when absent.
func (s *Store) UserByUsername(ctx context.Context, username string) (*UserWithHash, error) {
	var u UserWithHash
	err := s.Pool.QueryRow(ctx,
		`SELECT id::text, username, is_admin, hashed_password FROM users WHERE username = $1`, username).
		Scan(&u.ID, &u.Username, &u.IsAdmin, &u.HashedPassword)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	return &u, err
}

// UserByID returns the user, or (nil, nil) when absent.
func (s *Store) UserByID(ctx context.Context, id string) (*User, error) {
	var u User
	err := s.Pool.QueryRow(ctx, `SELECT id::text, username, is_admin FROM users WHERE id = $1::uuid`, id).
		Scan(&u.ID, &u.Username, &u.IsAdmin)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	return &u, err
}

// CountUsers returns the number of users.
func (s *Store) CountUsers(ctx context.Context) (int, error) {
	var n int
	err := s.Pool.QueryRow(ctx, `SELECT count(*) FROM users`).Scan(&n)
	return n, err
}

// AssignOrphans gives legacy receipts/images with no owner to userID.
func (s *Store) AssignOrphans(ctx context.Context, userID string) error {
	if _, err := s.Pool.Exec(ctx, `UPDATE receipts SET user_id = $1::uuid WHERE user_id IS NULL`, userID); err != nil {
		return err
	}
	_, err := s.Pool.Exec(ctx, `UPDATE images SET user_id = $1::uuid WHERE user_id IS NULL`, userID)
	return err
}
