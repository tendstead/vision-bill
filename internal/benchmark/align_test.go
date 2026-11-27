package benchmark

import (
	"reflect"
	"testing"
)

func TestOpcodes(t *testing.T) {
	cases := []struct {
		a, b []string
		want []opcode
	}{
		{[]string{"a", "b", "c"}, []string{"a", "b", "c"}, []opcode{{"equal", 0, 3, 0, 3}}},
		{[]string{"a", "b", "c"}, []string{"a", "x", "c"}, []opcode{{"equal", 0, 1, 0, 1}, {"replace", 1, 2, 1, 2}, {"equal", 2, 3, 2, 3}}},
		{[]string{"a", "b"}, []string{"a", "b", "c"}, []opcode{{"equal", 0, 2, 0, 2}, {"insert", 2, 2, 2, 3}}},
		{[]string{"a", "b", "c"}, []string{"a", "c"}, []opcode{{"equal", 0, 1, 0, 1}, {"delete", 1, 2, 1, 1}, {"equal", 2, 3, 1, 2}}},
		{nil, []string{"a"}, []opcode{{"insert", 0, 0, 0, 1}}},
		{[]string{"a"}, nil, []opcode{{"delete", 0, 1, 0, 0}}},
		{nil, nil, nil},
		// Repeated purchases: the longest block wins, ties go to the leftmost.
		{[]string{"x", "x", "y"}, []string{"x", "y"}, []opcode{{"delete", 0, 1, 0, 0}, {"equal", 1, 3, 0, 2}}},
	}
	for i, c := range cases {
		if got := opcodes(c.a, c.b); !reflect.DeepEqual(got, c.want) {
			t.Errorf("case %d: got %v, want %v", i, got, c.want)
		}
	}
}
