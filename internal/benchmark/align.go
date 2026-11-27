package benchmark

import "sort"

// opcode describes how a[i1:i2] relates to b[j1:j2]. Row alignment uses the
// classic longest-matching-block algorithm (leftmost, longest first), so
// ties are resolved deterministically and rows are paired in order.
type opcode struct {
	tag            string // equal | replace | delete | insert
	i1, i2, j1, j2 int
}

type block struct{ a, b, size int }

func longestMatch(a, b []string, b2j map[string][]int, alo, ahi, blo, bhi int) block {
	best := block{alo, blo, 0}
	j2len := map[int]int{}
	for i := alo; i < ahi; i++ {
		newLen := map[int]int{}
		for _, j := range b2j[a[i]] {
			if j < blo {
				continue
			}
			if j >= bhi {
				break
			}
			k := j2len[j-1] + 1
			newLen[j] = k
			if k > best.size {
				best = block{i - k + 1, j - k + 1, k}
			}
		}
		j2len = newLen
	}
	return best
}

func matchingBlocks(a, b []string) []block {
	b2j := map[string][]int{}
	for j, s := range b {
		b2j[s] = append(b2j[s], j)
	}
	type span struct{ alo, ahi, blo, bhi int }
	queue := []span{{0, len(a), 0, len(b)}}
	var matches []block
	for len(queue) > 0 {
		s := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		m := longestMatch(a, b, b2j, s.alo, s.ahi, s.blo, s.bhi)
		if m.size == 0 {
			continue
		}
		matches = append(matches, m)
		if s.alo < m.a && s.blo < m.b {
			queue = append(queue, span{s.alo, m.a, s.blo, m.b})
		}
		if m.a+m.size < s.ahi && m.b+m.size < s.bhi {
			queue = append(queue, span{m.a + m.size, s.ahi, m.b + m.size, s.bhi})
		}
	}
	sort.Slice(matches, func(x, y int) bool {
		if matches[x].a != matches[y].a {
			return matches[x].a < matches[y].a
		}
		return matches[x].b < matches[y].b
	})
	var out []block
	cur := block{}
	for _, m := range matches {
		if cur.a+cur.size == m.a && cur.b+cur.size == m.b {
			cur.size += m.size
			continue
		}
		if cur.size > 0 {
			out = append(out, cur)
		}
		cur = m
	}
	if cur.size > 0 {
		out = append(out, cur)
	}
	return append(out, block{len(a), len(b), 0})
}

func opcodes(a, b []string) []opcode {
	var out []opcode
	i, j := 0, 0
	for _, m := range matchingBlocks(a, b) {
		tag := ""
		switch {
		case i < m.a && j < m.b:
			tag = "replace"
		case i < m.a:
			tag = "delete"
		case j < m.b:
			tag = "insert"
		}
		if tag != "" {
			out = append(out, opcode{tag, i, m.a, j, m.b})
		}
		i, j = m.a+m.size, m.b+m.size
		if m.size > 0 {
			out = append(out, opcode{"equal", m.a, i, m.b, j})
		}
	}
	return out
}
