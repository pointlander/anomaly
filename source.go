// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package anomaly

import (
	"math"
	"math/rand"
)

const (
	// One is odds for 1
	One = math.MaxUint32 / 6
	// MinusOne is odds for -1
	MinusOne = 2 * One
)

// Source is a random source of 1, -1, and 0
type Source interface {
	Int() int8
}

// SourceFactory generates new random number sources
type SourceFactory func(seed uint64) Source

// LFSR32 is a 32 bit linear feedback shift register
type LFSR32 uint32

// Uint64 generates a 32 bit random number
func (l *LFSR32) Uint64() uint64 {
	const polynomial = 0x80000057
	ll := *l
	r := (ll >> 1) ^ (-(ll & 1) & polynomial)
	*l = r
	return uint64(r)
}

// Int randomly returns 1, -1 or 0
func (l *LFSR32) Int() int8 {
	r := l.Uint64()
	if r < One {
		return 1
	} else if r < MinusOne {
		return -1
	}
	return 0
}

// NewLFSR32Source create a new LFSR32 based source
func NewLFSR32Source(seed uint64) Source {
	lfsr := LFSR32(seed)
	return &lfsr
}

// Rand is the golang random number generator
type Rand struct {
	*rand.Rand
}

// Int randomly returns 1, -1 or 0
func (r Rand) Int() int8 {
	// https://en.wikipedia.org/wiki/Random_projection#More_computationally_efficient_random_projections
	// make below distribution function of vector element index
	switch r.Intn(6) {
	case 0:
		return 1
	case 1:
		return -1
	}
	return 0
}

// NewRandSource creates a new Rand based source
func NewRandSource(seed uint64) Source {
	return Rand{
		Rand: rand.New(rand.NewSource(int64(seed))),
	}
}
