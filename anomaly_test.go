// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package anomaly

import (
	"math/rand"
	"testing"
)

func BenchmarkLFSR(b *testing.B) {
	lfsr := LFSR32(1)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		lfsr.Uint64()
	}
}

func BenchmarkSource(b *testing.B) {
	source := rand.NewSource(1)
	lfsr := source.(rand.Source64)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		lfsr.Uint64()
	}
}

func BenchmarkVectorizer(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	vectorizer := NewVectorizer(1024, true, NewRandSource)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		b.StartTimer()
		vectorizer.Vectorize(object)
	}
}

func BenchmarkVectorizerLFSR(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	vectorizer := NewVectorizer(1024, true, NewLFSR32Source)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		b.StartTimer()
		vectorizer.Vectorize(object)
	}
}

func BenchmarkVectorizerNoCache(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	vectorizer := NewVectorizer(1024, false, NewRandSource)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		b.StartTimer()
		vectorizer.Vectorize(object)
	}
}

func BenchmarkVectorizerLFSRNoCache(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	vectorizer := NewVectorizer(1024, false, NewLFSR32Source)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		b.StartTimer()
		vectorizer.Vectorize(object)
	}
}
