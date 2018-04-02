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

func BenchmarkAverageSimilarity(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	vectorizer := NewVectorizer(1024, true, NewLFSR32Source)
	network := NewAverageSimilarity(1024, rnd)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		vector := vectorizer.Vectorize(object)
		unit := Normalize(vector)
		if len(network.(*AverageSimilarity).Vectors) > 1000 {
			network.(*AverageSimilarity).Vectors = network.(*AverageSimilarity).Vectors[:1000]
		}
		b.StartTimer()
		network.Train(unit)
	}
}

func BenchmarkNeuron(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	vectorizer := NewVectorizer(1024, true, NewLFSR32Source)
	network := NewNeuron(1024, rnd)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		vector := vectorizer.Vectorize(object)
		unit := Normalize(vector)
		b.StartTimer()
		network.Train(unit)
	}
}

func BenchmarkAutoencoder(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	vectorizer := NewVectorizer(1024, true, NewLFSR32Source)
	network := NewAutoencoder(1024, rnd)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		vector := vectorizer.Vectorize(object)
		unit := Normalize(vector)
		b.StartTimer()
		network.Train(unit)
	}
}
