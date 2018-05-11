// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package anomaly

import (
	"encoding/json"
	"math/rand"
	"testing"

	"github.com/pointlander/anomaly/gru"
	"github.com/pointlander/anomaly/lstm"
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
	network := NewAverageSimilarity(rnd, vectorizer)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		input, err := json.Marshal(object)
		if err != nil {
			panic(err)
		}
		b.StartTimer()
		network.Train(input)
	}
}

func BenchmarkNeuron(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	vectorizer := NewVectorizer(1024, true, NewLFSR32Source)
	network := NewNeuron(rnd, vectorizer)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		input, err := json.Marshal(object)
		if err != nil {
			panic(err)
		}
		b.StartTimer()
		network.Train(input)
	}
}

func BenchmarkAutoencoder(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	vectorizer := NewVectorizer(1024, true, NewLFSR32Source)
	network := NewAutoencoder(rnd, vectorizer)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		input, err := json.Marshal(object)
		if err != nil {
			panic(err)
		}
		b.StartTimer()
		network.Train(input)
	}
}

func BenchmarkLSTM(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	network := lstm.NewLSTM()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		data, err := json.Marshal(object)
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		network.Train(data)
	}
}

func BenchmarkGRU(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	network := gru.NewGRU()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		data, err := json.Marshal(object)
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		network.Train(data)
	}
}

func BenchmarkComplexity(b *testing.B) {
	rnd := rand.New(rand.NewSource(1))
	vectorizer := NewVectorizer(1024, true, NewLFSR32Source)
	network := NewComplexity(rnd, vectorizer)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		object := GenerateRandomJSON(rnd)
		data, err := json.Marshal(object)
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		network.Train(data)
	}
}
