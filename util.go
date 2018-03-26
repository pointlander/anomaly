// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package anomaly

import (
	"math"
	"math/rand"
)

// GenerateRandomJSON generates random JSON
func GenerateRandomJSON(rnd *rand.Rand) map[string]interface{} {
	sample := func(stddev float64) int {
		return int(math.Abs(rnd.NormFloat64()) * stddev)
	}
	sampleCount := func() int {
		return sample(1) + 1
	}
	sampleName := func() string {
		const symbols = 'z' - 'a'
		s := sample(8)
		if s > symbols {
			s = symbols
		}
		return string('a' + s)
	}
	sampleValue := func() string {
		value := sampleName()
		return value + value
	}
	sampleDepth := func() int {
		return sample(3)
	}
	var generate func(hash map[string]interface{}, depth int)
	generate = func(hash map[string]interface{}, depth int) {
		count := sampleCount()
		if depth > sampleDepth() {
			for i := 0; i < count; i++ {
				hash[sampleName()] = sampleValue()
			}
			return
		}
		for i := 0; i < count; i++ {
			array := make([]interface{}, sampleCount())
			for j := range array {
				sub := make(map[string]interface{})
				generate(sub, depth+1)
				array[j] = sub
			}
			hash[sampleName()] = array
		}
	}
	object := make(map[string]interface{})
	generate(object, 0)
	return object
}

func sigmoid32(x float32) float32 {
	return 1 / (1 + float32(math.Exp(-float64(x))))
}

// Normalize converts a vector to a unit vector
func Normalize(a []int64) []float32 {
	sum := 0.0
	for _, v := range a {
		sum += float64(v) * float64(v)
	}
	sum = math.Sqrt(sum)
	b := make([]float32, len(a))
	for i, v := range a {
		b[i] = float32(v) / float32(sum)
	}
	return b
}

// Adapt prepares a vector for input into a neural network
func Adapt(a []float32) []float32 {
	b := make([]float32, len(a))
	for i, v := range a {
		b[i] = sigmoid32(v)
	}
	return b
}

// Similarity computes the cosine similarity between two vectors
// https://en.wikipedia.org/wiki/Cosine_similarity
func Similarity(a, b []float32) float64 {
	dot, xx, yy := 0.0, 0.0, 0.0
	for i, j := range b {
		x, y := float64(a[i]), float64(j)
		dot += x * y
		xx += x * x
		yy += y * y
	}
	return dot / math.Sqrt(xx*yy)
}
