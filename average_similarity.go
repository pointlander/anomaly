// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package anomaly

import (
	"encoding/json"
	"math"
	"math/rand"
)

const vectorsSize = 1024

// AverageSimilarity computes surpise by calculation the average cosine
// similarity across all Vectors
type AverageSimilarity struct {
	vectors       [][]float32
	begin, length int
	*Vectorizer
}

// NewAverageSimilarity creates a new average similarity surprise engine
func NewAverageSimilarity(rnd *rand.Rand, vectorizer *Vectorizer) Network {
	return &AverageSimilarity{
		vectors:    make([][]float32, vectorsSize),
		Vectorizer: vectorizer,
	}
}

// Train computes the surprise with average similarity
func (a *AverageSimilarity) Train(input []byte) (surprise, uncertainty float32) {
	var object map[string]interface{}
	err := json.Unmarshal(input, &object)
	if err != nil {
		panic(err)
	}
	vector := a.Vectorizer.Vectorize(object)
	unit := Normalize(vector)

	sum, c := 0.0, a.begin
	for i := 0; i < a.length; i++ {
		sum += math.Abs(Similarity(unit, a.vectors[c]))
		c = (c + 1) % vectorsSize
	}
	averageSimilarity := float32(sum / float64(a.length))

	if a.length < vectorsSize {
		a.vectors[a.begin+a.length] = unit
		a.length++
	} else {
		a.vectors[a.begin] = unit
		a.begin = (a.begin + 1) % vectorsSize
	}

	return averageSimilarity, 0
}
