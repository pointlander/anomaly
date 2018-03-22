// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "hash/fnv"

// Vectorizer converts JSON documents to vectors
type Vectorizer struct {
	MatrixColumnCache map[uint64][]int8
	Source            SourceFactory
}

// NewVectorizer creates a new vectorizer
// https://en.wikipedia.org/wiki/Random_projection
func NewVectorizer(source SourceFactory) *Vectorizer {
	return &Vectorizer{
		MatrixColumnCache: make(map[uint64][]int8),
		Source:            source,
	}
}

func hash(a []string) uint64 {
	h := fnv.New64()
	for _, s := range a {
		h.Write([]byte(s))
	}
	return h.Sum64()
}

// GetMatrixColumn finds or generates a matrix column
func (v *Vectorizer) GetMatrixColumn(a []string) []int8 {
	h := hash(a)
	transform, found := v.MatrixColumnCache[h]
	if found {
		return transform
	}
	transform = make([]int8, VectorSize)
	rnd := v.Source(h)
	for i := range transform {
		transform[i] = rnd.Int()
	}
	v.MatrixColumnCache[h] = transform
	return transform
}

// Vectorize produces a vector from a JSON object
func (v *Vectorizer) Vectorize(object map[string]interface{}) []int64 {
	vector := make([]int64, VectorSize)
	var process func(object map[string]interface{}, context []string)
	process = func(object map[string]interface{}, context []string) {
		for k, val := range object {
			sub := append(context, k)
			switch value := val.(type) {
			case []interface{}:
				for _, i := range value {
					process(i.(map[string]interface{}), sub)
				}
			case string:
				sub = append(sub, value)
				for i := range sub {
					transform := v.GetMatrixColumn(sub[i:])
					for x, y := range transform {
						vector[x] += int64(y)
					}
				}
			}
		}
	}
	process(object, make([]string, 0))
	return vector
}
