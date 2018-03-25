// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"sync"
)

// Vectorizer converts JSON documents to vectors
type Vectorizer struct {
	UseCache          bool
	MatrixColumnCache map[uint64][]int8
	Source            SourceFactory
	sync.RWMutex
}

// NewVectorizer creates a new vectorizer
// https://en.wikipedia.org/wiki/Random_projection
func NewVectorizer(useCache bool, source SourceFactory) *Vectorizer {
	return &Vectorizer{
		UseCache:          useCache,
		MatrixColumnCache: make(map[uint64][]int8, 256),
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

// AddMatrixColumn finds or generates a matrix column and adds it to a vector
func (v *Vectorizer) AddMatrixColumn(a []string, b []int64) {
	h := hash(a)
	if !v.UseCache {
		rnd := v.Source(h)
		for i := range b {
			b[i] += int64(rnd.Int())
		}
		return
	}

	v.RLock()
	transform, found := v.MatrixColumnCache[h]
	v.RUnlock()
	if found {
		for i := range b {
			b[i] += int64(transform[i])
		}
		return
	}
	transform = make([]int8, VectorSize)
	rnd := v.Source(h)
	for i := range transform {
		x := rnd.Int()
		transform[i] = x
		b[i] += int64(x)
	}
	v.Lock()
	v.MatrixColumnCache[h] = transform
	v.Unlock()
}

// Vectorize produces a vector from a JSON object
func (v *Vectorizer) Vectorize(object map[string]interface{}) []int64 {
	vector := make([]int64, VectorSize)
	var process func(object map[string]interface{}, context []string)
	process = func(object map[string]interface{}, context []string) {
		for key, value := range object {
			sub := append(context, key)
			sum := func(value string) {
				subvalue := append(sub, value)
				for i := range subvalue {
					v.AddMatrixColumn(subvalue[i:], vector)
				}
			}
			switch value := value.(type) {
			case []interface{}:
				for _, value := range value {
					switch value := value.(type) {
					case map[string]interface{}:
						process(value, sub)
					case string:
						sum(value)
					case float64:
						sum(fmt.Sprintf("%f", value))
					case json.Number:
						sum(value.String())
					}
				}
			case string:
				sum(value)
			case float64:
				sum(fmt.Sprintf("%f", value))
			case json.Number:
				sum(value.String())
			}
		}
	}
	process(object, make([]string, 0, 256))
	return vector
}
