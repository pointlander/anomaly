// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
)

const (
	symbols    = 'z' - 'a'
	vectorSize = 1024
)

func generateJSON() map[string]interface{} {
	sample := func(stddev float64) int {
		return int(math.Abs(rand.NormFloat64()) * stddev)
	}
	sampleCount := func() int {
		return sample(1) + 1
	}
	sampleName := func() string {
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

func hash(a []string) uint64 {
	h := fnv.New64()
	for _, s := range a {
		h.Write([]byte(s))
	}
	return h.Sum64()
}

var cache = make(map[uint64][]int8)

func lookup(a []string) []int8 {
	h := hash(a)
	transform, found := cache[h]
	if found {
		return transform
	}
	transform = make([]int8, vectorSize)
	rnd := rand.New(rand.NewSource(int64(h)))
	for i := range transform {
		// https://en.wikipedia.org/wiki/Random_projection#More_computationally_efficient_random_projections
		// make below distribution function of vector element index
		switch rnd.Intn(6) {
		case 0:
			transform[i] = 1
		case 1:
			transform[i] = -1
		}
	}
	cache[h] = transform
	return transform
}

func hashJSON(object map[string]interface{}) []int64 {
	hash := make([]int64, vectorSize)
	var process func(object map[string]interface{}, context []string)
	process = func(object map[string]interface{}, context []string) {
		for k, v := range object {
			sub := append(context, k)
			switch value := v.(type) {
			case []interface{}:
				for _, i := range value {
					process(i.(map[string]interface{}), sub)
				}
			case string:
				sub = append(sub, value)
				for i := range sub {
					transform := lookup(sub[i:])
					for x, y := range transform {
						hash[x] += int64(y)
					}
				}
			}
		}
	}
	process(object, make([]string, 0))
	return hash
}

var images = flag.Bool("images", false, "run images demo")

func main() {
	flag.Parse()

	if *images {
		imagesDemo()
	}

	object := generateJSON()
	data, err := json.MarshalIndent(object, "", " ")
	if err != nil {
		panic(err)
	}
	fmt.Println(string(data))
	hash := hashJSON(object)
	fmt.Println(hash)
}
