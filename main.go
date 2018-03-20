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

	"github.com/pointlander/neural"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

const (
	// Symbols is the number of symbols to use for JSON generation
	Symbols = 'z' - 'a'
	// VectorSize is the size of the JSON document vector
	VectorSize = 1024
	// Samples is the number of JSON documents to generate per trial
	Samples = 1000
	// Trials is the number of trials
	Trials = 100
	// Parallelization is how many trials to perform in parallel
	Parallelization = 10
)

var tests = []string{`{
 "alfa": [
  {"alfa": "1"},
	{"bravo": "2"}
 ],
 "bravo": [
  {"alfa": "3"},
	{"bravo": "4"}
 ]
}`, `{
 "a": [
  {"a": "aa"},
	{"b": "bb"}
 ],
 "b": [
  {"a": "aa"},
	{"b": "bb"}
 ]
}`}

func generateJSON(rnd *rand.Rand) map[string]interface{} {
	sample := func(stddev float64) int {
		return int(math.Abs(rnd.NormFloat64()) * stddev)
	}
	sampleCount := func() int {
		return sample(1) + 1
	}
	sampleName := func() string {
		s := sample(8)
		if s > Symbols {
			s = Symbols
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

// Vectorizer converts JSON documents to vectors
type Vectorizer struct {
	Cache map[uint64][]int8
}

// Lookup looks a vector up
func (v *Vectorizer) Lookup(a []string) []int8 {
	h := hash(a)
	transform, found := v.Cache[h]
	if found {
		return transform
	}
	transform = make([]int8, VectorSize)
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
	v.Cache[h] = transform
	return transform
}

// Hash produces a vector from a JSON object
func (v *Vectorizer) Hash(object map[string]interface{}) []int64 {
	hash := make([]int64, VectorSize)
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
					transform := v.Lookup(sub[i:])
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

func sigmoid32(x float32) float32 {
	return 1 / (1 + float32(math.Exp(-float64(x))))
}

func tanh32(x float32) float32 {
	a, b := math.Exp(float64(x)), math.Exp(-float64(x))
	return float32((a - b) / (a + b))
}

func dtanh32(x float32) float32 {
	return 1 - x*x
}

func normalize(a []int64) []float32 {
	sum := 0.0
	for _, v := range a {
		sum += float64(v) * float64(v)
	}
	sum = math.Sqrt(sum)
	b := make([]float32, len(a))
	for i, v := range a {
		b[i] = sigmoid32(float32(v) / float32(sum))
		//b[i] = float32(v) / float32(sum)
	}
	return b
}

func similarity(a, b []float32) float64 {
	dot, xx, yy := 0.0, 0.0, 0.0
	for i, j := range b {
		x, y := float64(a[i]), float64(j)
		dot += x * y
		xx += x * x
		yy += y * y
	}
	return dot / math.Sqrt(xx*yy)
}

var images = flag.Bool("images", false, "run images demo")

func anomaly(seed int) (values, sims plotter.Values, correct bool) {
	rnd := rand.New(rand.NewSource(int64(seed)))

	config := func(n *neural.Neural32) {
		random32 := func(a, b float32) float32 {
			return (b-a)*rnd.Float32() + a
		}
		weightInitializer := func(in, out int) float32 {
			return random32(-1, 1) / float32(math.Sqrt(float64(in)))
		}
		n.Init(weightInitializer, VectorSize, VectorSize/2, VectorSize)
		/*for f := range n.Functions {
			n.Functions[f] = neural.FunctionPair32{
				F:  tanh32,
				DF: dtanh32,
			}
		}*/
		//n.EnableDropout(.5)
	}
	nn := neural.NewNeural32(config)
	context := nn.NewContext()
	vectorizer := &Vectorizer{
		Cache: make(map[uint64][]int8),
	}
	values, sims = make(plotter.Values, Samples), make(plotter.Values, Samples)
	for i := 0; i < Samples; i++ {
		object := generateJSON(rnd)
		hash := vectorizer.Hash(object)
		input := normalize(hash)

		context.SetInput(input)
		context.Infer()
		outputs := context.GetOutput()
		sims[i] = float64(similarity(input, outputs))

		source := func(iterations int) [][][]float32 {
			data := make([][][]float32, 1)
			data[0] = [][]float32{input, input}
			return data
		}
		e := nn.Train(source, 1, 0.6, 0.4)
		//e := nn.Train(source, 1, 0.1, 0.0001)
		values[i] = float64(e[0])
	}
	values = values[100:]
	sims = sims[100:]

	sum, sumSquared, length := 0.0, 0.0, float64(len(values))
	for _, v := range values {
		value := float64(v)
		sum += value
		sumSquared += value * value
	}
	average := sum / length
	stddev := math.Sqrt(sumSquared/length - average*average)
	fmt.Printf("avg=%v stddev=%v\n", average, stddev)

	surprise := make([]float64, len(tests))
	for i, test := range tests {
		var object map[string]interface{}
		err := json.Unmarshal([]byte(test), &object)
		if err != nil {
			panic(err)
		}
		hash := vectorizer.Hash(object)
		input := normalize(hash)

		context.SetInput(input)
		context.Infer()
		outputs := context.GetOutput()
		sim := float64(similarity(input, outputs))
		fmt.Printf("sim before=%v\n", sim)

		source := func(iterations int) [][][]float32 {
			data := make([][][]float32, 1)
			data[0] = [][]float32{input, input}
			return data
		}
		e := nn.Train(source, 1, 0.6, 0.4)
		//e := nn.Train(source, 1, 0.1, 0.0001)
		surprise[i] = math.Abs((float64(e[0]) - average) / stddev)
		fmt.Println(surprise[i])

		context.SetInput(input)
		context.Infer()
		outputs = context.GetOutput()
		sim = float64(similarity(input, outputs))
		fmt.Printf("sim after=%v\n", sim)
	}
	correct = surprise[0] > surprise[1]

	return
}

func main() {
	flag.Parse()

	if *images {
		imagesDemo()
	}

	values, sims, _ := anomaly(1)
	plot := func(title, name string, values plotter.Values) {
		p, err := plot.New()
		if err != nil {
			panic(err)
		}
		p.Title.Text = title

		h, err := plotter.NewHist(values, 20)
		if err != nil {
			panic(err)
		}
		h.Normalize(1)
		p.Add(h)

		err = p.Save(8*vg.Inch, 8*vg.Inch, name)
		if err != nil {
			panic(err)
		}
	}
	plot("Output", "output.png", values)
	plot("Sims", "sims.png", sims)

	count, results, j := 0, make(chan bool, Parallelization), 1
	for i := 0; i < Parallelization; i++ {
		go func(j int) {
			_, _, correct := anomaly(j)
			results <- correct
		}(j)
		j++
	}
	total := 0
	for j <= Trials {
		correct := <-results
		if correct {
			count++
		}
		total++
		go func(j int) {
			_, _, correct := anomaly(j)
			results <- correct
		}(j)
		j++
	}
	for total < Trials {
		correct := <-results
		if correct {
			count++
		}
		total++
	}
	fmt.Printf("count=%v/%v\n", count, total)
}
