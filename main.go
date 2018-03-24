// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"

	"github.com/pointlander/neural"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

const (
	// VectorSize is the size of the JSON document vector
	VectorSize = 1024
	// Samples is the number of JSON documents to generate per trial
	Samples = 1000
	// Trials is the number of trials
	Trials = 100
	// Parallelization is how many trials to perform in parallel
	Parallelization = 10
	// Cutoff is the number of initial samples to ignore
	Cutoff = 100
)

// Tests are basic tests for anomaly detection
var Tests = []string{`{
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

func tanh32(x float32) float32 {
	a, b := math.Exp(float64(x)), math.Exp(-float64(x))
	return float32((a - b) / (a + b))
}

func dtanh32(x float32) float32 {
	return 1 - x*x
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

// TestResult is a test result
type TestResult struct {
	Surprise         float64
	SimilarityBefore float64
	SimilarityAfter  float64
}

// TestResults are the test results from Anomaly
type TestResults struct {
	Seed              int
	AutoencoderError  plotter.Values
	Average, STDDEV   float64
	Similarity        plotter.Values
	AverageSimilarity []float64
	Results           []TestResult
}

// Anomaly tests the anomaly detection algorithm
func Anomaly(seed int) *TestResults {
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
	vectorizer := NewVectorizer(true, NewLFSR32Source)
	vectors, averageSimilarity := make([][]float32, 0, Samples), make([]float64, Samples)
	autoencoderError, similarity := make(plotter.Values, Samples), make(plotter.Values, Samples)
	for i := 0; i < Samples; i++ {
		object := GenerateRandomJSON(rnd)
		vector := vectorizer.Vectorize(object)
		unit := Normalize(vector)
		input := Adapt(unit)

		if length := len(vectors); length > 0 {
			sum := 0.0
			for _, v := range vectors {
				sum += Similarity(unit, v) + 1
			}
			averageSimilarity[i] = sum / float64(length)
		}

		context.SetInput(input)
		context.Infer()
		outputs := context.GetOutput()
		similarity[i] = float64(Similarity(input, outputs))

		source := func(iterations int) [][][]float32 {
			data := make([][][]float32, 1)
			data[0] = [][]float32{input, input}
			return data
		}
		e := nn.Train(source, 1, 0.6, 0.4)
		//e := nn.Train(source, 1, 0.1, 0.0001)
		autoencoderError[i] = float64(e[0])
		vectors = append(vectors, unit)
	}
	autoencoderError = autoencoderError[Cutoff:]
	similarity = similarity[Cutoff:]

	sum, sumSquared, length := 0.0, 0.0, float64(len(autoencoderError))
	for _, v := range autoencoderError {
		value := float64(v)
		sum += value
		sumSquared += value * value
	}
	average := sum / length
	stddev := math.Sqrt(sumSquared/length - average*average)

	results := make([]TestResult, len(Tests))
	for i, test := range Tests {
		var object map[string]interface{}
		err := json.Unmarshal([]byte(test), &object)
		if err != nil {
			panic(err)
		}
		vector := vectorizer.Vectorize(object)
		unit := Normalize(vector)
		input := Adapt(unit)

		context.SetInput(input)
		context.Infer()
		outputs := context.GetOutput()
		results[i].SimilarityBefore = float64(Similarity(input, outputs))

		source := func(iterations int) [][][]float32 {
			data := make([][][]float32, 1)
			data[0] = [][]float32{input, input}
			return data
		}
		e := nn.Train(source, 1, 0.6, 0.4)
		//e := nn.Train(source, 1, 0.1, 0.0001)
		results[i].Surprise = math.Abs((float64(e[0]) - average) / stddev)

		context.SetInput(input)
		context.Infer()
		outputs = context.GetOutput()
		results[i].SimilarityAfter = float64(Similarity(input, outputs))
	}

	return &TestResults{
		Seed:              seed,
		AutoencoderError:  autoencoderError,
		Average:           average,
		STDDEV:            stddev,
		Similarity:        similarity,
		AverageSimilarity: averageSimilarity[Cutoff:],
		Results:           results,
	}
}

// IsCorrect determines if a result is IsCorrect
func (t *TestResults) IsCorrect() bool {
	return t.Results[0].Surprise > t.Results[1].Surprise
}

// Process processes the results from Anomaly
func (t *TestResults) Process() {
	graph := 1
	histogram := func(title, name string, values plotter.Values) {
		p, err := plot.New()
		if err != nil {
			panic(err)
		}
		p.Title.Text = title

		h, err := plotter.NewHist(values[100:], 20)
		if err != nil {
			panic(err)
		}
		h.Normalize(1)
		p.Add(h)

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("graph_%v_%v", graph, name))
		if err != nil {
			panic(err)
		}

		graph++
	}

	histogram("Autoencoder Error Distribution", "autoencoder_error_distribution.png", t.AutoencoderError)
	histogram("Similarity Distribution", "similarity_distribution.png", t.Similarity)

	scatterPlot := func(xTitle, yTitle, name string, xys plotter.XYs) {
		x, y, x2, y2, xy, n := 0.0, 0.0, 0.0, 0.0, 0.0, float64(len(xys))
		for i := range xys {
			x += xys[i].X
			y += xys[i].Y
			x2 += xys[i].X * xys[i].X
			y2 += xys[i].Y * xys[i].Y
			xy += xys[i].X * xys[i].Y
		}
		corr := (n*xy - x*y) / (math.Sqrt(n*x2-x*x) * math.Sqrt(n*y2-y*y))

		p, err := plot.New()
		if err != nil {
			panic(err)
		}

		p.Title.Text = fmt.Sprintf("%v vs %v corr=%v", yTitle, xTitle, corr)
		p.X.Label.Text = xTitle
		p.Y.Label.Text = yTitle

		s, err := plotter.NewScatter(xys)
		if err != nil {
			panic(err)
		}
		p.Add(s)

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("graph_%v_%v", graph, name))
		if err != nil {
			panic(err)
		}

		graph++
	}

	xys := make(plotter.XYs, len(t.AutoencoderError))
	for i, v := range t.AverageSimilarity {
		xys[i].X = v
		xys[i].Y = t.AutoencoderError[i]
	}
	scatterPlot("Average Similarity", "Autoencoder Error", "autoencoder_error_vs_average_similarity.png", xys)

	for i, v := range t.AutoencoderError {
		xys[i].X = float64(i)
		xys[i].Y = v
	}
	scatterPlot("Time", "Autoencoder Error", "autoencoder_error.png", xys)

	for i, v := range t.AverageSimilarity {
		xys[i].X = float64(i)
		xys[i].Y = v
	}
	scatterPlot("Time", "Average Similarity", "average_similarity.png", xys)
}

// Print prints test results
func (t *TestResults) Print() {
	results := t.Results
	fmt.Printf("%v %v %v\n", t.Seed, results[0].Surprise, results[1].Surprise)
}

var images = flag.Bool("images", false, "run images demo")
var lfsr = flag.Bool("lfsr", false, "run lfsr")

func main() {
	flag.Parse()

	if *images {
		imagesDemo()
		return
	}

	if *lfsr {
		searchLFSR32()
		return
	}

	result := Anomaly(1)
	result.Process()
	result.Print()

	count, total, results, j := 0, 0, make(chan *TestResults, Parallelization), 1
	process := func() {
		result := <-results
		result.Print()
		if result.IsCorrect() {
			count++
		}
		total++
	}
	for i := 0; i < Parallelization; i++ {
		go func(j int) {
			results <- Anomaly(j)
		}(j)
		j++
	}
	for j <= Trials {
		process()
		go func(j int) {
			results <- Anomaly(j)
		}(j)
		j++
	}
	for total < Trials {
		process()
	}
	fmt.Printf("count=%v/%v\n", count, total)
}
