// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"

	"github.com/pointlander/neural"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	"github.com/pointlander/anomaly"
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

func tanh32(x float32) float32 {
	a, b := math.Exp(float64(x)), math.Exp(-float64(x))
	return float32((a - b) / (a + b))
}

func dtanh32(x float32) float32 {
	return 1 - x*x
}

// TestResult is a test result
type TestResult struct {
	Surprise         float64
	Sim              float64
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
	Sim               plotter.Values
	Results           []TestResult
}

func statistics(values plotter.Values) (average, stddev float64) {
	sum, sumSquared, length := 0.0, 0.0, float64(len(values))
	for _, v := range values {
		value := float64(v)
		sum += value
		sumSquared += value * value
	}
	average = sum / length
	stddev = math.Sqrt(sumSquared/length - average*average)
	return
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
	vectorizer := anomaly.NewVectorizer(VectorSize, true, anomaly.NewLFSR32Source)
	vectors, averageSimilarity := make([][]float32, 0, Samples), make([]float64, Samples)
	autoencoderError, similarity := make(plotter.Values, Samples), make(plotter.Values, Samples)
	sim := make(plotter.Values, Samples)
	neuron := anomaly.NewNeuron(VectorSize, rnd)
	for i := 0; i < Samples; i++ {
		object := anomaly.GenerateRandomJSON(rnd)
		vector := vectorizer.Vectorize(object)
		unit := anomaly.Normalize(vector)
		input := anomaly.Adapt(unit)

		if length := len(vectors); length > 0 {
			sum := 0.0
			for _, v := range vectors {
				sum += anomaly.Similarity(unit, v) + 1
			}
			averageSimilarity[i] = sum / float64(length)
		}

		context.SetInput(input)
		context.Infer()
		outputs := context.GetOutput()
		similarity[i] = float64(anomaly.Similarity(input, outputs))

		source := func(iterations int) [][][]float32 {
			data := make([][][]float32, 1)
			data[0] = [][]float32{input, input}
			return data
		}
		e := nn.Train(source, 1, 0.6, 0.4)
		//e := nn.Train(source, 1, 0.1, 0.0001)
		autoencoderError[i] = float64(e[0])
		sim[i] = float64(neuron.Train(unit))
		vectors = append(vectors, unit)
	}
	autoencoderError = autoencoderError[Cutoff:]
	similarity = similarity[Cutoff:]
	sim = sim[Cutoff:]

	average, stddev := statistics(autoencoderError)
	simaverage, simstddev := statistics(sim)

	results := make([]TestResult, len(Tests))
	for i, test := range Tests {
		var object map[string]interface{}
		err := json.Unmarshal([]byte(test), &object)
		if err != nil {
			panic(err)
		}
		vector := vectorizer.Vectorize(object)
		unit := anomaly.Normalize(vector)
		input := anomaly.Adapt(unit)

		context.SetInput(input)
		context.Infer()
		outputs := context.GetOutput()
		results[i].SimilarityBefore = float64(anomaly.Similarity(input, outputs))

		source := func(iterations int) [][][]float32 {
			data := make([][][]float32, 1)
			data[0] = [][]float32{input, input}
			return data
		}
		e := nn.Train(source, 1, 0.6, 0.4)
		//e := nn.Train(source, 1, 0.1, 0.0001)
		results[i].Surprise = math.Abs((float64(e[0]) - average) / stddev)
		results[i].Sim = math.Abs((float64(neuron.Train(unit)) - simaverage) / simstddev)

		context.SetInput(input)
		context.Infer()
		outputs = context.GetOutput()
		results[i].SimilarityAfter = float64(anomaly.Similarity(input, outputs))
	}

	return &TestResults{
		Seed:              seed,
		AutoencoderError:  autoencoderError,
		Average:           average,
		STDDEV:            stddev,
		Similarity:        similarity,
		AverageSimilarity: averageSimilarity[Cutoff:],
		Sim:               sim,
		Results:           results,
	}
}

// IsCorrect determines if a result is IsCorrect
func (t *TestResults) IsCorrect() bool {
	return t.Results[0].Surprise > t.Results[1].Surprise
}

// IsSimCorrect determines if a result is IsCorrect
func (t *TestResults) IsSimCorrect() bool {
	return t.Results[0].Sim > t.Results[1].Sim
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

	for i, v := range t.Sim {
		xys[i].X = v
		xys[i].Y = t.AutoencoderError[i]
	}
	scatterPlot("Similarity", "Autoencoder Error", "sim.png", xys)
}

// Print prints test results
func (t *TestResults) Print() {
	results := t.Results
	fmt.Printf("%v %v %v %v %v\n", t.Seed, results[0].Surprise, results[1].Surprise, results[0].Sim, results[1].Sim)
}

func main() {
	result := Anomaly(1)
	result.Process()
	result.Print()

	count, simcount, total, results, j := 0, 0, 0, make(chan *TestResults, Parallelization), 1
	process := func() {
		result := <-results
		result.Print()
		if result.IsCorrect() {
			count++
		}
		if result.IsSimCorrect() {
			simcount++
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
	fmt.Printf("count=%v %v / %v\n", count, simcount, total)
}
