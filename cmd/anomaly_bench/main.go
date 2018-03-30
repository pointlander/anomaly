// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"

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
	Surprise float64
}

// TestResults are the test results from Anomaly
type TestResults struct {
	Seed            int
	Surprise        plotter.Values
	Average, STDDEV float64
	Results         []TestResult
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
func Anomaly(seed int, factory anomaly.NetworkFactory) *TestResults {
	rndGenerator := rand.New(rand.NewSource(int64(seed)))
	rndNetwork := rand.New(rand.NewSource(int64(seed)))
	vectorizer := anomaly.NewVectorizer(VectorSize, true, anomaly.NewLFSR32Source)
	network := factory(VectorSize, rndNetwork)

	surprise := make(plotter.Values, Samples)
	for i := 0; i < Samples; i++ {
		object := anomaly.GenerateRandomJSON(rndGenerator)
		vector := vectorizer.Vectorize(object)
		unit := anomaly.Normalize(vector)
		surprise[i] = float64(network.Train(unit))
	}
	surprise = surprise[Cutoff:]

	average, stddev := statistics(surprise)

	results := make([]TestResult, len(Tests))
	for i, test := range Tests {
		var object map[string]interface{}
		err := json.Unmarshal([]byte(test), &object)
		if err != nil {
			panic(err)
		}
		vector := vectorizer.Vectorize(object)
		unit := anomaly.Normalize(vector)
		e := float64(network.Train(unit))
		results[i].Surprise = math.Abs((e - average) / stddev)
	}

	return &TestResults{
		Seed:     seed,
		Surprise: surprise,
		Average:  average,
		STDDEV:   stddev,
		Results:  results,
	}
}

// IsCorrect determines if a result is IsCorrect
func (t *TestResults) IsCorrect() bool {
	return t.Results[0].Surprise > t.Results[1].Surprise
}

// Print prints test results
func (t *TestResults) Print() {
	results := t.Results
	fmt.Printf("%v %v %v\n", t.Seed, results[0].Surprise, results[1].Surprise)
}

func main() {
	graph := 1

	histogram := func(title, name string, values plotter.Values) {
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

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("graph_%v_%v", graph, name))
		if err != nil {
			panic(err)
		}

		graph++
	}

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

	averageSimilarityResult := Anomaly(1, anomaly.NewAverageSimilarity)
	histogram("Average Similarity Distribution", "average_similarity_distribution.png", averageSimilarityResult.Surprise)
	xys := make(plotter.XYs, len(averageSimilarityResult.Surprise))
	for i, v := range averageSimilarityResult.Surprise {
		xys[i].X = float64(i)
		xys[i].Y = v
	}
	scatterPlot("Time", "Average Similarity", "average_similarity.png", xys)
	averageSimilarityResult.Print()

	autoencoderErrorResult := Anomaly(1, anomaly.NewAutoencoder)
	histogram("Autoencoder Error Distribution", "autoencoder_error_distribution.png", autoencoderErrorResult.Surprise)
	for i, v := range autoencoderErrorResult.Surprise {
		xys[i].X = float64(i)
		xys[i].Y = v
	}
	scatterPlot("Time", "Autoencoder Error", "autoencoder_error.png", xys)
	for i, v := range averageSimilarityResult.Surprise {
		xys[i].X = v
		xys[i].Y = autoencoderErrorResult.Surprise[i]
	}
	scatterPlot("Average Similarity", "Autoencoder Error", "autoencoder_error_vs_average_similarity.png", xys)
	autoencoderErrorResult.Print()

	neuronResult := Anomaly(1, anomaly.NewNeuron)
	histogram("Neuron Distribution", "neuron_distribution.png", neuronResult.Surprise)
	for i, v := range neuronResult.Surprise {
		xys[i].X = float64(i)
		xys[i].Y = v
	}
	scatterPlot("Time", "Neuron", "neuron.png", xys)
	for i, v := range averageSimilarityResult.Surprise {
		xys[i].X = v
		xys[i].Y = neuronResult.Surprise[i]
	}
	scatterPlot("Average Similarity", "Neuron", "neuron_vs_average_similarity.png", xys)
	neuronResult.Print()

	test := func(factory anomaly.NetworkFactory) {
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
				results <- Anomaly(j, factory)
			}(j)
			j++
		}
		for j <= Trials {
			process()
			go func(j int) {
				results <- Anomaly(j, factory)
			}(j)
			j++
		}
		for total < Trials {
			process()
		}
		fmt.Printf("count=%v / %v\n", count, total)
	}
	test(anomaly.NewAverageSimilarity)
	test(anomaly.NewNeuron)
	test(anomaly.NewAutoencoder)
}
