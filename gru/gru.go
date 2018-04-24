package gru

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// GRU is a GRU based anomaly detection engine
type GRU struct {
	*Model
	learner *CharRNN
	solver  *G.RMSPropSolver
}

// NewGRU creates a new GRU anomaly detection engine
func NewGRU() *GRU {
	steps := 4
	vocabulary := NewVocabularyFromRange(0, 256)

	inputSize := len(vocabulary.List)
	embeddingSize := 10
	outputSize := len(vocabulary.List)
	hiddenSizes := []int{10}
	stddev := 0.08
	gru := NewModel(inputSize, embeddingSize, outputSize, hiddenSizes, stddev)

	learner := NewCharRNN(gru, vocabulary)
	err := learner.ModeLearn(steps)
	if err != nil {
		panic(err)
	}

	learnrate := 0.01
	l2reg := 0.000001
	clipVal := 5.0
	solver := G.NewRMSPropSolver(G.WithLearnRate(learnrate), G.WithL2Reg(l2reg), G.WithClip(clipVal))

	return &GRU{
		Model:   gru,
		learner: learner,
		solver:  solver,
	}
}

// Train trains the GRU
func (g *GRU) Train(input []byte) float32 {
	data := make([]rune, len(input))
	for i, v := range input {
		data[i] = rune(v)
	}
	cost, _, err := g.learner.Learn(data, 0, g.solver)
	if err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
	average := 0.0
	for _, v := range cost {
		average += v
	}
	return float32(average / float64(len(cost)))
}
