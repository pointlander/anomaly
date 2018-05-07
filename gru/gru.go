package gru

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// GRU is a GRU based anomaly detection engine
type GRU struct {
	*Model
	learner, inference *CharRNN
	solver             *G.RMSPropSolver
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
	inference := NewCharRNN(gru, vocabulary)
	err = inference.ModeInference()
	if err != nil {
		panic(err)
	}

	learnrate := 0.01
	l2reg := 0.000001
	clipVal := 5.0
	solver := G.NewRMSPropSolver(G.WithLearnRate(learnrate), G.WithL2Reg(l2reg), G.WithClip(clipVal))

	return &GRU{
		Model:     gru,
		learner:   learner,
		inference: inference,
		solver:    solver,
	}
}

// Train trains the GRU
func (g *GRU) Train(input []byte) float32 {
	cost := g.inference.Cost(input)

	data := make([]rune, len(input))
	for i, v := range input {
		data[i] = rune(v)
	}
	_, _, err := g.learner.Learn(data, 0, g.solver)
	if err != nil {
		panic(fmt.Sprintf("%+v", err))
	}

	return float32(cost) / float32(len(input))
}
