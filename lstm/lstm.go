package lstm

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// LSTM is a LSTM based anomaly detection engine
type LSTM struct {
	*Model
	learner, inference *CharRNN
	solver             *G.RMSPropSolver
}

// NewLSTM creates a new LSTM anomaly detection engine
func NewLSTM() *LSTM {
	steps := 4
	vocabulary := NewVocabularyFromRange(0, 256)

	inputSize := len(vocabulary.List)
	embeddingSize := 10
	outputSize := len(vocabulary.List)
	hiddenSizes := []int{10}
	stddev := 0.08
	lstm := NewLSTMModel(inputSize, embeddingSize, outputSize, hiddenSizes, stddev)

	learner := NewCharRNN(lstm, vocabulary)
	err := learner.ModeLearn(steps)
	if err != nil {
		panic(err)
	}
	inference := NewCharRNN(lstm, vocabulary)
	err = inference.ModeInference()
	if err != nil {
		panic(err)
	}

	learnrate := 0.01
	l2reg := 0.000001
	clipVal := 5.0
	solver := G.NewRMSPropSolver(G.WithLearnRate(learnrate), G.WithL2Reg(l2reg), G.WithClip(clipVal))

	return &LSTM{
		Model:     lstm,
		learner:   learner,
		inference: inference,
		solver:    solver,
	}
}

// Train trains the LSTM
func (l *LSTM) Train(input []byte) (surprise, uncertainty float32) {
	cost := l.inference.Cost(input)

	data := make([]rune, len(input))
	for i, v := range input {
		data[i] = rune(v)
	}
	_, _, err := l.learner.Learn(data, 0, l.solver)
	if err != nil {
		panic(fmt.Sprintf("%+v", err))
	}

	return float32(cost) / float32(len(input)), 0
}
