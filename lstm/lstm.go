package lstm

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// LSTM is a LSTM based anomaly detection engine
type LSTM struct {
	*model
	learner *charRNN
	solver  *G.RMSPropSolver
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

	learnrate := 0.01
	l2reg := 0.000001
	clipVal := 5.0
	solver := G.NewRMSPropSolver(G.WithLearnRate(learnrate), G.WithL2Reg(l2reg), G.WithClip(clipVal))

	return &LSTM{
		model:   lstm,
		learner: learner,
		solver:  solver,
	}
}

// Train trains the LSTM
func (l *LSTM) Train(input []byte) float32 {
	data := make([]rune, len(input))
	for i, v := range input {
		data[i] = rune(v)
	}
	cost, _, err := l.learner.Learn(data, 0, l.solver)
	if err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
	average := 0.0
	for _, v := range cost {
		average += v
	}
	return float32(average / float64(len(cost)))
}
