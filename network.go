// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package anomaly

import (
	"math/rand"

	"github.com/pointlander/anomaly/gru"
	"github.com/pointlander/anomaly/lstm"
)

// Network is a network for calculating surprise
type Network interface {
	Train(input []byte) (surprise, uncertainty float32)
}

// NetworkFactory produces new networks
type NetworkFactory func(rnd *rand.Rand, vectorizer *Vectorizer) Network

// NewLSTM creates a new LSTM network
func NewLSTM(rnd *rand.Rand, vectorizer *Vectorizer) Network {
	return lstm.NewLSTM()
}

// NewGRU creates a new GRU network
func NewGRU(rnd *rand.Rand, vectorizer *Vectorizer) Network {
	return gru.NewGRU()
}
