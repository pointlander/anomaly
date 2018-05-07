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
	Train(input []float32) float32
}

// NetworkFactory produces new networks
type NetworkFactory func(width int, rnd *rand.Rand) Network

// ByteNetwork is a network for calculating surprise from bytes
type ByteNetwork interface {
	Train(input []byte) float32
}

// ByteNetworkFactory produces new byte networks
type ByteNetworkFactory func() ByteNetwork

// NewLSTM creates a new LSTM network
func NewLSTM() ByteNetwork {
	return lstm.NewLSTM()
}

// NewGRU creates a new GRU network
func NewGRU() ByteNetwork {
	return gru.NewGRU()
}
