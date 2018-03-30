package anomaly

import (
	"math/rand"
)

// Network is a network for calculating surprise
type Network interface {
	Train(input []float32) float32
}

// NetworkFactory produces new networks
type NetworkFactory func(width int, rnd *rand.Rand) Network
