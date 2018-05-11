// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package anomaly

// Autoencoder is a autoencoding neural network
import (
	"encoding/json"
	"math"
	"math/rand"

	"github.com/pointlander/neural"
)

// Autoencoder is an autoencoding neural network
type Autoencoder struct {
	*neural.Neural32
	*Vectorizer
}

// NewAutoencoder creates an autoencoder
func NewAutoencoder(rnd *rand.Rand, vectorizer *Vectorizer) Network {
	width := vectorizer.Size
	config := func(n *neural.Neural32) {
		random32 := func(a, b float32) float32 {
			return (b-a)*rnd.Float32() + a
		}
		weightInitializer := func(in, out int) float32 {
			return random32(-1, 1) / float32(math.Sqrt(float64(in)))
		}
		n.Init(weightInitializer, width, width/2, width)
	}
	nn := neural.NewNeural32(config)
	return &Autoencoder{
		Neural32:   nn,
		Vectorizer: vectorizer,
	}
}

// Train calculates the surprise with the autoencoder
func (a *Autoencoder) Train(input []byte) (surprise, uncertainty float32) {
	var object map[string]interface{}
	err := json.Unmarshal(input, &object)
	if err != nil {
		panic(err)
	}
	vector := a.Vectorizer.Vectorize(object)
	unit := Normalize(vector)

	unit = Adapt(unit)
	source := func(iterations int) [][][]float32 {
		data := make([][][]float32, 1)
		data[0] = [][]float32{unit, unit}
		return data
	}
	e := a.Neural32.Train(source, 1, 0.6, 0.4)
	return e[0], 0
}
