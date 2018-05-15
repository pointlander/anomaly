// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package anomaly

import (
	"math"
	"math/bits"
	"math/rand"
)

// Meta is a meta anomaly detection engine that uses other engines
type Meta struct {
	Models []*CDF16
	Rand   *rand.Rand
}

// NewMeta creates a new meta engine
func NewMeta(rnd *rand.Rand, vectorizer *Vectorizer) Network {
	models := make([]*CDF16, 8)
	for i := range models {
		models[i] = NewCDF16()
	}
	return &Meta{
		Models: models,
		Rand:   rand.New(rand.NewSource(1)),
	}
}

// Train trains the meta engine
func (m *Meta) Train(input []byte) (surprise, uncertainty float32) {
	sum, sumSquared := 0.0, 0.0
	for _, model := range m.Models {
		var total uint64
		for _, s := range input {
			cdf := model.Model()
			total += uint64(bits.Len16(cdf[s+1] - cdf[s]))
			model.AddContext(uint16(s))
		}
		model.ResetContext()

		sample := float64(CDF16Fixed+1) - (float64(total) / float64(len(input)))
		sum += sample
		sumSquared += sample * sample

		if m.Rand.Intn(2) == 0 {
			for _, s := range input {
				model.Update(uint16(s))
			}
			model.ResetContext()
		}
	}

	length := float64(len(m.Models))
	average := sum / length
	surprise = float32(average)
	uncertainty = float32(math.Sqrt(sumSquared/length - average*average))
	return
}
