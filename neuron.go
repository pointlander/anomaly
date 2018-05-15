// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package anomaly

import (
	"encoding/json"
	"math"
	"math/rand"

	gg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Neuron is a single neuron
type Neuron struct {
	I, W *tensor.Dense
	CS   *gg.Node
	*gg.ExprGraph
	gg.VM
	gg.Nodes
	*gg.VanillaSolver
	*Vectorizer
}

// NewNeuron creates a new neuron
func NewNeuron(rnd *rand.Rand, vectorizer *Vectorizer) Network {
	width := vectorizer.Size
	ii := tensor.NewDense(tensor.Float32, tensor.Shape{width})
	ww := tensor.NewDense(tensor.Float32, tensor.Shape{width})

	random32 := func(a, b float32) float32 {
		return (b-a)*rnd.Float32() + a
	}
	fanIn := float32(math.Sqrt(float64(width)))
	for i := 0; i < width; i++ {
		ww.SetAt(random32(-1, 1)/fanIn, i)
	}

	g := gg.NewGraph()
	i := gg.NewVector(g, tensor.Float32, gg.WithShape(width), gg.WithName("i"), gg.WithValue(ii))
	w := gg.NewVector(g, tensor.Float32, gg.WithShape(width), gg.WithName("w"), gg.WithValue(ww))
	iw := gg.Must(gg.Mul(i, w))
	mi := gg.Must(gg.Sqrt(gg.Must(gg.Sum(gg.Must(gg.Square(i))))))
	mw := gg.Must(gg.Sqrt(gg.Must(gg.Sum(gg.Must(gg.Square(w))))))
	cs := gg.Must(gg.Div(iw, gg.Must(gg.Mul(mi, mw))))

	one := gg.NewScalar(g, tensor.Float32, gg.WithValue(float32(1.0)))
	cost := gg.Must(gg.Square(gg.Must(gg.Sub(one, cs))))

	_, err := gg.Grad(cost, w)
	if err != nil {
		panic(err)
	}

	return &Neuron{
		I:             ii,
		W:             ww,
		CS:            cs,
		ExprGraph:     g,
		VM:            gg.NewTapeMachine(g, gg.BindDualValues(w)),
		Nodes:         gg.Nodes{w},
		VanillaSolver: gg.NewVanillaSolver(gg.WithLearnRate(0.5)),
		Vectorizer:    vectorizer,
	}
}

// Train trains the neuron
func (n *Neuron) Train(input []byte) (surprise, uncertainty float32) {
	var object map[string]interface{}
	err := json.Unmarshal(input, &object)
	if err != nil {
		panic(err)
	}
	vector := n.Vectorizer.Vectorize(object)
	unit := Normalize(vector)

	for i, v := range unit {
		n.I.SetAt(v, i)
	}

	err = n.RunAll()
	if err != nil {
		panic(err)
	}
	defer n.Reset()

	cs := n.CS.Value().Data().(float32)

	err = n.Step(n.Nodes)
	if err != nil {
		panic(err)
	}

	return float32(math.Abs(float64(cs))), 0
}
