// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package anomaly

import (
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
}

// NewNeuron creates a new neuron
func NewNeuron(width int, rnd *rand.Rand) Network {
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
	}
}

// Train trains the neuron
func (n *Neuron) Train(input []float32) float32 {
	for i, v := range input {
		n.I.SetAt(v, i)
	}

	err := n.RunAll()
	if err != nil {
		panic(err)
	}
	defer n.Reset()

	cs := n.CS.Value().Data().(float32)

	err = n.Step(n.Nodes)
	if err != nil {
		panic(err)
	}

	return float32(math.Abs(float64(cs)))
}
