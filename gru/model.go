package gru

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"strconv"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// prediction params
var softmaxTemperature = 1.0
var maxCharGen = 100

type contextualError interface {
	error
	Node() *G.Node
	Value() G.Value
	InstructionID() int
}

type layer struct {
	wf *tensor.Dense
	uf *tensor.Dense
	br *tensor.Dense

	wh *tensor.Dense
	uh *tensor.Dense
	bh *tensor.Dense

	ones *tensor.Dense
}

// Model is a GRU model
type Model struct {
	layers []*layer
	we     *tensor.Dense
	wo     *tensor.Dense
	bo     *tensor.Dense

	inputSize, embeddingSize, outputSize int
	layerSizes                           []int
}

// NewModel creates a new GRU model
func NewModel(inputSize, embeddingSize, outputSize int, layerSizes []int, stddev float64) *Model {
	model := &Model{
		inputSize:     inputSize,
		embeddingSize: embeddingSize,
		outputSize:    outputSize,
		layerSizes:    layerSizes,
	}
	model.we = tensor.New(tensor.WithShape(embeddingSize, inputSize),
		tensor.WithBacking(G.Gaussian32(0.0, stddev, embeddingSize, inputSize)))

	previous := embeddingSize
	for _, size := range layerSizes {
		layer := &layer{}
		model.layers = append(model.layers, layer)

		layer.wf = tensor.New(tensor.WithShape(size, previous),
			tensor.WithBacking(G.Gaussian32(0.0, stddev, size, previous)))
		layer.uf = tensor.New(tensor.WithShape(size, size),
			tensor.WithBacking(G.Gaussian32(0.0, stddev, size, size)))
		layer.br = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(size))

		layer.wh = tensor.New(tensor.WithShape(size, previous),
			tensor.WithBacking(G.Gaussian32(0.0, stddev, size, previous)))
		layer.uh = tensor.New(tensor.WithShape(size, size),
			tensor.WithBacking(G.Gaussian32(0.0, stddev, size, size)))
		layer.bh = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(size))

		layer.ones = tensor.Ones(tensor.Float32, size)

		previous = size
	}

	model.wo = tensor.New(tensor.WithShape(outputSize, previous),
		tensor.WithBacking(G.Gaussian32(0.0, stddev, outputSize, previous)))
	model.bo = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(outputSize))

	return model
}

type gru struct {
	wf *G.Node
	uf *G.Node
	br *G.Node

	wh *G.Node
	uh *G.Node
	bh *G.Node

	ones *G.Node
}

func (l *layer) NewGRULayer(g *G.ExprGraph, name string) *gru {
	wf := G.NodeFromAny(g, l.wf, G.WithName("wf_"+name))
	uf := G.NodeFromAny(g, l.uf, G.WithName("uf_"+name))
	br := G.NodeFromAny(g, l.br, G.WithName("br_"+name))

	wh := G.NodeFromAny(g, l.wh, G.WithName("wh_"+name))
	uh := G.NodeFromAny(g, l.uh, G.WithName("uh_"+name))
	bh := G.NodeFromAny(g, l.bh, G.WithName("bh_"+name))

	ones := G.NodeFromAny(g, l.ones, G.WithName("ones_"+name))
	return &gru{
		wf:   wf,
		uf:   uf,
		br:   br,
		wh:   wh,
		uh:   uh,
		bh:   bh,
		ones: ones,
	}
}

func (g *gru) fwd(input, previous *G.Node) *G.Node {
	x := G.Must(G.Mul(g.wf, input))
	y := G.Must(G.Mul(g.uf, previous))
	f := G.Must(G.Sigmoid(G.Must(G.Add(G.Must(G.Add(x, y)), g.br))))

	x = G.Must(G.Mul(g.wh, input))
	y = G.Must(G.Mul(g.uh, G.Must(G.HadamardProd(f, previous))))
	z := G.Must(G.Tanh(G.Must(G.Add(G.Must(G.Add(x, y)), g.bh))))

	a := G.Must(G.HadamardProd(G.Must(G.Sub(g.ones, f)), z))
	b := G.Must(G.HadamardProd(f, previous))

	return G.Must(G.Add(a, b))
}

type gruOut struct {
	hiddens       G.Nodes
	probabilities *G.Node
}

// CharRNN is a LSTM that takes characters as input
type CharRNN struct {
	*Model
	layers []*gru

	*Vocabulary

	g       *G.ExprGraph
	we      *G.Node
	wo      *G.Node
	bo      *G.Node
	hiddens G.Nodes

	steps            int
	inputs           []*tensor.Dense
	outputs          []*tensor.Dense
	previous         []*gruOut
	cost, perplexity *G.Node
	machine          G.VM
}

// NewCharRNN create a new GRU for characters as inputs
func NewCharRNN(model *Model, vocabulary *Vocabulary) *CharRNN {
	g := G.NewGraph()
	var layers []*gru
	var hiddens G.Nodes
	for i, v := range model.layerSizes {
		name := strconv.Itoa(i)
		layer := model.layers[i].NewGRULayer(g, name)
		layers = append(layers, layer)

		hiddenTensor := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(v))
		hidden := G.NewVector(g, G.Float32, G.WithName("prevHidden_"+name),
			G.WithShape(v), G.WithValue(hiddenTensor))
		hiddens = append(hiddens, hidden)
	}
	we := G.NodeFromAny(g, model.we, G.WithName("we"))
	wo := G.NodeFromAny(g, model.wo, G.WithName("wo"))
	bo := G.NodeFromAny(g, model.bo, G.WithName("bo"))
	return &CharRNN{
		Model:      model,
		layers:     layers,
		Vocabulary: vocabulary,
		g:          g,
		we:         we,
		wo:         wo,
		bo:         bo,
		hiddens:    hiddens,
	}
}

func (r *CharRNN) learnables() (value G.Nodes) {
	for _, l := range r.layers {
		nodes := G.Nodes{
			l.wf,
			l.uf,
			l.br,
			l.wh,
			l.uh,
			l.bh,
		}
		value = append(value, nodes...)
	}

	value = append(value, r.we)
	value = append(value, r.wo)
	value = append(value, r.bo)

	return
}

func (r *CharRNN) fwd(previous *gruOut) (inputTensor *tensor.Dense, retVal *gruOut, err error) {
	previousHiddens := r.hiddens
	if previous != nil {
		previousHiddens = previous.hiddens
	}

	var hiddens G.Nodes
	for i, v := range r.layers {
		var inputVector *G.Node
		if i == 0 {
			inputTensor = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(r.inputSize))
			input := G.NewVector(r.g, tensor.Float32, G.WithShape(r.inputSize), G.WithValue(inputTensor))
			inputVector = G.Must(G.Mul(r.we, input))
		} else {
			inputVector = hiddens[i-1]
		}

		hidden := v.fwd(inputVector, previousHiddens[i])
		hiddens = append(hiddens, hidden)
	}
	lastHidden := hiddens[len(hiddens)-1]
	var output *G.Node
	if output, err = G.Mul(r.wo, lastHidden); err == nil {
		if output, err = G.Add(output, r.bo); err != nil {
			G.WithName("LAST HIDDEN")(lastHidden)
			ioutil.WriteFile("err.dot", []byte(lastHidden.RestrictedToDot(3, 10)), 0644)
			panic(fmt.Sprintf("ERROR: %v", err))
		}
	}

	var probs *G.Node
	probs = G.Must(G.SoftMax(output))

	retVal = &gruOut{
		hiddens:       hiddens,
		probabilities: probs,
	}

	return
}

func (r *CharRNN) feedback(tap int) {
	prev := r.previous[tap]
	for i := range r.hiddens {
		input := r.hiddens[i].Value().(*tensor.Dense)
		output := prev.hiddens[i].Value().(*tensor.Dense)
		err := output.CopyTo(input)
		if err != nil {
			panic(err)
		}
	}
}

func (r *CharRNN) reset() {
	for i := range r.hiddens {
		r.hiddens[i].Value().(*tensor.Dense).Zero()
	}
}

// ModeLearn puts the CharRNN into a learning mode
func (r *CharRNN) ModeLearn(steps int) (err error) {
	inputs := make([]*tensor.Dense, steps-1)
	outputs := make([]*tensor.Dense, steps-1)
	previous := make([]*gruOut, steps-1)
	var cost, perplexity *G.Node

	for i := 0; i < steps-1; i++ {
		var loss, perp *G.Node

		var prev *gruOut
		if i > 0 {
			prev = previous[i-1]
		}
		inputs[i], previous[i], err = r.fwd(prev)
		if err != nil {
			return
		}

		logprob := G.Must(G.Neg(G.Must(G.Log(previous[i].probabilities))))
		outputs[i] = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(r.outputSize))
		output := G.NewVector(r.g, tensor.Float32, G.WithShape(r.outputSize), G.WithValue(outputs[i]))
		loss = G.Must(G.Mul(logprob, output))
		log2prob := G.Must(G.Neg(G.Must(G.Log2(previous[i].probabilities))))
		perp = G.Must(G.Mul(log2prob, output))

		if cost == nil {
			cost = loss
		} else {
			cost = G.Must(G.Add(cost, loss))
		}
		G.WithName("Cost")(cost)

		if perplexity == nil {
			perplexity = perp
		} else {
			perplexity = G.Must(G.Add(perplexity, perp))
		}
	}

	r.steps = steps
	r.inputs = inputs
	r.outputs = outputs
	r.previous = previous
	r.cost = cost
	r.perplexity = perplexity

	_, err = G.Grad(cost, r.learnables()...)
	if err != nil {
	}

	r.machine = G.NewTapeMachine(r.g, G.BindDualValues(r.learnables()...))
	return
}

// ModeInference puts the CharRNN into inference mode
func (r *CharRNN) ModeInference() (err error) {
	inputs := make([]*tensor.Dense, 1)
	outputs := make([]*tensor.Dense, 1)
	previous := make([]*gruOut, 1)

	inputs[0], previous[0], err = r.fwd(nil)
	if err != nil {
		return
	}
	logprob := G.Must(G.Neg(G.Must(G.Log(previous[0].probabilities))))
	outputs[0] = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(r.outputSize))
	output := G.NewVector(r.g, tensor.Float32, G.WithShape(r.outputSize), G.WithValue(outputs[0]))
	cost := G.Must(G.Mul(logprob, output))

	r.inputs = inputs
	r.outputs = outputs
	r.previous = previous
	r.cost = cost
	r.machine = G.NewTapeMachine(r.g)
	return
}

// Predict genreates a string
func (r *CharRNN) Predict() {
	var sentence []rune
	var err error

	r.reset()
	for {
		var id int
		if len(sentence) > 0 {
			id = r.Index[sentence[len(sentence)-1]]
		}
		r.inputs[0].Zero()
		r.inputs[0].SetF32(id, 1.0)

		// f, _ := os.Create("log1.log")
		// logger := log.New(f, "", 0)
		// machine := NewLispMachine(g, ExecuteFwdOnly(), WithLogger(logger), WithWatchlist(), LogBothDir())
		if err = r.machine.RunAll(); err != nil {
			if ctxerr, ok := err.(contextualError); ok {
				ioutil.WriteFile("FAIL1.dot", []byte(ctxerr.Node().RestrictedToDot(3, 3)), 0644)
			}
			log.Printf("ERROR1 while predicting with %v %+v", r.machine, err)
		}

		sampledID := sample(r.previous[0].probabilities.Value())
		//fmt.Println(r.previous[0].probs.Value())
		var char rune // hur hur varchar
		if char = r.List[sampledID]; char == END {
			break
		}

		if len(sentence) > maxCharGen {
			break
		}

		sentence = append(sentence, char)
		r.feedback(0)
		r.machine.Reset()
	}

	var sentence2 []rune
	r.reset()
	for {
		var id int
		if len(sentence2) > 0 {
			id = r.Index[sentence2[len(sentence2)-1]]
		}
		r.inputs[0].Zero()
		r.inputs[0].SetF32(id, 1.0)

		// f, _ := os.Create("log2.log")
		// logger := log.New(f, "", 0)
		// machine := NewLispMachine(g, ExecuteFwdOnly(), WithLogger(logger), WithWatchlist(), LogBothDir())
		if err = r.machine.RunAll(); err != nil {
			if ctxerr, ok := err.(contextualError); ok {
				log.Printf("Instruction ID %v", ctxerr.InstructionID())
				ioutil.WriteFile("FAIL2.dot", []byte(ctxerr.Node().RestrictedToDot(3, 3)), 0644)
			}
			log.Printf("ERROR2 while predicting with %v: %+v", r.machine, err)
		}

		sampledID := maxSample(r.previous[0].probabilities.Value())

		var char rune // hur hur varchar
		if char = r.List[sampledID]; char == END {
			break
		}

		if len(sentence2) > maxCharGen {
			break
		}

		sentence2 = append(sentence2, char)
		r.feedback(0)
		r.machine.Reset()
	}

	fmt.Printf("Sampled: %q; \nArgMax: %q\n", string(sentence), string(sentence2))
}

// Cost computes the cost of the input
func (r *CharRNN) Cost(input []byte) float32 {
	var cost float32
	r.reset()
	for i := range input[:len(input)-1] {
		r.inputs[0].Zero()
		r.inputs[0].SetF32(int(input[i]), 1.0)
		r.outputs[0].Zero()
		r.outputs[0].SetF32(int(input[i+1]), 1.0)
		err := r.machine.RunAll()
		if err != nil {
			panic(err)
		}
		if cv, ok := r.cost.Value().(G.Scalar); ok {
			cost += cv.Data().(float32)
		}
		r.feedback(0)
		r.machine.Reset()
	}
	return cost
}

// Learn learns strings
func (r *CharRNN) Learn(sentence []rune, iter int, solver G.Solver) (retCost, retPerp []float64, err error) {
	n := len(sentence)

	r.reset()
	steps := r.steps - 1
	for x := 0; x < n-steps; x++ {
		for j := 0; j < steps; j++ {
			source := sentence[x+j]
			target := sentence[x+j+1]

			r.inputs[j].Zero()
			r.inputs[j].SetF32(r.Index[source], 1.0)
			r.outputs[j].Zero()
			r.outputs[j].SetF32(r.Index[target], 1.0)
		}

		// f, _ := os.Create("FAIL.log")
		// logger := log.New(f, "", 0)
		// machine := NewLispMachine(g, WithLogger(logger), WithValueFmt("%-1.1s"), LogBothDir(), WithWatchlist())

		if err = r.machine.RunAll(); err != nil {
			if ctxerr, ok := err.(contextualError); ok {
				ioutil.WriteFile("FAIL.dot", []byte(ctxerr.Node().RestrictedToDot(3, 3)), 0644)

			}
			return
		}

		err = solver.Step(r.learnables())
		if err != nil {
			return
		}

		if sv, ok := r.perplexity.Value().(G.Scalar); ok {
			v := sv.Data().(float32)
			retPerp = append(retPerp, math.Pow(2, float64(v)/(float64(n)-1)))
		}
		if cv, ok := r.cost.Value().(G.Scalar); ok {
			retCost = append(retCost, float64(cv.Data().(float32)))
		}
		r.feedback(0)
		r.machine.Reset()
	}

	return
}
