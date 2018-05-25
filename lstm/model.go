package lstm

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
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
	wix   G.Value
	wih   G.Value
	biasI G.Value

	wfx   G.Value
	wfh   G.Value
	biasF G.Value

	wox   G.Value
	woh   G.Value
	biasO G.Value

	wcx   G.Value
	wch   G.Value
	biasC G.Value
}

type lstm struct {
	wix   *G.Node
	wih   *G.Node
	biasI *G.Node

	wfx   *G.Node
	wfh   *G.Node
	biasF *G.Node

	wox   *G.Node
	woh   *G.Node
	biasO *G.Node

	wcx   *G.Node
	wch   *G.Node
	biasC *G.Node
}

func newLSTMLayer(g *G.ExprGraph, l *layer, name string) *lstm {
	retVal := new(lstm)
	retVal.wix = G.NodeFromAny(g, l.wix, G.WithName("wix_"+name))
	retVal.wih = G.NodeFromAny(g, l.wih, G.WithName("wih_"+name))
	retVal.biasI = G.NodeFromAny(g, l.biasI, G.WithName("bias_i_"+name))

	retVal.wfx = G.NodeFromAny(g, l.wfx, G.WithName("wfx_"+name))
	retVal.wfh = G.NodeFromAny(g, l.wfh, G.WithName("wfh_"+name))
	retVal.biasF = G.NodeFromAny(g, l.biasF, G.WithName("bias_f_"+name))

	retVal.wox = G.NodeFromAny(g, l.wox, G.WithName("wox_"+name))
	retVal.woh = G.NodeFromAny(g, l.woh, G.WithName("woh_"+name))
	retVal.biasO = G.NodeFromAny(g, l.biasO, G.WithName("bias_o_"+name))

	retVal.wcx = G.NodeFromAny(g, l.wcx, G.WithName("wcx_"+name))
	retVal.wch = G.NodeFromAny(g, l.wch, G.WithName("wch_"+name))
	retVal.biasC = G.NodeFromAny(g, l.biasC, G.WithName("bias_c_"+name))
	return retVal
}

func (l *lstm) fwd(inputVector, prevHidden, prevCell *G.Node) (hidden, cell *G.Node) {
	var h0, h1, inputGate *G.Node
	h0 = G.Must(G.Mul(l.wix, inputVector))
	h1 = G.Must(G.Mul(l.wih, prevHidden))
	inputGate = G.Must(G.Sigmoid(G.Must(G.Add(G.Must(G.Add(h0, h1)), l.biasI))))

	var h2, h3, forgetGate *G.Node
	h2 = G.Must(G.Mul(l.wfx, inputVector))
	h3 = G.Must(G.Mul(l.wfh, prevHidden))
	forgetGate = G.Must(G.Sigmoid(G.Must(G.Add(G.Must(G.Add(h2, h3)), l.biasF))))

	var h4, h5, outputGate *G.Node
	h4 = G.Must(G.Mul(l.wox, inputVector))
	h5 = G.Must(G.Mul(l.woh, prevHidden))
	outputGate = G.Must(G.Sigmoid(G.Must(G.Add(G.Must(G.Add(h4, h5)), l.biasO))))

	var h6, h7, cellWrite *G.Node
	h6 = G.Must(G.Mul(l.wcx, inputVector))
	h7 = G.Must(G.Mul(l.wch, prevHidden))
	cellWrite = G.Must(G.Tanh(G.Must(G.Add(G.Must(G.Add(h6, h7)), l.biasC))))

	// cell activations
	var retain, write *G.Node
	retain = G.Must(G.HadamardProd(forgetGate, prevCell))
	write = G.Must(G.HadamardProd(inputGate, cellWrite))
	cell = G.Must(G.Add(retain, write))
	hidden = G.Must(G.HadamardProd(outputGate, G.Must(G.Tanh(cell))))
	return
}

// Model single LSTM layer
type Model struct {
	ls []*layer

	// decoder
	whd   G.Value
	biasD G.Value

	embedding G.Value

	// metadata
	inputSize, embeddingSize, outputSize int
	hiddenSizes                          []int

	prefix string
	free   bool
}

type lstmOut struct {
	hiddens G.Nodes
	cells   G.Nodes

	probs *G.Node
}

// NewLSTMModel creates a new LSTM model
func NewLSTMModel(rnd *rand.Rand, inputSize, embeddingSize, outputSize int, hiddenSizes []int) *Model {
	m := new(Model)
	m.inputSize = inputSize
	m.embeddingSize = embeddingSize
	m.outputSize = outputSize
	m.hiddenSizes = hiddenSizes

	gaussian32 := func(s ...int) []float32 {
		size := tensor.Shape(s).TotalSize()
		weights, stdev := make([]float32, size), math.Sqrt(2/float64(s[len(s)-1]))
		for i := range weights {
			weights[i] = float32(rnd.NormFloat64() * stdev)
		}
		return weights
	}

	for depth := 0; depth < len(hiddenSizes); depth++ {
		prevSize := embeddingSize
		if depth > 0 {
			prevSize = hiddenSizes[depth-1]
		}
		hiddenSize := hiddenSizes[depth]
		l := new(layer)
		m.ls = append(m.ls, l) // add layer to model

		// input gate weights

		l.wix = tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(gaussian32(hiddenSize, prevSize)))
		l.wih = tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(gaussian32(hiddenSize, hiddenSize)))
		l.biasI = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))

		// output gate weights

		l.wox = tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(gaussian32(hiddenSize, prevSize)))
		l.woh = tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(gaussian32(hiddenSize, hiddenSize)))
		l.biasO = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))

		// forget gate weights

		l.wfx = tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(gaussian32(hiddenSize, prevSize)))
		l.wfh = tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(gaussian32(hiddenSize, hiddenSize)))
		l.biasF = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))

		// cell write

		l.wcx = tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(gaussian32(hiddenSize, prevSize)))
		l.wch = tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(gaussian32(hiddenSize, hiddenSize)))
		l.biasC = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
	}

	lastHiddenSize := hiddenSizes[len(hiddenSizes)-1]

	m.whd = tensor.New(tensor.WithShape(outputSize, lastHiddenSize), tensor.WithBacking(gaussian32(outputSize, lastHiddenSize)))
	m.biasD = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(outputSize))

	m.embedding = tensor.New(tensor.WithShape(embeddingSize, inputSize), tensor.WithBacking(gaussian32(embeddingSize, inputSize)))
	return m
}

// CharRNN is a LSTM that takes characters as input
type CharRNN struct {
	*Model
	*Vocabulary

	g  *G.ExprGraph
	ls []*lstm

	// decoder
	whd   *G.Node
	biasD *G.Node

	embedding *G.Node

	prevHiddens G.Nodes
	prevCells   G.Nodes

	steps            int
	inputs           []*tensor.Dense
	outputs          []*tensor.Dense
	previous         []*lstmOut
	cost, perplexity *G.Node
	machine          G.VM
}

// NewCharRNN create a new LSTM for characters as inputs
func NewCharRNN(m *Model, vocabulary *Vocabulary) *CharRNN {
	r := new(CharRNN)
	r.Model = m
	r.Vocabulary = vocabulary
	g := G.NewGraph()
	r.g = g

	var hiddens, cells G.Nodes
	for depth := 0; depth < len(m.hiddenSizes); depth++ {
		hiddenSize := m.hiddenSizes[depth]
		layerID := strconv.Itoa(depth)
		l := newLSTMLayer(r.g, r.Model.ls[depth], layerID)
		r.ls = append(r.ls, l)

		// this is to simulate a default "previous" state
		hiddenT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		cellT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		hidden := G.NewVector(g, G.Float32, G.WithName("prevHidden_"+layerID), G.WithShape(hiddenSize), G.WithValue(hiddenT))
		cell := G.NewVector(g, G.Float32, G.WithName("prevCell_"+layerID), G.WithShape(hiddenSize), G.WithValue(cellT))

		hiddens = append(hiddens, hidden)
		cells = append(cells, cell)
	}
	r.whd = G.NodeFromAny(r.g, m.whd, G.WithName("whd"))
	r.biasD = G.NodeFromAny(r.g, m.biasD, G.WithName("bias_d"))
	r.embedding = G.NodeFromAny(r.g, m.embedding, G.WithName("Embedding"))

	// these are to simulate a previous state
	r.prevHiddens = hiddens
	r.prevCells = cells

	return r
}

func (r *CharRNN) learnables() (retVal G.Nodes) {
	for _, l := range r.ls {
		lin := G.Nodes{
			l.wix,
			l.wih,
			l.biasI,
			l.wfx,
			l.wfh,
			l.biasF,
			l.wox,
			l.woh,
			l.biasO,
			l.wcx,
			l.wch,
			l.biasC,
		}

		retVal = append(retVal, lin...)
	}

	retVal = append(retVal, r.whd)
	retVal = append(retVal, r.biasD)
	retVal = append(retVal, r.embedding)
	return
}

func (r *CharRNN) fwd(prev *lstmOut) (inputTensor *tensor.Dense, retVal *lstmOut, err error) {
	prevHiddens := r.prevHiddens
	prevCells := r.prevCells
	if prev != nil {
		prevHiddens = prev.hiddens
		prevCells = prev.cells
	}

	var hiddens, cells G.Nodes
	for i, l := range r.ls {
		var inputVector *G.Node
		if i == 0 {
			inputTensor = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(r.inputSize))
			input := G.NewVector(r.g, tensor.Float32, G.WithShape(r.inputSize), G.WithValue(inputTensor))
			inputVector = G.Must(G.Mul(r.embedding, input))
		} else {
			inputVector = hiddens[i-1]
		}
		prevHidden := prevHiddens[i]
		prevCell := prevCells[i]

		hidden, cell := l.fwd(inputVector, prevHidden, prevCell)
		hiddens = append(hiddens, hidden)
		cells = append(cells, cell)
	}
	lastHidden := hiddens[len(hiddens)-1]
	var output *G.Node
	if output, err = G.Mul(r.whd, lastHidden); err == nil {
		if output, err = G.Add(output, r.biasD); err != nil {
			G.WithName("LAST HIDDEN")(lastHidden)
			ioutil.WriteFile("err.dot", []byte(lastHidden.RestrictedToDot(3, 10)), 0644)
			panic(fmt.Sprintf("ERROR: %v", err))
		}
	}

	var probs *G.Node
	probs = G.Must(G.SoftMax(output))

	retVal = &lstmOut{
		hiddens: hiddens,
		cells:   cells,
		probs:   probs,
	}
	return
}

func (r *CharRNN) feedback(tap int) {
	prev := r.previous[tap]
	for i := range r.prevHiddens {
		input := r.prevHiddens[i].Value().(*tensor.Dense)
		output := prev.hiddens[i].Value().(*tensor.Dense)
		err := output.CopyTo(input)
		if err != nil {
			panic(err)
		}
	}
	for i := range r.prevCells {
		input := r.prevCells[i].Value().(*tensor.Dense)
		output := prev.cells[i].Value().(*tensor.Dense)
		err := output.CopyTo(input)
		if err != nil {
			panic(err)
		}
	}
}

func (r *CharRNN) reset() {
	for i := range r.prevHiddens {
		r.prevHiddens[i].Value().(*tensor.Dense).Zero()
	}
	for i := range r.prevCells {
		r.prevCells[i].Value().(*tensor.Dense).Zero()
	}
}

// ModeLearn puts the CharRNN into a learning mode
func (r *CharRNN) ModeLearn(steps int) (err error) {
	inputs := make([]*tensor.Dense, steps-1)
	outputs := make([]*tensor.Dense, steps-1)
	previous := make([]*lstmOut, steps-1)
	var cost, perplexity *G.Node

	for i := 0; i < steps-1; i++ {
		var loss, perp *G.Node
		// cache

		var prev *lstmOut
		if i > 0 {
			prev = previous[i-1]
		}
		inputs[i], previous[i], err = r.fwd(prev)
		if err != nil {
			return
		}

		logprob := G.Must(G.Neg(G.Must(G.Log(previous[i].probs))))
		outputs[i] = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(r.outputSize))
		output := G.NewVector(r.g, tensor.Float32, G.WithShape(r.outputSize), G.WithValue(outputs[i]))
		loss = G.Must(G.Mul(logprob, output))
		log2prob := G.Must(G.Neg(G.Must(G.Log2(previous[i].probs))))
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
		return
	}

	r.machine = G.NewTapeMachine(r.g, G.BindDualValues(r.learnables()...))
	return
}

// ModeInference puts the CharRNN into inference mode
func (r *CharRNN) ModeInference() (err error) {
	inputs := make([]*tensor.Dense, 1)
	outputs := make([]*tensor.Dense, 1)

	previous := make([]*lstmOut, 1)
	inputs[0], previous[0], err = r.fwd(nil)
	if err != nil {
		return
	}
	logprob := G.Must(G.Neg(G.Must(G.Log(previous[0].probs))))
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

// ModeInferencePredict puts the CharRNN into inference prediction mode
func (r *CharRNN) ModeInferencePredict() (err error) {
	inputs := make([]*tensor.Dense, 1)
	previous := make([]*lstmOut, 1)

	inputs[0], previous[0], err = r.fwd(nil)
	if err != nil {
		return
	}

	r.inputs = inputs
	r.previous = previous
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

		sampledID := sample(r.previous[0].probs.Value())
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

		sampledID := maxSample(r.previous[0].probs.Value())

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
