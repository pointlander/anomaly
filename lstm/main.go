package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"runtime/pprof"
	"syscall"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	T "gorgonia.org/gorgonia"

	"net/http"
	_ "net/http/pprof"
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
var memprofile = flag.String("memprofile", "", "write memory profile to this file")

// prediction params
var softmaxTemperature = 1.0
var maxCharGen = 100

// various global variable inits
var epochSize = -1
var inputSize = -1
var outputSize = -1

// gradient update stuff
var l2reg = 0.000001
var learnrate = 0.0001
var clipVal = 5.0

type contextualError interface {
	error
	Node() *T.Node
	Value() T.Value
	InstructionID() int
}

func cleanup(sigChan chan os.Signal, doneChan chan bool, profiling bool) {
	select {
	case <-sigChan:
		log.Println("EMERGENCY EXIT!")
		graph()
		if profiling {
			pprof.StopCPUProfile()
		}
		os.Exit(1)

	case <-doneChan:
		return
	}
}

var costValues, perpValues = make(plotter.Values, 0, 1000), make(plotter.Values, 0, 10000)

func graph() {
	graph := 0

	scatterPlot := func(xTitle, yTitle, name string, yy plotter.Values) {
		xys := make(plotter.XYs, len(yy))
		for i, v := range yy {
			xys[i].X = float64(i)
			xys[i].Y = v
		}

		x, y, x2, y2, xy, n := 0.0, 0.0, 0.0, 0.0, 0.0, float64(len(xys))
		for i := range xys {
			x += xys[i].X
			y += xys[i].Y
			x2 += xys[i].X * xys[i].X
			y2 += xys[i].Y * xys[i].Y
			xy += xys[i].X * xys[i].Y
		}
		corr := (n*xy - x*y) / (math.Sqrt(n*x2-x*x) * math.Sqrt(n*y2-y*y))

		p, err := plot.New()
		if err != nil {
			panic(err)
		}

		p.Title.Text = fmt.Sprintf("%v vs %v corr=%v", yTitle, xTitle, corr)
		p.X.Label.Text = xTitle
		p.Y.Label.Text = yTitle

		s, err := plotter.NewScatter(xys)
		if err != nil {
			panic(err)
		}
		p.Add(s)

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("graph_%v_%v", graph, name))
		if err != nil {
			panic(err)
		}

		graph++
	}

	scatterPlot("Time", "Cost", "cost_vs_time.png", costValues)
	scatterPlot("Time", "Perplexity", "cost_vs_perplexity.png", perpValues)
}

func main() {
	flag.Parse()
	rand.Seed(1337)

	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	// intercept Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	doneChan := make(chan bool, 1)
	// defer func() {
	// 	nn, cc, ec := T.GraphCollisionStats()
	// 	log.Printf("COLLISION COUNT: %d/%d. Expected : %d", cc, nn, ec)
	// }()

	var profiling bool
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		profiling = true
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	go cleanup(sigChan, doneChan, profiling)

	m := NewLSTMModel(inputSize, embeddingSize, outputSize, hiddenSizes)
	r := newCharRNN(m)

	solver := T.NewRMSPropSolver(T.WithLearnRate(learnrate), T.WithL2Reg(l2reg), T.WithClip(clipVal))
	start := time.Now()
	eStart := start
	for i := 0; i <= 100000; i++ {
		// log.Printf("Iter: %d", i)
		// _, _, err := m.run(i, solver)
		cost, perp, err := run(r, i, solver)
		if err != nil {
			panic(fmt.Sprintf("%+v", err))
		}

		if i%1000 == 0 {
			log.Printf("Going to predict now")
			r.predict()
			log.Printf("Done predicting")

			old := r
			r = newCharRNN(m)
			old.cleanup()
			log.Printf("New RNN - m.embeddint %v", m.embedding.Shape())
		}

		if i%100 == 0 {
			timetaken := time.Since(eStart)
			fmt.Printf("Time Taken: %v\tCost: %v\tPerplexity: %v\n", timetaken, cost, perp)
			eStart = time.Now()
			costValues = append(costValues, float64(cost))
			perpValues = append(perpValues, float64(perp))
		}

		if *memprofile != "" && i == 1000 {
			f, err := os.Create(*memprofile)
			if err != nil {
				log.Fatal(err)
			}
			pprof.WriteHeapProfile(f)
			f.Close()
			return
		}

	}

	graph()

	end := time.Now()
	fmt.Printf("%v", end.Sub(start))
	fmt.Printf("%+3.3s", m.embedding)
}
