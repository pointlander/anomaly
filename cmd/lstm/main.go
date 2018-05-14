package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"runtime/pprof"
	"strings"
	"syscall"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	T "gorgonia.org/gorgonia"

	"github.com/pointlander/anomaly/lstm"
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
var memprofile = flag.String("memprofile", "", "write memory profile to this file")

func cleanup(sigChan chan os.Signal, doneChan chan bool, profiling bool) {
	select {
	case <-sigChan:
		log.Println("EMERGENCY EXIT!")
		stop = true
	case <-doneChan:
		return
	}
}

var stop = false
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
		s.Radius = 1
		p.Add(s)

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("graph_%v_%v", graph, name))
		if err != nil {
			panic(err)
		}

		graph++
	}

	scatterPlot("Time", "Cost", "cost_vs_time.png", costValues)
	scatterPlot("Time", "Perplexity", "perplexity_vs_time.png", perpValues)
}

func main() {
	//T.Use(blase.Implementation())
	fmt.Println(T.WhichBLAS())
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

	steps := 8
	var sentences [][]rune
	sentencesRaw := strings.Split(lstm.Corpus, "\n")
	sentencesRaw = []string{strings.Join(sentencesRaw, " ")}
	//sentencesRaw = []string{"abababababababababab"}
	for _, s := range sentencesRaw {
		s2 := []rune(strings.TrimSpace(s))
		length := len(s2) + steps
		s3 := make([]rune, length)
		copy(s3[1:], s2)
		s3[0] = lstm.START
		for i := 1; i < steps; i++ {
			s3[length-i] = lstm.END
		}
		sentences = append(sentences, s3)
	}
	vocabulary := lstm.NewVocabulary(sentences, 1)

	inputSize := len(vocabulary.List)
	embeddingSize := 10
	outputSize := len(vocabulary.List)
	hiddenSizes := []int{100, 100}
	rnd := rand.New(rand.NewSource(1))
	m := lstm.NewLSTMModel(rnd, inputSize, embeddingSize, outputSize, hiddenSizes)
	r := lstm.NewCharRNN(m, vocabulary)
	err := r.ModeLearn(steps)
	if err != nil {
		panic(err)
	}

	predict := lstm.NewCharRNN(m, vocabulary)
	err = predict.ModeInference()
	if err != nil {
		panic(err)
	}

	learnrate := 0.000001
	l2reg := 0.000001
	clipVal := 5.0
	solver := T.NewRMSPropSolver(T.WithLearnRate(learnrate), T.WithL2Reg(l2reg), T.WithClip(clipVal))
	start := time.Now()
	eStart := start
	for i := 0; i <= 100000 && !stop; i++ {
		// log.Printf("Iter: %d", i)
		// _, _, err := m.run(i, solver)
		j := rand.Intn(len(sentences))
		cost, perp, err := r.Learn(sentences[j], i, solver)
		if err != nil {
			panic(fmt.Sprintf("%+v", err))
		}
		costAvg, perpAvg := 0.0, 0.0
		for _, v := range cost {
			costAvg += v
			costValues = append(costValues, v)
		}
		for _, v := range perp {
			perpAvg += v
			perpValues = append(perpValues, v)
		}
		costAvg /= float64(len(cost))
		perpAvg /= float64(len(perp))

		if i%1 == 0 {
			log.Printf("Going to predict now")
			predict.Predict()
			log.Printf("Done predicting")

			timetaken := time.Since(eStart)
			fmt.Printf("Time Taken: %v\tCost: %.3f\tPerplexity: %.3f\n", timetaken, costAvg, perpAvg)
			eStart = time.Now()
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
}
