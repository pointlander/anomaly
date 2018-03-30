package anomaly

import (
	"math"
	"math/rand"
)

// AverageSimilarity computes surpise by calculation the average cosine
// similarity across all Vectors
type AverageSimilarity struct {
	Vectors [][]float32
}

// NewAverageSimilarity creates a new average similarity surprise engine
func NewAverageSimilarity(width int, rnd *rand.Rand) Network {
	return &AverageSimilarity{
		Vectors: make([][]float32, 0, 1024),
	}
}

// Train computes the surprise with average similarity
func (a *AverageSimilarity) Train(input []float32) float32 {
	var averageSimilarity float32
	if length := len(a.Vectors); length > 0 {
		sum := 0.0
		for _, v := range a.Vectors {
			sum += math.Abs(Similarity(input, v))
		}
		averageSimilarity = float32(sum / float64(length))
	}
	a.Vectors = append(a.Vectors, input)
	return averageSimilarity
}
