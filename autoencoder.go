package anomaly

// Autoencoder is a autoencoding neural network
import (
	"math"
	"math/rand"

	"github.com/pointlander/neural"
)

// Autoencoder is an autoencoding neural network
type Autoencoder struct {
	*neural.Neural32
}

// NewAutoencoder creates an autoencoder
func NewAutoencoder(width int, rnd *rand.Rand) Network {
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
		Neural32: nn,
	}
}

// Train calculates the surprise with the autoencoder
func (a *Autoencoder) Train(input []float32) float32 {
	input = Adapt(input)
	source := func(iterations int) [][][]float32 {
		data := make([][][]float32, 1)
		data[0] = [][]float32{input, input}
		return data
	}
	e := a.Neural32.Train(source, 1, 0.6, 0.4)
	return e[0]
}
