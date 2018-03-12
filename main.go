// Copyright 2017 The Anomaly Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/pointlander/neural"
)

const (
	testImage = "images/pexels-photo-103573.jpeg"
	blockSize = 8
	netWidth  = 3 * blockSize * blockSize
	hiddens   = netWidth / 64
)

func main() {
	file, err := os.Open(testImage)
	if err != nil {
		log.Fatal(err)
	}

	input, _, err := image.Decode(file)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()

	width, height := input.Bounds().Max.X, input.Bounds().Max.Y

	config := func(n *neural.Neural32) {
		n.Init(neural.WeightInitializer32FanIn, netWidth, hiddens, netWidth)
	}
	codec := neural.NewNeural32(config)

	fmt.Println("Load")
	var patterns [][][]float32
	for j := 0; j < height; j += blockSize {
		for i := 0; i < width; i += blockSize {
			pixels, p := make([]float32, netWidth), 0
			for y := 0; y < blockSize; y++ {
				for x := 0; x < blockSize; x++ {
					r, g, b, _ := input.At(i+x, j+y).RGBA()
					pixels[p] = float32(r) / 0xFFFF
					p++
					pixels[p] = float32(g) / 0xFFFF
					p++
					pixels[p] = float32(b) / 0xFFFF
					p++
				}
			}
			patterns = append(patterns, [][]float32{pixels, pixels})
		}
	}

	size := len(patterns)
	randomized := make([][][]float32, size)
	copy(randomized, patterns)
	source := func(iterations int) [][][]float32 {
		for i := 0; i < size; i++ {
			j := i + rand.Intn(size-i)
			randomized[i], randomized[j] = randomized[j], randomized[i]
		}
		return randomized
	}
	fmt.Println("Train")
	errors := codec.Train(source, 3, 0.6, 0.4)
	fmt.Println(errors)

	mse, context := make([]float32, size), codec.NewContext()
	min, max := float32(math.MaxFloat32), float32(0)
	for p, pattern := range patterns {
		context.SetInput(pattern[0])
		context.Infer()
		outputs := context.GetOutput()
		sum := float32(0)
		for i, j := range outputs {
			k := pattern[0][i] - j
			sum += k * k
		}
		mse[p] = sum / float32(len(outputs))
		if mse[p] < min {
			min = mse[p]
		} else if mse[p] > max {
			max = mse[p]
		}
	}
	fmt.Printf("min = %v max = %v\n", min, max)

	width, height = width/blockSize, height/blockSize
	img := image.NewGray(image.Rect(0, 0, width, height))
	c := 0
	for j := 0; j < height; j++ {
		for i := 0; i < width; i++ {
			pix := (mse[c] - min) / (max - min)
			img.SetGray(i, j, color.Gray{Y: uint8(255 * pix)})
			c++
		}
	}

	file, err = os.Create("mse.png")
	if err != nil {
		log.Fatal(err)
	}

	err = png.Encode(file, img)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()

	img = image.NewGray(image.Rect(0, 0, width, height))
	c = 0
	for j := 0; j < height; j++ {
		for i := 0; i < width; i++ {
			pix := (mse[c] - min) / (max - min)
			if pix > .05 {
				pix = 1
			}
			img.SetGray(i, j, color.Gray{Y: uint8(255 * pix)})
			c++
		}
	}

	file, err = os.Create("mset.png")
	if err != nil {
		log.Fatal(err)
	}

	err = png.Encode(file, img)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()
}
