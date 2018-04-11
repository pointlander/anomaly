package main

import (
	"strings"
)

const (
	// START is the start symbol
	START rune = 0x02
	// END is the end symbol
	END rune = 0x03
	// Steps is how many steps for backpropagation in time there is
	Steps = 8
)

// vocab related
var sentences [][]rune
var vocab []rune
var vocabIndex map[rune]int

func initVocab(ss [][]rune, thresh int) {
	dict := make(map[rune]int)
	for _, s := range ss {
		for _, r := range s {
			dict[r]++
		}
	}

	vocab = append(vocab, START)
	vocabIndex = make(map[rune]int)

	for ch, c := range dict {
		if c >= thresh && ch != START && ch != END {
			// then add letter to vocab
			vocab = append(vocab, ch)
		}
	}

	vocab = append(vocab, END)

	for i, v := range vocab {
		vocabIndex[v] = i
	}

	inputSize = len(vocab)
	outputSize = len(vocab)
	epochSize = len(ss)
}

func init() {
	sentencesRaw := strings.Split(corpus, "\n")
	//sentencesRaw = []string{"abababababababababab"}
	for _, s := range sentencesRaw {
		s2 := []rune(strings.TrimSpace(s))
		length := len(s2) + 2
		s3 := make([]rune, length)
		copy(s3[1:], s2)
		s3[0] = START
		s3[length-1] = END
		sentences = append(sentences, s3)
	}

	initVocab(sentences, 1)
}
