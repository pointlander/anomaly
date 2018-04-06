package main

import "strings"

const (
	// START is the start symbol
	START rune = 0x02
	// END is the end symbol
	END rune = 0x03
	// Steps is how many steps for backpropagation in time there is
	Steps = 8
)

// vocab related
var sentences []string
var vocab []rune
var vocabIndex map[rune]int

func initVocab(ss []string, thresh int) {
	s := strings.Join(ss, "")
	dict := make(map[rune]int)
	for _, r := range s {
		dict[r]++
	}

	vocab = append(vocab, START)
	vocabIndex = make(map[rune]int)

	for ch, c := range dict {
		if c >= thresh {
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
	for _, s := range sentencesRaw {
		s2 := strings.TrimSpace(s)
		length := len(s2)
		if length >= Steps {
			start := make([]rune, Steps)
			start[0] = START
			copy(start[1:], []rune(s2))
			sentences = append(sentences, string(start))
			for i := 0; i <= length-Steps; i++ {
				sentences = append(sentences, s2[i:i+Steps])
			}
			end := make([]rune, Steps)
			end[Steps-1] = END
			copy(end, []rune(s2[length-Steps+1:]))
			sentences = append(sentences, string(end))
		}
	}

	initVocab(sentences, 1)
}
