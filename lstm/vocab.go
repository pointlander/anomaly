package main

const (
	// START is the start symbol
	START rune = 0x02
	// END is the end symbol
	END rune = 0x03
)

// Vocabulary maps between runes and ints
type Vocabulary struct {
	List  []rune
	Index map[rune]int
}

// NewVocabulary create a new vocabulary list
func NewVocabulary(ss [][]rune, thresh int) *Vocabulary {
	dict := make(map[rune]int)
	for _, s := range ss {
		for _, r := range s {
			dict[r]++
		}
	}

	list, index := []rune{START}, make(map[rune]int)

	for ch, c := range dict {
		if c >= thresh && ch != START && ch != END {
			// then add letter to vocab
			list = append(list, ch)
		}
	}

	list = append(list, END)

	for i, v := range list {
		index[v] = i
	}

	return &Vocabulary{
		List:  list,
		Index: index,
	}
}

// NewVocabularyFromRange create a new vocabulary list using a range
func NewVocabularyFromRange(start, stop rune) *Vocabulary {
	list, index := make([]rune, 0), make(map[rune]int)
	for i := start; i < stop; i++ {
		list = append(list, i)
	}
	for i, v := range list {
		index[v] = i
	}

	return &Vocabulary{
		List:  list,
		Index: index,
	}
}
