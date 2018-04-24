# Anomaly detection for JSON documents

## Organization
* cmd/anomaly_bench - Anomaly detection for JSON documents prototype code
* cmd/anomaly_image - Anomaly detection for images
* cmd/lstm - LSTM test code
* cmd/search_lfsr - Code for finding maximal length lfsr
* images - Test images for anomaly_image
* lstm - LSTM implementation

## Abstract
Standard statistical methods can be used for anomaly detection of one dimensional real valued data. The multidimensional nature of JSON documents makes anomaly detection more difficult. Firstly, this README proposes a two stage algorithm for the anomaly detection of JSON documents. The first stage of the algorithm uses [random matrix dimensionality reduction](https://en.wikipedia.org/wiki/Random_projection) to vectorize a JSON document into a fixed length vector (JSON document vector). The second stage of the algorithm uses one of three methods: average [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), a single neuron, or an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) to determine how surprising the JSON document vector is. Secondly, this README proposes using a [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) for anomaly detection. Simple statistical analysis can then be used for determining which JSON documents the user should be alerted to.

## Background
* [Random projection](https://en.wikipedia.org/wiki/Random_projection)
* [Anomaly detection with autoencoders](http://philipperemy.github.io/anomaly-detection/)

## Two stage algorithm
### Vectorization (first stage)
#### Converting a JSON document into a vector
The typical approach for converting a document into a vector is to count words. Each entry in the vector corresponds to the count for a particular word. In order to capture more document struct word pairs could be counted instead. JSON documents have explicit struct that should be captured in the computed vector. For example the below JSON:
```json
{
 "a": [
  {"a": "aa"},
  {"b": "bb"}
 ],
 "b": [
  {"a": "aa"},
  {"b": "bb"}
 ]
}
```
would be converted into the below 12 "words" using a recursive algorithm:
1. a a aa
2. a aa
3. aa
4. a b bb
5. b bb
6. bb
7. b a aa
8. a aa
9. aa
10. b b bb
11. b bb
12. bb

The 8 unique words are ["a a aa", "a aa", "aa", "a b bb", "b bb", "bb", "b a aa", "b b bb"], and their vector is [1, 2, 2, 1, 2, 2, 1, 1]. For all possible JSON documents the vector would be very large, so an algorithm is needed for compressing this vector.

#### Random matrix dimensionality reduction
Dimensionality reduction compresses a large vector into a smaller vector. This is done by multiplying a vector by a matrix. [Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA) is the algorithm typically used, but it doesn't scale well for larger vectors. For very large vectors a random matrix can be used instead of a matrix generated by PCA. The random matrix is filled with (-1, 1, 0) with respective odds (1/6, 1/6, 4/6). For the vector from the previous section the matrix would look like: [0 1 0 0 -1 0 0 1; 1 0 0 -1 0 1 0 0]. This matrix would compress the 8 entry vector down to a 2 entry vector. As an optimization the matrix columns can be generated on the fly using a random number generator seeded with the hash of a particular word. Addition can then be used to replace multiplication.

The code for the vectorizer can be found [here](https://github.com/pointlander/anomaly/blob/master/vectorizer.go).

### Anomaly detection (second stage)
Three anomaly detection methods are described below. Each method computes a surprise metric from a JSON document vector.

#### With average cosine similarity
The average cosine similarity method works by computing the cosine similarity between a given JSON document vector and each member of a JSON document vector database. The average is then computed. This metric represents how close the given JSON document vector is to the database of document vectors on average. After computing the average cosine similarity the JSON document vector is added to the database. The algorithm gets slower with time.

The code for the average cosine similarity algorithm can be found [here](https://github.com/pointlander/anomaly/blob/master/average_similarity.go).

#### With a single neuron
A single neuron implemented with the cosine similarity formula can be used for anomaly detection. The single valued output of the neuron represents how surprising the inputed JSON document vector is. The single neuron is trained with a JSON document vector as input and 1 as the output.

The code for the single neuron algorithm can be found [here](https://github.com/pointlander/anomaly/blob/master/neuron.go).

#### With autoencoders
An autoencoding neural network isn't trained with labeled data, instead it is trained to output the input vector. The standard autoencoder has three layers. The top and bottom layers are the same size, and the middle layer is typically more narrow than the top and bottom layers. The narrow middle layer creates an information bottleneck. It is possible to compute an autoencoder error metric for a particular JSON document vector. This "surprise" metric is computed by inputing the JSON document vector into the neural network and then computing the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) at the output. The neural network can then be trained on the JSON document vector, so the neural network isn't surprised by similar JSON document vectors in the future.

The code for the autoencoder algorithm can be found [here](https://github.com/pointlander/anomaly/blob/master/autoencoder.go).

## LSTM algorithm
The LSTM takes a series of bytes as input and outputs a predicted next byte. The LSTM algorithm works by training a LSTM on JSON data. The cost of training is then used as a surprise metric of the JSON data. Unlike the above algorithms, the LSTM based solution is capable of anomaly detection for non-JSON binary protocols. The LSTM has a state which could be used as a JSON document vector as in the above algorithm.

The code for the LSTM algorithm can be found [here](https://github.com/pointlander/anomaly/blob/master/lstm/lstm.go).

## Benchmarks
The benchmarks are executed with:
```
go test -bench=.
```

```
BenchmarkLFSR-4                    	1000000000	         1.99 ns/op
BenchmarkSource-4                  	500000000	         3.77 ns/op
BenchmarkVectorizer-4              	    2000	    537426 ns/op
BenchmarkVectorizerLFSR-4          	   10000	    103953 ns/op
BenchmarkVectorizerNoCache-4       	    2000	   1075383 ns/op
BenchmarkVectorizerLFSRNoCache-4   	   10000	    200837 ns/op
BenchmarkAverageSimilarity-4       	   10000	   1260944 ns/op
BenchmarkNeuron-4                  	    5000	    222802 ns/op
BenchmarkAutoencoder-4             	     200	   7721645 ns/op
BenchmarkLSTM-4                    	      20	 113401111 ns/op
```
As can been seen the LSTM based algorithm is much slower than the two stage algorithm. The single neuron based approach is the fastest solution.

## Verification
### Two stage algorithm
Verification is accomplished by generating random JSON documents from a gaussian random variable and feeding them into the anomaly detection algorithm. The below graph shows the distribution resulting from average cosine similarity method being fed random JSON documents:

![Graph 1 average cosine similarity distribution](graph_1_average_similarity_distribution.png?raw=true)

For single neuron:

![Graph 3 neuron distribution](graph_3_neuron_distribution.png?raw=true)

For autoencoder:

![Graph 6 autoencoder error distribution](graph_6_autoencoder_error_distribution.png?raw=true)

As should be expected the graphs appears to be gaussian. Another test implemented feeds the below two JSON documents into each of the above three anomaly detection algorithms after they have been trained on 1000 random JSON documents:

First test JSON document:
```json
{
 "alfa": [
  {"alfa": "1"},
  {"bravo": "2"}
 ],
 "bravo": [
  {"alfa": "3"},
  {"bravo": "4"}
 ]
}
```

Second test JSON document:
```json
{
 "a": [
  {"a": "aa"},
  {"b": "bb"}
 ],
 "b": [
  {"a": "aa"},
  {"b": "bb"}
 ]
}
```

The second JSON document is more similar to the randomly generated JSON documents than the first JSON document. This idea is tested 100 times by changing the random seed used to generate the 1000 random JSON documents. All of the methods pass the test with a score near 100. This shows that the anomaly detection algorithm isn't a random number generator. One final test is performed by graphing the output of the single neuron method and the autoencoder method against the average cosine similarity method. The below graph 5 shows that the output of the single neuron correlates with average cosine similarity:

![Graph 5 neuron vs average similarity](graph_5_neuron_vs_average_similarity.png?raw=true)

The below graph 8 shows that the autoencoder method correlates with average cosine similarity:

![Graph 8 autoencoder error vs average similarity](graph_8_autoencoder_error_vs_average_similarity.png?raw=true)

The below three graphs show average cosine similarity, single neuron, and autoencoder are not correlated through time:

![Graph 2 average similarity vs time](graph_2_average_similarity.png?raw=true)

![Graph 2 neuron vs time](graph_4_neuron.png?raw=true)

![Graph 7 autoencoder error vs time](graph_7_autoencoder_error.png?raw=true)

### LSTM algorithm
The distribution of the LSTM surprise metrics for random JSON documents is very narrow:

![Graph 9 LSTM distribution](graph_9_lstm_distribution.png?raw=true)

The LSTM learning the random JSON documents can be seen in the below graph:

![Graph 10 LSTM vs time](graph_10_lstm.png?raw=true)

The LSTM algorithm doesn't correlate with the average similarity approach, so it is not clear what is being computed. This is to be expected as LSTMs are Turing complete:

![Graph 11 LSTM vs average similarity](graph_11_lstm_vs_average_similarity.png?raw=true)

The LSTM does correctly determine the surprise metrics of the above two test JSON documents. The surprise metric of the first test JSON document is greater than the surprise metric of the second test JSON document.

## Conclusion
An anomaly detection engine has been demonstrated. The first two stage algorithm is made up of two components: a vectorizer and an algorithm for computing surprise. After vectorization the single neuron and autoencoder algorithms have a fixed cost for determining if a JSON document is an anomaly. The single neuron and autoencoder methods are suitable for taking a real time learning approach. The single neuron method is faster than the other two methods. The LSTM based algorithm works, but it is slower than the other approaches.

## Future work
- [ ] The LSTM algorithm could be used to replace the vectorizer of the two stage algorithm. In theory this should result in more optimal vectorization.
- [ ] Instead of a LSTM based recurrent neural network, a [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) based recurrent neural network could be used. The GRU would be faster than the LSTM.
- [ ] Use a recurrent neural network to create a heat map of the JSON document. This would show the parts that are anomalies.
- [ ] [Ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning) could be used to combine multiple algorithms.
- [ ] Use [models for adaptive arithmetic coding](https://fgiesen.wordpress.com/2015/05/26/models-for-adaptive-arithmetic-coding/) for anomaly detection.
