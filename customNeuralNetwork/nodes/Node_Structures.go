package nodes


/*
This defines the structure for a node within the network, complete with indexing information, I/O information, error checking information, and feeding information
We use a float32 for feeding to speed up compute on specific architechures, however error handling requires float64 precision, so that will be handled and converted
  Index:          What node within the layer
  LastLayerNodes: How many nodes in the last layer, avaiable for error checking
  NumInputs:      How many inputs are expected, used for error checking in real time
  Connections:    List of integers, refer to node indexes of the previous layers, so when feeding respect only the inputs who's index matches a number within the list
					Only used for partially connected networks, Nil for fully connected networks
  Weights:        The node's weights within a list, used for feeding, ex. Weights[1] * Inputs[1]
  Bias:           The node's assiocated bias
  Activation:     The number of activation function to use, for use when comparing activations (Sigmoid, Relu, Custom, etc)
  Lay:            The layer index we are in
*/

type Node struct {
  Index          int
  Lay            int
  LastLayerNodes int
  NumInputs      int
  Connections    []int
  Weights        []float32
  Bias           float32
  Activation     int
  NumWeights     int //TO BE DONE AT CREATE TIME FOR SPEED, SEE OUTPUT BELOW
}

/*
  Used during feeding, indexes the node output for use when spraying go routines
  Index:  What node's output
  Output: Node's output after activation
  Pre:    Node's weighted sum
  Dead:   Did the node turn off randomly during training (bool)
  Inputs: What inputs did the node recieve, this is to make backpropogation easy by eliminating the need to match inputs with nodes, isolating the node's BP4 eq error from everything except its node's BP2 eq error
  Lay:    What layer are we in (used for BP4)
*/

type NodeOutput struct {
  Index      int
  Lay        int
  Output     float32
  Pre        float32
  Dead       bool
  Inputs     []float32
  NumWeights int //FOR BP COUNTING ONLY
}
