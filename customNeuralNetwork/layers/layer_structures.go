package layers

//define the structure for a layer
import(
  . "customNeuralNetwork/nodes"
)

/*
  The structure for a layer
  NumInputs:  How many inputs are expected when feeding
  Nodes:      Nodes within layer
  Index:      What layer is this
  NumOutputs: How many nodes are here, each will output, used to expect an output length
  LastLayer:  Is the the last layer (output), if so, NEVER allow dead nodes
  Activation: Which function the layer is using. ALL nodes in a layer must use the same activation function, but the layers can vary. BP2 function will handle this, but needs this record
*/

type Layer struct {
  NumInputs  int
  Nodes      []Node
  NumOutputs int
  Index      int
  LastLayer  bool
  Activation int
}

/*
  The structure for layer outputs
  index:       What layer output this
  output:      The slice of outputs from the nodes (used here for easier BP)
  pre:         weighted inputs to each node (z), used here for easier bp
  DeadNodes:   What nodes died
  NodeOutputs: The raw node outputs; unordered. During backprop, each node will be assigned an error structure via BP2
  * 					 --->During BP4, each node error will be matched to its node output via nodeoutput's .Lay and .Index members
  *						 --->NodeOutput structure contians the inputs to the node (ordered slice) which is all that is needed to produce weight-specific errors
  Activation:  What function is used (for reference in packprop)
*/

type LayerOutput struct {
  Index       int
  Output      []float32
  Pre         []float32
  DeadNodes   []bool
  NodeOutputs []NodeOutput
  Activation  int
  NumNodes    int //FOR BP COUNTING ONLY
}
