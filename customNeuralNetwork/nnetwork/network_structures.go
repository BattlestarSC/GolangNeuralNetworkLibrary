package nnetwork

import (
  l "customNeuralNetwork/layers"
)

/*
Define the structure for a network

Records:
NumInputs:      How many inputs can the network take, all the internals are handled in creation
NumLayers:      How many layers do we have in the network
NumOutputs:     The expected number of output values
FullyConnected: If the network is fully connected, saving this should save some CPU cycles
Layers:         The attached layers
LayerListing:   Left for faster printing of network dimentions and for debug during creation
Bias:           Is the bias enabled for the network, used during updating 
*/

type Network struct {
  NumInputs      int
  NumLayers      int
  NumOutputs     int
  FullyConnected bool
  Layers         []l.Layer
  LayerListing   []int //FOR DEBUG AND PRINTING TO SCREEN ONLY
  Bias           bool
}

/*
 * Also define a structure for network outputs
 *
 * This saves the outputs (for cost compute and prediction after training)
 * The layer outputs are saved for per-layer/per-node error calculation (BP2; http://neuralnetworksanddeeplearning.com/chap2.html)
 * NumDead is the number of nodes that died that run, kept here in case its needed later
 */

type NetworkOutput struct {
  Outputs      []float32
  LayerOutputs []l.LayerOutput
  NumDead      int
}
