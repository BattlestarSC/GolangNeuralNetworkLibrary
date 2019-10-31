package bp4

import (
  l "customNeuralNetwork/layers"
  e "customNeuralNetwork/error_structures"
  n "customNeuralNetwork/nodes"
)

/*
  This function will return two structures, an error structure for weights and one for biases
  Runs BP4 from neuralnetworksanddeeplearning book, and as BP3 == the node's error

*/

//ERROR: ONLY RETURNING BP4 FOR 1 NODE PER LAYER

func BP4(layerRuntime l.LayerOutput, layerError e.LayerError, output chan WeightError, biasOutput chan BiasError) {
  //fmt.Println("length:", len(layerRuntime.NodeOutputs), "Layer:", layerRuntime.Index)
  for index,ndeout := range layerRuntime.NodeOutputs {
    //fmt.Println("running layerRuntime", index)
    go bp4_per_node(layerRuntime.Index, layerError.Error[index], ndeout, output)
    go makeBiasOutput(layerRuntime.Index, index, layerError.Error[index], layerRuntime.DeadNodes[index], biasOutput)
  }

}

func bp4_per_node(lI int, error float64, nde n.NodeOutput, output chan WeightError) {
  //fmt.Println("input length:", len(nde.Inputs), "Layer:", lI)
  for weightIndex,value := range nde.Inputs {
    temp := float64(value) * error
    //fmt.Println("Running:", lI, nde.Index, weightIndex)
    output <- WeightError{lI, nde.Index, weightIndex, temp, nde.Dead}
  }
}

//Ok, so making this output inline within BP4 caused the goroutine to hang until it's output was grabbed, which is NOT what the goal was so make this its own hanging goroutine (feat. OSX's reported 200,000+ idle wakeups)
func makeBiasOutput(layerIndex, nodeIndex int, error float64, dead bool, biasoutput chan BiasError ) {
  biasoutput <- BiasError{layerIndex, nodeIndex, error, dead}
}
