package bp2

import (
  n "customNeuralNetwork/nnetwork"
  l "customNeuralNetwork/layers"
)

/*
  This will read though network 'net' for the weights from layer 'i' that connect to node 'x' from layer 'i'-1
  will also return the connections list
*/

func getWeights(net n.Network, i, x int) ([]float32,[]int) {
  //first select the layer
  targetLayer := net.Layers[i]
  //find out if layer is fully connected
  firstNode := targetLayer.Nodes[0]
  var fulCon bool
  if firstNode.Connections == nil {
    fulCon = true
  } else {
    fulCon = false
  }

  //get result for fully connected
  if fulCon {
    return fullycon(targetLayer, x),nil
  }

  //make output var
  var output []float32
  var output_connections []int

  //For each node in the not fully connected layer
  for ind,nde := range targetLayer.Nodes {
    //if the node is connected to input x
    location := ifxinslice(nde.Connections, x)
    if location != -1 {
      output = append(output, nde.Weights[location])
      output_connections = append(output_connections, ind)
    }
  }

  return output,output_connections
}

//easy if in sclice
func ifxinslice(s []int, x int) int {
  for ind,value := range s {
    if value == x {
      return ind
    }
  }
  return -1
}

//for easy FullyConnected networks, know that weight index x will equal the weight to node index x of last layer
func fullycon(lay l.Layer, x int) []float32 {
  var output []float32
  output = make([]float32, len(lay.Nodes))

  for ind,nde := range lay.Nodes {
    output[ind] = nde.Weights[x]
  }

  return output

}
