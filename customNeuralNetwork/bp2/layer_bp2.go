package bp2

import (
  n "customNeuralNetwork/nnetwork"
  e "customNeuralNetwork/error_structures"
  d "customNeuralNetwork/derivitives"
  l "customNeuralNetwork/layers"
  b "customNeuralNetwork/bp4"
)

/*
  this package is to run the BP2 equation for every layer after len(layers)-1, which is handled by BP1
  BP2 is equal to ( weights^l+1 * error^l+1 ) * ActivationDerivitie(pre^l)
  Thus, for layer index 4, node index 2, the error is equal to weightsThatConnectToIndex2 of layer 5 times the errors of layer 5 all times activation derivitive of layer 4 node 2 pre

  Will also launch all BP4 go routines
*/

func BP2(net n.Network, out n.NetworkOutput, output_layer_error e.LayerError, wo chan b.WeightError, bo chan b.BiasError) []e.LayerError {

  //first make the output variable
  var output []e.LayerError
  output = make([]e.LayerError, net.NumLayers)

  //put the first error in
  output[output_layer_error.Index] = output_layer_error

  //go though each layer, inverted starting by specifying l+1, launching BP4 along the way
  go b.BP4(out.LayerOutputs[output_layer_error.Index], output[output_layer_error.Index], wo, bo)
  for i:=output_layer_error.Index;i>0;i-- {
    output[i-1] = bp2_layer(net, net.Layers[i], out.LayerOutputs[i-1], output[i])
    go b.BP4(out.LayerOutputs[i-1],output[i-1],wo,bo)
  }

  return output

}

//link the per node into a layer
//layer is layer l+1, lOut is for layer l, x is node
func bp2_layer(net n.Network, layer l.Layer, lOut l.LayerOutput, lastError e.LayerError) e.LayerError {

  //make an output channel
  output_channel := make(chan singleNodeError)
  //for each node in the layer
  for i:=0;i<layer.NumInputs;i++ {
    go bp2_node(net, layer.Index, lOut, i, lastError, output_channel)
  }

  //recieve and organize
  var error []singleNodeError
  for i:=0;i<layer.NumInputs;i++ {
    temp := <- output_channel
    error = append(error, temp)
  }

  //organize
  var output_error []float64
  for i:=0;i<layer.NumInputs;i++ {
    for _,val := range error {
      if val.index == i {
        output_error = append(output_error, val.error)
        break
      }
    }
  }

  //make the structure

  var output e.LayerError
  output.Index = lastError.Index - 1
  output.Error = output_error

  return output

}

//ONLY USED FOR THE BELOW BP2_NODE INTO BP2_LAYER
//DO NOT USE ELSEWHERE
type singleNodeError struct {
  index int
  error float64
}

//This is by single node
//lay is layer l+1, lOut is for layer l, x is node
//as specified in the book, this is for matrix form, so perform matrix math and ADD the multiplications
func bp2_node(net n.Network, lay int, lOut l.LayerOutput, x int, lastLayer e.LayerError, out chan singleNodeError) {
  //First get weights
  weights,connections := getWeights(net, lay, x)
  //Then multiply by error and add together
  var error float64
  if connections == nil {
    //fmt.Println("made it 1") //DEBUG
    for ind,_ := range weights {
      //fmt.Println("At: ", ind, "Weight len:", len(weights), "Error len:", len(lastLayer.Error), "Layer:", lay, "Layer Inputs:", net.Layers[lay].NumInputs, "Layer Nodes:", net.Layers[lay].NumOutputs) //debug
      temp := float64(weights[ind]) * lastLayer.Error[ind]
      error = error + temp
    }
  } else {
    for ind,con := range connections {
      temp := float64(weights[ind]) * lastLayer.Error[con]
      error = error + temp
    }
  }

  //then get the activation derivitive
  act := d.ActivationDerivities(lOut.Activation, lOut.Pre[x])

  //multiply
  result := error * act

  var output singleNodeError
  output.index = x
  output.error = result

  out <- output
}
