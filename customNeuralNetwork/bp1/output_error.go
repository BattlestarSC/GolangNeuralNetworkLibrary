package bp1

import (
  c "customNeuralNetwork/costs"
  n "customNeuralNetwork/nnetwork"
  s "customNeuralNetwork/error_structures"
  d "customNeuralNetwork/derivitives"
)

//from the neural network and deep learning book, this implments BP1 from chapter 2
//This does the error for the output layer, equal to partial derivitive of the cost function with the respect to each activation times the derivitive of the activation function at the weighted sum
func OutputError(expected_output []float32, network_runtime n.NetworkOutput) s.LayerError {
  //sanity checking, length of expected_output should equal to the network_runtime's length of outputs
  if len(expected_output) != len(network_runtime.Outputs) {panic("Malformed inputs to OutputError; the length of the expected data is incorrect")}

  //Get the last layer's index number
  target_index := len(network_runtime.LayerOutputs) - 1

  //make the output variable
  var output s.LayerError
  //and load the Index
  output.Index = target_index //This is the last layer's index

  //cost Derivitives
  cstDer := c.MSEDerPerNode(network_runtime.Outputs, expected_output)

  //for each index in the outputs and network outputs and network pre-activations, run the compute
  for index:=0;index<len(expected_output);index++ {

    //first lets do the derivitive of the weighted sum
    der := d.ActivationDerivities(network_runtime.LayerOutputs[target_index].Activation, network_runtime.LayerOutputs[target_index].Pre[index])

    //The cost partial derivitive
    cst := cstDer[index]

    //multiply
    done := der * cst

    //append to output
    output.Error = append(output.Error, done)

  } //end foreach

  return output
}
