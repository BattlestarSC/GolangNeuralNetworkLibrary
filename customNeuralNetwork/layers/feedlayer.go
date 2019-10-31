package layers

/*
  This will feed a layer forward, handle interfacing between a layer and nodes
*/

import (
  n "customNeuralNetwork/nodes"
  "strconv"
)

func FeedLayer(l Layer, inputs []float32, training bool) LayerOutput {
  //quick error checking
  if len(inputs) != l.NumInputs {panic("Inproper feeding length\nExpected: " + strconv.Itoa(l.NumInputs) + " Actually got: " + strconv.Itoa(len(inputs)) )}

  //start feeding nodes
  //make the output channel
  nodeOut := make(chan n.NodeOutput)
  //feednode usage:  FeedNode(n Node, inputs []float32, allow_deaths bool, outputChannel chan NodeOutput)
  for _,nde := range l.Nodes {//For every single node, feed
		allowDead := !l.LastLayer && training //if its not the output layer and we are training, we will allow deaths
		go n.FeedNode(nde, inputs, allowDead, nodeOut) //Launch the single node feeding
	} //end foreach

	//make somewhere to hold the outputs
	nodeData := make([]n.NodeOutput, l.NumOutputs)
	//recieve the output from the channel
	for i:=0;i<l.NumOutputs;i++ { //for the number of nodes that were launched to feed, take and save them
		nodeData[i] = <- nodeOut //take the channel result and load it
	} //end for loop

	//create the actual output variable
	var output LayerOutput
	output.Output    = make([]float32,l.NumOutputs)
	output.Pre       = make([]float32,l.NumOutputs)
	output.DeadNodes = make([]bool, l.NumOutputs)

	//Now to organize layer outputs and pre-activations

  for i:=0;i<l.NumOutputs;i++ { //for each output, this tracks order
    //where to keep temps before loading
    var out   float32
    var pre   float32
    var death bool

		//Now go through each output var
		for _,ndeout := range nodeData {
			//if its a match
			if ndeout.Index == i {
				//get the data
				out   = ndeout.Output
				pre   = ndeout.Pre
				death = ndeout.Dead
				//no need to waste CPU time
				break
			} //end if

		}//end for each output

		//load the data
    output.Output[i]    = out
    output.Pre[i]       = pre
    output.DeadNodes[i] = death
	} //end for i...l.NumOutputs loop

	//save the list
	output.NodeOutputs = nodeData //This slice was created here and will not be reused, so just save its address
  //Index
	output.Index = l.Index
  //activation
  output.Activation = l.Nodes[0].Activation //there must be at least one node
  //number of nodes
  output.NumNodes = l.NumOutputs

	return output
}
