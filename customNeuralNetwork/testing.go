package main

//TODO: Interface section of the package
//Interface: contains a predict function, train function, and backprop (OutputError into BP2 into BP4 into train function)

import (
  "fmt"
  //. "customNeuralNetwork/nodes"
  //. "customNeuralNetwork/activationFunctions"
  //. "customNeuralNetwork/layers"
  . "customNeuralNetwork/nnetwork"
  //. "customNeuralNetwork/costs"
  "math/rand"
  "math"
  "time"
  //. "customNeuralNetwork/bp1"
  //. "customNeuralNetwork/bp2"
  //. "customNeuralNetwork/bp4"
  //. "customNeuralNetwork/organization"
  //. "customNeuralNetwork/updateNetwork"
  . "customNeuralNetwork/training"
  . "customNeuralNetwork/iris"
  . "customNeuralNetwork/metro"
 )

func main() {
  //Seed rand
  rand.Seed(time.Now().UTC().UnixNano())

  //test timing
  Start := time.Now()


  //Testing of make nodes and activation - PASS
    //fmt.Println(MakeNode(0,5,true,true,nil,"custom"))
    //fmt.Println(Activation(float32(1.23), 5))

  //Testing of feed node - PASS
  /*
    nde1 := MakeNode(0,5,true,true,nil,"custom")
    nde2 := MakeNode(0,5,false,true,[]int{1,3,4},"custom")
    nde3 := MakeNode(0,5,true,true,nil,"sigmoid")
    nde4 := MakeNode(0,5,true,true,nil,"relu")
    nde5 := MakeNode(0,5,true,true,nil,"leaky relu")
    nde6 := MakeNode(0,5,true,true,nil,"relu 6")
    inputs := []float32{0.1,0.4,0.5,0.7,4.3}
    output := make(chan NodeOutput)
    go FeedNode(nde1, inputs, false, output)
    go FeedNode(nde2, inputs, false, output)
    go FeedNode(nde3, inputs, false, output)
    go FeedNode(nde4, inputs, false, output)
    go FeedNode(nde5, inputs, true , output)
    go FeedNode(nde6, inputs, true , output)

    for i:=0;i<6;i++{
	  fmt.Println(<-output)
	}
  */

/*
  //Testing of make a layer -PASS
  layer1 := MakeLayer(1,4,5,true,true,false,"custom",nil)
  layer2 := MakeLayer(2,5,12,true,false,false,"sigmoid",nil)
  fmt.Println(layer1)
  fmt.Println("\n")
  //fmt.Println(layer2)

  //testing of feeding a layer -PASS
  fmt.Println("\n")
  inputs := []float32{2.453,2.432,0.2342,2.312}
  outs := FeedLayer(layer1,inputs,false)
  fmt.Println(outs.Index)
  fmt.Println(outs.Output)
  fmt.Println(outs.Pre)
  fmt.Println(outs.DeadNodes)
  fmt.Println(outs.NodeOutputs)
  fmt.Println("\n\n")
  outs2 := FeedLayer(layer2, outs.Output, false)
  fmt.Println(outs2.Index)
  fmt.Println(outs2.Output)
  fmt.Println(outs2.Pre)
  fmt.Println(outs2.DeadNodes)
  fmt.Println(outs2.NodeOutputs)
  fmt.Println("\n\n")
*/

/*
  //testing of make a network -PASS
  net1 := MakeNetwork(4, 2, []int{2,4,3}, true, true, false, "custom", nil, nil)
  //net1 := MakeNetwork(4, 2, []int{784,15,10},true,true,false,"custom",nil,nil)
  fmt.Println("Network 1 stats")
  fmt.Println("Number of Inputs:  ",net1.NumInputs)
  fmt.Println("Number of Layers:  ",net1.NumLayers)
  fmt.Println("Number of Outputs: ",net1.NumOutputs)
  fmt.Println("Full Connection Status: ", net1.FullyConnected)
  fmt.Println("Layer Details",net1.Layers)

  //break
  fmt.Println("\n\n\nFeeding Data:\n")

  //testing feeding a network -PASS
  //Also testing cost functions -PASS
  //Also testing output error (bp1) function -PASS
  //Also testing for bp2 -PROBABLY PASS (worked for one test, not sure of accuracy)
  //Also testing BP4 -PASS
  wo := make(chan WeightError)
  bo := make(chan BiasError)
  inputs1 := []float32{2.453,2.432,0.2342,2.312}
  inputs2 := []float32{9.234,2.141,5.2342,-3.243}
  inputs3 := []float32{-8.234,14.3234,-1.032,0.234}
  inputs4 := []float32{0,3.432,-4.24,9.234}
  inputs5 := []float32{-4.243,4.23,9.423,-2.42}
  expect := []float32{0.5,75.5}
  netout := FeedNetwork(net1, inputs1, false)
  netout2 := FeedNetwork(net1, inputs2, false)
  netout3 := FeedNetwork(net1, inputs3, false)
  netout4 := FeedNetwork(net1, inputs4, false)
  netout5 := FeedNetwork(net1, inputs5, false)
  outerr := OutputError(expect, netout)
  layerr := BP2(net1, netout, outerr, wo, bo)
  BP2(net1, netout2, OutputError(expect,netout2),wo,bo)
  BP2(net1, netout3, OutputError(expect,netout3),wo,bo)
  BP2(net1, netout4, OutputError(expect,netout4),wo,bo)
  BP2(net1, netout5, OutputError(expect,netout5),wo,bo)

  //numWeights, numBiases := numWeightsAndBiases(netout)
  //defer fmt.Println("\n\nNumber of weights in the network", numWeights, "\nNumber of biases in the network", numBiases)
  fmt.Println(netout.Outputs)
  fmt.Println(netout.LayerOutputs)
  fmt.Println(netout.NumDead)
  fmt.Println("Cost:", MeansSquared(netout.Outputs, expect))
  fmt.Println("Output Error:", outerr)
  fmt.Println("\n\nLayer Error:",layerr)
  fmt.Println("Layers:", net1.LayerListing)
  fmt.Println("\n\n\nBP4 output:\n")
  wErr, bErr := ErrorForNetwork(net1, 0.001,5, wo,bo)
  fmt.Println("\n\n", wErr, "\n\n", bErr)
  fmt.Println("\n\nNetwork before update:\n\n", net1, "\n\n\n")
  UpdateNetwork(net1, wErr, bErr)
  fmt.Println("\n\nNetwork after update:\n\n", net1, "\n\n\n")
  */

  //network1 := MakeNetwork(5,5,[]int{10,15,20,25,20,15,10}, true, true, false, "leaky relu", nil, nil)
  network1new := MakeNetwork(4,3,[]int{10,15,20,25,20,15,10}, true, true, false, "leaky relu", nil, nil)
  network2 := MakeNetwork(4,3,[]int{10,10,10}, true, true, false, "custom", nil, nil)
  network2origional := MakeNetwork(4,3,[]int{10,15,20,25,20,15,10}, true, true, false, "custom", nil, nil)
  network2Metro := MakeNetwork(19, 2, []int{20,20,10,5}, true, true, false, "sigmoid", nil, nil)
  //network3 := MakeNetwork(5,5,[]int{10,15,20,25,20,15,10}, true, true, false, "relu", nil, nil)
  //network4 := MakeNetwork(5,5,[]int{10,15,20,25,20,15,10}, true, true, false, "sigmoid", nil, nil)
  //network5 := MakeNetwork(5,5,[]int{10,15,20,25,20,15,10}, true, true, false, "relu 6", nil, nil)

/*
  inputs := make([][]float32, 0)
  expect := make([][]float32, 0)
  for i:=0;i<15000;i++{
    var newIn []float32
    var newEx []float32
    for j:=0;j<5;j++{
      randomValue := randomInput()
      newIn = append(newIn,(randomValue * float32(j+1)))
      //output is equal to sigmoid of value + j
      newEx = append(newEx,sigmoid(randomValue + float32(j)))
    }
    inputs = append(inputs, newIn)
    expect = append(expect, newEx)
  }

  //firstOutput := FeedNetwork(network2, []float32{0.5,0.5,0.5,0.5,0.5}, false)
  //fmt.Println("\n\n\nNetwork1")
  //TrainSupervisedNetwork(network1, inputs, expect, 100, 1000, 0.001)
  //secondOutput := FeedNetwork(network2, []float32{0.5,0.5,0.5,0.5,0.5}, false)
  //fmt.Println("Difference in feeding data\nFirst run of 5x 0.5: ", firstOutput.Outputs, "\nSecond rung of 5x 0.5: ", secondOutput.Outputs)
  fmt.Println("\n\n\nNetwork2") //This works really really well
  TrainSupervisedNetwork(network2, inputs, expect, 2000, 1750, 0.001)
  fmt.Println("\n\n\nNetwork3")
  TrainSupervisedNetwork(network3, inputs, expect, 1250, 100, 0.00001)
  fmt.Println("\n\n\nNetwork4")
  TrainSupervisedNetwork(network4, inputs, expect, 1250, 1000, 0.00001)
  fmt.Println("\n\n\nNetwork5")
  TrainSupervisedNetwork(network5, inputs, expect, 1250, 1000, 0.00001)
*/

irisData := IrisDataSet()
metroData := TrafficDataSet()
//fix metro data
metDatIn := [][]float32{}
metDatOut := [][]float32{}
for _,value := range metroData {
  metDatIn = append(metDatIn, value.NetInputs)
  metDatOut = append(metDatOut, value.NetOutputs)
}

TrainSupervisedNetwork(network2origional, irisData.Inputs, irisData.Outputs, 25, 25, 0.15)
TrainSupervisedNetwork(network2, irisData.Inputs, irisData.Outputs, 15, 50000, 0.001)

TrainSupervisedNetwork(network1new, irisData.Inputs, irisData.Outputs, 25, 500, 0.001)
TrainSupervisedNetwork(network2Metro, metDatIn, metDatOut, 3250, 50000, 0.27)



  /*
  weightChannel := make(chan WeightError)
  biasChannel   := make(chan BiasError)
  costChannel   := make(chan float64)
  netChannel    := make(chan NetworkOutput)
  for _,input := range inputs {
    go TrainOnceSupervised(network2, input, expect, true, true, weightChannel, biasChannel, costChannel, netChannel)
  }
  for i:=0;i<len(inputs);i++ {
    fmt.Println("Error ", i, " == ", <-costChannel)
  }
  fmt.Println("\n\n\nExample test network output\n",<-netChannel)
  allWeight, allBias := ErrorForNetwork(network2, 0.001, len(inputs), weightChannel, biasChannel)
  fmt.Println("\n\n\n")
  fmt.Println("All weight errors:")
  for index,item := range allWeight {
    fmt.Println("Weight Error", index+1, " ==> ", item)
  }
  fmt.Println("All bias errors:")
  for index,item := range allBias {
    fmt.Println("Bias Error", index+1, " ==> ", item)
  }
*/


  //end timing
  defer fmt.Println("\n", time.Since(Start))
}

/*
// debug from organization
func numWeightsAndBiases(input NetworkOutput) (int, int) {
  //running total vars
  var runningWeights int
  var runningBiases  int

  //now for each layer in network output, which contains every layer output including the output layer
  for _,layerRunData := range input.LayerOutputs {
    //the number of nodes is equal to the number of biases
    runningBiases = runningBiases + layerRunData.NumNodes
    //for each node, add its number of weights
    for _,nde := range layerRunData.NodeOutputs {
      //the node saves the needed data in NumWeights
      runningWeights = runningWeights + nde.NumWeights
    }
  } //end for each layer output in the input network output
  return runningWeights,runningBiases
}
*/

func randomInput() float32 {
  num := rand.Float64() * float64(rand.Intn(20))
  if rand.Intn(2) == 1 {
    num = num * float64(-1)
  }
  return float32(num)
}

func sigmoid (x float32) float32 {
  eTo    := math.Exp(float64(-x))
  bottom := float64(1) + eTo
  all    := float64(1) / bottom
  return float32(all)
}
