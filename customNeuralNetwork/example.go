package main

import (
  "fmt"
  . "customNeuralNetwork/nnetwork"
  "math/rand"
  "time"
  . "customNeuralNetwork/training"
  . "customNeuralNetwork/iris"
  . "customNeuralNetwork/metro"
)


//Example runtime, some errors are resolved
//No good interface is provided
func main() {
  rand.Seed(time.Now().UTC().UnixNano())


  issue1Network := MakeNetwork(4,3,[]int{10,15,20,25,20,15,10}, true, true, false, "relu", nil, nil) //this leaky relu will reach an Inf cost then a NaN cost

  issue2Network := MakeNetwork(4,3,[]int{15,15,15}, true, true, false, "relu", nil, nil) //
  //these are to show the issues with the metro dataset
  issue3Network1 := MakeNetwork(19, 2, []int{20,20,10,5}, true, true, false, "custom", nil, nil) //
  issue3Network2 := MakeNetwork(19, 2, []int{20,20,10,5}, true, true, false, "sigmoid", nil, nil) //
  issue3Network3 := MakeNetwork(19, 2, []int{20,20,10,5}, true, true, false, "relu", nil, nil) //
  metroIssueLong := MakeNetwork(19, 2, []int{40,60,40,20,10,5}, true, true, false, "custom", nil, nil)

  //this is to show the custom function working
  customSuccess1 := MakeNetwork(4,3,[]int{10,10,10}, true, true, false, "custom", nil, nil)
  customSuccess2 := MakeNetwork(4,3,[]int{10,10,10}, true, true, false, "custom", nil, nil)
  customSuccess3 := MakeNetwork(4,3,[]int{10,10,10}, true, true, false, "custom", nil, nil)
  customSuccess4 := MakeNetwork(4,3,[]int{10,10,10}, true, true, false, "custom", nil, nil)
  customSuccess5 := MakeNetwork(4,3,[]int{10,10,10}, true, true, false, "custom", nil, nil)


  //iris dataset
  fmt.Println("Loading iris dataset")
  irisData := IrisDataSet()

  //train the issue networks 1 and 2
  fmt.Println("Running a network with the leaky relu function and the iris dataset")
  TrainSupervisedNetwork(issue1Network, irisData.Inputs, irisData.Outputs, 25, 25, 0.01)

  fmt.Println("Running a network with the regular relu function and the iris dataset")
  TrainSupervisedNetwork(issue2Network, irisData.Inputs, irisData.Outputs, 25,25,0.15)
  //Show the custom function
  fmt.Println("Running the custom function network and iris dataset a few times (hoping for high starting error to see learning)")
  TrainSupervisedNetwork(customSuccess1, irisData.Inputs, irisData.Outputs, 25, 50, 0.15)
  fmt.Println("next")
  TrainSupervisedNetwork(customSuccess2, irisData.Inputs, irisData.Outputs, 25, 50, 0.15)
  fmt.Println("next")
  TrainSupervisedNetwork(customSuccess3, irisData.Inputs, irisData.Outputs, 15, 50, 0.15)
  fmt.Println("next")
  TrainSupervisedNetwork(customSuccess4, irisData.Inputs, irisData.Outputs, 25, 50, 0.15)
  fmt.Println("And the slow version")
  TrainSupervisedNetwork(customSuccess5, irisData.Inputs, irisData.Outputs, 25, 1500, 0.001)


  //show the failures to learn with the other data
  //metro dataset
  metro := TrafficDataSet()
  var metroDataIn, metroDataOut [][]float32
  for _,value := range metro {
    metroDataIn = append(metroDataIn, value.NetInputs)
    metroDataOut = append(metroDataOut, value.NetOutputs)
  }


  //train the issues
  fmt.Println("Training a custom activation function network for the metro dataset (https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume ) ")
  TrainSupervisedNetwork(issue3Network1, metroDataIn, metroDataOut, 1500, 500, 0.25)
  fmt.Println("Training another network, with a sigmoid function this time")
  TrainSupervisedNetwork(issue3Network2, metroDataIn, metroDataOut, 1500, 500, 0.25)
  fmt.Println("And one more, with the relu function")
  TrainSupervisedNetwork(issue3Network3, metroDataIn, metroDataOut, 1500, 500, 0.25)
  fmt.Println("Running a long metro sim")
  TrainSupervisedNetwork(metroIssueLong, metroDataIn, metroDataOut, 2500, 1500, 0.15)
  fmt.Println("Finished")


}
