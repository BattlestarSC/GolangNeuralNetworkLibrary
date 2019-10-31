package error_structures

/*
  This is simply here to make error structures for error equations
*/

//This is for layer errors BP1/BP2
type LayerError struct  {
  Index int       //What layer
  Error []float64 //The error per node (ordered)
}
