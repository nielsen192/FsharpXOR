// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
open Encog.ML.Data.Basic
open Encog.Engine.Network.Activation
open Encog.Neural.Networks
open Encog.Neural.Networks.Layers
open Encog.Neural.Networks.Training.Propagation.Resilient

let createNetwork() =
    let network = BasicNetwork()
    network.AddLayer( BasicLayer( null, true, 2 ))
    network.AddLayer( BasicLayer( ActivationSigmoid(), true, 2 ))
    network.AddLayer( BasicLayer( ActivationSigmoid(), false, 1 ))
    network.Structure.FinalizeStructure()
    network.Reset()
    network

let train trainingSet (network: BasicNetwork) =
    let trainedNetwork = network.Clone() :?> BasicNetwork
    let trainer = ResilientPropagation(trainedNetwork, trainingSet)

    let rec trainIteration epoch error =
        match error > 0.001 with
        | false -> ()
        | true -> trainer.Iteration()
                  printfn "Iteration no : %d, Error: %f" epoch error
                  trainIteration (epoch + 1) trainer.Error

    trainIteration 1 1.0
    trainedNetwork

[<EntryPoint>]
let main argv = 
    
    let xor_input =
        [|
            [| 0.0 ; 0.0 |]
            [| 1.0 ; 0.0 |]
            [| 0.0 ; 1.0 |]
            [| 1.0 ; 1.0 |]
        |]

    let xor_ideal =
        [|
            [| 0.0 |]
            [| 1.0 |]
            [| 1.0 |]
            [| 0.0 |]
        |]

    let trainingSet = BasicMLDataSet(xor_input, xor_ideal)
    let network = createNetwork()

    let trainedNetwork = network |> train trainingSet

    trainingSet
    |> Seq.iter (
        fun item ->
            let output = trainedNetwork.Compute(item.Input)
            printfn "Input: %f, %f Ideal: %f Actual: %f"
                item.Input.[0] item.Input.[1] item.Ideal.[0] output.[0])

    printfn "Press return to exit.."
    System.Console.Read() |> ignore

    0 // return an integer exit code
