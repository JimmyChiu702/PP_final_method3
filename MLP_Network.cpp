#include "MLP_Network.h"

void MLP_Network::Allocate(int nInputUnit,   int nHiddenUnit, int nOutputUnit, int nHiddenLayer,
                           int nTrainingSet, int max_num_threads)
{
    this->nTrainingSet  = nTrainingSet;
    this->nInputUnit    = nInputUnit;
    this->nHiddenUnit   = nHiddenUnit;
    this->nOutputUnit   = nOutputUnit;
    this->nHiddenLayer  = nHiddenLayer;
    
    layerNetwork = new MLP_Layer[nHiddenLayer+1]();
    
    layerNetwork[0].Allocate(nInputUnit, nHiddenUnit, max_num_threads);
    for (int i = 1; i < nHiddenLayer; i++)
    {
        layerNetwork[i].Allocate(nHiddenUnit, nHiddenUnit, max_num_threads);
    }
    layerNetwork[nHiddenLayer].Allocate(nHiddenUnit, nOutputUnit, max_num_threads);
}

void MLP_Network::Delete()
{
    for (int i = 0; i < nHiddenLayer+1; i++)
    {
        layerNetwork[i].Delete();
    }
}

void MLP_Network::ForwardPropagateNetwork(float* inputNetwork, int threadID, int round)
{
    float* outputOfHiddenLayer=NULL;
    
    outputOfHiddenLayer=layerNetwork[0].ForwardPropagate(inputNetwork, threadID, round);
    for (int i=1; i < nHiddenLayer ; i++)
    {
        outputOfHiddenLayer=layerNetwork[i].ForwardPropagate(outputOfHiddenLayer, threadID, round);                  //hidden forward
    }
    layerNetwork[nHiddenLayer].ForwardPropagate(outputOfHiddenLayer, threadID, round);      // output forward
}

void MLP_Network::BackwardPropagateNetwork(float* desiredOutput, int threadID, int round)
{
    layerNetwork[nHiddenLayer].BackwardPropagateOutputLayer(desiredOutput, threadID, round);  // back_propa_output
    for (int i= nHiddenLayer-1; i >= 0  ; i--)
        layerNetwork[i].BackwardPropagateHiddenLayer(&layerNetwork[i+1], threadID, round);              // back_propa_hidden
}

void MLP_Network::UpdateWeight(float learningRate, int round)
{
    for (int i = 0; i < nHiddenLayer; i++)
        layerNetwork[i].UpdateWeight(learningRate, round);
    
    layerNetwork[nHiddenLayer].UpdateWeight(learningRate, round);
}

float MLP_Network::CostFunction(float* inputNetwork, float* desiredOutput, int threadID, int round)
{
    float *outputNetwork = &layerNetwork[nHiddenLayer].GetOutput()[round][threadID*nOutputUnit];
    float err=0.F;
    for (int j = 0; j < nOutputUnit; ++j)
        err += (desiredOutput[j] - outputNetwork[j])*(desiredOutput[j] - outputNetwork[j]);
    err /= 2;
        
    return err;
}

 
float MLP_Network::CalculateResult(float* inputNetwork,float* desiredOutput, int threadID, int round)
{
    int maxIdx = 0;
    
    maxIdx = layerNetwork[nHiddenLayer].GetMaxOutputIndex(threadID, round);
    
    if(desiredOutput[maxIdx] == 1.0f)
        return 1;
    return 0;
}
