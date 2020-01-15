#ifndef MLP_Network_H
#define MLP_Network_H
#include "MLP_Layer.h"


#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

class MLP_Network {
    
private:
    MLP_Layer *layerNetwork;
    
    
    int nTrainingSet;
    int nInputUnit;
    int nHiddenUnit;
    int nOutputUnit;
    int nHiddenLayer;

public:
    MLP_Network(){}
    ~MLP_Network(){Delete();}

    void Allocate(int nInputUnit,   int nHiddenUnit, int nOutputUnit, int nHiddenLayer,
                  int nTrainingSet, int max_num_threads);
    void Delete();
    
    void Train();
    void ForwardPropagateNetwork(float* inputNetwork, int threadID, int round);
    void BackwardPropagateNetwork(float* desiredOutput, int threadID, int round);
    void UpdateWeight(float learningRate, int round);
    float CostFunction(float* inputNetwork,float* desiredOutput, int threadID, int round);
    
    float CalculateResult(float* inputNetwork,float* desiredOutput, int threadID, int round);

    void test() {
        layerNetwork[1].test();
    }
};

#endif
