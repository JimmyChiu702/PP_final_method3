#ifndef MLP_Layer_H
#define MLP_Layer_H

#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>
#include <string.h>

#include <iomanip>

class  MLP_Layer {
    int nPreviousUnit;
    int nCurrentUnit;

    int max_num_threads;
    
    float** inputLayer;
    float** outputLayer;
    float** weight;
    float** gradient;
    float** delta;
    
    float** biasWeight;    
    float** biasGradient;
    
public:
    MLP_Layer(){};
    ~MLP_Layer()    {   Delete();   }
    
    void Allocate(int previous_node_num, int nCurrentUnit, int max_num_threads);
    void Delete();
    
    float* ForwardPropagate(float* inputLayer, int threadID, int round);
    void BackwardPropagateHiddenLayer(MLP_Layer* previousLayer, int threadID, int round);
    void BackwardPropagateOutputLayer(float* desiredValues, int threadID, int round);
    
    void UpdateWeight(float learningRate, int round);
    
	float** GetOutput()  {   return outputLayer; }
    float** GetWeight()  {   return weight;      }
    float** GetDelta()   {   return delta;       }
    int GetNumCurrent() {   return nCurrentUnit;}
	int GetMaxOutputIndex(int threadID, int round);
    // Sigmoid
    float ActivationFunction(float net)		{ return 1.F/(1.F + (float)exp(-net)); }
    
    float DerActivationFromOutput(float output){ return output * (1.F-output); }
    float DerActivation(float net)	{ return DerActivationFromOutput(ActivationFunction(net)); }

    void test() {
        for (int i=0; i<max_num_threads; i++) {
            for (int j=0; j<100; j++) {
                printf("%1.5f ", gradient[i][j]);
                // std::cout << outputLayer[i][j] << " ";
                // std::cout << weight[i][j] << " ";
                // std::cout << gradient[i][j] << " ";
                // std::cout << delta[i][j] << " ";
                // std::cout << biasWeight[i][j] << " ";
                // std::cout << biasGradient[i][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n\n";
    }
};

#endif