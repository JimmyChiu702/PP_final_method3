#include <omp.h>

#include <string.h>

#include "MLP_Network.h"
#include "MLP_Layer.h"
#include "MNIST.h"

#define NUM_THREADS 1

int main()
{
    int nInputUnit      = 784;
    int nHiddenUnit     = 512;
    int nOutputUnit     = 10;
    
    int nHiddenLayer    = 1;
    int nMiniBatch      = 10;
    float learningRate  = 0.05;
    
    int nTrainingSet    = 60000;
    int nTestSet        = 10000;
    
    float errMinimum = 0.01;    
    int maxEpoch = 1;

    omp_set_num_threads(NUM_THREADS);
    
    //Allocate
    float **inputTraining			= new float*[nTrainingSet];
    float **desiredOutputTraining	= new float*[nTrainingSet];
    
    for(int i = 0;i < nTrainingSet;i++){
        inputTraining[i]			= new float[nInputUnit];
        desiredOutputTraining[i]	= new float[nOutputUnit];
    }
    float **inputTest			= new float*[nTestSet];
    float **desiredOutputTest	= new float*[nTestSet];
    
    for(int i = 0;i < nTestSet;i++){
        inputTest[i]			= new float[nInputUnit];
        desiredOutputTest[i]	= new float[nOutputUnit];
    }
    
    //MNIST Input Array Allocation and Initialization
    MNIST mnist;
    mnist.ReadMNIST_Input("./datasets/train-images-idx3-ubyte", nTrainingSet, inputTraining);
    mnist.ReadMNIST_Label("./datasets/train-labels-idx1-ubyte",nTrainingSet, desiredOutputTraining);
    
    mnist.ReadMNIST_Input("./datasets/t10k-images-idx3-ubyte",nTestSet, inputTest);
    mnist.ReadMNIST_Label("./datasets/t10k-labels-idx1-ubyte",nTestSet, desiredOutputTest);
    
    MLP_Network mlp;
    
    mlp.Allocate(nInputUnit,nHiddenUnit,nOutputUnit,nHiddenLayer,nTrainingSet, NUM_THREADS);

    //Start clock
    clock_t start, finish;
    double elapsed_time;
    start = clock();
    
    float initialLR = learningRate;
   
    int epoch = 0;
    while (epoch < maxEpoch)
    {
        float error[NUM_THREADS-1] = {0.0};
        int batchCount=0;

        for (int i=0; i<nTrainingSet/(nMiniBatch*(NUM_THREADS-1)); i++) {
            #pragma omp parallel 
            {
                int threadID = omp_get_thread_num();
                if (threadID==NUM_THREADS-1) {          // parameter node
                    mlp.UpdateWeight(learningRate, i%2);
                } else {
                    for (int j=0; j<nMiniBatch; j++) {
                        mlp.ForwardPropagateNetwork(inputTraining[i*nMiniBatch+j], threadID, i%2);
                        mlp.BackwardPropagateNetwork(desiredOutputTraining[i*nMiniBatch+j], threadID, i%2);
                        error[threadID] += mlp.CostFunction(inputTraining[i*nMiniBatch+j],desiredOutputTraining[i*nMiniBatch+j], threadID, i%2);
                    }
                }
            }
        }
        float sumError = 0;
        for (int i=0; i<NUM_THREADS-1; i++) {
            sumError += error[i];
        }
        
        sumError /= nTrainingSet;
        
        cout<<epoch<<" | "<<sumError<<" | "<<errMinimum<<endl;
        
        if (sumError < errMinimum)
            break;
        
        learningRate = initialLR/(1+epoch*learningRate);    // learning rate progressive decay
        ++epoch;
    }

    
    
    //Finish clock
    finish = clock();
    elapsed_time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"time: "<<elapsed_time<<" sec"<<endl;

    
    // // Train Set Result
    // cout<<"[Result]"<<endl<<endl;

    // int sums=0;
    // float accuracyRate=0.F;

    // #pragma omp parallel for num_threads(NUM_THREADS-1)
    // for (int i = 0; i < nTrainingSet; i++)
    // {
    //     int threadID = omp_get_thread_num();
    //     mlp.ForwardPropagateNetwork(inputTraining[i], threadID, 0);
    //     sums += mlp.CalculateResult(inputTraining[i],desiredOutputTraining[i], threadID, 0);
    // }
    
    // accuracyRate = (sums / (float)nTrainingSet) * 100;
    
    // cout << "[Training Set]\t"<<"Accuracy Rate: " << accuracyRate << " %"<<endl;
    
    // Test Set Result
    // int sums=0;
    // float accuracyRate=0.F;
    // #pragma omp parallel for reduction(+: sums) num_threads(NUM_THREADS-1)
    // for (int i = 0; i < nTestSet; i++)
    // {
    //     int threadID = omp_get_thread_num();

    //     mlp.ForwardPropagateNetwork(inputTest[i], threadID, 0);
        
    //     sums += mlp.CalculateResult(inputTest[i], desiredOutputTest[i], threadID, 0);
    // }
    // accuracyRate = (sums / (float)nTestSet) * 100;
    
    // cout << "[Test Set]\t"<<"Accuracy Rate: " << accuracyRate << " %"<<endl;
    
    
    for (int i = 0; i < nTrainingSet; i++)
    {
        delete [] desiredOutputTraining[i];
        delete [] inputTraining[i];
    }

    for (int i = 0; i < nTestSet; i++)
    {
        delete [] desiredOutputTest[i];
        delete [] inputTest[i];
    }

    delete[] inputTraining;
    delete[] desiredOutputTraining;
    delete[] inputTest;
    delete[] desiredOutputTest;
    
 
    return 0;
}
