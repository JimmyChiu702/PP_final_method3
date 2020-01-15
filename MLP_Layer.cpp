#include "MLP_Layer.h"

void MLP_Layer::Allocate(int previous_num, int current_num, int max_num_threads)
{
    this->nPreviousUnit   =  previous_num;
    this->nCurrentUnit    =  current_num;

    this->max_num_threads = max_num_threads;

    inputLayer   = new float*[(max_num_threads-1)];

    weight       = new float*[2];
    gradient     = new float*[2];
    outputLayer  = new float*[2];
    delta        = new float*[2];
    biasWeight   = new float*[2];
    biasGradient = new float*[2];
    for (int i=0; i<2; i++) {
        weight[i]        = new float[nPreviousUnit * nCurrentUnit];
        gradient[i]      = new float[nPreviousUnit * nCurrentUnit * (max_num_threads-1)];
        outputLayer[i]   = new float[nCurrentUnit * (max_num_threads-1)];
        delta[i]         = new float[nCurrentUnit * (max_num_threads-1)];
        biasWeight[i]    = new float[nCurrentUnit]; 
        biasGradient[i]  = new float[nCurrentUnit * (max_num_threads-1)];

        memset(gradient[i], 0.0, sizeof(float)*nPreviousUnit * nCurrentUnit * (max_num_threads-1));
        memset(outputLayer[i], 0.0, sizeof(float)*nCurrentUnit*(max_num_threads-1));
        memset(delta[i], 0.0, sizeof(float)*nCurrentUnit * (max_num_threads-1));
        memset(biasGradient[i], 0.0, sizeof(float)*nCurrentUnit * (max_num_threads-1));
    }
    int seed = 1;
    srand(seed);
    for (int j=0; j<nCurrentUnit; j++) {
        for (int k=0; k<nPreviousUnit; k++) {
            weight[0][j*k] = 0.2*rand()/RAND_MAX - 0.1;
        }
        biasWeight[0][j] = 0.2*rand()/RAND_MAX - 0.1;
    }
    for (int j=0; j<nCurrentUnit; j++) {
        for (int k=0; k<nPreviousUnit; k++) {
            weight[1][j*k] = 0.2*rand()/RAND_MAX - 0.1;
        }
        biasWeight[1][j] = 0.2*rand()/RAND_MAX - 0.1;
    }
}

void MLP_Layer::Delete(){
    for (int i=0; i<2; i++) {
        delete [] weight;
        delete [] gradient;
        delete [] delta;
        delete [] outputLayer;
        delete [] biasGradient;
        delete [] biasWeight;
    }
}

float* MLP_Layer::ForwardPropagate(float* inputLayers, int threadID, int round)      // f( sigma(weights * inputs) + bias )
{
    this->inputLayer[threadID] = inputLayers;

    float* _weight = weight[round];
    float* _biasWeight = biasWeight[round];
    float* _outputLayer = &outputLayer[round][threadID*nCurrentUnit];

    for(int j = 0 ; j < nCurrentUnit ; j++)
    {
        float net= 0;
        for(int i = 0 ; i < nPreviousUnit ; i++)
        {
            net += inputLayer[threadID][i] * _weight[j*nPreviousUnit+i];
        }
        net += _biasWeight[j];
        
        _outputLayer[j] = ActivationFunction(net);
    }
    return _outputLayer;
}

void MLP_Layer::BackwardPropagateOutputLayer(float* desiredValues, int threadID, int round)
{
    float* _delta = &delta[round][threadID*nCurrentUnit];
    float* _outputLayer = &outputLayer[round][threadID*nCurrentUnit];
    float* _gradient = &gradient[round][threadID*nCurrentUnit*nPreviousUnit];
    float* _biasGradient = &biasGradient[round][threadID*nCurrentUnit];

    for (int k = 0; k < nCurrentUnit; k++) {
        float fnet_derivative = _outputLayer[k] * (1 - _outputLayer[k]);
        _delta[k] = fnet_derivative * (desiredValues[k] - _outputLayer[k]);
    }
    
    for (int k = 0 ; k < nCurrentUnit ; k++) {
        for (int j = 0 ; j < nPreviousUnit; j++){
            _gradient[k*nPreviousUnit + j] += - (_delta[k] * inputLayer[threadID][j]);
        }
    }
    
    for (int k = 0 ; k < nCurrentUnit   ; k++)
        _biasGradient[k] += - _delta[k];
}

void MLP_Layer::BackwardPropagateHiddenLayer(MLP_Layer* previousLayer, int threadID, int round)
{
    float** previousLayer_weight = previousLayer->GetWeight();
    float** previousLayer_delta = previousLayer->GetDelta();
    int previousLayer_node_num = previousLayer->GetNumCurrent();
    
    float* _inputLayer = inputLayer[threadID];
    float* _delta = &delta[round][threadID*nCurrentUnit];
    float* _outputLayer = &outputLayer[round][threadID*nCurrentUnit];
    float* _gradient = &gradient[round][threadID*nCurrentUnit*nPreviousUnit];
    float* _biasGradient = &biasGradient[round][threadID*nCurrentUnit];
    float* _previousLayer_weight = previousLayer_weight[round];
    float* _previousLayer_delta = &previousLayer_delta[round][threadID*previousLayer_node_num];

    for (int j = 0; j < nCurrentUnit; j++)
    {
        float previous_sum=0;
        for (int k = 0; k < previousLayer_node_num; k++)
        {
            previous_sum += _previousLayer_delta[k] * _previousLayer_weight[k*nCurrentUnit + j];
        }
        _delta[j] =  _outputLayer[j] * (1 - _outputLayer[j])* previous_sum;
    }
    
    for (int j = 0; j < nCurrentUnit; j++) {
        for (int i = 0; i < nPreviousUnit ; i++) {
            _gradient[j*nPreviousUnit + i] +=  -_delta[j] * _inputLayer[i];
        }
    }

    for (int j = 0 ; j < nCurrentUnit   ; j++)
        _biasGradient[j] += -_delta[j] ;
}

void MLP_Layer::UpdateWeight(float learningRate, int round)
{
    float* _weight_cur = weight[round];
    float* _weight_new = weight[1-round];
    float* _biasWeight_cur = biasWeight[round];
    float* _biasWeight_new = biasWeight[1-round];
    float* _gradient = gradient[1-round];
    float* _biasGradient = biasGradient[1-round];

    for (int j = 0; j < nCurrentUnit; j++) {
        for (int i = 0; i < nPreviousUnit; i++) {
            float tmp = 0.0;
            for (int k=0; k < max_num_threads-1; k++) {
                tmp += _gradient[k*nCurrentUnit*nPreviousUnit+j*nPreviousUnit+i];
            }
            _weight_new[j*nPreviousUnit + i] = -learningRate*tmp + _weight_cur[j*nPreviousUnit + i];
        }
    }
    
    for (int j = 0; j < nCurrentUnit; j++) {
        float tmp = 0.0;
        for (int k=0; k < max_num_threads; k++) {
            tmp += -_biasGradient[k*nCurrentUnit+j];
        }
        _biasWeight_new[j] = _biasWeight_cur[j] + tmp;
    }
    
    memset(_gradient, 0.0, sizeof(float)*nPreviousUnit*nCurrentUnit*(max_num_threads-1));
    memset(_biasGradient, 0.0, sizeof(float)*nCurrentUnit*(max_num_threads-1));
}


int MLP_Layer::GetMaxOutputIndex(int threadID, int round)
{
    float* _outputLayer = &outputLayer[round][threadID*nCurrentUnit];
    int maxIdx = threadID*nCurrentUnit;
    for(int o = maxIdx+1; o < maxIdx+nCurrentUnit; o++){
        if(_outputLayer[o] > _outputLayer[maxIdx])
            maxIdx = o;
    }
    
    return maxIdx;
}


