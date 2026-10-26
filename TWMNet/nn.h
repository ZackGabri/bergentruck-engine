#pragma once

#include <stdio.h>

typedef struct _TWM_RNN_LAYER
{
	double *pNeuronValues;
	double **pNeuronHiddenStates;

	double **pNeuronWeights;
	double **pNeuronRecurrentWeights;
	double *pNeuronBiases;

	double *pNeuronErrors;

	double *pNeuronOutputGradients;
	double *pNeuronBiasGradients;
	double **pNeuronWeightGradients;
	double **pNeuronRecurrentWeightGradients;

	double NeuronMinimumWeight;
	double NeuronMaximumWeight;

	unsigned long long CurrentNeuronCount;
	unsigned long long PreviousNeuronCount;

	double LayerLearningRate;
} TWM_RNN_LAYER, *PTWM_RNN_LAYER;

typedef struct _TWM_RNN_NETWORK
{
	PTWM_RNN_LAYER *pLayers;

	unsigned long long LayerCount;
	unsigned long long TotalTimeSteps;
} TWM_RNN_NETWORK, *PTWM_RNN_NETWORK;

__declspec(dllexport) PTWM_RNN_NETWORK TWMNetworkRNNCreateNetwork(unsigned long long LayerCount, PTWM_RNN_LAYER pLayer, ...);
__declspec(dllexport) double *TWMNetworkRNNAdvanceNetwork(PTWM_RNN_NETWORK pNetwork, double *pInputs, unsigned long long InputLength);
__declspec(dllexport) void TWMNetworkRNNResetNetwork(PTWM_RNN_NETWORK pNetwork);
__declspec(dllexport) char TWMNetworkRNNUpdateNetwork(PTWM_RNN_NETWORK pNetwork, double *pExpectedOutput);
__declspec(dllexport) void TWMNetworkRNNUpdateLearningRate(PTWM_RNN_NETWORK pNetwork, double LearningRate);

__declspec(dllexport) char TWMNetworkRNNExportNetwork(PTWM_RNN_NETWORK pNetwork, char *pFilePath);
__declspec(dllexport) PTWM_RNN_NETWORK TWMNetworkRNNImportNetwork(char *pFilePath);

PTWM_RNN_LAYER TWMNetworkRNNCreateLayer(unsigned long long CurrentNeuronCount, unsigned long long PreviousNeuronCount, double WeightMinimum, double WeightMaximum, double LearningRate);
double *TWMNetworkRNNAdvanceLayer(PTWM_RNN_LAYER pLayer, unsigned long long CurrentTimeStep, double *pInputs, unsigned long long InputLength);
char TWMNetworkRNNUpdateLayer(PTWM_RNN_LAYER pCurrentLayer);
char TWMNetworkRNNBackpropagateHidden(PTWM_RNN_LAYER pLayer, PTWM_RNN_LAYER pPreviousLayer, PTWM_RNN_LAYER pNextLayer, unsigned long long CurrentTimeStep);
char TWMNetworkRNNBackpropagateOutput(PTWM_RNN_LAYER pLayer, PTWM_RNN_LAYER pPreviousLayer, double *pExpectedOutput, unsigned long long CurrentTimeStep);

char SaveDoubleArray(FILE *f, double *arr, unsigned long long size);
char LoadDoubleArray(FILE *f, double *arr, unsigned long long size);
char SaveDouble2DArray(FILE *f, double **arr, unsigned long long rows, unsigned long long cols);
char LoadDouble2DArray(FILE *f, double **arr, unsigned long long rows, unsigned long long cols);