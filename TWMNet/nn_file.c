#include <Windows.h>
#include <math.h>
#include <stdio.h>

#include "nn.h"

// Highkey, these helper functions were AI generated because I just do not care enough right now
// I'll come back and fix up a lot of the code once I know everything works as intended ><".

// Sanity limits so a corrupt file can't trigger a gigantic allocation. Adjust to taste.
#define TWM_MAX_LAYERS 1024ULL
#define TWM_MAX_NEURONS (1ULL << 20)

char SaveDoubleArray(FILE *f, double *arr, unsigned long long size)
{
	if (size == 0)
		return 1;
	return fwrite(arr, sizeof(double), size, f) == size;
}

char LoadDoubleArray(FILE *f, double *arr, unsigned long long size)
{
	if (size == 0)
		return 1;
	return fread(arr, sizeof(double), size, f) == size;
}

char SaveDouble2DArray(FILE *f, double **arr, unsigned long long rows, unsigned long long cols)
{
	if (rows == 0 || cols == 0)
		return 1;
	for (unsigned long long i = 0; i < rows; i++)
	{
		if (fwrite(arr[i], sizeof(double), cols, f) != cols)
			return 0;
	}
	return 1;
}

char LoadDouble2DArray(FILE *f, double **arr, unsigned long long rows, unsigned long long cols)
{
	if (rows == 0 || cols == 0)
		return 1;
	for (unsigned long long i = 0; i < rows; i++)
	{
		if (fread(arr[i], sizeof(double), cols, f) != cols)
			return 0;
	}
	return 1;
}

char TWMNetworkRNNExportNetwork(PTWM_RNN_NETWORK pNetwork, char *pFilePath)
{
	FILE *File = fopen(pFilePath, "wb");
	if (!File)
	{
		goto Error;
	}

	if (fwrite(&pNetwork->LayerCount, sizeof(unsigned long long), 1, File) != 1)
		goto Error;

	for (unsigned long long i = 0; i < pNetwork->LayerCount; i++)
	{
		PTWM_RNN_LAYER pLayer = pNetwork->pLayers[i];
		if (!pLayer)
			goto Error;

		if (fwrite(&pLayer->NeuronMinimumWeight, sizeof(double), 1, File) != 1)
			goto Error;
		if (fwrite(&pLayer->NeuronMaximumWeight, sizeof(double), 1, File) != 1)
			goto Error;
		if (fwrite(&pLayer->CurrentNeuronCount, sizeof(unsigned long long), 1, File) != 1)
			goto Error;
		if (fwrite(&pLayer->PreviousNeuronCount, sizeof(unsigned long long), 1, File) != 1)
			goto Error;
		if (fwrite(&pLayer->LayerLearningRate, sizeof(double), 1, File) != 1)
			goto Error;

		unsigned long long N = pLayer->CurrentNeuronCount;
		unsigned long long P = pLayer->PreviousNeuronCount;

		if (!SaveDoubleArray(File, pLayer->pNeuronBiases, N))
			goto Error;
		if (!SaveDouble2DArray(File, pLayer->pNeuronWeights, N, P))
			goto Error;
		if (!SaveDouble2DArray(File, pLayer->pNeuronRecurrentWeights, N, P))
			goto Error;
	}

	fclose(File);
	return 0;

Error:
	printf("wtf happened?\n");
	if (File)
	{
		fclose(File);
	}
	return 1;
}

// Frees a layer created by TWMNetworkRNNCreateLayer (or by import).
static void FreeLayer(PTWM_RNN_LAYER pLayer)
{
	if (!pLayer)
		return;

	unsigned long long N = pLayer->CurrentNeuronCount;

	FreeDouble2DArray(pLayer->pNeuronWeights, N);
	FreeDouble2DArray(pLayer->pNeuronRecurrentWeights, N);
	FreeDouble2DArray(pLayer->pNeuronWeightGradients, N);
	FreeDouble2DArray(pLayer->pNeuronRecurrentWeightGradients, N);

	free(pLayer->pNeuronBiases);
	free(pLayer->pNeuronBiasGradients);
	free(pLayer->pNeuronErrors);
	free(pLayer->pNeuronOutputGradients);
	free(pLayer->pNeuronValues);

	// Hidden states are only allocated while running; normally NULL here.
	if (pLayer->pNeuronHiddenStates)
	{
		// Time step count isn't stored on the layer, so this path should be
		// handled by TWMNetworkRNNResetNetwork before freeing a live network.
		free(pLayer->pNeuronHiddenStates);
	}

	free(pLayer);
}

PTWM_RNN_NETWORK TWMNetworkRNNImportNetwork(char *pFilePath)
{
	PTWM_RNN_NETWORK pNetwork = 0;
	PTWM_RNN_LAYER *pLayers = 0;
	unsigned long long LayerCount = 0;
	unsigned long long LoadedLayers = 0;

	FILE *File = fopen(pFilePath, "rb");
	if (!File)
		goto Error;

	if (fread(&LayerCount, sizeof(unsigned long long), 1, File) != 1)
		goto Error;
	if (LayerCount == 0 || LayerCount > TWM_MAX_LAYERS)
		goto Error;

	pLayers = malloc(LayerCount * sizeof(PTWM_RNN_LAYER));
	if (!pLayers)
		goto Error;
	memset(pLayers, 0, LayerCount * sizeof(PTWM_RNN_LAYER));

	for (unsigned long long i = 0; i < LayerCount; i++)
	{
		double MinWeight = 0, MaxWeight = 0, LearningRate = 0;
		unsigned long long N = 0, P = 0;

		// Same order the export writes them in.
		if (fread(&MinWeight, sizeof(double), 1, File) != 1)
			goto Error;
		if (fread(&MaxWeight, sizeof(double), 1, File) != 1)
			goto Error;
		if (fread(&N, sizeof(unsigned long long), 1, File) != 1)
			goto Error;
		if (fread(&P, sizeof(unsigned long long), 1, File) != 1)
			goto Error;
		if (fread(&LearningRate, sizeof(double), 1, File) != 1)
			goto Error;

		if (N == 0 || P == 0 || N > TWM_MAX_NEURONS || P > TWM_MAX_NEURONS)
			goto Error;

		// Each layer's input size must match the previous layer's output size.
		if (i > 0 && pLayers[i - 1]->CurrentNeuronCount != P)
			goto Error;

		// Allocates every buffer (values, errors, gradients, weights, ...).
		// The random weights it generates get overwritten below.
		pLayers[i] = TWMNetworkRNNCreateLayer(N, P, MinWeight, MaxWeight, LearningRate);
		if (!pLayers[i])
			goto Error;
		LoadedLayers = i + 1;

		if (!LoadDoubleArray(File, pLayers[i]->pNeuronBiases, N))
			goto Error;
		if (!LoadDouble2DArray(File, pLayers[i]->pNeuronWeights, N, P))
			goto Error;
		if (!LoadDouble2DArray(File, pLayers[i]->pNeuronRecurrentWeights, N, P))
			goto Error;
	}

	pNetwork = malloc(sizeof(TWM_RNN_NETWORK));
	if (!pNetwork)
		goto Error;
	memset(pNetwork, 0, sizeof(TWM_RNN_NETWORK));

	pNetwork->LayerCount = LayerCount;
	pNetwork->pLayers = pLayers;

	fclose(File);
	return pNetwork;

Error:
	printf("wtf happened?\n");
	if (File)
		fclose(File);

	if (pLayers)
	{
		for (unsigned long long i = 0; i < LoadedLayers; i++)
			FreeLayer(pLayers[i]);
		free(pLayers);
	}

	return 0;
}