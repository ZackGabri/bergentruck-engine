#include <Windows.h>
#include <math.h>
#include <stdio.h>

#include "nn.h"

// Highkey, these helper functions were AI generated because I just do not care enough right now
// I'll come back and fix up a lot of the code once I know everything works as intended ><".

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
			return 1;

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

PTWM_RNN_NETWORK TWMNetworkRNNImportNetwork(char *pFilePath)
{
	PTWM_RNN_NETWORK pNetwork = 0;
	PTWM_RNN_LAYER *pLayers = 0;

	unsigned long long LayerCount = 0;

	FILE *File = fopen(pFilePath, "rb");
	if (!File)
	{
		goto Error;
	}

	if (fread(&LayerCount, sizeof(unsigned long long), 1, File) != 1)
		goto Error;

	pLayers = malloc(LayerCount * sizeof(PTWM_RNN_LAYER));
	if (!pLayers)
	{
		goto Error;
	}
	memset(pLayers, 0, LayerCount * sizeof(PTWM_RNN_LAYER));

	for (unsigned long long i = 0; i < LayerCount; i++)
	{
		pLayers[i] = malloc(sizeof(TWM_RNN_LAYER)); // fix 1
		if (!pLayers[i])
		{
			goto Error;
		}
		memset(pLayers[i], 0, sizeof(TWM_RNN_LAYER)); // fix 1

		if (fread(&pLayers[i]->NeuronMinimumWeight, sizeof(double), 1, File) != 1)
			goto Error;
		if (fread(&pLayers[i]->NeuronMaximumWeight, sizeof(double), 1, File) != 1)
			goto Error;
		if (fread(&pLayers[i]->CurrentNeuronCount, sizeof(unsigned long long), 1, File) != 1)
			goto Error;
		if (fread(&pLayers[i]->PreviousNeuronCount, sizeof(unsigned long long), 1, File) != 1)
			goto Error;
		if (fread(&pLayers[i]->LayerLearningRate, sizeof(double), 1, File) != 1)
			goto Error;

		unsigned long long N = pLayers[i]->CurrentNeuronCount;
		unsigned long long P = pLayers[i]->PreviousNeuronCount;

		// ---- fix 2: allocate the arrays before loading into them ----
		pLayers[i]->pNeuronBiases = malloc(N * sizeof(double));
		pLayers[i]->pNeuronWeights = malloc(N * sizeof(double *));
		pLayers[i]->pNeuronRecurrentWeights = malloc(N * sizeof(double *));
		if (!pLayers[i]->pNeuronBiases ||
			!pLayers[i]->pNeuronWeights ||
			!pLayers[i]->pNeuronRecurrentWeights)
		{
			goto Error;
		}

		// zero the row pointers so the Error cleanup can safely free partial results
		memset(pLayers[i]->pNeuronWeights, 0, N * sizeof(double *));
		memset(pLayers[i]->pNeuronRecurrentWeights, 0, N * sizeof(double *));

		for (unsigned long long j = 0; j < N; j++)
		{
			pLayers[i]->pNeuronWeights[j] = malloc(P * sizeof(double));
			pLayers[i]->pNeuronRecurrentWeights[j] = malloc(P * sizeof(double));
			if (!pLayers[i]->pNeuronWeights[j] || !pLayers[i]->pNeuronRecurrentWeights[j])
			{
				goto Error;
			}
		}
		// -------------------------------------------------------------

		if (!LoadDoubleArray(File, pLayers[i]->pNeuronBiases, N))
			goto Error;
		if (!LoadDouble2DArray(File, pLayers[i]->pNeuronWeights, N, P))
			goto Error;
		if (!LoadDouble2DArray(File, pLayers[i]->pNeuronRecurrentWeights, N, P))
			goto Error;
	}

	pNetwork = malloc(sizeof(TWM_RNN_NETWORK));
	if (!pNetwork)
	{
		goto Error;
	}
	memset(pNetwork, 0, sizeof(TWM_RNN_NETWORK));

	pNetwork->LayerCount = LayerCount;
	pNetwork->pLayers = pLayers;

	fclose(File);
	return pNetwork;

Error:
	printf("wtf happened?\n");
	if (File)
	{
		fclose(File);
	}

	if (!pNetwork)
	{
		return 0;
	}

	for (unsigned long long i = 0; i < pNetwork->LayerCount; i++)
	{
		if (!pNetwork->pLayers[i])
		{
			continue;
		}

		for (unsigned long long j = 0; j < pNetwork->pLayers[i]->CurrentNeuronCount; j++)
		{
			free(pNetwork->pLayers[i]->pNeuronWeights[i]);
			free(pNetwork->pLayers[i]->pNeuronRecurrentWeights[i]);
		}

		free(pNetwork->pLayers[i]->pNeuronBiases);
		free(pNetwork->pLayers[i]);
	}

	free(pNetwork);
	return 0;
}