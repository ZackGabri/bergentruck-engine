#include <math.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "nn.h"

// This function will stop iterating over the variadic arguments when either
// a) The number of iterations has reached the value of LayerCount
// b) It encounters a null pointer (in which case it will return a null pointer as the function's return value)
PTWM_RNN_NETWORK TWMNetworkRNNCreateNetwork(unsigned long long LayerCount, PTWM_RNN_LAYER pLayer, ...)
{
	PTWM_RNN_NETWORK pNetwork = 0;
	PTWM_RNN_LAYER pAddedLayer = 0;
	va_list VarArgs = 0;

	if (!LayerCount || !pLayer)
	{
		return 0;
	}

	pNetwork = malloc(sizeof(TWM_RNN_NETWORK));
	if (!pNetwork)
	{
		return 0;
	}
	memset(pNetwork, 0, sizeof(TWM_RNN_NETWORK));

	pNetwork->LayerCount = LayerCount;

	// Allocate LayerCount * sizeof(PTWM_RNN_LAYER) bytes of memory.
	// This is because this array stores a list of layer pointers that define the network.
	// As such, to reference layer 2, use pNetwork->pLayers[1].
	pNetwork->pLayers = malloc(LayerCount * sizeof(PTWM_RNN_LAYER));
	if (!pNetwork->pLayers)
	{
		goto Cleanup;
	}
	memset(pNetwork->pLayers, 0, LayerCount * sizeof(PTWM_RNN_LAYER));

	pNetwork->pLayers[0] = pLayer;

	va_start(VarArgs, pLayer);
	for (unsigned long long i = 1; i < LayerCount; i++)
	{
		pAddedLayer = va_arg(VarArgs, PTWM_RNN_LAYER);
		if (!pAddedLayer)
		{
			__debugbreak();

			goto Cleanup;
		}

		pNetwork->pLayers[i] = pAddedLayer;
	}
	va_end(VarArgs);

	return pNetwork;

	// This is placed below the above return statement as the program should not clean-up after building the entire object.
	// If all of the memory allocations are successful, then the above return statement alone is perfect.
Cleanup:
	if (pNetwork->pLayers)
	{
		for (unsigned long long i = 0; i < LayerCount; i++)
		{
			if (!pNetwork->pLayers[i])
			{
				continue;
			}

			free(pNetwork->pLayers[i]);
		}

		free(pNetwork->pLayers);
	}

	free(pNetwork);

	return 0;
}

PTWM_RNN_LAYER TWMNetworkRNNCreateLayer(unsigned long long CurrentNeuronCount, unsigned long long PreviousNeuronCount, double WeightMinimum, double WeightMaximum, double LearningRate)
{
	PTWM_RNN_LAYER pLayer = 0;

	double RandomValue = 0;

	if (!CurrentNeuronCount || !PreviousNeuronCount)
	{
		return 0;
	}

	pLayer = malloc(sizeof(TWM_RNN_LAYER));
	if (!pLayer)
	{
		return 0;
	}
	memset(pLayer, 0, sizeof(TWM_RNN_LAYER));

	pLayer->CurrentNeuronCount = CurrentNeuronCount;
	pLayer->PreviousNeuronCount = PreviousNeuronCount;
	pLayer->NeuronMinimumWeight = WeightMinimum;
	pLayer->NeuronMaximumWeight = WeightMaximum;
	pLayer->LayerLearningRate = LearningRate;

	pLayer->pNeuronValues = malloc(CurrentNeuronCount * sizeof(double));
	if (!pLayer->pNeuronValues)
	{
		goto Cleanup;
	}
	memset(pLayer->pNeuronValues, 0, CurrentNeuronCount * sizeof(double));

	pLayer->pNeuronOutputGradients = malloc(CurrentNeuronCount * sizeof(double));
	if (!pLayer->pNeuronOutputGradients)
	{
		goto Cleanup;
	}
	memset(pLayer->pNeuronOutputGradients, 0, CurrentNeuronCount * sizeof(double));

	pLayer->pNeuronErrors = malloc(CurrentNeuronCount * sizeof(double));
	if (!pLayer->pNeuronErrors)
	{
		goto Cleanup;
	}
	memset(pLayer->pNeuronErrors, 0, CurrentNeuronCount * sizeof(double));

	pLayer->pNeuronBiases = malloc(CurrentNeuronCount * sizeof(double));
	if (!pLayer->pNeuronBiases)
	{
		goto Cleanup;
	}
	memset(pLayer->pNeuronBiases, 0, CurrentNeuronCount * sizeof(double));

	pLayer->pNeuronBiasGradients = malloc(CurrentNeuronCount * sizeof(double));
	if (!pLayer->pNeuronBiasGradients)
	{
		goto Cleanup;
	}
	memset(pLayer->pNeuronBiasGradients, 0, CurrentNeuronCount * sizeof(double));

	pLayer->pNeuronWeights = malloc(CurrentNeuronCount * sizeof(double));
	if (!pLayer->pNeuronWeights)
	{
		goto Cleanup;
	}
	memset(pLayer->pNeuronWeights, 0, CurrentNeuronCount * sizeof(double));
	for (unsigned long long i = 0; i < CurrentNeuronCount; i++)
	{
		pLayer->pNeuronWeights[i] = malloc(PreviousNeuronCount * sizeof(double));
		if (!pLayer->pNeuronWeights[i])
		{
			goto Cleanup;
		}
		memset(pLayer->pNeuronWeights[i], 0, PreviousNeuronCount * sizeof(double));
	}

	pLayer->pNeuronRecurrentWeights = malloc(CurrentNeuronCount * sizeof(double));
	if (!pLayer->pNeuronRecurrentWeights)
	{
		goto Cleanup;
	}
	memset(pLayer->pNeuronRecurrentWeights, 0, CurrentNeuronCount * sizeof(double));
	for (unsigned long long i = 0; i < CurrentNeuronCount; i++)
	{
		pLayer->pNeuronRecurrentWeights[i] = malloc(PreviousNeuronCount * sizeof(double));
		if (!pLayer->pNeuronRecurrentWeights[i])
		{
			goto Cleanup;
		}
		memset(pLayer->pNeuronRecurrentWeights[i], 0, PreviousNeuronCount * sizeof(double));
	}

	pLayer->pNeuronWeightGradients = malloc(CurrentNeuronCount * sizeof(double));
	if (!pLayer->pNeuronWeightGradients)
	{
		goto Cleanup;
	}
	memset(pLayer->pNeuronWeightGradients, 0, CurrentNeuronCount * sizeof(double));
	for (unsigned long long i = 0; i < CurrentNeuronCount; i++)
	{
		pLayer->pNeuronWeightGradients[i] = malloc(PreviousNeuronCount * sizeof(double));
		if (!pLayer->pNeuronWeightGradients[i])
		{
			goto Cleanup;
		}
		memset(pLayer->pNeuronWeightGradients[i], 0, PreviousNeuronCount * sizeof(double));
	}

	pLayer->pNeuronRecurrentWeightGradients = malloc(CurrentNeuronCount * sizeof(double));
	if (!pLayer->pNeuronRecurrentWeightGradients)
	{
		goto Cleanup;
	}
	memset(pLayer->pNeuronRecurrentWeightGradients, 0, CurrentNeuronCount * sizeof(double));
	for (unsigned long long i = 0; i < CurrentNeuronCount; i++)
	{
		pLayer->pNeuronRecurrentWeightGradients[i] = malloc(PreviousNeuronCount * sizeof(double));
		if (!pLayer->pNeuronRecurrentWeightGradients[i])
		{
			goto Cleanup;
		}
		memset(pLayer->pNeuronRecurrentWeightGradients[i], 0, PreviousNeuronCount * sizeof(double));
	}

	// Randomize the network weights and biases.
	// Was originally going to be its own function, but I will keep it here for now.
	for (unsigned long long i = 0; i < CurrentNeuronCount; i++)
	{
		for (unsigned long long j = 0; j < PreviousNeuronCount; j++)
		{
			RandomValue = 0.0;

			while (!RandomValue || RandomValue < WeightMinimum || RandomValue > WeightMaximum)
			{
				RandomValue = rand();
				RandomValue /= 10000.0;

				if (!(rand() % 2))
				{
					RandomValue *= -1.0;
				}

				pLayer->pNeuronWeights[i][j] = RandomValue;
			}
		}
	}

	for (unsigned long long i = 0; i < CurrentNeuronCount; i++)
	{
		for (unsigned long long j = 0; j < PreviousNeuronCount; j++)
		{
			RandomValue = 0.0;

			while (!RandomValue || RandomValue < WeightMinimum || RandomValue > WeightMaximum)
			{
				RandomValue = rand();
				RandomValue /= 10000.0;

				if (!(rand() % 2))
				{
					RandomValue *= -1.0;
				}

				pLayer->pNeuronRecurrentWeights[i][j] = RandomValue;
			}
		}
	}

	for (unsigned long long i = 0; i < CurrentNeuronCount; i++)
	{
		RandomValue = 0.0;

		while (!RandomValue || RandomValue < WeightMinimum || RandomValue > WeightMaximum)
		{
			RandomValue = rand();
			RandomValue /= 10000.0;

			if (!(rand() % 2))
			{
				RandomValue *= -1.0;
			}

			pLayer->pNeuronBiases[i] = RandomValue;
		}
	}

	return pLayer;

	// This is placed below the above return statement as the program should not clean-up after building the entire object.
	// If all of the memory allocations are successful, then the above return statement alone is perfect.
Cleanup:
	if (pLayer->pNeuronWeightGradients)
	{
		for (unsigned long long i = 0; i < CurrentNeuronCount; i++)
		{
			if (!pLayer->pNeuronWeightGradients[i])
			{
				continue;
			}

			free(pLayer->pNeuronWeightGradients[i]);
		}

		free(pLayer->pNeuronWeightGradients);
	}

	if (pLayer->pNeuronRecurrentWeightGradients)
	{
		for (unsigned long long i = 0; i < CurrentNeuronCount; i++)
		{
			if (!pLayer->pNeuronRecurrentWeightGradients[i])
			{
				continue;
			}

			free(pLayer->pNeuronRecurrentWeightGradients[i]);
		}

		free(pLayer->pNeuronRecurrentWeightGradients);
	}

	if (pLayer->pNeuronWeights)
	{
		for (unsigned long long i = 0; i < CurrentNeuronCount; i++)
		{
			if (!pLayer->pNeuronWeights[i])
			{
				continue;
			}

			free(pLayer->pNeuronWeights[i]);
		}

		free(pLayer->pNeuronWeights);
	}

	if (pLayer->pNeuronRecurrentWeights)
	{
		for (unsigned long long i = 0; i < CurrentNeuronCount; i++)
		{
			if (!pLayer->pNeuronRecurrentWeights[i])
			{
				continue;
			}

			free(pLayer->pNeuronRecurrentWeights[i]);
		}

		free(pLayer->pNeuronRecurrentWeights);
	}

	if (pLayer->pNeuronBiasGradients)
	{
		free(pLayer->pNeuronBiasGradients);
	}

	if (pLayer->pNeuronBiases)
	{
		free(pLayer->pNeuronBiases);
	}

	if (pLayer->pNeuronErrors)
	{
		free(pLayer->pNeuronErrors);
	}

	if (pLayer->pNeuronOutputGradients)
	{
		free(pLayer->pNeuronOutputGradients);
	}

	if (pLayer->pNeuronValues)
	{
		free(pLayer->pNeuronValues);
	}

	free(pLayer);

	return 0;
}

double TWMActivationSigmoid(double Value)
{
	return 1.0 / (1.0 + exp(-Value));
}

double TWMActivationSigmoidPrime(double Value)
{
	return TWMActivationSigmoid(Value) * (1.0 - TWMActivationSigmoid(Value));
}

void TWMNetworkRNNResetNetwork(PTWM_RNN_NETWORK pNetwork)
{
	if (!pNetwork)
	{
		return;
	}

	for (unsigned long long i = 0; i < pNetwork->LayerCount; i++)
	{
		if (!pNetwork->pLayers[i]->pNeuronHiddenStates)
		{
			continue;
		}

		for (unsigned long long t = 0; t < pNetwork->TotalTimeSteps; t++)
		{
			if (!pNetwork->pLayers[i]->pNeuronHiddenStates[t])
			{
				continue;
			}

			free(pNetwork->pLayers[i]->pNeuronHiddenStates[t]);
			pNetwork->pLayers[i]->pNeuronHiddenStates[t] = 0;
		}

		free(pNetwork->pLayers[i]->pNeuronHiddenStates);
		pNetwork->pLayers[i]->pNeuronHiddenStates = 0;
	}

	pNetwork->TotalTimeSteps = 0;

	return;
}

double *TWMNetworkRNNAdvanceNetwork(PTWM_RNN_NETWORK pNetwork, double *pInputs, unsigned long long InputLength)
{
	if (!pNetwork || !pInputs || !InputLength)
	{
		return 0;
	}

	if (!TWMNetworkRNNAdvanceLayer(pNetwork->pLayers[0], pNetwork->TotalTimeSteps, pInputs, InputLength))
	{
		return 0;
	}

	for (unsigned long long i = 1; i < pNetwork->LayerCount; i++)
	{
		if (!TWMNetworkRNNAdvanceLayer(pNetwork->pLayers[i], pNetwork->TotalTimeSteps, pNetwork->pLayers[i - 1]->pNeuronValues, pNetwork->pLayers[i]->PreviousNeuronCount))
		{
			return 0;
		}
	}

	pNetwork->TotalTimeSteps++;

	return pNetwork->pLayers[pNetwork->LayerCount - 1]->pNeuronValues;
}

double *TWMNetworkRNNAdvanceLayer(PTWM_RNN_LAYER pLayer, unsigned long long CurrentTimeStep, double *pInputs, unsigned long long InputLength)
{
	if (!pLayer || !pInputs || !InputLength)
	{
		return 0;
	}

	// Allocate memory for each hidden state
	if (!pLayer->pNeuronHiddenStates)
	{
		pLayer->pNeuronHiddenStates = malloc(sizeof(double **));
	}
	else
	{
		pLayer->pNeuronHiddenStates = realloc(pLayer->pNeuronHiddenStates, sizeof(double **) * (CurrentTimeStep + 1));
	}
	if (!pLayer->pNeuronHiddenStates)
	{
		__debugbreak();
		return 0;
	}

	pLayer->pNeuronHiddenStates[CurrentTimeStep] = (double *)malloc(sizeof(double) * pLayer->CurrentNeuronCount);
	if (!pLayer->pNeuronHiddenStates[CurrentTimeStep])
	{
		__debugbreak();
		return 0;
	}
	memset(pLayer->pNeuronHiddenStates[CurrentTimeStep], 0, sizeof(double) * pLayer->CurrentNeuronCount);

	for (unsigned long long i = 0; i < pLayer->CurrentNeuronCount; i++)
	{
		pLayer->pNeuronValues[i] = 0;

		for (unsigned long long j = 0; j < InputLength; j++)
		{
			pLayer->pNeuronValues[i] += pInputs[j] * pLayer->pNeuronWeights[i][j];

			// This is my new addition, the hidden states will be the values of each neuron from the previous calculation
			if (CurrentTimeStep)
			{
				// Fetch the hidden state for the previous time step
				pLayer->pNeuronValues[i] += pLayer->pNeuronHiddenStates[CurrentTimeStep - 1][i] * pLayer->pNeuronRecurrentWeights[i][j];
			}
		}

		pLayer->pNeuronValues[i] += pLayer->pNeuronBiases[i];

		pLayer->pNeuronValues[i] = TWMActivationSigmoid(pLayer->pNeuronValues[i]);

		// Record the hidden states for this time step
		// This will be used in the backpropagation process, and it will also be used for future calculations as well!
		pLayer->pNeuronHiddenStates[CurrentTimeStep][i] = pLayer->pNeuronValues[i];
	}

	return pLayer->pNeuronValues;
}

// `a = sigmoid(h_t - 1 * W_hh + x * W_xh + b)`
//
// gradient of `L` with respect to `h_t-1` = `sigmoid'(h_t - 1 * W_hh + x * W_xh + b)* W_hh`
// gradient of `L` with respect to `W_hh` = `sigmoid'(h_t - 1 * W_hh + x * W_xh + b)* h_t - 1`
// gradient of `L` with respect to `x` = `sigmoid'(h_t - 1 * W_hh + x * W_xh + b)* W_xh`
// gradient of `L` with respect to `W_xh` = `sigmoid'(h_t - 1 * W_hh + x * W_xh + b)* x`
// gradient of `L` with respect to `b` = `sigmoid'(h_t - 1 * W_hh + x * W_xh + b) * 1`

char TWMNetworkRNNUpdateNetwork(PTWM_RNN_NETWORK pNetwork, double *pExpectedOutput)
{
	unsigned long long t = 0;

	if (!pNetwork || !pExpectedOutput)
	{
		return 1;
	}

	t = pNetwork->TotalTimeSteps;

	while (t)
	{
		t--;

		// Do not use greater than OR equal to as that would lead to an infinite loop.
		// Old code: for (int i = this.FeedForwardLayers.Length - 1; i >= 0; i--)
		for (unsigned long long i = pNetwork->LayerCount - 1; i > 0; i--)
		{
			if (i == pNetwork->LayerCount - 1)
			{
				// Backprop output
				if (TWMNetworkRNNBackpropagateOutput(pNetwork->pLayers[i], pNetwork->pLayers[i - 1], pExpectedOutput, t))
				{
					return 1;
				}
			}
			else
			{
				// Backprop hidden
				if (TWMNetworkRNNBackpropagateHidden(pNetwork->pLayers[i], pNetwork->pLayers[i - 1], pNetwork->pLayers[i + 1], t))
				{
					return 1;
				}
			}
		}

		for (unsigned long long i = 0; i < pNetwork->LayerCount; i++)
		{
			if (TWMNetworkRNNUpdateLayer(pNetwork->pLayers[i]))
			{
				return 1;
			}
		}
	}

	TWMNetworkRNNResetNetwork(pNetwork);
	return 0;
}

char TWMNetworkRNNUpdateLayer(PTWM_RNN_LAYER pLayer)
{
	if (!pLayer || !pLayer->CurrentNeuronCount || !pLayer->PreviousNeuronCount)
	{
		return 1;
	}

	for (unsigned long long i = 0; i < pLayer->CurrentNeuronCount; i++)
	{
		for (unsigned long long j = 0; j < pLayer->PreviousNeuronCount; j++)
		{
			pLayer->pNeuronWeights[i][j] -= pLayer->pNeuronWeightGradients[i][j] * pLayer->LayerLearningRate;

			pLayer->pNeuronRecurrentWeights[i][j] -= pLayer->pNeuronRecurrentWeightGradients[i][j] * pLayer->LayerLearningRate;
		}

		pLayer->pNeuronBiases[i] -= pLayer->pNeuronBiasGradients[i] * pLayer->LayerLearningRate;
	}

	return 0;
}

char TWMNetworkRNNBackpropagateHidden(PTWM_RNN_LAYER pLayer, PTWM_RNN_LAYER pPreviousLayer, PTWM_RNN_LAYER pNextLayer, unsigned long long CurrentTimeStep)
{
	for (unsigned long long i = 0; i < pLayer->CurrentNeuronCount; i++)
	{
		pLayer->pNeuronOutputGradients[i] = 0;

		for (unsigned long long j = 0; j < pNextLayer->CurrentNeuronCount; j++)
		{
			pLayer->pNeuronOutputGradients[i] += pNextLayer->pNeuronOutputGradients[j] * pNextLayer->pNeuronWeights[j][i];
			pLayer->pNeuronOutputGradients[i] += pNextLayer->pNeuronOutputGradients[j] * pNextLayer->pNeuronRecurrentWeights[j][i];

			pLayer->pNeuronOutputGradients[i] += pNextLayer->pNeuronOutputGradients[j] * pNextLayer->pNeuronBiases[j];
		}

		pLayer->pNeuronOutputGradients[i] *= TWMActivationSigmoidPrime(pLayer->pNeuronValues[i]);
	}

	for (unsigned long long i = 0; i < pLayer->CurrentNeuronCount; i++)
	{
		for (unsigned long long j = 0; j < pPreviousLayer->CurrentNeuronCount; j++)
		{
			pLayer->pNeuronWeightGradients[i][j] = pLayer->pNeuronOutputGradients[i] * pPreviousLayer->pNeuronValues[j];
			pLayer->pNeuronRecurrentWeightGradients[i][j] = pLayer->pNeuronOutputGradients[i] * pPreviousLayer->pNeuronHiddenStates[CurrentTimeStep][j];

			pLayer->pNeuronBiasGradients[i] = pLayer->pNeuronOutputGradients[i] * pPreviousLayer->pNeuronValues[j];
		}
	}

	return 0;
}

char TWMNetworkRNNBackpropagateOutput(PTWM_RNN_LAYER pLayer, PTWM_RNN_LAYER pPreviousLayer, double *pExpectedOutput, unsigned long long CurrentTimeStep)
{
	for (unsigned long long i = 0; i < pLayer->CurrentNeuronCount; i++)
	{
		// Errors
		// NeuronValue - ExpectedValue;
		pLayer->pNeuronErrors[i] = pLayer->pNeuronValues[i] - pExpectedOutput[i];
	}

	for (unsigned long long i = 0; i < pLayer->CurrentNeuronCount; i++)
	{
		// z = h_t - 1 * W_hh + x * W_xh + b
		// pLayer->pNeuronValues[i] is equal to sigmoid(z)
		//
		// As such, for the hidden states we need to multiply by the neuron output gradients calculated here.
		pLayer->pNeuronOutputGradients[i] = pLayer->pNeuronErrors[i] * TWMActivationSigmoidPrime(pLayer->pNeuronValues[i]);
	}

	for (unsigned long long i = 0; i < pLayer->CurrentNeuronCount; i++)
	{
		for (unsigned long long j = 0; j < pPreviousLayer->CurrentNeuronCount; j++)
		{
			pLayer->pNeuronWeightGradients[i][j] = pLayer->pNeuronOutputGradients[i] * pPreviousLayer->pNeuronValues[j];
			// See notes mentioned above
			pLayer->pNeuronRecurrentWeightGradients[i][j] = pLayer->pNeuronOutputGradients[i] * pPreviousLayer->pNeuronHiddenStates[CurrentTimeStep][j];

			pLayer->pNeuronBiasGradients[i] = pLayer->pNeuronOutputGradients[i] * pPreviousLayer->pNeuronValues[j];
		}
	}

	return 0;
}

// Added this function upon request as it seems to be helpful.
// You're welcome :3
void TWMNetworkRNNUpdateLearningRate(PTWM_RNN_NETWORK pNetwork, double LearningRate)
{
	if (!pNetwork)
	{
		__debugbreak();
		return;
	}

	for (unsigned long long i = 0; i < pNetwork->LayerCount; i++)
	{
		if (!pNetwork->pLayers[i])
		{
			__debugbreak();
			return;
		}

		pNetwork->pLayers[i]->LayerLearningRate = LearningRate;
	}
}