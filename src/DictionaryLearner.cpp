#include "DictionaryLearner.hpp"

namespace SparseCoding
{
	const double DictionaryLearner::DEFAULT_STEP = 1.0;
	const double DictionaryLearner::DEFAULT_ISTA_STEP = 0.1;
	const double DictionaryLearner::DEFAULT_NOISE_VARIANCE = 0.1;
	const double DictionaryLearner::DEFAULT_TOLERANCE = 0.1;

	DictionaryLearner::DictionaryLearner(std::vector<cv::Mat>& input, double basisFunctionCount, unsigned int patchLength, unsigned int batchCount)
	{
		images = input;
		patchSize = patchLength;
		sampleCount = batchCount;

		step = DEFAULT_STEP;
		istaStep = DEFAULT_ISTA_STEP;
		noiseVariance = DEFAULT_NOISE_VARIANCE;
		tolerance = DEFAULT_TOLERANCE;

		//Intialize the dictionary to random values between -0.5 and 0.5.
		std::mt19937 randGen(time(nullptr));
		std::uniform_real_distribution<double> dist(-0.5, 0.5);

		dictionary = cv::Mat(patchSize * patchSize, basisFunctionCount, CV_64FC1);

		for (unsigned int i = 0; i < dictionary.rows; i++)
		{
			for (unsigned int j = 0; j < dictionary.cols; j++)
				dictionary.at<double>(i, j) = dist(randGen);
		}

		//Make the basis functions of unit norm.
		for (unsigned int i = 0; i < basisFunctionCount; i++)
			dictionary.col(i) /= cv::norm(dictionary.col(i));
	}

	void DictionaryLearner::SetParameters(double _step, double _istaStep, double _noiseVariance)
	{
		step = _step;
		istaStep = _istaStep;
		noiseVariance = _noiseVariance;
	}

	cv::Mat DictionaryLearner::Train(unsigned int iterations)
	{
		//Random distribution for the images.
		std::mt19937 randGen(time(nullptr));
		std::uniform_int_distribution<unsigned int> imageDist(0, images.size() - 1);

		cv::Mat visualization;

		for (unsigned int i = 0; i < iterations; i++)
		{
			unsigned int imageIndex = imageDist(randGen);
			cv::Mat currentImage = images[imageIndex];

			//Choose random patches from a randomly selected image.
			cv::Mat patches = ChooseRandomPatches(currentImage);

			//Infer the sparse coefficients for every image patch.
			cv::Mat coefficients = ISTA(patches, dictionary, istaStep, noiseVariance, tolerance);

			//Compute the reconstruction error for the dictionary update rule.
			cv::Mat reconstError = patches - dictionary * coefficients;

			//The dictionary is updated with a scalar multiple of it's average gradient.
			cv::Mat averageDictionaryGradient = cv::Mat::zeros(dictionary.size(), dictionary.type());

			//Compute the average gradient.
			for (unsigned int j = 0; j < patches.cols; j++)
				averageDictionaryGradient += reconstError.col(j) * coefficients.col(j).t();

			averageDictionaryGradient /= static_cast<double>(patches.cols);

			//Take one step in the direction of the average gradient.
			dictionary += step * averageDictionaryGradient;

			//Normalize the basis functions to keep them from growing unbounded.
			for (unsigned int j = 0; j < dictionary.cols; j++)
				dictionary.col(j) /= cv::norm(dictionary.col(j));

			//How long for the opencv imshow window to wait.
			//Small values seem to not let some of the first images display.
			int waitTime = !(i == (iterations - 1)) * DISPLAY_DELAY;

			visualization = Visualize(dictionary, waitTime);
		}

		return visualization.clone();
	}

	//Add the L1 penalty to the sparse codes or clamp them to zero.
	void DictionaryLearner::Shrink(cv::Mat sparseCodes, double step, double noiseVariance)
	{
		double shrinkStep = step * noiseVariance;

		for (unsigned int i = 0; i < sparseCodes.rows; i++)
		{
			int codeSign = Sign(sparseCodes.at<double>(i, 0));

			//Result of applying the L1 penalty.
			double result = sparseCodes.at<double>(i, 0) - (double)(shrinkStep * codeSign);

			//Clamp to 0 if the sign of the code changed.
			if (std::abs(result) < noiseVariance)
				sparseCodes.at<double>(i, 0) = 0.0;
			else
				sparseCodes.at<double>(i, 0) = result;
		}
	}

	int DictionaryLearner::Sign(double d)
	{
		if (d >= 0.0)
			return 1;
		else
			return -1;
	}


	cv::Mat DictionaryLearner::ISTA(cv::Mat samples, cv::Mat basisFunctions, double step, double noiseVariance, double tolerance)
	{
		//Initialize sparse codes to zero.
		cv::Mat sparseCodes = cv::Mat::zeros(basisFunctions.cols, samples.cols, CV_64FC1);

		for (unsigned int i = 0; i < samples.cols; i++)
		{
			while (true)
			{
				//Magnitude used for checking for convergence.
				double originalMagnitude = cv::norm(sparseCodes.col(i));

				//Compute the reconstruction error of the input.
				cv::Mat reconstError = samples.col(i) - basisFunctions * sparseCodes.col(i);

				sparseCodes.col(i) += step * basisFunctions.t() * reconstError;

				//Add the L1 penalty or clamp to zero if necessary.
				Shrink(sparseCodes.col(i), step, noiseVariance);

				double newMagnitude = cv::norm(sparseCodes.col(i));

				double magnitudeChange = newMagnitude - originalMagnitude;

				if (std::abs(magnitudeChange) < tolerance)
					break;
			}
		}

		return sparseCodes;
	}

	cv::Mat DictionaryLearner::ChooseRandomPatches(cv::Mat image)
	{
		std::mt19937 randGen(time(nullptr));
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		cv::Mat patches = cv::Mat(dictionary.cols, sampleCount, CV_64FC1);

		unsigned int lastRow = image.rows - patchSize;
		unsigned int lastCol = image.cols - patchSize;

		for (unsigned int j = 0; j < sampleCount; j++)
		{
			unsigned int row = static_cast<unsigned int>(ceil(dist(randGen) * lastRow));
			unsigned int col = static_cast<unsigned int>(ceil(dist(randGen) * lastCol));

			cv::Mat subMatrix = image(cv::Rect(row, col, patchSize, patchSize));
			subMatrix = subMatrix.clone().reshape(1, dictionary.rows);

			subMatrix.copyTo(patches.col(j));
		}

		return patches;
	}

	cv::Mat DictionaryLearner::Visualize(cv::Mat& basisFunctions, int wait)
	{
		unsigned int functionsPerDim = std::sqrt(basisFunctions.cols);
		unsigned int imageDim = BORDER + (BORDER * patchSize) + (patchSize * functionsPerDim);

		cv::Mat image = cv::Mat::zeros(imageDim, imageDim, CV_64FC1);

		for (unsigned int j = 0; j < basisFunctions.cols; j++)
		{
			cv::Mat reshaped = basisFunctions.col(j).clone().reshape(1, patchSize);

			double minValue = 0.0;

			for (unsigned int a = 0; a < reshaped.rows; a++)
			{
				for (unsigned int b = 0; b < reshaped.cols; b++)
				{
					double currentVal = reshaped.at<double>(a, b);

					if (minValue > currentVal)
						minValue = currentVal;
				}
			}

			reshaped -= minValue;

			double maxValue = 0.0;

			for (unsigned int a = 0; a < reshaped.rows; a++)
			{
				for (unsigned int b = 0; b < reshaped.cols; b++)
				{
					double currentVal = reshaped.at<double>(a, b);

					if (maxValue < currentVal)
						maxValue = currentVal;
				}
			}

			reshaped /= maxValue;

			//OpenCV reshapes mats differently than octave;
			//keep consistency with it for now.
			reshaped = reshaped.t();

			unsigned int row = BORDER + ((j % functionsPerDim) * (patchSize + BORDER));
			unsigned int col = BORDER + ((j / functionsPerDim) * (patchSize + BORDER));
			reshaped.copyTo(image(cv::Rect(row, col, patchSize, patchSize)));
		}

		cv::resize(image, image, cv::Size(350, 350), 0.0, 0.0, cv::INTER_NEAREST);

		cv::imshow("basis functions", image);

		cv::waitKey(wait);

		return image;
	}
}
