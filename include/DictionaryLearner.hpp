#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace SparseCoding
{
	class DictionaryLearner
	{
		public:
			DictionaryLearner(std::vector<cv::Mat>& input, double basisFunctionCount, unsigned int patchLength, unsigned int batchCount);
			void SetParameters(double _step, double _istaStep, double _noiseVariance);
			cv::Mat Train(unsigned int iterations);
		private:
			const static unsigned int DISPLAY_DELAY = 100;
			const static unsigned int BORDER = 1;
			const static double DEFAULT_STEP;
			const static double DEFAULT_ISTA_STEP;
			const static double DEFAULT_NOISE_VARIANCE;
			const static double DEFAULT_TOLERANCE;
			unsigned int patchSize;
			unsigned int sampleCount;
			double step;
			double istaStep;
			double noiseVariance;
			double tolerance;
			cv::Mat dictionary;
			std::vector<cv::Mat> images;
			void Shrink(cv::Mat sparseCodes, double step, double noiseVariance);
			int Sign(double d);
			cv::Mat ISTA(cv::Mat samples, cv::Mat basisFunctions, double step, double noiseVariance, double tolerance);
			cv::Mat ChooseRandomPatches(cv::Mat image);
			cv::Mat Visualize(cv::Mat& basisFunctions, int wait);
	};
}
