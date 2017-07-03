#include <string>
#include <sstream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "DictionaryLearner.hpp"

namespace SparseCoding
{
	class App
	{
		public:
			App();
			int Execute();
		private:
			const static int EXIT_S = 0;
			const static int EXIT_F = 1;
			const static unsigned int IMAGE_COUNT = 10;
			const static unsigned int BASIS_FUNCTION_COUNT = 64;
			const static unsigned int SAMPLE_COUNT = 100;
			const static unsigned int PATCH_SIZE = 8;
			const static unsigned int IMAGE_ROWS = 512;
			const static unsigned int IMAGE_COLS = 512;
			const static unsigned int ITERATION_COUNT = 2000;
			const static std::string DATA_FOLDER;
			const static std::string IMAGE_PREFIX;
			const static std::string OUTPUT_IMAGE_FILE;
			const static std::string EXT_CSV;
			cv::Mat LoadFromCSV(std::string fileName, unsigned int rows, unsigned int cols);
	};
}
