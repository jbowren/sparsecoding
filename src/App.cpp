#include "App.hpp"

namespace SparseCoding
{
	const std::string App::DATA_FOLDER = "images/";
	const std::string App::IMAGE_PREFIX = "image";
	const std::string App::OUTPUT_IMAGE_FILE = "basisfunctions.png";
	const std::string App::EXT_CSV = ".csv";

	App::App()
	{

	}

	int App::Execute()
	{
		std::vector<cv::Mat> images;

		for (unsigned int i = 0; i < IMAGE_COUNT; i++)
		{
			std::ostringstream fileNameStream;
			fileNameStream << DATA_FOLDER << IMAGE_PREFIX << i << EXT_CSV;

			images.push_back(LoadFromCSV(fileNameStream.str(), IMAGE_ROWS, IMAGE_COLS));
		}

		DictionaryLearner sparseCoding(images, BASIS_FUNCTION_COUNT, PATCH_SIZE, SAMPLE_COUNT);
		cv::Mat dictionary = sparseCoding.Train(ITERATION_COUNT);

		dictionary.convertTo(dictionary, CV_8UC1, 255.0);
		cv::imwrite(OUTPUT_IMAGE_FILE, dictionary);

		return EXIT_S;
	}

	cv::Mat App::LoadFromCSV(std::string fileName, unsigned int rows, unsigned int cols)
	{
		cv::Mat matrix = cv::Mat(rows, cols, CV_64FC1);

		std::ifstream inputFile;
		inputFile.open(fileName);

		std::string line;

		unsigned int i = 0;

		while (std::getline(inputFile, line))
		{
			std::istringstream lineStream;
			lineStream.str(line);

			std::string dataString;

			unsigned int j = 0;

			while (std::getline(lineStream, dataString, ','))
			{
				matrix.at<double>(j, i) = std::stod(dataString);

				j++;
			}

			i++;
		}

		return matrix;
	}
}
