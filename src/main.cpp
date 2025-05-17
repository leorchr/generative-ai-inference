#include <iostream>
#include <ostream>

#include "M2M100Manager.h"
#include "T5Manager.h"

int main(int argc, char** argv)
{
	std::cout << "Enter 'stop' to leave the program." <<std::endl;

	ModelManager* manager = new T5Manager();
	if (!manager->Load())
	{
		return 1;
	};

	std::string inputText = "Hello!";
	std::cout << "Input Text : " << inputText << std::endl;
	manager->Run(inputText);
	std::cout << std::endl;

	while (true)
	{
		std::string input {};
		std::cout << "Input Text : ";
		std::getline(std::cin, input);
		std::cout << std::endl;
		if (input == "stop") break;
		manager->Run(input);
	}
	return 0;
}