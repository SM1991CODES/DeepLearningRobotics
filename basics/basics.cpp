// basics.cpp : Defines the entry point for the application.
//

#include "basics.h"

#define BOOK_NAME	"A Complete Guide to Programming in C++"

using namespace std;

int main()
{
	cout << "Welcome to learning C++" << endl;
	std::cout << "Book name -> " << BOOK_NAME << std::endl;
	//
	cout << "sizeof bool -> " << sizeof(bool) << endl;
	//
	cout << "size of int -> min : [" << INT_MIN << "], max [ " << INT_MAX << " ]" << endl;
	//
	// characters are represented by ascii numbers - each char is 1 byte
	uint8_t A = 'A';
	printf("A = [%d] / [%c]\n", A, A);
	//
	for (int i = 0; i < 255; i++)
	{
		if ((A + 1) < 255)
		{
			A = A + 1;
			printf("%d / %c\n", A, A); // prints ascii and number for each character
		}
			
	}
	//
	uint8_t msg[] = "Hello World!"; // this is possible since internally only ascii numbers are stored
	char msg2[] = "Hello World2!"; // this is more intuitive and conventional
	cout << msg << endl;
	cout << sizeof(msg) << endl;
	cout << strlen((const char*)msg) << endl;

	return 0;
}
