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

	// displaying hex in C++
	cout << "hex of 255 = " << hex << 255 << endl; // no need to call hex(255)

	// scientific notation
	float c_mps = 3E8F;  // 10^8
	cout << "Speed of light -> " << c_mps << endl;

	// strings are always \0 terminated, so actual length to store is strlen() + 1

	// variable init - both below are ok. All global vars are default init to 0, but locals have undefined value
	float x(3.14);
	unsigned long int d = 5UL;
	cout << x << endl << d << endl;

	// constants are readonly variables - must be initialized at creation and cannot be modified later
	const float PI = 3.14f;
	cout << "PI = " << PI << endl;

	// volatiles variables can be modified later by the program and from other programs and sources, e.g., clocks, interrupts etc
	volatile int regx = 0xFA;
	cout << "regx = 0x" << hex << regx << endl;

	// const valoatile vars cannot be changed by the program but by other sources
	volatile const int regy = 0xAA;
	cout << "regy = 0x" << hex << regy << endl; // NOTE: the hex form stays in effect till changed
	cout << dec << endl;

	chapter3();

	return 0;
}
