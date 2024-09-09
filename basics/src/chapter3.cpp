#include "chapter3.h"

#include <iostream>
#include <cstdio>
#include <cstdio>
#include <cstring>

using namespace std;

void chapter3()
{
    // string operations
    string name("Sambit"), surname;
    cout << "Enter your surname:" << endl;
    cin >> surname;
    cout << "Hello " << name + '\t' + surname << endl; // concat with +

    float c = 3E8f;  // scientifc notation
    float inv_C = 3e-8F;
    cout << scientific << c << endl << inv_C << endl; // this enables printing in scientific notation, to reset, use fixed

    cout << fixed << endl;
    cout.precision(3); // set 3 places to display after decimal for floats
    cout << 3.1235343 << endl;

    string sentence;
    cout << "Enter a sentence or paragraph. Type < to terminate" << endl;
    getline(cin, sentence, '<'); // reads a stream of characters including \n from console and writes into sentence. Stops reading when < is found

    cout << sentence << endl;

    // ------- looops ---------//
    int k = 0;
    for(; k < 10;) // this is same as while(k < 10)
    {
        cout << "k = " << k << endl;
        k += 1;

    }

}