#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

using namespace std;

typedef int ID;
typedef int VALUE;
typedef float WEIGHT;
typedef unsigned long long offset_t;

const offset_t BUFF_SIZE = 0x4000; //16K

enum Status{
    OK = 1,
    ERROR = 0
};

int floattoint(void* f_a){
 int i_a = *((int *)f_a);
 return i_a;
}

class OutFile 
{
protected:
    int fos;
    offset_t wp;
    char buffer[BUFF_SIZE];
public:
    string dir, name, dir_name;

    OutFile(string name, string dir):
        dir(dir), name(name), dir_name(dir + "/" + name),wp(0){
        fos = open((dir_name).c_str(), 
                O_WRONLY | O_CREAT | O_TRUNC, 
                S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    }

    Status write(const char* buff, size_t len)
    {
        //Fill the buffer
        if(wp + len > BUFF_SIZE)
        {
            offset_t remain = BUFF_SIZE - wp;
            memcpy(buffer + wp, buff, remain);
            ::write(fos, buffer, BUFF_SIZE);
            wp = 0;
            len -= remain;
            buff += remain;

            //write big chunks if any   
            if(len > BUFF_SIZE)
            {
                remain = len - (len / BUFF_SIZE) * BUFF_SIZE;
                ::write(fos, buff, len - remain);
                buff += (len - remain);
                len = remain;
            }
        }
        memcpy(buffer, buff, len);
        wp += len;
        return OK;
    }

    Status flush()
    {
        if(::write(fos, buffer, wp) == -1)
            return ERROR;
        wp = 0;
        return OK;
    }

    Status close()
    {
        Status s = flush();
        if(s != OK) return s;
        if(::close(fos) == -1)
        {
            cout << "close OutFile " << dir_name << " failed." << endl;
            return ERROR;
        }
        return OK;
    }

    template<class T>
    Status write_unit(T unit)
    {
   
        for(int i = 0; i < sizeof(T); i++)
        {
            if(wp >= BUFF_SIZE)
            {   
                ::write(fos, buffer, wp);
                wp = 0;
            }
            char c = static_cast<unsigned char> (unit & 0xff);
            buffer[wp++] = c;
            unit >>= 8;
        }
        return OK;
    } 

};


void convert(char *fname, char *to_fname)
{
    FILE *f = fopen(fname, "r");
    OutFile binary_file(string(to_fname), string("."));

    //Start read dataset
    cout << "Start read dataset " << endl;
    ID from;
    long long maxlen = 10000000;
    char *s = (char*)malloc(maxlen);
    char delims[] = " \t\n";

    long long edge_cnt = 0;
    ID max_id = 0;

    while (fgets(s, maxlen, f) != NULL)
    {
        if (s[0] == '#' || s[0] == '%' || s[0] == '\n')
            continue; //Comment

        char *t = strtok(s, delims);
        from = atoi(t);
        max_id = max(max_id, from);
        binary_file.write_unit(from);

        ID to;
        if ((t = strtok(NULL, delims)) != NULL)
        {
            to = atoi(t);
            max_id = max(max_id, to);

            binary_file.write_unit(to);   
        }
        VALUE cost;
        WEIGHT w;

        if ((t = strtok(NULL, delims)) != NULL)
        {
            w = atof(t);
            cost = floattoint(&w);

            binary_file.write_unit(cost);
            edge_cnt++;    
        }
        /*
        if (from % 100000 == 0)
            cout << from << endl;
        //*/
        ///*
        //if (edge_cnt % 1000000 == 0)
        //cout << "Read edge num: " << edge_cnt << endl;
        
    }

    binary_file.close();

    fclose(f);
    free(s);
    cout << "Edge number: " << edge_cnt << endl;    
    cout << "End read dataset max id: " << max_id <<endl;
    cout << "vertices num: "<< max_id + 1 << endl;
}


int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Usage ./convert2binary <dataset.txt> <binary edgelist file name>" << endl;
        return 0;
    }

    convert(argv[1], argv[2]);

    return 0;
}
