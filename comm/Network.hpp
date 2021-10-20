/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Network.hpp
 * Author: wangqg
 *
 * Created on November 18, 2019, 9:24 AM
 */

#ifndef NETWORK_HPP
#define NETWORK_HPP
#include "core/type.hpp"
template <typename t_v>
class Network
{
public:
    t_v *recv_buffer;
    t_v *buffer;
    int worknum;
    int workid = -1;
    int weight_row = 0;
    int weight_col = 0;

    Network(int partitions, int partition_id, int weight_row_, int weight_col_)
    {
        // std::cout<<"constract network"<<std::endl;
        worknum = partitions;
        workid = partition_id;
        weight_row = weight_row_;
        weight_col = weight_col_;
        if (workid == 0)
        {
            recv_buffer = new t_v[worknum * weight_row * weight_col];
        }
        buffer = new t_v[weight_row * weight_col];
    }
    void setWsize(int weight_row_, int weight_col_)
    {
        weight_row = weight_row_;
        weight_col = weight_col_;
        realloc(buffer, weight_row * weight_col * sizeof(t_v));
    }
    //    void wrtWtoBuff(weightVector *w) {
    //        memcpy(buffer, w->weight, sizeof (t_v)*weight_row * weight_col);
    //       // memcpy(buffer, weight, sizeof (float)*WEIGHT_ROW * WEIGHT_COL);
    //    }
    void wrtBuffertoBuff(ValueType *buffer_)
    {
        memcpy(buffer, buffer_, sizeof(t_v) * weight_row * weight_col);
        // memcpy(buffer, weight, sizeof (float)*WEIGHT_ROW * WEIGHT_COL);
    }

    void wrtBuffertoBuff(ValueType *buffer_, int row, int col)
    {
        memcpy(buffer, buffer_, sizeof(t_v) * row * col);
        // memcpy(buffer, weight, sizeof (float)*WEIGHT_ROW * WEIGHT_COL);
    }

    void gatherW()
    {

        if (workid == 0)
        {
            //接收数组
            memset(recv_buffer, 0, sizeof(t_v) * (worknum)*weight_row * weight_col);
            memcpy(recv_buffer, buffer, sizeof(t_v) * weight_row * weight_col);

            for (int i = 1; i < (worknum); i++)
            {
                MPI_Recv(recv_buffer + i * weight_row * weight_col, weight_row * weight_col * sizeof(t_v), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        else
        {
            //发送数组
            MPI_Send(buffer, weight_row * weight_col * sizeof(t_v), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    void gatherW(int row, int col)
    {
        //std::cout<<"network"<<buffer[0]<<" "<<workid<<" "<<worknum<<std::endl;
        if (workid == 0)
        {
            //接收数组
            memset(recv_buffer, 0, sizeof(t_v) * (worknum)*row * col);
            memcpy(recv_buffer, buffer, sizeof(t_v) * row * col);

            for (int i = 1; i < (worknum); i++)
            {
                MPI_Recv(recv_buffer + i * row * col, row * col * sizeof(t_v), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        else
        {
            //发送数组
            MPI_Send(buffer, row * col * sizeof(t_v), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
        //MPI_Barrier(MPI_COMM_WORLD);
    }

    void computeW()
    {
        if (workid == 0)
        {
            for (int i = 1; i < worknum; i++)
            {
                for (int j = 0; j < weight_row * weight_col; j++)
                {
                    recv_buffer[j] = recv_buffer[j] + recv_buffer[j + i * weight_row * weight_col];
                }
            }
            //  printf("\n");
            //            for (int i = 0; i < weight_row; i++) {
            //                for (int j = 0; j < weight_col; j++) {
            //                    recv_buffer[weight_col * i + j] /= worknum;
            //         //           printf("%f\t", recv_buffer[4 * i + j]);
            //                }
            //           //     printf("\n");
            //            }
        }
    }
    void computeW(int row, int col)
    {
        if (workid == 0)
        {
            for (int i = 1; i < worknum; i++)
            {
                for (int j = 0; j < row * col; j++)
                {
                    recv_buffer[j] = recv_buffer[j] + recv_buffer[j + i * row * col];
                }
            }
        }
    }

    void broadcastW(t_v *buffer_)
    {

        if (workid == 0)
        {
            memcpy(buffer, buffer_, sizeof(t_v) * weight_row * weight_col);
            for (int i = 1; i < (worknum); i++)
            {
                MPI_Send(buffer, weight_row * weight_col * sizeof(t_v), MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
            MPI_Recv(buffer, weight_row * weight_col * sizeof(t_v), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void broadcastW()
    {
        //  memcpy(buffer, recv_buffer, sizeof (float)*WEIGHT_ROW * WEIGHT_COL);
        if (workid == 0)
        {
            for (int i = 1; i < (worknum); i++)
            {
                MPI_Send(buffer, weight_row * weight_col * sizeof(t_v), MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
            MPI_Recv(buffer, weight_row * weight_col * sizeof(t_v), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //MPI_Barrier(MPI_COMM_WORLD);
    }
    void broadcastW(t_v *buffer_, int row, int col)
    {

        //  memcpy(buffer, recv_buffer, sizeof (float)*WEIGHT_ROW * WEIGHT_COL);
        if (workid == 0)
        {
            memcpy(buffer, buffer_, sizeof(t_v) * row * col);
            for (int i = 1; i < (worknum); i++)
            {
                MPI_Send(buffer, row * col * sizeof(t_v), MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
            MPI_Recv(buffer, row * col * sizeof(t_v), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //MPI_Barrier(MPI_COMM_WORLD);
    }
    void all_reduce_sum(t_v *buffer, int row, int col)
    {
        MPI_Datatype f_vid_t = get_mpi_data_type<t_v>();
        MPI_Allreduce(MPI_IN_PLACE, buffer, sizeof(f_vid_t) * row * col, f_vid_t, MPI_SUM, MPI_COMM_WORLD);
    }
};

template <typename t_v>
class Network_simple
{
public:
    t_v *recv_buffer;
    t_v *buffer;
    int worknum;
    int workid = -1;
    int weight_row = 0;
    int weight_col = 0;

    Network_simple(int weight_row_, int weight_col_)
    {
        weight_row = weight_row_;
        weight_col = weight_col_;
    }

    void all_reduce_sum(t_v *buffer)
    {
        MPI_Datatype f_vid_t = get_mpi_data_type<float>();
        MPI_Allreduce(MPI_IN_PLACE, buffer, weight_row * weight_col, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        //printf("%d sd%f\n", weight_row * weight_col, buffer[3]);
    }
    void broadcast(t_v *buffer)
    {
        MPI_Datatype f_vid_t = get_mpi_data_type<t_v>();
        MPI_Bcast(buffer, weight_row * weight_col, f_vid_t, 0, MPI_COMM_WORLD);
    }
};

#endif /* NETWORK_HPP */
