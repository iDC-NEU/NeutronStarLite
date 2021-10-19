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