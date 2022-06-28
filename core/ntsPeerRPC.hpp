//
// Created by toao on 2022/6/20.
//

#ifndef GNNMINI_NTSPEERRPC_HPP
#define GNNMINI_NTSPEERRPC_HPP
/*
 * @Date: 2022-06-03 16:40:17
 * @LastEditors: Toao
 * @LastEditTime: 2022-06-21 20:57:31
 * @FilePath: /peerRPC/PeerRPC.hpp
 * @Description: Toao's code file
 */
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <mpi.h>
#include <functional>
#include <cassert>
#include <string>
#include <cstring>
#include <tuple>
#include <atomic>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <limits>





template<typename DataType, typename ArgType>
class ntsPeerRPC
{
private:
    struct RPCMessage {
        uint32_t host;
        uint32_t tag;
        std::vector<uint8_t> data;
        MPI_Request req;
        // mpiMessage(message&& _m, MPI_Request _req) : m(std::move(_m)), req(_req)
        // {}
        RPCMessage(uint32_t host, uint32_t tag, std::vector<uint8_t>&& data)
                : host(host), tag(tag), data(std::move(data)) {}
        RPCMessage(uint32_t host, uint32_t tag, size_t len)
                : host(host), tag(tag), data(len) {}
    };
    using vector_function = std::function<std::vector<std::vector<DataType>>(std::vector<ArgType>)>;
    std::unordered_map<uint64_t, vector_function> function_map;
    const int RPC_CALL = 33;
    const int RPC_RESP = 34;
    const int RPC_STOP = 35;
    int host_num;
    int host_id;
    int comm_num;
    bool is_automatic;
    bool old_is_automatic;
    std::deque<RPCMessage> flight_messages;
    std::thread server_thread;
    enum ServerState{Running, Sleeping, Exit} ;
    std::atomic<ServerState> server_state;
    std::mutex mutex;
    // std::shared_mutex shared_mutex;
    std::condition_variable stop_condition;


    void initMPI()
    {
        int supportProvided;
        int initSuccess =
                MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &supportProvided);
        if (initSuccess != MPI_SUCCESS)
        {
            MPI_Abort(MPI_COMM_WORLD, initSuccess);
        }

        assert(supportProvided == MPI_THREAD_MULTIPLE);
        // if (supportProvided != MPI_THREAD_MULTIPLE)
        // {
        //     printf("MPI_THREAD_MULTIPLE not supported.\n");
        //     exit(3);
        // }
    }

    std::vector<uint8_t> serialize(uint64_t& function_id, std::vector<ArgType>&function_args) {
        size_t data_size = sizeof(uint64_t) + sizeof(ArgType) * function_args.size();
        std::vector<uint8_t> data;
        data.resize(data_size);

        uint8_t* data_ptr = data.data();
        memcpy((void*)data_ptr, &function_id, sizeof(uint64_t));
        memcpy((void*)(data_ptr + sizeof(uint64_t)), (void*)function_args.data(), sizeof(ArgType) * function_args.size());

        return data;
    }

    std::vector<uint8_t> serialize_response(std::vector<std::vector<DataType>>& arrs) {
        std::vector<uint8_t> serialize_data;
        size_t size = 0;
        if(arrs.size() != 0) {
            size = arrs[0].size() * arrs.size();
        }
        serialize_data.resize(size * sizeof(DataType));

        uint8_t* data_ptr = serialize_data.data();
        size_t pos = 0;
        size_t arr_size = 0;
        if(arrs.size() != 0) {
            arr_size = arrs[0].size() * sizeof(DataType);
        }
        for(auto& arr : arrs) {
            memcpy(data_ptr + pos, arr.data(), arr_size);
            pos += arr_size;
        }
        assert(pos == size * sizeof(DataType));
        return serialize_data;

    }

    std::vector<std::vector<DataType>> deserialize_response(std::vector<uint8_t>& serialize_data, int row){
        std::vector<std::vector<DataType>> deserialize_data;
        if(serialize_data.size() == 0) {
            deserialize_data.resize(0);
            return deserialize_data;
        }
        int pos = 0;
        int row_size = row * sizeof(DataType);
        int col = serialize_data.size() / row_size;
        int col_size = col * sizeof(DataType);
        uint8_t* data_ptr = serialize_data.data();
        for(int i = 0; i < row; i++) {
            std::vector<DataType> arr((DataType*)data_ptr, (DataType*)(data_ptr + col_size));
            deserialize_data.emplace_back(std::move(arr));
            data_ptr += col_size;
        }
        return deserialize_data;
    }

    std::tuple<uint64_t, std::vector<ArgType>> deserialize(std::vector<uint8_t>& serialize_data) {
        uint64_t function_id;
        std::vector<ArgType>function_args;
        auto args_size = serialize_data.size() - sizeof(uint64_t);
        auto data_ptr = serialize_data.data();

        function_args.resize(args_size/(sizeof(ArgType)/sizeof(uint8_t)));
        function_id = *((uint64_t*)data_ptr);
        memcpy(function_args.data(), (data_ptr + sizeof(uint64_t)), args_size);

        return std::make_tuple(function_id, function_args);
    }

    void rpc_process() {
        // 定义一些传输过程中用到的变量
        char* no_message = "";
        int count;
        int flag;
        int* indices = new int[host_num];
        int index_count;
        int total_send = 0;
        int had_sent = 0;

        std::unique_lock<std::mutex> lock(mutex, std::defer_lock);

        // 将线程设为运行态
        // server_state = Running;
        // server_state.store(Running);
        while(!(server_state == Exit && flight_messages.size() == 0)) {
            // 检测是否有消息到来并查看之前的消息是否完成
            MPI_Status status;
            flag = 0;
            had_sent += complete();
            if(!is_automatic) {
                had_sent = 0;
            }
            // MPI_Improbe(MPI_ANY_SOURCE, RPC_CALL, MPI_COMM_WORLD, &flag, )
            lock.lock();
            if((server_state.load() == Sleeping && flight_messages.size() == 0) || (is_automatic && had_sent == comm_num)) {
//                std::printf("  process %d 进入了睡眠, server_state: %d, is_auto: %d\n", this->host_id,
//                            server_state.load(), is_automatic);
                stop_condition.wait(lock);

                had_sent = 0;
                // 唤醒后如果是退出了则打破循环
                if(server_state.load() == Exit) {
                    lock.unlock();
                    break;
                }
            }
            lock.unlock();
            if(server_state == Exit && flight_messages.size() == 0) {
                break;
            }

            // MPI_Iprobe(MPI_ANY_SOURCE, RPC_CALL, MPI_COMM_WORLD, &flag, &status);
            probe(flag, status);
            if(!flag){
                std::this_thread::yield();
                continue;
            }
            MPI_Get_count(&status, MPI_UINT8_T, &count);

            // 接收到来的消息
            int host_id = status.MPI_SOURCE;
            std::vector<uint8_t> data;
            data.resize(count);
            // MPI_Recv(data.data(), count, MPI_UINT8_T, host_id, RPC_CALL, MPI_COMM_WORLD, &status);
            MPI_Recv(data.data(), count, MPI_UINT8_T, host_id, RPC_CALL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // 将消息进行反序列化提取
            uint64_t function_id;
            std::vector<ArgType> function_args;
            std::tie(function_id, function_args) = deserialize(data);

            // 检查接收参数
//            if(function_args.size() != 0) {
//                std::printf("  %d 收到 %d，第一个参数为: %u, 最后一个为：%u\n", this->host_id, host_id, function_args.front(), function_args.back());
//
//            }

            // 执行对应函数并将消息进行序列化
            auto result = function_map[function_id](function_args);
            if(result.size() == 0){
                count = 0;
            } else {
                count = result.size() * result[0].size();
            }

//            // 测试函数执行结果
//            for(int i = 0; i < result.size(); i++) {
//                float sum = 0;
//                for(int j = 0; j < result[i].size(); j++) {
//                    sum += result[i][j];
//                }
//                std::printf("%u sum %f\n", function_args[i], sum);
//            }

            flight_messages.emplace_back(host_id, RPC_RESP, serialize_response(result));

            // 发送序列化之后的消息
            auto& resp_data = flight_messages.back();
            if(count == 0) {
                MPI_Issend(resp_data.data.data(), 0, MPI_UINT8_T, host_id, RPC_RESP, MPI_COMM_WORLD, &resp_data.req);
            } else {
                MPI_Issend(resp_data.data.data(), resp_data.data.size(), MPI_UINT8_T, host_id, RPC_RESP, MPI_COMM_WORLD, &resp_data.req);
            }

        }
        delete indices;
    }

    void probe(int& flag, MPI_Status& status) {
        MPI_Iprobe(MPI_ANY_SOURCE, RPC_CALL, MPI_COMM_WORLD, &flag, &status);
    }

    int complete() {
        int count = 0;
        while(!flight_messages.empty()) {
            int flag = 0;
            MPI_Status status;
            auto& f = flight_messages.front();
            int rv = MPI_Test(&f.req, &flag, &status);

            if(flag) {
                count++;
                flight_messages.pop_front();
            } else {
                break;
            }
        }
        return count;
    }

    void init() {
        int flag = 0;
        MPI_Initialized(&flag);

        if(!flag) {
             initMPI();
//            int provide;
//            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provide);
        }
        MPI_Comm_size(MPI_COMM_WORLD, &host_num);
        MPI_Comm_rank(MPI_COMM_WORLD, &host_id);

        // server_state = Sleeping;
        server_thread = std::thread(&ntsPeerRPC::rpc_process, this);
        old_is_automatic = is_automatic;
    }


public:
    ntsPeerRPC() {
        server_state.store(Sleeping);
        is_automatic = false;
        comm_num = std::numeric_limits<int>::max();
        init();
    }

    ntsPeerRPC(int comm_num) {
        this->comm_num = comm_num;
        is_automatic = true;
        server_state.store(Running);
        init();
    }
    void set_comm_num(int comm_num) {
        this->comm_num = comm_num;
        is_automatic = true;
        old_is_automatic = is_automatic;
    }

    void register_function(std::string function_name, vector_function function ) {
        uint64_t function_id = std::hash<std::string>{}(function_name);
        function_map.insert({function_id, function});
    }

    std::vector<std::vector<DataType>> call_function(std::string function_name, std::vector<ArgType>&function_args, int host_id) {
        if(host_id == this->host_id) {
            uint64_t function_id = std::hash<std::string>{}(function_name);
            assert(function_map.find(function_id) != function_map.end());
            return function_map[function_id](function_args);
        }
//        if(function_args.size() != 0) {
//            std::printf("  %d 发给 %d，第一个参数为: %u, 最后一个为：%u\n", this->host_id, host_id, function_args.front(), function_args.back());
//
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
        server_state.store(Running);
        stop_condition.notify_all();

        uint64_t function_id = std::hash<std::string>{}(function_name);
        assert(function_map.find(function_id) != function_map.end());
        auto data = serialize(function_id, function_args);
        MPI_Send((void*)data.data(), data.size(), MPI_UINT8_T, host_id, RPC_CALL, MPI_COMM_WORLD);

        MPI_Status status;
        int count;
        MPI_Probe(host_id, RPC_RESP, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_UINT8_T, &count);

        std::vector<uint8_t> response_data;
        response_data.resize(count);
        MPI_Recv(response_data.data(), count, MPI_UINT8_T, host_id, RPC_RESP, MPI_COMM_WORLD, &status);

        auto result = deserialize_response(response_data, function_args.size());

//        // 测试函数执行结果
//        for(int i = 0; i < result.size(); i++) {
//            float sum = 0;
//            for(int j = 0; j < result[i].size(); j++) {
//                sum += result[i][j];
//            }
//            std::printf("%u sum %f\n", function_args[i], sum);
//        }


        return result;

    }

    void start() {
        if(is_automatic) {
            return;
        }
        // this lock is used for synchronization, not for atomicity
        std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
        lock.lock();
        // server_state = Running;
        server_state.store(Running);
        lock.unlock();
        stop_condition.notify_all();
    }

    void keep_running() {
        std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
        lock.lock();
        server_state.store(Running);
        is_automatic = false;
        lock.unlock();
        stop_condition.notify_all();
    }

    void stop_running() {
        MPI_Barrier(MPI_COMM_WORLD);
        std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
        lock.lock();
        server_state.store(Sleeping);
        is_automatic = old_is_automatic;
        lock.unlock();
    }

    void finish(){
        if(is_automatic){
            return;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
        lock.lock();
        // server_state = Sleeping;
        server_state.store(Sleeping);
        lock.unlock();
    }

    void exit() {
        MPI_Barrier(MPI_COMM_WORLD);
        std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
        lock.lock();
        // server_state = Exit;
        server_state.store(Exit);
        lock.unlock();
        stop_condition.notify_all();
    }

    ~ntsPeerRPC() {
        // server_state = Exit;
        std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
        lock.lock();
        server_state.store(Exit);
        lock.unlock();
        stop_condition.notify_all();
        server_thread.join();
    }
};


#endif //GNNMINI_NTSPEERRPC_HPP
