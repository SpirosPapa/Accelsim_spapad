#pragma once
#include <atomic>
#include <string>
#include <functional>
#include <thread>




class IpcServer {
    public:
        using Handler = std::function<std::string(const std::string &)>;
        explicit IpcServer(const std::string& socket_p);
        ~IpcServer();
        bool start(Handler h);
        void stop();
    private:
        void run_accept_loop();
        std::string socket_path;
        int listen_fd =-1;
        std::atomic<bool> running{false};
        std::thread th;
        Handler handler;
};