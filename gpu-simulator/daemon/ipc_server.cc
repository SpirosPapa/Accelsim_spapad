#include "ipc_server.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <csignal>
#include <sys/stat.h>



#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>


namespace{
    std::string read_line(int fd){
        std::string out;
        char ch;
        while (true){
            ssize_t n= ::read(fd, &ch , 1);
            if (n==0) return out;
            if (n<0){
                if (errno==EINTR) continue;
                return std::string();

            }
            if (ch == '\n') break;
            out.push_back(ch);
            if(out.size()> 1<<20) break; 
        }
        return out;
    }


    bool write_line(int fd , const std::string& s){
        std::string tmp=s;
        tmp.push_back('\n');
        const char* p = tmp.data();
        size_t left = tmp.size();
        while(left){
            ssize_t n = ::write(fd,p,left);
            if (n<0){
                if(errno == EINTR) continue;
                return false;
            } 
            left -= static_cast<size_t>(n);
            p    += static_cast<size_t>(n);
        } 
        return true;
    }
}

IpcServer::IpcServer(const std::string& socket_p): socket_path(socket_p){}

IpcServer::~IpcServer(){stop();}


bool IpcServer::start(Handler h){
    handler = std::move(h);

    std::signal(SIGPIPE, SIG_IGN);

    listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd<0) { std::perror("socket"); return false; }

    ::unlink(socket_path.c_str());

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socket_path.c_str());

    if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("bind");
        ::close(listen_fd);
        listen_fd = -1;
        return false;
    }

    ::chmod(socket_path.c_str(), 0700);

    if(::listen(listen_fd, 16) < 0) {
        std::perror("listen");
        ::close(listen_fd);
        listen_fd = -1;
        return false;
    }
    running = true;
    th = std::thread(&IpcServer::run_accept_loop, this);
    return true;
}

void IpcServer::stop(){
    bool was_running = running.exchange(false);
    if(was_running){
        if(listen_fd>=0){
            ::close(listen_fd);
            listen_fd=-1;
        }
    }
    if(th.joinable())th.join();
    ::unlink(socket_path.c_str());
}



void IpcServer::run_accept_loop(){
    while(running){
        int client_fd = ::accept(listen_fd, nullptr, nullptr);
        if(client_fd<0){
            if(errno == EINTR) continue;
            if(!running) break;
            std::perror("accept");
            continue;
        }

        // set a recv timeout 
        timeval tv{5,0}; 
        ::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        std::string line = read_line(client_fd);
        if(!line.empty()){
            std::cout << "[ipc] received line: '" << line
                      << "' (len=" << line.size() << ")\n";

            std::string reply;
            try {
                reply = handler ? handler(line)
                                : std::string("{\"ok\":false,\"error\":\"no handler\"}");
            } catch (...) {
                reply = "{\"ok\":false,\"error\":\"exception in handler\"}";
            }

            std::cout << "[ipc] reply: '" << reply
                      << "' (len=" << reply.size() << ")\n";

            (void)write_line(client_fd, reply);
        }
        ::close(client_fd);

    }
}
