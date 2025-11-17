#pragma once 
#include <memory>
#include <string>

#include "context.h"

class gpgpu_context;
class gpgpu_sim;
class stream_manager;


class ExecutionEngine {
    public:
        bool initialize(const std::string& cfg_path);
        bool reset();
        bool bind(const GPUContext& ctx);
        bool run_to_completion();
        void unbind();
        void shutdown();
        void set_output_root(const std::string& path) { output_root = path; }
    private:
        bool file_exists(const std::string& p) const;
        gpgpu_context* ctx_ = nullptr;
        gpgpu_sim* gpu_ = nullptr;
        std::string output_root;
        std::string current_trace_dir;
        
};