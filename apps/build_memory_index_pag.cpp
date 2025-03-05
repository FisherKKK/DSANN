// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"
#include "pa_graph.h"
#include "program_options_utils.hpp"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"
#include "index_factory.h"

namespace po = boost::program_options;

// 构建内存索引
int main(int argc, char **argv)
{
    std::string data_type, dist_fn, data_path, index_path_prefix;
    uint32_t sample_freq, num_threads, R, L, redundant_num;
    float sample_rate, extra_rate, sigma, delta, early_stop_factor, alpha, scale_factor;
    uint8_t use_routing_path_redundancy;

    // 参数描述
    po::options_description desc{
        program_options_utils::make_program_description("build_memory_index", "Build a memory-based PAG index.")};
    try
    {
        // --help/h参数的作用
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters, 必选参数
        po::options_description required_configs("Required");
        // data_type -> string data_type, 相当于下面的参数都是一一对应关系
        // 数据类型
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        // 距离函数
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        // index的路径前缀
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        // 数据的路径
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH);

        // Optional parameters, 可选参数
        po::options_description optional_configs("Optional");
        // 线程数
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        // 每个节点最大的出度
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE);
        // 图构建过程中搜索的邻居数目
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);
        // 长边的松弛
        optional_configs.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                                       program_options_utils::GRAPH_BUILD_ALPHA);

        optional_configs.add_options()("sample_freq", po::value<uint32_t>(&sample_freq)->default_value(32),
                                       "sample radius frequency");
        optional_configs.add_options()("redundant_num", po::value<uint32_t>(&redundant_num)->default_value(4),
                                       "redundant number of partition");
        optional_configs.add_options()("scale_factor", po::value<float>(&scale_factor)->default_value(5),
                                       "pre-scale up the ivf size");
        optional_configs.add_options()("sample_rate", po::value<float>(&sample_rate)->default_value(0.04),
                                       "sample rate of aggregation point");
        optional_configs.add_options()("extra_rate", po::value<float>(&extra_rate)->default_value(0.1),
                                       "allow extra overflow of ivf");
        optional_configs.add_options()("sigma", po::value<float>(&sigma)->default_value(0.8),
                                       "relax RNG-base rule");
        optional_configs.add_options()("delta", po::value<float>(&delta)->default_value(0.1),
                                       "select nearly the same partition portion");
        optional_configs.add_options()("early_stop_factor", po::value<float>(&early_stop_factor)->default_value(1.2),
                                       "early stop condition: d  > stop_factor * (minimal_d + 2 * r)");
        optional_configs.add_options()("use_routing_path_redundancy", po::bool_switch()->default_value(false),
                                       "redundancy with path");

        // Merge required and optional parameters
        // 参数合并
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        // 将参数存储到map中
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        // 是否使用PQ
        use_routing_path_redundancy = vm["use_routing_path_redundancy"].as<bool>() ? 1 : 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    // DiskANN使用的度量
    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {   // 内积
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {   // l2范数
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        // 余弦
        metric = diskann::Metric::COSINE;
    }
    else
    {   // 报错
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }


    try
    {
        // 参数描述
        diskann::cout << "Starting index build" << std::endl;

        if (data_type == std::string("uint8")) {
            std::unique_ptr<diskann::Distance<uint8_t>> fn;
            fn.reset(diskann::get_distance_function<uint8_t >(metric));

            auto index = std::make_unique<diskann::PointAggregationGraph<uint8_t>>(
                sample_rate, sample_freq, L, R, num_threads, alpha, use_routing_path_redundancy, extra_rate,
                redundant_num, sigma, delta, scale_factor, early_stop_factor, std::move(fn));

            index->BuildInMemoryFromFile(data_path.c_str());
            index->SaveFromMemoryPAG(index_path_prefix.c_str());
            index.reset();
        } else if (data_type == std::string("float")) {
            std::unique_ptr<diskann::Distance<float>> fn;

            if (metric == diskann::Metric::COSINE)
                fn.reset(new diskann::AVXNormalizedCosineDistanceFloat());
            else
                fn.reset(diskann::get_distance_function<float>(metric));

            auto index = std::make_unique<diskann::PointAggregationGraph<float>>(
                sample_rate, sample_freq, L, R, num_threads, alpha, use_routing_path_redundancy, extra_rate,
                redundant_num, sigma, delta, scale_factor, early_stop_factor, std::move(fn));

            index->BuildInMemoryFromFile(data_path.c_str());
            index->SaveFromMemoryPAG(index_path_prefix.c_str());
            index.reset();
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
