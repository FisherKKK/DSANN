// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"
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
    std::string data_type, dist_fn, data_path, index_path_prefix, label_file, universal_label, label_type;
    uint32_t num_threads, R, L, Lf, build_PQ_bytes;
    float alpha;
    bool use_pq_build, use_opq;

    // 参数描述
    po::options_description desc{
        program_options_utils::make_program_description("build_memory_index", "Build a memory-based DiskANN index.")};
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
        // PQ编码的字节数
        optional_configs.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ_bytes)->default_value(0),
                                       program_options_utils::BUIlD_GRAPH_PQ_BYTES);
        // 是否使用优化的PQ
        optional_configs.add_options()("use_opq", po::bool_switch()->default_value(false),
                                       program_options_utils::USE_OPQ);
        optional_configs.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
                                       program_options_utils::LABEL_FILE);
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL);

        optional_configs.add_options()("FilteredLbuild", po::value<uint32_t>(&Lf)->default_value(0),
                                       program_options_utils::FILTERED_LBUILD);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);

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
        use_pq_build = (build_PQ_bytes > 0);
        use_opq = vm["use_opq"].as<bool>();
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
        diskann::cout << "Starting index build with R: " << R << "  Lbuild: " << L << "  alpha: " << alpha
                      << "  #threads: " << num_threads << std::endl;

        size_t data_num, data_dim;
        // 获取元数据信息: 数据路径, 数据集大小, 数据维度
        diskann::get_bin_metadata(data_path, data_num, data_dim);
        
        // 构建索引参数, 就是默认的索引参数, 返回的也是一个可以随意访问属性的Class对象
        auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                      .with_filter_list_size(Lf)
                                      .with_alpha(alpha)
                                      .with_saturate_graph(false)
                                      .with_num_threads(num_threads)
                                      .build();
        
        //TODO 目前可以不用管Filter
        auto filter_params = diskann::IndexFilterParamsBuilder()
                                 .with_universal_label(universal_label)
                                 .with_label_file(label_file)
                                 .with_save_path_prefix(index_path_prefix)
                                 .build();

        // 综合所有的索引配置参数, 构建之后返回Config对象, 属于结构体, 所有的配置信息都能访问
        auto config = diskann::IndexConfigBuilder()
                          .with_metric(metric)
                          .with_dimension(data_dim)
                          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                          .with_max_points(data_num)
                          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                          .with_data_type(data_type)
                          .with_label_type(label_type)
                          .is_dynamic_index(false)
                          .with_index_write_params(index_build_params)
                          .is_enable_tags(false)
                          .is_use_opq(use_opq)
                          .is_pq_dist_build(use_pq_build)
                          .with_num_pq_chunks(build_PQ_bytes)
                          .build();

        // 按照config构建索引工厂
        auto index_factory = diskann::IndexFactory(config);

        // 创建索引对象实例, 目前index里面也都是空的, 存在构建索引所需要的配置信息
        auto index = index_factory.create_instance();

        // 盲猜测是构建索引
        index->build(data_path, data_num, filter_params);
        index->save(index_path_prefix.c_str());
        index.reset();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
