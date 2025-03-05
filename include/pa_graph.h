//
// Created by ECNU on 2024/8/21.
//

#ifndef DISKANN_PA_GRAPH_H
#define DISKANN_PA_GRAPH_H
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include <memory>
#include "distance.h"
#include "index.h"
#include "pa_aio.h"
namespace diskann
{


// workflow:
//  1. construct index instance
//  2. build index
//  3. save index
//  4. load index

template<typename dist_t>
class PointAggregationGraph
{
  public:
    PointAggregationGraph(float sample_rate,
                          uint32_t sample_freq,
                          uint32_t L,
                          uint32_t R,
                          uint32_t num_threads,
                          float alpha,
                          bool use_routing_path_redundancy,
                          float extra_rate,
                          size_t redundant_num,
                          float sigma,
                          float delta,
                          float scale_factor,
                          float early_stop_factor,
                          std::unique_ptr<Distance<dist_t>> distance_fn);

    PointAggregationGraph() = default;
    ~PointAggregationGraph() {
        if (fd_ > 0)
            close(fd_);
    }

    void BuildInMemory(dist_t *data, size_t num, size_t dim);

    void BuildInMemoryFromFile(const char* filename);

    void BuildInDisk(const char* data_file, const char* index_file);

    void BuildInDFS(){};

    std::pair<uint32_t, uint32_t> SearchInMemoryByTopK(const dist_t *query, size_t k, size_t l, size_t nprobe,
                                                       uint32_t *idx, float *dx = nullptr);

    std::pair<uint32_t, uint32_t> SearchInMemoryByRoute(const dist_t *query, size_t k, size_t l,
                                                        uint32_t *idx, float *dx = nullptr);

    std::pair<uint32_t, uint32_t> SearchInDiskByTopK(dist_t *query, size_t k, size_t l, size_t nprobe,
                                                     uint32_t *idx, float *dx = nullptr);

    std::pair<size_t, size_t> SearchInDiskByRoute(dist_t *query, size_t k, size_t l,
                                                      uint32_t *idx, float *dx = nullptr);

    void SaveFromMemoryPAG(const char *filename);

    void LoadFromMemoryPAG(const char *filename, uint32_t L);

    void SaveFromDiskPAG();

    void LoadFromDiskPAG(const char* filename, uint32_t L);

    bool RoutingPathSearch() { return use_routing_path_redundancy_; }

    using Element = std::pair<float, uint32_t>;

  private:
    void sample_radius_of_every_agg();
    float sample_radius_based_on_edge();
    void update_radius(size_t id);
    float sample_radius();
    void sample_aggregation();
    void sample_aggregation_k_means();
    void sample_aggregation_k_means_in_disk();
    void sample_aggregation_in_disk();
    void build_pg();
    void build_pag();
    void build_pag_in_disk();
    std::vector<uint32_t> limit_prune(std::vector<uint32_t> &idx, std::vector<float> &dist);
    std::vector<uint32_t> limit_prune_with_fine_grained_radius(std::vector<uint32_t> &idx, std::vector<float> &dist);
    std::vector<uint32_t> limit_prune_with_prob(std::vector<uint32_t> &idx, std::vector<float> &dist);
    uint32_t promote_as_aggregation(uint32_t residual_id);

    void distances_batch_4(const dist_t* partition,
                           const dist_t* query,
                           const uint32_t idx0,
                           const uint32_t idx1,
                           const uint32_t idx2,
                           const uint32_t idx3,
                           float& dis0,
                           float& dis1,
                           float& dis2,
                           float& dis3) const;

    void initial_ivf_disk(std::ofstream &ivf_writer);
    void write_vec_to_partition(const dist_t *x, uint32_t p, unsigned int sz, std::ofstream &ivf_appender);

    void promote_as_aggregation_disk(uint32_t id, const dist_t *vec);

  private:
    size_t dim_;
    size_t aligned_dim_;
    size_t num_;
    float sample_rate_ = 0.01;
    float radius_;
    std::vector<float> fine_grained_radius_;
    uint32_t use_fine_grained_ = 1;
    float radius_rate_ = 0.5;
    float overview_rate_ = 0.15;
    std::mt19937 mt_;
    uint32_t sample_freq_ = 32;
    std::vector<std::vector<uint32_t>> ivf_;
//    std::vector<std::vector<float>> ivf_vec_;
    std::vector<uint32_t> ivf_size_;
    std::vector<uint32_t> agg2id_;
    std::vector<uint32_t> res2id_;
    dist_t *data_;
    std::vector<dist_t> dataset_;
    std::vector<dist_t> aggregation_point_;
    std::vector<dist_t> residual_point_;
    std::vector<uint32_t> residual_keep_offset_;
    std::unordered_map<uint32_t, uint32_t> id2offset_;
    std::vector<size_t> ivf_offset_;
    std::unique_ptr<Distance<dist_t>> distance_fn_;
    int64_t ag_num_;
    int64_t res_num_;

    bool is_build = false;


    /** PG construction parameter **/
    uint32_t L_ = 24;
    uint32_t R_ = 16;
    float alpha_ = 1;
    uint32_t num_threads_ = 1;
    diskann::Metric metric_ = diskann::Metric::L2;
    std::unique_ptr<diskann::Index<dist_t>> index_;

    /** PAG-Disk Parameter **/
    std::string data_file_;
    std::string index_file_;

    /** PAG construction parameter **/

    // whether use routing path to select redundant partition? else use TOPK
    uint8_t use_routing_path_redundancy_ = 1;

    // capacity of partition
    size_t capacity_;

    // allow extra overflow
    float extra_rate_ = 0.1;

    // the number to place redundancy
    size_t redundant_num_ = 4;

    // relax RNG-based rule
    float sigma_ = 1.2;

    // select nearly the same partition portion
    float delta_ = 0.1;

    // pre-scale up the ivf in case of expansion caused by dynamic representation selection
    float scale_factor_ = 2;

    // early stop condition: d * stop_factor > (d + 2 * r)
    float early_stop_factor_ = 1.2;

    /** Async IO Manager */
    int fd_;
    std::unique_ptr<AsyncIOManager<dist_t>> io_mgr_;
    std::unique_ptr<IOManagerPool<dist_t>> pool_;
};

}



#endif // DISKANN_PA_GRAPH_H
