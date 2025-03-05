//
// Created by ECNU on 2024/8/21.
//
#include <algorithm>
#include "pa_aio.h"
#include "pa_graph.h"
#include "parameters.h"
#include "index.h"
#include "index_factory.h"
#include "simd_utils.h"
#include <immintrin.h>
namespace diskann
{

template<typename dist_t>
PointAggregationGraph<dist_t>::PointAggregationGraph(float sample_rate,
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
                                             std::unique_ptr<Distance<dist_t>> distance_fn)
    : sample_rate_(sample_rate), sample_freq_(sample_freq), L_(L),
      R_(R), num_threads_(num_threads), alpha_(alpha), use_routing_path_redundancy_(use_routing_path_redundancy),
      extra_rate_(extra_rate), redundant_num_(redundant_num), sigma_(sigma),
      delta_(delta), scale_factor_(scale_factor), early_stop_factor_(early_stop_factor),
      distance_fn_(std::move(distance_fn))
{
    metric_ = distance_fn_->get_metric();
}

template<typename dist_t>
void PointAggregationGraph<dist_t>::sample_aggregation()
{
    ag_num_ = static_cast<uint32_t>(sample_rate_ * num_);
    res_num_ = num_ - ag_num_;
    std::vector<uint32_t> position(num_);
    for (uint32_t i = 0; i < num_; i++)
        position[i] = i;
    // random permulation
    for (uint32_t i = 0; i + 1 < num_; i++)
    {
        int i2 = i + mt_() % (num_ - i);
        std::swap(position[i], position[i2]);
    }

    aggregation_point_.resize(scale_factor_ * ag_num_ * aligned_dim_, 0);
//    residual_point_.resize(res_num_ * dim_);
    agg2id_.resize(scale_factor_ * ag_num_);
    res2id_.resize(res_num_);
    memcpy(agg2id_.data(), position.data(), sizeof(uint32_t) * ag_num_);
    memcpy(res2id_.data(), position.data() + ag_num_, sizeof (uint32_t) * res_num_);
    for (size_t i = 0; i < num_; i++) {
        if (i < ag_num_) {
            memcpy(aggregation_point_.data() + i * aligned_dim_, data_ + position[i] * aligned_dim_, sizeof (dist_t) * aligned_dim_);
        } else {
//            memcpy(residual_point_.data() + (i - ag_num_) * dim_, data_ + position[i] * dim_, sizeof (float) * dim_);
        }
    }
}


template<typename dist_t>
void PointAggregationGraph<dist_t>::sample_aggregation_k_means()
{

}

template<typename dist_t>
void PointAggregationGraph<dist_t>::sample_aggregation_k_means_in_disk()
{

}


template<typename dist_t>
void PointAggregationGraph<dist_t>::build_pg() {
    diskann::IndexWriteParameters low_degree_params = diskann::IndexWriteParametersBuilder(L_, R_)
                                                          .with_filter_list_size(0)
                                                          .with_alpha(alpha_)
                                                          .with_saturate_graph(false)
                                                          .with_num_threads(num_threads_)
                                                          .build();
    // expand the max point because of the dynamic representation selection
    index_ = std::make_unique<diskann::Index<dist_t>>(metric_, aligned_dim_, ag_num_ * scale_factor_,
                                                          std::make_shared<diskann::IndexWriteParameters>(low_degree_params), nullptr,
                                                          diskann::defaults::NUM_FROZEN_POINTS_STATIC, false, false, false, false,
                                                          0, false);
    index_->build(aggregation_point_.data(), ag_num_, std::vector<uint32_t >());

    decltype(aggregation_point_) tmp;
    aggregation_point_.swap(tmp);
}

/**
 * @param idx   id of aggregation point (id after map in graph)
 * @param dist  distance of the query to aggregation point
 * @return pruned idx
 */
template<typename dist_t>
std::vector<uint32_t> PointAggregationGraph<dist_t>::limit_prune(std::vector<uint32_t> &idx, std::vector<float> &dist) {

    dist_t *vi, *vj;
    std::vector<uint32_t> result;
    for (uint32_t i = 0; i < idx.size(); i++) {
        bool occlude = false;
        float dij;
        index_->get_vector(idx[i], &vi);
//            const dist_t *vi = aggregation_point_.data() + idx[i] * dim_;
        for (uint32_t j = 0; j < result.size(); j++) {
            index_->get_vector(result[j], &vj);
//                const dist_t *vj = aggregation_point_.data() + result[j] * dim_;
            dij = distance_fn_->compare(vi, vj, aligned_dim_);
            if (dij < sigma_ * dist[i]) {
                occlude = true;
                break;
            }
        }
        if (!occlude) result.push_back(idx[i]);
    }
    return result;
}

template<typename dist_t>
std::vector<uint32_t> PointAggregationGraph<dist_t>::limit_prune_with_prob(std::vector<uint32_t> &idx, std::vector<float> &dist) {
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::random_device rd;
    std::mt19937 g(rd());


    auto calculate_score = [](float d, float r, size_t size, float sample_rate, float extra_rate,
                              float alpha = 0.4f, float beta = 2.0f, float gamma = 0.05f) -> float {
        auto capacity = extra_rate / (sample_rate + 0.00001f);
        return std::pow(2, -alpha * (d / (r + 0.0001f))) *
                    std::pow(2, -beta * std::abs(size / capacity - gamma));
    };


    std::vector<uint32_t> result;
    dist_t* vi, *vj;
    for (uint32_t i = 0; i < idx.size(); i++) {
        float dij;
        float radius = fine_grained_radius_[idx[i]], d = dist[i];
        auto ivf_size = ivf_size_[idx[i]];
        auto score = calculate_score(d, radius, ivf_size, sample_rate_, extra_rate_);
        auto probability = distribution(g);

        if (probability < score || d < radius * 0.5f) {
            result.push_back(idx[i]);
        } else if (d < radius * 2) {
            bool occlude = false;
            index_->get_vector(idx[i], &vi);
            for (uint32_t j = 0; j < result.size(); j++) {
                index_->get_vector(result[j], &vj);
                dij = distance_fn_->compare(vi, vj, aligned_dim_);
                if (dij < sigma_ * dist[i]) {
                    occlude = true;
                    break;
                }
            }
            if (!occlude) result.push_back(idx[i]);
        }
    }
    return result;
}




template<typename dist_t>
std::vector<uint32_t> PointAggregationGraph<dist_t>::limit_prune_with_fine_grained_radius(std::vector<uint32_t> &idx, std::vector<float> &dist)
{


//    auto overlap_fn = [](std::vector<uint32_t> &s1, std::vector<uint32_t> &s2) {
//        std::unordered_set<uint32_t> all, one;
//        all.insert(s1.begin(), s1.end());
//        all.insert(s2.begin(), s2.end());
//
//        one.insert(s1.begin(), s1.end());
//        float overlap = 0;
//        for (size_t i = 0; i < s2.size(); i++) {
//            overlap += one.count(s2[i]);
//        }
//
//        return overlap; /// all.size();
//    };

//    std::unordered_map<uint32_t, float> scores;
//    for (auto id: idx) {
//        auto neighbor = index_->get_neighbors(id);
//        scores[id] = overlap_fn(idx, neighbor);
//    }
//
//   std::vector<std::pair<uint32_t, float>> candidates(idx.size());
//   for (int i = 0; i < candidates.size(); i++) {
//       candidates[i].first = idx[i];
//       candidates[i].second = dist[i];
//   }
//
//   std::sort(candidates.begin(), candidates.end(), [&](std::pair<uint32_t, float> &p1, std::pair<uint32_t, float> &p2) {
//       return scores[p1.first] > scores[p2.first];
//   });

    std::vector<uint32_t> result;


//    for (uint32_t i = 0; i < idx.size(); i++) {
//        bool occlude = false;
//        float dij, radius = fine_grained_radius_[idx[i]];
//        if (dist[i] < radius || (radius_ != radius && dist[i] < std::min((double)radius_, radius * (1 + ag_num_ / (num_ * (sample_rate_ + 0.0001) * scale_factor_)))))
//        {
//            const float *vi = aggregation_point_.data() + idx[i] * dim_;
//            for (uint32_t j = 0; j < result.size(); j++)
//            {
//                const float *vj = aggregation_point_.data() + result[j] * dim_;
//                dij = distance_fn_->compare(vi, vj, dim_);
//                if (dij < sigma_ * dist[i])
//                {
//                    occlude = true;
//                    break;
//                }
//            }
//            if (!occlude)
//                result.push_back(idx[i]);
//            else if (dist[i] / radius < delta_)
//                result.push_back(idx[i]); // 如果距离非常接近
//        }
//    }




//    for (int i = 0; i < redundant_num_ * 0.4 && i < candidates.size(); i++) {
//        auto id = candidates[i].first;
//        if (std::find(result.begin(), result.end(), id) == result.end())
//        result.push_back(id);
//    }




//    std::vector<uint32_t> recommendation;
    dist_t* vi, *vj;
    for (uint32_t i = 0; i < idx.size(); i++) {
        bool occlude = false;
        float dij, radius = fine_grained_radius_[idx[i]];
//        (dist[i] - radius) < radius_ * (ag_num_ / (num_ * (sample_rate_ + 0.0001) * 2))
        if (dist[i] < radius || (radius_ != radius && dist[i] < std::min((double)radius_, radius * (1 + ag_num_ / (num_ * (sample_rate_ + 0.0001) * 2 * scale_factor_))))) {
//            index_->get_vector(idx[i], vi.data());
//            const dist_t *vi = aggregation_point_.data() + idx[i] * dim_;
            index_->get_vector(idx[i], &vi);
            for (uint32_t j = 0; j < result.size(); j++) {
//                const dist_t *vj = aggregation_point_.data() + result[j] * dim_;
//                index_->get_vector(result[j], vj.data());
                index_->get_vector(result[j], &vj);
                dij = distance_fn_->compare(vi, vj, aligned_dim_);
                if (dij < sigma_ * dist[i]) { //&& dij / radius > delta_) {
                    occlude = true;
                    break;
                }
            }
            if (!occlude) result.push_back(idx[i]);
            else if (dist[i] / radius < delta_) result.push_back(idx[i]); // 如果距离非常接近
        } else if (i == 0 ||
                 (i < idx.size() * 0.2 && dist[i] < radius * 1.5 && ivf_size_[idx[i]] < capacity_ * 0.4)) {
            result.push_back(idx[i]);
        }
        // else if (dist[i] < radius * 2) {
//            auto id = idx[i];
//            auto sz = ivf_size_[id];
//            for (int k = 0; k < sz; k++) {
//                auto res_id = ivf_[id][k];
//                const float *x = data_ + res_id * dim_;
//                if (distance_fn_->compare(x, q, dim_) < radius * 0.5) {
//                    result.push_back(id);
//                    break;
//                }
//            }
//        }

    }


    if (use_routing_path_redundancy_)
        std::stable_sort(result.begin(), result.end(), [&](uint32_t a, uint32_t b) {
            if (ivf_size_[a] < ivf_size_[b]) return true;
            return false;
        });

//    for (auto id: result) {
//        index_->prune_neighbors_for(id, recommendation);
//    }

    return result;
}

template<typename dist_t>
uint32_t PointAggregationGraph<dist_t>::promote_as_aggregation(uint32_t real_id) {
    if (ag_num_ >= ivf_.size())
        diskann::cout << "Too many promotion\n";
    agg2id_[ag_num_] = real_id;
//    memcpy(aggregation_point_.data() + ag_num_ * dim_,
//           data_ + real_id * dim_,
//           sizeof (dist_t) * dim_);
//    std::vector<dist_t> dummy_vec(aligned_dim_, 0);
//    memcpy(dummy_vec.data(), data_ + real_id * dim_, dim_ * sizeof(dist_t));
//    index_->insert_point(dummy_vec.data(), ag_num_);
    index_->insert_point(data_ + real_id * aligned_dim_, ag_num_);
    update_radius(ag_num_);
    ag_num_++;
    res_num_--;
    return ag_num_ - 1;
}

template<typename dist_t>
void PointAggregationGraph<dist_t>::build_pag()
{
    capacity_ = (1 + extra_rate_) / sample_rate_;
    ivf_.resize(ag_num_ * scale_factor_, std::vector<uint32_t>(capacity_, -1));
    ivf_size_.resize(ag_num_ * scale_factor_, 0);
    residual_keep_offset_.reserve(res_num_);

#pragma omp parallel for
    for (size_t i = 0; i < res_num_; i++) {
        uint32_t id = res2id_[i];
        const dist_t *x = data_ + id * aligned_dim_;

        std::vector<uint32_t> idx;
        std::vector<float> dist;
        if (use_routing_path_redundancy_) {
            // search return the path
            std::vector<Element> path;
            index_->search_with_path(x, path, 1, L_);
            idx.resize(path.size()); dist.resize(path.size());
            for (size_t r = 0; r < path.size(); r++) {
                idx[r] = path[r].second;
                dist[r] = path[r].first;
            }
//            sigma_ = sigma_* 2;
        } else {
            // search nearest
            idx.resize(L_ / 2); dist.resize(L_ / 2);
            index_->search(x, L_ / 2, L_, idx.data(), dist.data());
        }

        std::vector<uint32_t> partitions;
        if (!use_fine_grained_)
            partitions = limit_prune(idx, dist);
        else
            partitions = limit_prune_with_fine_grained_radius(idx, dist);
#pragma omp critical
        {
            bool place = false;
            for (int r = 0; r < redundant_num_ && r < partitions.size(); r++)
            {
                uint32_t p = partitions[r];
                auto &sz = ivf_size_[p];
                if (sz < capacity_)
                {
                    ivf_[p][sz] = id;
                    /** for memory save
                     *  memcpy(ivf_vec_[p].data() + sz * dim_, data_ + id * dim_, sizeof(float) * dim_);
                     */
                    sz++;
                    place = true;
                }
            }
            if (!place)
            {
                promote_as_aggregation(id);
            } else {
                residual_keep_offset_.push_back(id);
            }
        }
    }
}

template<typename dist_t>
void PointAggregationGraph<dist_t>::BuildInMemoryFromFile(const char* filename) {
    size_t num, dim;
    diskann::get_bin_metadata(filename, num, dim);
    aligned_dim_ = ROUND_UP(dim, 8);
    dataset_.resize(num * aligned_dim_, 0);
    if (!file_exists(filename))
    {
        std::stringstream stream;
        stream << "ERROR: Data file " << filename << " does not exist." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    std::ifstream reader(filename, std::ios::binary);
    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));

    for (size_t i = 0; i < num; i++)
    {
        reader.read((char *)(dataset_.data() + i * aligned_dim_), dim * sizeof(dist_t));
    }
    BuildInMemory(dataset_.data(), num, dim);
}

template<typename dist_t>
void PointAggregationGraph<dist_t>::BuildInMemory(dist_t *data, size_t num, size_t dim)
{
    /// init param
    data_ = data;
    num_ = num;
    dim_ = dim;
    if (distance_fn_->preprocessing_required()) // in case of diskann preprocess
        distance_fn_->preprocess_base_points(data, aligned_dim_, num);
    // sample aggregation points
    sample_aggregation();
    // construct proximity graph on aggregation points
    build_pg();
    // sample radius
    fine_grained_radius_.resize(ag_num_ * scale_factor_);
    radius_ = sample_radius_based_on_edge();
    if (use_fine_grained_)
        sample_radius_of_every_agg();

    // aggregation
    build_pag();
    is_build = true;
}

template<typename dist_t>
void PointAggregationGraph<dist_t>::update_radius(size_t id)
{
    if (!use_fine_grained_) {
        fine_grained_radius_[id] = radius_;
        return;
    }
    std::vector<float> dists;
    dists.reserve(R_);
    index_->sample_neighbor_distance(id, dists);
    uint32_t thres_pos = dists.size() * radius_rate_;
    std::nth_element(dists.begin(), dists.begin() + thres_pos, dists.end());
    fine_grained_radius_[id] = std::min(*(dists.begin() + thres_pos), radius_);
}

template<typename dist_t>
void PointAggregationGraph<dist_t>::sample_radius_of_every_agg()
{
    for (size_t i = 0; i < ag_num_; i++) {
        update_radius(i);
    }
}


template<typename dist_t>
float PointAggregationGraph<dist_t>::sample_radius_based_on_edge() {
    std::vector<uint32_t> position(ag_num_);

#pragma omp parallel for
    for (uint32_t i = 0; i < ag_num_; i++)
        position[i] = i;

    // random permulation
    for (uint32_t i = 0; i + 1 < ag_num_; i++)
    {
        int i2 = i + mt_() % (ag_num_ - i);
        std::swap(position[i], position[i2]);
    }

    std::vector<float> dists;
    dists.reserve(1024);

    for (size_t i = 0; i < sample_freq_; i++) {
        index_->sample_neighbor_distance(position[i], dists);
    }

    uint32_t thres_pos = dists.size() * overview_rate_;

    std::nth_element(dists.begin(), dists.begin() + thres_pos, dists.end());
    return *(dists.begin() + thres_pos);
}

template<typename dist_t>
float PointAggregationGraph<dist_t>::sample_radius()
{
    std::vector<uint32_t> position(num_);

#pragma omp parallel for
    for (uint32_t i = 0; i < num_; i++)
        position[i] = i;
    // random permulation
    for (uint32_t i = 0; i + 1 < num_; i++)
    {
        int i2 = i + mt_() % (num_ - i);
        std::swap(position[i], position[i2]);
    }


    float radius_sum = .0;
    uint32_t thres_pos = static_cast<uint32_t>(sample_rate_ * 100);
    for (int sf = 0; sf < sample_freq_; sf++)
    {
        std::vector<float> dis(100);
        auto reference = data_ + dim_ * position[sf * 100];
        for (uint32_t i = 0; i < 100; i++)
        {
            auto tmp = data_ + dim_ * position[sf * 100 + i + 1];
            float d = distance_fn_->compare(reference, tmp, dim_);
            dis[i] = d;
        }
        std::nth_element(dis.begin(), dis.begin() + thres_pos, dis.end());
        radius_sum += *(dis.begin() + thres_pos);
    }
    return radius_sum / sample_freq_;
}

namespace {
#if defined(__GNUC__)
#define PAG_PRAGMA_IMPRECISE_LOOP
#define PAG_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define PAG_PRAGMA_IMPRECISE_FUNCTION_END \
    _Pragma("GCC pop_options")
#endif


PAG_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template<typename dist_t>
void fvec_batch_4(
    const dist_t* x,
    const dist_t* y0,
    const dist_t* y1,
    const dist_t* y2,
    const dist_t* y3,
    const size_t d,
    float& dis0,
    float& dis1,
    float& dis2,
    float& dis3,
    const std::unique_ptr<Distance<dist_t>> &dist_fn
    ) {
    PAG_PRAGMA_IMPRECISE_LOOP
    dis0 = dist_fn->compare(x, y0, d);
    dis1 = dist_fn->compare(x, y1, d);
    dis2 = dist_fn->compare(x, y2, d);
    dis3 = dist_fn->compare(x, y3, d);
}
PAG_PRAGMA_IMPRECISE_FUNCTION_END


//TODO
//template<typename dist_t>
//void fvec_L2sqr_batch_4(
//    const dist_t* x,
//    const dist_t* y0,
//    const dist_t* y1,
//    const dist_t* y2,
//    const dist_t* y3,
//    const size_t d,
//    float& dis0,
//    float& dis1,
//    float& dis2,
//    float& dis3
//) {
//    x  = (const float *)__builtin_assume_aligned(x, 32);
//    y0 = (const float *)__builtin_assume_aligned(y0, 32);
//    y1 = (const float *)__builtin_assume_aligned(y1, 32);
//    y2 = (const float *)__builtin_assume_aligned(y2, 32);
//    y3 = (const float *)__builtin_assume_aligned(y3, 32);
//
//
//    uint16_t niters = (uint16_t)(d / 8);
//
//    __m256 sum0 = _mm256_setzero_ps();
//    __m256 sum1 = _mm256_setzero_ps();
//    __m256 sum2 = _mm256_setzero_ps();
//    __m256 sum3 = _mm256_setzero_ps();
//
//    for (uint16_t j = 0; j < niters; j++)
//    {
//        if (j < (niters - 1))
//        {
//            _mm_prefetch((char *)(x + 8 * (j + 1)), _MM_HINT_T0);
//            _mm_prefetch((char *)(y0 + 8 * (j + 1)), _MM_HINT_T0);
//            _mm_prefetch((char *)(y1 + 8 * (j + 1)), _MM_HINT_T0);
//            _mm_prefetch((char *)(y2 + 8 * (j + 1)), _MM_HINT_T0);
//            _mm_prefetch((char *)(y3 + 8 * (j + 1)), _MM_HINT_T0);
//        }
//        __m256 x_vec = _mm256_load_ps(x + 8 * j);
//        // load b_vec
//        __m256 y0_vec = _mm256_load_ps(y0 + 8 * j);
//        __m256 y1_vec = _mm256_load_ps(y1 + 8 * j);
//        __m256 y2_vec = _mm256_load_ps(y2 + 8 * j);
//        __m256 y3_vec = _mm256_load_ps(y3 + 8 * j);
//        // a_vec - b_vec
//        __m256 tmp0_vec = _mm256_sub_ps(x_vec, y0_vec);
//        __m256 tmp1_vec = _mm256_sub_ps(x_vec, y1_vec);
//        __m256 tmp2_vec = _mm256_sub_ps(x_vec, y2_vec);
//        __m256 tmp3_vec = _mm256_sub_ps(x_vec, y3_vec);
//
//
//        sum0 = _mm256_fmadd_ps(tmp0_vec, tmp0_vec, sum0);
//        sum1 = _mm256_fmadd_ps(tmp1_vec, tmp1_vec, sum1);
//        sum2 = _mm256_fmadd_ps(tmp2_vec, tmp2_vec, sum2);
//        sum3 = _mm256_fmadd_ps(tmp3_vec, tmp3_vec, sum3);
//    }
//    dis0 =  _mm256_reduce_add_ps(sum0);
//    dis1 =  _mm256_reduce_add_ps(sum1);
//    dis2 =  _mm256_reduce_add_ps(sum2);
//    dis3 =  _mm256_reduce_add_ps(sum3);
//}

//TODO
//void fvec_ip_batch_4(
//    const float* x,
//    const float* y0,
//    const float* y1,
//    const float* y2,
//    const float* y3,
//    const size_t d,
//    float& dis0,
//    float& dis1,
//    float& dis2,
//    float& dis3
//) {
//    x  = (const float *)__builtin_assume_aligned(x, 32);
//    y0 = (const float *)__builtin_assume_aligned(y0, 32);
//    y1 = (const float *)__builtin_assume_aligned(y1, 32);
//    y2 = (const float *)__builtin_assume_aligned(y2, 32);
//    y3 = (const float *)__builtin_assume_aligned(y3, 32);
//
//#define AVX_DOT_BATCH(addr1, addr2, dest, tmp1, tmp2)                                                                  \
//    tmp1 = _mm256_loadu_ps(addr1);                                                                                     \
//    tmp2 = _mm256_loadu_ps(addr2);                                                                                     \
//    tmp1 = _mm256_mul_ps(tmp1, tmp2);                                                                                  \
//    dest = _mm256_add_ps(dest, tmp1);
//
//    __m256 sum0, sum1, sum2, sum3;
//    __m256 l00, l01, l10, l11, l20, l21, l30, l31;
//    __m256 r00, r01, r10, r11, r20, r21, r30, r31;
//    uint32_t D = (d + 7) & ~7U;
//    uint32_t DR = D % 16;
//    uint32_t DD = D - DR;
//
//    const float *e_x = x + DD;
//    const float *e_y0 = y0 + DD;
//    const float *e_y1 = y1 + DD;
//    const float *e_y2 = y2 + DD;
//    const float *e_y3 = y3 + DD;
//    float unpack0[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
//    float unpack1[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
//    float unpack2[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
//    float unpack3[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
//
//
//    sum0 = _mm256_loadu_ps(unpack0);
//    sum1 = _mm256_loadu_ps(unpack1);
//    sum2 = _mm256_loadu_ps(unpack2);
//    sum3 = _mm256_loadu_ps(unpack3);
//    if (DR)
//    {
//        AVX_DOT_BATCH(e_x, e_y0, sum0, l00, r00);
//        AVX_DOT_BATCH(e_x, e_y1, sum1, l10, r10);
//        AVX_DOT_BATCH(e_x, e_y2, sum2, l20, r20);
//        AVX_DOT_BATCH(e_x, e_y3, sum3, l30, r30);
//    }
//
//    for (uint32_t i = 0; i < DD; i += 16, x += 16, y0 += 16, y1 += 16, y2 += 16, y3 += 16)
//    {
//        AVX_DOT_BATCH(x, y0, sum0, l00, r00);
//        AVX_DOT_BATCH(x + 8, y0 + 8, sum0, l01, r01);
//
//        AVX_DOT_BATCH(x, y1, sum1, l10, r10);
//        AVX_DOT_BATCH(x + 8, y1 + 8, sum1, l11, r11);
//
//        AVX_DOT_BATCH(x, y2, sum2, l20, r20);
//        AVX_DOT_BATCH(x + 8, y2 + 8, sum2, l21, r21);
//
//        AVX_DOT_BATCH(x, y3, sum3, l30, r30);
//        AVX_DOT_BATCH(x + 8, y3 + 8, sum3, l31, r31);
//    }
//    _mm256_storeu_ps(unpack0, sum0);
//    _mm256_storeu_ps(unpack1, sum1);
//    _mm256_storeu_ps(unpack2, sum2);
//    _mm256_storeu_ps(unpack3, sum3);
//
//    dis0 =  -(unpack0[0] + unpack0[1] + unpack0[2] + unpack0[3] + unpack0[4] + unpack0[5] + unpack0[6] + unpack0[7]);
//    dis1 =  -(unpack1[0] + unpack1[1] + unpack1[2] + unpack1[3] + unpack1[4] + unpack1[5] + unpack1[6] + unpack1[7]);
//    dis2 =  -(unpack2[0] + unpack2[1] + unpack2[2] + unpack2[3] + unpack2[4] + unpack2[5] + unpack2[6] + unpack2[7]);
//    dis3 =  -(unpack3[0] + unpack3[1] + unpack3[2] + unpack3[3] + unpack3[4] + unpack3[5] + unpack3[6] + unpack3[7]);
//}

//PAG_PRAGMA_IMPRECISE_FUNCTION_BEGIN
//template<typename dist_t>
//void fvec_angular_batch_4(
//    const dist_t* x,
//    const dist_t* y0,
//    const dist_t* y1,
//    const dist_t* y2,
//    const dist_t* y3,
//    const size_t d,
//    float& dis0,
//    float& dis1,
//    float& dis2,
//    float& dis3
//) {
//    float d0, d1, d2, d3;
//    fvec_ip_batch_4(x, y0, y1, y2, y3, d, d0, d1, d2, d3);
//    dis0 = 1 + d0;
//    dis1 = 1 + d1;
//    dis2 = 1 + d2;
//    dis3 = 1 + d3;
//}
//PAG_PRAGMA_IMPRECISE_FUNCTION_END

};


//TODO
template<typename dist_t>
void PointAggregationGraph<dist_t>::distances_batch_4(const dist_t* partition,
                                              const dist_t* query,
                                              const uint32_t idx0,
                                              const uint32_t idx1,
                                              const uint32_t idx2,
                                              const uint32_t idx3,
                                              float& dis0,
                                              float& dis1,
                                              float& dis2,
                                              float& dis3) const {
    const dist_t* __restrict x0 = partition + idx0 * aligned_dim_;
    const dist_t* __restrict x1 = partition + idx1 * aligned_dim_;
    const dist_t* __restrict x2 = partition + idx2 * aligned_dim_;
    const dist_t* __restrict x3 = partition + idx3 * aligned_dim_;
    float dp0 = 0;
    float dp1 = 0;
    float dp2 = 0;
    float dp3 = 0;

    fvec_batch_4(query,
                 x0,
                 x1,
                 x2,
                 x3,
                 aligned_dim_,
                 dp0,
                 dp1,
                 dp2,
                 dp3,
                 distance_fn_);
//    switch (metric_)
//    {
//    case L2:
//        fvec_L2sqr_batch_4(query, x0, x1, x2, x3, dim_, dp0, dp1, dp2, dp3);
//        break;
//    case INNER_PRODUCT:
//        fvec_ip_batch_4(query, x0, x1, x2, x3, dim_, dp0, dp1, dp2, dp3);
//        break;
//    case COSINE:
//        fvec_angular_batch_4(query, x0, x1, x2, x3, dim_, dp0, dp1, dp2, dp3);
////        fvec_cosine_batch_4(query, x0, x1, x2, x3, dim_, dp0, dp1, dp2, dp3, distance_fn_);
//        break;
//    case FAST_L2:
//        fvec_L2sqr_batch_4(query, x0, x1, x2, x3, dim_, dp0, dp1, dp2, dp3);
//        break;
//    }

    dis0 = dp0;
    dis1 = dp1;
    dis2 = dp2;
    dis3 = dp3;
}


template<typename dist_t>
std::pair<uint32_t, uint32_t> PointAggregationGraph<dist_t>::SearchInMemoryByTopK(const dist_t *query, size_t k, size_t l,
                                                                          size_t nprobe, uint32_t *idx, float *dx) {
    if (!is_build) return {};
    std::vector<Element> q;
    q.reserve(k * 1.2);
    std::vector<uint32_t> keys(nprobe);
    std::vector<float> dists(nprobe);
    auto [hops, cmps] = index_->search(query, nprobe, l, keys.data(), dists.data());
    hops = cmps;
    std::vector<uint8_t> visited(num_, 0);
    for (size_t i = 0; i < nprobe; i++) {
        _mm_prefetch((char *)(agg2id_.data() + keys[i]), _MM_HINT_T0);
    }

    for (size_t i = 0; i < nprobe; i++)
    {
        auto key = keys[i];
        auto key_id = agg2id_[keys[i]];
        auto key_dist = dists[i];
        //        visited[key_id] = 1;
        if (q.size() < k)
        {
            q.emplace_back(key_dist, key_id);
            std::push_heap(q.begin(), q.end(), std::less<Element>());
        }
        else if (key_dist < q[0].first)
        {
            q.emplace_back(key_dist, key_id);
            std::push_heap(q.begin(), q.end(), std::less<Element>());
            std::pop_heap(q.begin(), q.end(), std::less<Element>());
            q.pop_back();
        }
    }

    for (size_t i = 0; i < nprobe; i++) {
        auto key = keys[i];
        auto sz = ivf_size_[key];
        cmps += sz;
        if (sz <= 0)
            continue;
        auto& ids_residual = ivf_[key];

        auto update_one = [&](uint32_t id, float d) {
            if (q.size() < k) {
                q.emplace_back(d, id);
                std::push_heap(q.begin(), q.end(), std::less<Element>());
            } else if (d < q[0].first) {
                q.emplace_back(d, id);
                std::push_heap(q.begin(), q.end(), std::less<Element>());
                std::pop_heap(q.begin(), q.end(), std::less<Element>());
                q.pop_back();
            }
        };

        int n_buffered = 0;
        uint32_t buffered_ids[4]; // keep inner id
        uint32_t buffered_real_ids[4]; // keep real id
        uint32_t buffered_offset[4];
        for (size_t si = 0; si < sz; ++si) {
            uint32_t real_id = ids_residual[si];
            bool get = visited[real_id];
            visited[real_id] = 1;
            buffered_ids[n_buffered] = si;
            buffered_real_ids[n_buffered] = real_id;
            buffered_offset[n_buffered] = id2offset_[real_id];
            n_buffered += get ? 0 : 1;
//            if (get) {
//                prefetch_vector_l2((char*)(residual_point_.data() + id2offset_[real_id]), sizeof(float) * dim_);
//            }

            if (n_buffered == 4) {
                float dist_buffered[4];
                distances_batch_4(residual_point_.data(), query, buffered_offset[0], buffered_offset[1], buffered_offset[2], buffered_offset[3],
                                  dist_buffered[0], dist_buffered[1], dist_buffered[2], dist_buffered[3]);
                for (size_t id4 = 0; id4 < 4; id4++) {
                    update_one(buffered_real_ids[id4], dist_buffered[id4]);
                }
                n_buffered = 0;
            }

            if (si + 1 < sz) {
                _mm_prefetch((char *)(visited.data() + ids_residual[si + 1]), _MM_HINT_T0);
            }
        }

        for (size_t icnt = 0; icnt < n_buffered; icnt++) {
            auto *x = residual_point_.data() + buffered_offset[icnt] * aligned_dim_;
            float dist = distance_fn_->compare(query, x, aligned_dim_);
            update_one(buffered_real_ids[icnt], dist);
        }

    }

    for (size_t i = 0; i < k; i++) {
        idx[i] = q[i].second;
        if (dx) dx[i] = q[i].first;
    }
    return {hops, cmps};
}

template<typename dist_t>
std::pair<uint32_t, uint32_t> PointAggregationGraph<dist_t>::SearchInMemoryByRoute(const dist_t *query, size_t k, size_t l,
                                                                           uint32_t *idx, float *dx) {
    if (!is_build) return {};
    std::vector<Element> keys;
//    if (use_fine_grained_)
//        index_->search_with_path_early_stop(query, keys, l, l, radius_, early_stop_factor_);
//    else
    index_->search_with_path_early_stop(query, keys, 1, l, radius_, early_stop_factor_);
    std::vector<uint8_t> visited(num_, 0);

    int trans_sz = std::min(k, keys.size());
    std::vector<Element> q(trans_sz);
    q.reserve(k * 1.2);
    std::transform(keys.begin(), keys.begin() + trans_sz, q.begin(), [&](const Element& e) -> Element {
        return {e.first, agg2id_[e.second]};
    });

    std::make_heap(q.begin(), q.end(), std::less<Element>());
    auto update_one = [&](uint32_t id, float d) {
        if (q.size() < k) {
            q.emplace_back(d, id);
            std::push_heap(q.begin(), q.end(), std::less<Element>());
        } else if (d < q[0].first) {
            q.emplace_back(d, id);
            std::push_heap(q.begin(), q.end(), std::less<Element>());
            std::pop_heap(q.begin(), q.end(), std::less<Element>());
            q.pop_back();
        }
    };
    float minimal = keys[0].first;
    for (auto &[key_dist, key]: keys) {
//        float r1, r2;
//        r1 = use_fine_grained_ ? fine_grained_radius_[keys[0].second] : radius_;
//        r2 = use_fine_grained_ ? fine_grained_radius_[key] : radius_;
//        _mm_prefetch((char *)(ivf_size_.data() + key), _MM_HINT_T0);
//        _mm_prefetch((char *)(ivf_.data() + key), _MM_HINT_T0);

        if (key_dist > early_stop_factor_ * (minimal + 2 * radius_))
            break;
        if (ivf_size_[key] < 1)
            continue;


        auto sz = ivf_size_[key];
        auto& ids_residual = ivf_[key];

        int n_buffered = 0;
        uint32_t buffered_ids[4]; // keep inner id
        uint32_t buffered_real_ids[4]; // keep real id
        uint32_t buffered_offset[4];

        // apply vector instruction to calculate the distance
        for (size_t si = 0; si < sz; si++) {
            uint32_t real_id = ids_residual[si];
            bool get = visited[real_id];
            visited[real_id] = 1;
            buffered_ids[n_buffered] = si;
            buffered_real_ids[n_buffered] = real_id;
            buffered_offset[n_buffered] = id2offset_[real_id];
            n_buffered += get ? 0 : 1;
//            if (get) {
//                prefetch_vector_l2((char*)(residual_point_.data() + id2offset_[real_id]), sizeof(float) * dim_);
//            }

            if (n_buffered == 4) {
                float dist_buffered[4];
                distances_batch_4(residual_point_.data(), query, buffered_offset[0], buffered_offset[1], buffered_offset[2], buffered_offset[3],
                                  dist_buffered[0], dist_buffered[1], dist_buffered[2], dist_buffered[3]);
                for (size_t id4 = 0; id4 < 4; id4++) {
                    update_one(buffered_real_ids[id4], dist_buffered[id4]);
                }
                n_buffered = 0;
            }

            if (si + 1 < sz) {
                _mm_prefetch((char *)(visited.data() + ids_residual[si + 1]), _MM_HINT_T0);
            }
        }

        for (size_t icnt = 0; icnt < n_buffered; icnt++) {
            const dist_t *x = residual_point_.data() + buffered_offset[icnt] * aligned_dim_;
            float dist = distance_fn_->compare(query, x, aligned_dim_);
            update_one(buffered_real_ids[icnt], dist);
        }
    }

    for (size_t i = 0; i < k; i++) {
        idx[i] = q[i].second;
        if (dx) dx[i] = q[i].first;
    }
    return {};
}


template<typename dist_t>
void PointAggregationGraph<dist_t>::LoadFromMemoryPAG(const char *filename, uint32_t L) {
    const auto graph_path = std::string(filename) + "_graph";
    const auto mapper_path = std::string(filename) + "_mapper";
    const auto ivf_vec_path = std::string(filename) + "_ivf_vec";
    const auto residual_path = std::string(filename) + "_residual";
    const auto residual_mapper_path = std::string(filename) + "_residual_mapper";
    const auto ivf_path = std::string(filename) + "_ivf";
    const auto parameter_path = std::string(filename) + "_params.txt";

    /** Read parameter
     */
    {
        int code;
        std::ifstream reader(parameter_path, std::ios::in);
        reader  >>  dim_  >> aligned_dim_       >>  radius_         >>  ag_num_
                >>  L_          >>  R_              >>  alpha_
                >>  use_routing_path_redundancy_
                >>  capacity_   >>  extra_rate_     >>  redundant_num_
                >>  sigma_      >>  delta_          >> scale_factor_
                >>  early_stop_factor_
                >>  num_         >> sample_rate_    >> code >> use_fine_grained_;
        reader.close();

        switch (code)
        {
        case 0:
            metric_ = diskann::Metric::L2;
            break;
        case 1:
            metric_ = diskann::Metric::INNER_PRODUCT;
            break;
        case 2:
            metric_ = diskann::Metric::COSINE;
            break;
        case 3:
            metric_ = diskann::Metric::FAST_L2;
            break;
        }
    }

    /** Read mapper
     */
    {
        int64_t dummy_ag_num;
        std::ifstream reader(mapper_path, std::ios::binary | std::ios::in);
        reader.read((char*)&dummy_ag_num, sizeof(dummy_ag_num));
        assert(dummy_ag_num == ag_num_);
        agg2id_.resize(ag_num_);
        reader.read((char*)agg2id_.data(), sizeof(uint32_t) * ag_num_);
        reader.close();
    }


    /** Read IVF Vector:
     * [size of ivf] [capacity of partition]  [dimension of vector]
     * [real size of partition] [vector_1] [vector_2]...[vector_size] [dummy]...[dummy] -> total capacity
     */
    {
        int64_t dummy_ag_num; size_t dummy_capacity, dummy_dim;
        std::ifstream reader(ivf_vec_path, std::ios::binary | std::ios::in);
        reader.read((char*)&dummy_ag_num, sizeof(dummy_ag_num));
        reader.read((char*)&dummy_capacity, sizeof(dummy_capacity));
        reader.read((char*)&dummy_dim, sizeof(dummy_dim));
        assert(ag_num_ == dummy_ag_num && capacity_ == dummy_capacity && dim_ == dummy_dim);
//        ivf_vec_.resize(ag_num_);
        ivf_size_.resize(ag_num_);

        for (size_t i = 0; i < ag_num_; i++) {
//            auto &ivf_i = ivf_vec_[i];
            auto &sz = ivf_size_[i];
            reader.read((char*)&sz, sizeof(sz));
//            ivf_i.resize(dim_ * sz);
//            reader.read((char*)ivf_i.data(), sizeof(float) * dim_ * sz);
        }
        reader.close();
    }

    /** Read seq vector and residual number
     */
    {
        std::ifstream res_reader(residual_path, std::ios::binary | std::ios::in);
        std::ifstream mapper_reader(residual_mapper_path, std::ios::binary | std::ios::in);
        size_t dummy_dim;
        res_reader.read((char*)&res_num_, sizeof(res_num_));
        res_reader.read((char*)&dummy_dim, sizeof(dummy_dim));
        residual_point_.resize(res_num_ * aligned_dim_);
        uint32_t id;
        for (size_t i = 0; i < res_num_; i++) {
            res_reader.read((char*)(residual_point_.data() + i * aligned_dim_), sizeof(dist_t) * aligned_dim_);
            mapper_reader.read((char*)&id, sizeof(id));
            id2offset_[id] = i;
        }
        res_reader.close();
        mapper_reader.close();
    }



    /** Read IVF Ids:
     * [size of ivf] [capacity of partition]
     * [real size of partition] [id_1]...[id_size] [dummy]...[dummy] -> total capacity
     */
    {
        int64_t dummy_ag_num; size_t dummy_capacity; auto dummy_sz = ivf_size_[0];
        std::ifstream reader(ivf_path, std::ios::binary | std::ios::in);
        reader.read((char*)&dummy_ag_num, sizeof(dummy_ag_num));
        reader.read((char*)&dummy_capacity, sizeof(capacity_));
        ivf_.resize(ag_num_);
        for (size_t i = 0; i < ag_num_; i++) {
            auto &ivf_i = ivf_[i];
            auto sz = ivf_size_[i];
            ivf_i.resize(sz);
            reader.read((char*)&dummy_sz, sizeof(dummy_sz));
            assert(sz == dummy_sz);
            reader.read((char*)ivf_i.data(), sizeof(uint32_t) * sz);
        }
        reader.close();
    }

    if (use_fine_grained_) {
        std::string radius_path = std::string(filename) + "_radius";
        std::ifstream writer(radius_path, std::ios::binary | std::ios::in);
        fine_grained_radius_.resize(ag_num_);
        writer.read((char*)fine_grained_radius_.data(), sizeof(float) * ag_num_);
        writer.close();
    }

    /** recovery diskann **/
    auto config = std::make_unique<diskann::IndexConfig>(diskann::IndexConfigBuilder()
                                                             .with_metric(metric_)
                                                             .with_dimension(aligned_dim_)
                                                             .with_max_points(0)
                                                             .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                                                             .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                                                             .with_data_type(diskann_type_to_name<dist_t>())
                                                             .is_concurrent_consolidate(false)
                                                             .is_pq_dist_build(false)
                                                             .is_use_opq(false)
                                                             .with_num_pq_chunks(0)
                                                             .build());
    // cosine must be float
    if (metric_ == diskann::Metric::COSINE && std::is_same_v<dist_t, float>) {
        distance_fn_.reset((Distance<dist_t> *)new AVXNormalizedCosineDistanceFloat());
    } else {
        distance_fn_.reset(diskann::get_distance_function<dist_t>(metric_));
    }

    std::unique_ptr<Distance<dist_t>> distance;
    if (metric_ == diskann::Metric::COSINE && std::is_same_v<dist_t, float>) {
        distance.reset((Distance<dist_t> *)new AVXNormalizedCosineDistanceFloat());
    } else {
        distance.reset(diskann::get_distance_function<dist_t>(metric_));
    }
    auto data_store = std::make_shared<diskann::InMemDataStore<dist_t>>((location_t)num_, aligned_dim_,
                                                                   std::move(distance));
    std::shared_ptr<AbstractDataStore<dist_t>> pq_data_store = data_store;
    size_t max_reserve_degree =
        (size_t)(defaults::GRAPH_SLACK_FACTOR * 1.05 *
                 (config->index_write_params == nullptr ? 0 : config->index_write_params->max_degree));
    std::unique_ptr<AbstractGraphStore> graph_store = std::make_unique<InMemGraphStore>(num_, max_reserve_degree);

    index_ = std::make_unique<diskann::Index<dist_t, uint32_t , uint32_t>>(*config, data_store,
                                                                      std::move(graph_store), pq_data_store);

    index_->load(graph_path.c_str(), num_threads_, L);

    is_build = true;
}


template<typename dist_t>
void PointAggregationGraph<dist_t>::SaveFromMemoryPAG(const char* filename) {
    const auto graph_path = std::string(filename) + "_graph";
    const auto mapper_path = std::string(filename) + "_mapper";
    const auto ivf_vec_path = std::string(filename) + "_ivf_vec";
    const auto residual_path = std::string(filename) + "_residual";
    const auto residual_mapper_path = std::string(filename) + "_residual_mapper";
    const auto ivf_path = std::string(filename) + "_ivf";
    const auto parameter_path = std::string(filename) + "_params.txt";


    index_->save(graph_path.c_str());

    /** Write mapper:
     * [number of aggregation point]
     * #aggregation id: [n1], [n2], [n3]
     * */
    {
        std::ofstream writer(mapper_path, std::ios::binary | std::ios::out);
        writer.write((char*)&ag_num_, sizeof(ag_num_));
        writer.write((char*)agg2id_.data(), sizeof(uint32_t) * ag_num_);
        writer.close();
    }

    /** Write IVF Vector:
     * [size of ivf] [capacity of partition]  [dimension of vector]
     * [real size of partition] [vector_1] [vector_2]...[vector_size]  ||||[dummy]...[dummy] -> total capacity||||
     */
    {
        std::ofstream writer(ivf_vec_path, std::ios::binary | std::ios::out);
        writer.write((char*)&ag_num_, sizeof(ag_num_));
        writer.write((char*)&capacity_, sizeof(capacity_));
        writer.write((char*)&dim_, sizeof(dim_));
        for (size_t i = 0; i < ag_num_; i++) {
           auto sz = ivf_size_[i];
           writer.write((char*)&sz, sizeof(sz));
//           writer.write((char*)ivf_vec_[i].data(), sizeof(float) * dim_ * sz);
        }
        writer.close();
    }

    /** Write seq vector of Residual points
     * [residual number] [dim]
     * [v1] [v2] ... [vn]
     *
     *
     * [residual number] [s1] [s2] ... [sn]
     */
    {
        std::ofstream res_writer(residual_path, std::ios::binary | std::ios::out);
        std::ofstream mapper_writer(residual_mapper_path, std::ios::binary | std::ios::out);
        assert(res_num_ == residual_keep_offset_.size());

        res_writer.write((char*)&res_num_, sizeof(res_num_));
        res_writer.write((char*)&dim_, sizeof(dim_));

        for (auto id: residual_keep_offset_) {
            res_writer.write((char*)(data_ + id * aligned_dim_), sizeof(*data_) * aligned_dim_);
            mapper_writer.write((char*)&id, sizeof(id));
        }
        res_writer.close();
        mapper_writer.close();
    }

    /** Write IVF Real Ids:
     * [size of ivf] [capacity of partition]
     * [real size of partition] [id_1]...[id_size] [dummy]...[dummy] -> total capacity
     */
    {
        std::ofstream writer(ivf_path, std::ios::binary | std::ios::out);
        writer.write((char*)&ag_num_, sizeof(ag_num_));
        writer.write((char*)&capacity_, sizeof(capacity_));
        for (size_t i = 0; i < ag_num_; i++) {
           auto sz = ivf_size_[i];
           writer.write((char*)&sz, sizeof(sz));
           writer.write((char*)ivf_[i].data(), sizeof(uint32_t) * sz);
        }
        writer.close();
    }

    if (use_fine_grained_) {
        std::string radius_path = std::string(filename) + "_radius";
        std::ofstream writer(radius_path, std::ios::binary | std::ios::out);
        writer.write((char*)fine_grained_radius_.data(), sizeof(float) * ag_num_);
        writer.close();
    }

    /** Write Parameter
     * dim_, radius_, ag_num_, L_, R_,
     * use_routing_path_redundancy_, capacity_,
     * extra_rate_ = 0.1, redundant_num_ = 4,
     * float alpha_ = 0.8, delta_ = 0.1,
     * scale_factor_ = 5, early_stop_factor = 1.2
     */
    {
        std::ofstream writer(parameter_path, std::ios::out);
        writer  <<  dim_  << " "  << aligned_dim_   <<  " " <<  radius_                     <<  " "
                <<  ag_num_                         <<  " " <<  L_                          <<  " "
                <<  R_                              <<  " " << alpha_                       <<  " "
                <<  use_routing_path_redundancy_    <<  " " <<  capacity_                   <<  " "
                <<  extra_rate_                     <<  " " <<  redundant_num_              <<  " "
                <<  sigma_                          <<  " " <<  delta_                      <<  " "
                <<  scale_factor_                   <<  " " <<  early_stop_factor_          <<  " "
                <<  num_                            <<  " " <<  sample_rate_                <<  " "
                <<  metric_                         <<  " " <<  use_fine_grained_;
        writer.close();
    }
}


template<typename dist_t>
void PointAggregationGraph<dist_t>::build_pag_in_disk()
{
    capacity_ = (1 + extra_rate_) / sample_rate_;
    /*
    std::string ivf_file = index_file_ + "_ivf_vec";
    std::ofstream ivf_writer(ivf_file, std::ios::binary | std::ios::out);
    initial_ivf_disk(ivf_writer);
     */


    std::unordered_set<uint32_t> not_belong;

for (int iter = 0; iter < 1; iter++)
{
    std::string residual_file = index_file_ + "_residual";
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream residual_reader(residual_file, read_blk_size);

    ivf_.resize(ag_num_ * scale_factor_, std::vector<uint32_t>(capacity_, -1));
    ivf_size_.resize(ag_num_ * scale_factor_, 0);

#pragma omp parallel for schedule(static, 2000)
    for (size_t i = 0; i < res_num_; i++)
    {
        if (i % 10000000 == 0) {
            std::cout << "#" << i << ": "
                      << "agg_num: " << ag_num_
                      << std::endl;
        }

        std::vector<dist_t> current_vec(aligned_dim_);
        uint32_t id;

#pragma omp critical
        {
            residual_reader.read((char*)&id, sizeof(id));
            residual_reader.read((char*)current_vec.data(), sizeof(dist_t) * aligned_dim_);
        }

        if (not_belong.count(id))
            continue;

        std::vector<uint32_t> idx;
        std::vector<float> dist;
        if (use_routing_path_redundancy_)
        {
            // search return the path
            std::vector<Element> path;
            index_->search_with_path(current_vec.data(), path, 1, L_);
            idx.resize(path.size());
            dist.resize(path.size());
            for (size_t r = 0; r < path.size(); r++)
            {
                idx[r] = path[r].second;
                dist[r] = path[r].first;
            }
        }
        else
        {
            // search nearest
            idx.resize(L_);
            dist.resize(L_);
            index_->search(current_vec.data(), L_, L_, idx.data(), dist.data());
        }

        std::vector<uint32_t> partitions;
        if (!use_fine_grained_)
            partitions = limit_prune(idx, dist);
        else
            partitions = limit_prune_with_prob(idx, dist);
//            partitions = limit_prune_with_fine_grained_radius(idx, dist);
#pragma omp critical
        {
            bool place = false;
            for (int r = 0; r < redundant_num_ && r < partitions.size(); r++)
            {
                uint32_t p = partitions[r];
                auto &sz = ivf_size_[p];
                if (sz < capacity_)
                {
                    // write when keep in disk
                    /* write_vec_to_partition(current_vec.data(), p, sz, ivf_writer); */
                    ivf_[p][sz] = id;
                    sz++;
                    place = true;
                }
            }
            if (!place)
            {
                promote_as_aggregation_disk(id, current_vec.data());
                not_belong.insert(id);
            }
        }
    }
}

    /*
    ivf_writer.close();
    */


    // record the ivf offset
    ivf_offset_.reserve(ag_num_);
    size_t prefix_sum = 0;
    for (size_t i = 0; i < ag_num_; i++) {
        ivf_offset_.push_back(prefix_sum);
        prefix_sum += ivf_size_[i] * (aligned_dim_ * sizeof(dist_t) + sizeof(uint32_t) * 8);
    }
}


template<typename dist_t>
void PointAggregationGraph<dist_t>::sample_aggregation_in_disk()
{
    std::random_device rd;
    std::mt19937 mt(rd());

    std::string residual_file = std::string(index_file_) + "_residual";
    ag_num_ = static_cast<uint32_t>(sample_rate_ * num_);
    res_num_ = num_ - ag_num_;


    std::vector<uint32_t> position;
    position.reserve(ag_num_);

    for (uint32_t i = 0; i < num_; i++) {
        if (i < ag_num_) position.push_back(i);
        else {
            std::uniform_int_distribution<uint32_t> dist(0, i);
            uint32_t j = dist(mt);
            if (j < ag_num_)
                position[j] = i;
        }
    }


    std::unordered_set<uint32_t> agg_set(position.begin(), position.begin() + ag_num_);

    aggregation_point_.resize(scale_factor_ * ag_num_ * aligned_dim_);
    agg2id_.reserve(scale_factor_ * ag_num_);
    res2id_.reserve(res_num_);

    size_t blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader(data_file_, blk_size);
    cached_ofstream residual_writer(residual_file, blk_size);
    uint32_t dummy_npt, dummy_dim;
    base_reader.read((char*)&dummy_npt, sizeof(uint32_t));
    base_reader.read((char*)&dummy_dim, sizeof(uint32_t));
    assert(dummy_npt == num_ && dummy_dim == dim_);

    std::vector<dist_t> current_vec(aligned_dim_, 0);
    size_t ag_counter = 0;
    for (size_t i = 0; i < num_; i++) {
        uint32_t id = static_cast<uint32_t>(i);
        base_reader.read((char*)current_vec.data(), dim_ * sizeof(dist_t));
        if (distance_fn_->preprocessing_required()) // in case of diskann preprocess
            distance_fn_->preprocess_base_points(current_vec.data(), aligned_dim_, 1);
        if (agg_set.count(id)) {
            agg2id_.push_back(id);
            memcpy(aggregation_point_.data() + ag_counter * aligned_dim_, current_vec.data(), sizeof(dist_t) * dim_);
            ag_counter += 1;
        } else {
            res2id_.push_back(id);
            residual_writer.write((char*)&id, sizeof(id));
            residual_writer.write((char*)current_vec.data(), sizeof(dist_t) * aligned_dim_);
        }
    }
    assert(ag_counter == ag_num_);
}


template<typename dist_t>
void PointAggregationGraph<dist_t>::BuildInDisk(const char* data_file, const char* index_file) {
    data_file_  = data_file;
    index_file_ = index_file;
    get_bin_metadata(data_file, num_, dim_);
    aligned_dim_ = ROUND_UP(dim_, 8);
    sample_aggregation_in_disk();
    build_pg();
    radius_ = sample_radius_based_on_edge();
    fine_grained_radius_.resize(ag_num_);
    if (use_fine_grained_)
    {
        sample_radius_of_every_agg();
    }


    build_pag_in_disk();
    is_build = true;
}


template<typename dist_t>
void PointAggregationGraph<dist_t>::initial_ivf_disk(std::ofstream &ivf_writer)
{
    std::vector<float> dummy_vecs(capacity_ * dim_, 0);
    //    size_t blk_size = 64 * 1024 * 1024;
    //    cached_ofstream ivf_writer(ivf_vec_path, blk_size);
    for (size_t i = 0; i < ag_num_; i++) {
        ivf_writer.write((char*)dummy_vecs.data(), capacity_ * dim_ * sizeof(float));
    }
    //    ivf_writer.close();
}


template<typename dist_t>
void PointAggregationGraph<dist_t>::write_vec_to_partition(const dist_t *x, uint32_t p, unsigned int sz, std::ofstream &ivf_appender)
{
    size_t line_size = capacity_ * dim_;
    size_t offset = (line_size * p + dim_ * sz) * sizeof(*x);
    ivf_appender.seekp(offset, ivf_appender.beg);
    ivf_appender.write((char*)x, sizeof(*x) * dim_);
}


template<typename dist_t>
void PointAggregationGraph<dist_t>::promote_as_aggregation_disk(uint32_t real_id, const dist_t *vec)
{
    if (ag_num_ >= ivf_.size())
    {
        diskann::cout << "Too many promotion\n";
        exit(1);
    }
    /*
    std::vector<float> dummy_vecs(dim_ * capacity_, 0);
    std::ofstream ivf_writer(index_file_ + "_ivf_vec", std::ios::app | std::ios::binary);
    ivf_writer.seekp(0, ivf_writer.end);
    ivf_writer.write((char*)dummy_vecs.data(), sizeof(float) * dim_ * capacity_);
     */
    agg2id_.push_back(real_id);
//    memcpy(aggregation_point_.data() + ag_num_ * dim_,
//           vec,
//           sizeof (dist_t) * dim_);
    index_->insert_point(vec, ag_num_);
    ag_num_++;
    res_num_--;
}

template<typename dist_t>
std::pair<uint32_t, uint32_t> PointAggregationGraph<dist_t>::SearchInDiskByTopK(dist_t *query, size_t k, size_t l, size_t nprobe,
                                                                        uint32_t *idx, float *dx) {
//    io_mgr_->set_query(query);
//    size_t line_sz = sizeof(float) * dim_ * capacity_;
    std::vector<Element> q;
    q.reserve(k * 1.2);
    std::vector<uint32_t> keys(nprobe);
    std::vector<float> dists(nprobe);
    index_->search(query, nprobe, l, keys.data(), dists.data());

    size_t id;
    auto& io_mgr = pool_->Get(id);
    io_mgr->set_query(query);

    for (size_t i = 0; i < nprobe; i++) {
        auto key = keys[i];
        if (ivf_size_[key] > 0)
            io_mgr->read(fd_, key);
    }

    std::vector<uint8_t> visited(num_, 0);
    for (size_t i = 0; i < nprobe; i++)
    {
        auto key_id = agg2id_[keys[i]];
        auto key_dist = dists[i];
        if (q.size() < k)
        {
            q.emplace_back(key_dist, key_id);
            std::push_heap(q.begin(), q.end(), std::less<Element>());
        }
        else if (key_dist < q[0].first)
        {
            q.emplace_back(key_dist, key_id);
            std::push_heap(q.begin(), q.end(), std::less<Element>());
            std::pop_heap(q.begin(), q.end(), std::less<Element>());
            q.pop_back();
        }
    }

//        auto sz = ivf_size_[key];
//        auto& ids_residual = ivf_[key];
//        std::vector<float> vec_residual(dim_ * sz);


    io_mgr->process(k, q);
    auto info = io_mgr->get_info();
    pool_->Put(id);

//        auto update_one = [&](uint32_t id, float d) {
//            if (q.size() < k) {
//                q.emplace_back(d, id);
//                std::push_heap(q.begin(), q.end(), std::less<Element>());
//            } else if (d < q[0].first) {
//                q.emplace_back(d, id);
//                std::push_heap(q.begin(), q.end(), std::less<Element>());
//                std::pop_heap(q.begin(), q.end(), std::less<Element>());
//                q.pop_back();
//            }
//        };

//        int n_buffered = 0;
//        uint32_t buffered_ids[4]; // keep inner id
//        uint32_t buffered_real_ids[4]; // keep real id
//        for (size_t si = 0; si < sz; ++si) {
//            uint32_t real_id = ids_residual[si];
//            bool get = visited[real_id];
//            visited[real_id] = 1;
//            buffered_ids[n_buffered] = si;
//            buffered_real_ids[n_buffered] = real_id;
//            n_buffered += get ? 0 : 1;
//
//            if (n_buffered == 4) {
//                float dist_buffered[4];
//                distances_batch_4(vec_residual.data(), query, buffered_ids[0], buffered_ids[1], buffered_ids[2], buffered_ids[3],
//                                  dist_buffered[0], dist_buffered[1], dist_buffered[2], dist_buffered[3]);
//                for (size_t id4 = 0; id4 < 4; id4++) {
//                    update_one(buffered_real_ids[id4], dist_buffered[id4]);
//                }
//                n_buffered = 0;
//            }
//
//            if (si + 1 < sz) {
//                _mm_prefetch((char *)(visited.data() + ids_residual[si + 1]), _MM_HINT_T0);
//            }
//        }
//
//        for (size_t icnt = 0; icnt < n_buffered; icnt++) {
//            const float *x = vec_residual.data() + buffered_ids[icnt] * dim_;
//            float dist = distance_fn_->compare(query, x, dim_);
//            update_one(buffered_real_ids[icnt], dist);
//        }
//
//    }

    size_t kv = std::min(k, q.size());
    for (size_t i = 0; i < kv; i++) {
        idx[i] = q[i].second;
        if (dx) dx[i] = q[i].first;
    }
    return info;
}

template<typename dist_t>
std::pair<size_t, size_t> PointAggregationGraph<dist_t>::SearchInDiskByRoute(dist_t *query, size_t k, size_t l,
                                                                        uint32_t *idx, float *dx)
{
    size_t id;
    auto& io_mgr = pool_->Get(id);
    io_mgr->set_query(query);
    index_->search_along_path_early_stop_disk(query, k, l, agg2id_, ivf_, ivf_size_, radius_, early_stop_factor_, fd_, io_mgr, idx, dx);
    auto info = io_mgr->get_info();
    pool_->Put(id);
//    std::string ivf_vec_file = index_file_ + "_ivf_vec";
//    index_->search_along_path_early_stop_disk(query, l, l, agg2id_, ivf_, ivf_size_, radius_, early_stop_factor_, ivf_vec_file);
//    std::vector<Element> keys;
//    index_->search_with_path_early_stop(query, keys, l, l, radius_, early_stop_factor_);
//    std::vector<uint8_t> visited(num_, 0);
//    std::vector<Element> q(k);
//    q.reserve(k * 1.2);
//    std::transform(keys.begin(), keys.begin() + k, q.begin(), [&](const Element& e) -> Element {
//        return {e.first, agg2id_[e.second]};
//    });
//
//    std::make_heap(q.begin(), q.end(), std::less<Element>());
//    for (auto &[key_dist, key]: keys) {
//        if (key_dist > early_stop_factor_ * (keys[0].first + 2 * radius_))
//            break;
//
//        auto sz = ivf_size_[key];
//        auto& ids_residual = ivf_[key];
////        auto& vec_residual = ivf_vec_[key];
//
//        auto update_one = [&](uint32_t id, float d) {
//            if (q.size() < k) {
//                q.emplace_back(d, id);
//                std::push_heap(q.begin(), q.end(), std::less<Element>());
//            } else if (d < q[0].first) {
//                q.emplace_back(d, id);
//                std::push_heap(q.begin(), q.end(), std::less<Element>());
//                std::pop_heap(q.begin(), q.end(), std::less<Element>());
//                q.pop_back();
//            }
//        };
//
//        int n_buffered = 0;
//        uint32_t buffered_ids[4]; // keep inner id
//        uint32_t buffered_real_ids[4]; // keep real id
//
//        // apply vector instruction to calculate the distance
//        for (size_t si = 0; si < sz; si++) {
//            uint32_t real_id = ids_residual[si];
//            bool get = visited[real_id];
//            visited[real_id] = 1;
//            buffered_ids[n_buffered] = si;
//            buffered_real_ids[n_buffered] = real_id;
//            n_buffered += get ? 0 : 1;
//
//            if (n_buffered == 4) {
//                float dist_buffered[4];
//                distances_batch_4(vec_residual.data(), query, buffered_ids[0], buffered_ids[1], buffered_ids[2], buffered_ids[3],
//                                  dist_buffered[0], dist_buffered[1], dist_buffered[2], dist_buffered[3]);
//                for (size_t id4 = 0; id4 < 4; id4++) {
//                    update_one(buffered_real_ids[id4], dist_buffered[id4]);
//                }
//                n_buffered = 0;
//            }
//
//            if (si + 1 < sz) {
//                _mm_prefetch((char *)(visited.data() + ids_residual[si + 1]), _MM_HINT_T0);
//            }
//        }
//
//        for (size_t icnt = 0; icnt < n_buffered; icnt++) {
//            const float *x = vec_residual.data() + buffered_ids[icnt] * dim_;
//            float dist = distance_fn_->compare(query, x, dim_);
//            update_one(buffered_real_ids[icnt], dist);
//        }
//    }
//
//    for (size_t i = 0; i < k; i++) {
//        idx[i] = q[i].second;
//        if (dx) dx[i] = q[i].first;
//    }
    return info;
}

template<typename dist_t>
void PointAggregationGraph<dist_t>::LoadFromDiskPAG(const char* filename, uint32_t L) {
    const auto graph_path = std::string(filename) + "_graph";
    const auto mapper_path = std::string(filename) + "_mapper";
    const auto ivf_offset_path = std::string(filename) + "_ivf_offset";
//    const auto ivf_vec_path = std::string(filename) + "_ivf_vec";
    const auto ivf_path = std::string(filename) + "_ivf";
    const auto parameter_path = std::string(filename) + "_params.txt";

    /** Read parameter
     */
    {
        int code;
        std::ifstream reader(parameter_path, std::ios::in);
        reader  >>  dim_     >> aligned_dim_   >>  radius_     >>  ag_num_
            >>  L_          >>  R_          >>  alpha_
            >>  use_routing_path_redundancy_
            >>  capacity_   >>  extra_rate_ >>  redundant_num_
            >>  sigma_      >>  delta_      >> scale_factor_
            >>  early_stop_factor_
            >>  num_         >> sample_rate_ >> code >> use_fine_grained_;
        reader.close();

        switch (code)
        {
        case 0:
            metric_ = diskann::Metric::L2;
            break;
        case 1:
            metric_ = diskann::Metric::INNER_PRODUCT;
            break;
        case 2:
            metric_ = diskann::Metric::COSINE;
            break;
        case 3:
            metric_ = diskann::Metric::FAST_L2;
            break;
        }
    }

    /** Read mapper
     */
    {
        int64_t dummy_ag_num;
        std::ifstream reader(mapper_path, std::ios::binary | std::ios::in);
        reader.read((char*)&dummy_ag_num, sizeof(dummy_ag_num));
//        assert(dummy_ag_num == ag_num_);
        agg2id_.resize(ag_num_);
        reader.read((char*)agg2id_.data(), sizeof(uint32_t) * ag_num_);
        reader.close();
    }


    if (use_fine_grained_) {
        std::string radius_path = std::string(filename) + "_radius";
        std::ifstream writer(radius_path, std::ios::binary | std::ios::in);
        fine_grained_radius_.resize(ag_num_);
        writer.read((char*)fine_grained_radius_.data(), sizeof(float) * ag_num_);
        writer.close();
    }


    /** read ivf offset:
     * [ag_number]
     * [off0] [off1] [off2] .... [offn]
     * [sz0] [sz1] [sz2] .... [szn]
     * */
    {
        int64_t dummy_ag_number;
        std::ifstream reader(ivf_offset_path, std::ios::binary | std::ios::in);

        reader.read((char*)&dummy_ag_number, sizeof(dummy_ag_number));
        assert(ag_num_ == dummy_ag_number);
        ivf_offset_.resize(ag_num_);
        ivf_size_.resize(ag_num_);
//        ivf_.resize(ag_num_);

        reader.read((char*)(ivf_offset_.data()), sizeof(size_t) * ag_num_);
        reader.read((char*)(ivf_size_.data()), sizeof(uint32_t) * ag_num_);
//        for (size_t i = 0; i < ag_num_; i++) {
//            auto& ivf_i = ivf_[i];
//            ivf_i.resize(ivf_size_[i]);
//            reader.read((char*)(ivf_i.data()), sizeof(uint32_t) * ivf_size_[i]);
//        }
        reader.close();
    }


    /** Read IVF Ids:
     * [size of ivf] [capacity of partition]
     * [real size of partition] [id_1]...[id_size] [dummy]...[dummy] -> total capacity
    {
        int64_t dummy_ag_num; size_t dummy_capacity;
        std::ifstream reader(ivf_path, std::ios::binary | std::ios::in);
        reader.read((char*)&dummy_ag_num, sizeof(dummy_ag_num));
        reader.read((char*)&dummy_capacity, sizeof(capacity_));
        ivf_.resize(ag_num_);
        for (size_t i = 0; i < ag_num_; i++) {
            auto &ivf_i = ivf_[i];
            auto &sz = ivf_size_[i];
            reader.read((char*)&sz, sizeof(sz));
            ivf_i.resize(sz);
//            assert(sz == dummy_sz);
            reader.read((char*)ivf_i.data(), sizeof(uint32_t) * sz);
        }
        reader.close();
    }
     */

    /** recovery diskann **/
    auto config = std::make_unique<diskann::IndexConfig>(diskann::IndexConfigBuilder()
                                                             .with_metric(metric_)
                                                             .with_dimension(aligned_dim_)
                                                             .with_max_points(0)
                                                             .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                                                             .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                                                             .with_data_type(diskann_type_to_name<dist_t>())
                                                             .is_concurrent_consolidate(false)
                                                             .is_pq_dist_build(false)
                                                             .is_use_opq(false)
                                                             .with_num_pq_chunks(0)
                                                             .build());

    if (metric_ == diskann::Metric::COSINE && std::is_same_v<dist_t, float>) {
        distance_fn_.reset((Distance<dist_t> *)new AVXNormalizedCosineDistanceFloat());
    } else {
        distance_fn_.reset(diskann::get_distance_function<dist_t>(metric_));
    }

    std::unique_ptr<Distance<dist_t>> distance;
    if (metric_ == diskann::Metric::COSINE && std::is_same_v<dist_t, float>) {
        distance.reset((Distance<dist_t> *)new AVXNormalizedCosineDistanceFloat());
    } else {
        distance.reset(diskann::get_distance_function<dist_t>(metric_));
    }
    auto data_store = std::make_shared<diskann::InMemDataStore<dist_t>>((location_t)ag_num_, aligned_dim_,
                                                                       std::move(distance));
    std::shared_ptr<AbstractDataStore<dist_t>> pq_data_store = data_store;
    size_t max_reserve_degree =
        (size_t)(defaults::GRAPH_SLACK_FACTOR * 1.05 *
                 (config->index_write_params == nullptr ? 0 : config->index_write_params->max_degree));
    std::unique_ptr<AbstractGraphStore> graph_store = std::make_unique<InMemGraphStore>(ag_num_, max_reserve_degree);

    index_ = std::make_unique<diskann::Index<dist_t, uint32_t , uint32_t>>(*config, data_store,
                                                                         std::move(graph_store), pq_data_store);

    index_->load(graph_path.c_str(), num_threads_, L);

//    io_mgr_.reset(new AsyncIOManagerUring(dim_, num_, ivf_, ivf_offset_, ivf_size_, distance_fn_));
    pool_.reset(new IOManagerPool(1, aligned_dim_, num_, ivf_, ivf_offset_, ivf_size_, distance_fn_));
    if ((fd_ = open(ivf_path.c_str(), O_RDONLY)) < 0) {
        std::cerr << "open file" << std::endl;
        exit(1);
    }
    is_build = true;
}

template<typename dist_t>
void PointAggregationGraph<dist_t>::SaveFromDiskPAG()
{
    const auto graph_path = index_file_ + "_graph";
    const auto mapper_path = index_file_ + "_mapper";
    const auto ivf_path = index_file_ + "_ivf";
    const auto ivf_offset_path = index_file_ + "_ivf_offset";
    const auto parameter_path = index_file_ + "_params.txt";
    const auto residual_file = std::string(index_file_) + "_residual";

    index_->save(graph_path.c_str());
    index_.reset();

    /** Write mapper:
     * [number of aggregation point]
     * #aggregation id: [n1], [n2], [n3]
     * */
    {
        std::ofstream writer(mapper_path, std::ios::binary | std::ios::out);
        writer.write((char*)&ag_num_, sizeof(ag_num_));
        writer.write((char*)agg2id_.data(), sizeof(uint32_t) * ag_num_);
        writer.close();
    }


    /** Write ivf offset:
     * [ag_number]
     * [off0] [off1] [off2] .... [offn]
     * [sz0] [sz1] [sz2] .... [szn]
     * */
    {
        std::ofstream writer(ivf_offset_path, std::ios::binary | std::ios::out);
        assert(ag_num_ == ivf_offset_.size());
        writer.write((char*)&ag_num_, sizeof(ag_num_));
        writer.write((char*)(ivf_offset_.data()), sizeof(size_t) * ag_num_);
        writer.write((char*)(ivf_size_.data()), sizeof(uint32_t) * ag_num_);
        for (size_t i = 0; i < ag_num_; i++) {
            const auto &ivf_i = ivf_[i];
            writer.write((char*)(ivf_i.data()), ivf_size_[i] * sizeof(uint32_t));
        }
        writer.close();
    }


    /** Write IVF Real Ids and vector:
     * [size of ivf]
     * [real size of partition] [id_1]...[id_size] -> sz
     */
    {
        std::vector<uint32_t> dummy_ids(7, 0);
        size_t dummy_offset = sizeof(uint32_t) * 2,
               line_size = dim_ * sizeof(dist_t);
//        std::ifstream reader(data_file_, std::ios::in | std::ios::binary);

        size_t num, dim;
        diskann::get_bin_metadata(data_file_, num, dim);
        dataset_.resize(num * aligned_dim_, 0);
        std::ifstream reader(data_file_, std::ios::binary);
        int npts_i32, dim_i32;
        reader.read((char *)&npts_i32, sizeof(int));
        reader.read((char *)&dim_i32, sizeof(int));

        for (size_t i = 0; i < num; i++)
        {
            reader.read((char *)(dataset_.data() + i * aligned_dim_), dim * sizeof(dist_t));
        }

        std::ofstream writer(ivf_path, std::ios::binary | std::ios::out);
//        std::vector<dist_t> dummy_vec(dim_);
        dist_t *dummy_vec;
//        std::vector<dist_t> dummy_vec(aligned_dim_, 0);
        for (size_t i = 0; i < ag_num_; i++) {
            const auto & ivf_i = ivf_[i];
            for (int j = 0; j < ivf_size_[i]; j++) {
                auto id = ivf_i[j];
//                memcpy(dummy_vec.data(), dataset_.data() + id * dim, sizeof(dist_t) * dim);
                dummy_vec = dataset_.data() + id * aligned_dim_;

//                reader.seekg(dummy_offset + line_size * id, reader.beg);
//                reader.read((char*)(dummy_vec.data()), sizeof(dist_t) * dim_);

                writer.write((char*)(&id), sizeof(id));
                writer.write((char*)(dummy_ids.data()), sizeof(id) * 7);
//                writer.write((char*)(&id), sizeof(id));
                if (distance_fn_->preprocessing_required())
                    distance_fn_->preprocess_base_points(dummy_vec, aligned_dim_, 1);
                writer.write((char*)(dummy_vec), sizeof(dist_t) * aligned_dim_);
            }
        }
        reader.close();
        writer.close();
    }


    if (use_fine_grained_) {
        std::string radius_path = std::string(index_file_) + "_radius";
        std::ofstream writer(radius_path, std::ios::binary | std::ios::out);
        writer.write((char*)fine_grained_radius_.data(), sizeof(float) * ag_num_);
        writer.close();
    }

    /** Write Parameter
     * dim_, radius_, ag_num_, L_, R_,
     * use_routing_path_redundancy_, capacity_,
     * extra_rate_ = 0.1, redundant_num_ = 4,
     * float alpha_ = 0.8, delta_ = 0.1,
     * scale_factor_ = 5, early_stop_factor = 1.2
     */
    {
        std::ofstream writer(parameter_path, std::ios::out);
        writer  <<  dim_  << " " << aligned_dim_   <<  " " <<  radius_                     <<  " "
               <<  ag_num_                         <<  " " <<  L_                          <<  " "
               <<  R_                              <<  " " << alpha_                       <<  " "
               <<  use_routing_path_redundancy_    <<  " " <<  capacity_                   <<  " "
               <<  extra_rate_                     <<  " " <<  redundant_num_              <<  " "
               <<  sigma_                          <<  " " <<  delta_                      <<  " "
               <<  scale_factor_                   <<  " " <<  early_stop_factor_          <<  " "
               <<  num_                            <<  " " <<  sample_rate_                <<  " "
               <<  metric_                         <<  " " <<  use_fine_grained_;
        writer.close();
    }
}


template class PointAggregationGraph<float>;

template class PointAggregationGraph<uint8_t>;
}





