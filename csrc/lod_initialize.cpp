// Block-local LoD initialization; no CUDA allocation or scene-sized scratch.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>

namespace py = pybind11;
using Vec = Eigen::Vector3d;
using Mat = Eigen::Matrix3d;
template<class T> using Array = py::array_t<T, py::array::c_style>;

static double clipped(double x, double lo, double hi) { return std::min(hi, std::max(lo,x)); }
static Mat rotation(const float* q) {
    double norm=0;
    for(int i=0;i<4;++i) norm+=double(q[i])*q[i];
    norm=std::max(std::sqrt(norm),1e-15);
    return Eigen::Quaterniond(q[0]/norm,q[1]/norm,q[2]/norm,q[3]/norm).toRotationMatrix();
}
static double area(const Vec& s) {
    // H-3DGS ellipseSurface: constant factors cancel in the weight ratios.
    return s.x()*s.y()+s.x()*s.z()+s.y()*s.z();
}
struct Leaf { Vec mean; Mat frame,covariance; double mass; };
struct Moments {
    double mass=0;
    int count=0;
    Vec mean=Vec::Zero();
    Mat scatter=Mat::Zero(); // weighted covariance sum about this mean
    std::array<double,48> sh{};
    void add(const Leaf& leaf,const float* features) {
        const double total=mass+leaf.mass;
        const Vec delta=leaf.mean-mean;
        scatter+=leaf.mass*leaf.covariance+(mass*(leaf.mass/total))*delta*delta.transpose();
        mean+=(leaf.mass/total)*delta;
        for(int k=0;k<48;++k) sh[k]+=leaf.mass*features[k];
        mass=total; ++count;
    }
    void combine(const Moments& other) {
        if(!other.count) return;
        if(!count) { *this=other; return; }
        const double total=mass+other.mass;
        const Vec delta=other.mean-mean;
        scatter+=other.scatter+(mass*(other.mass/total))*delta*delta.transpose();
        mean+=(other.mass/total)*delta;
        for(int k=0;k<48;++k) sh[k]+=other.sh[k];
        mass=total; count+=other.count;
    }
};
struct Frame { std::array<int,3> perm,sign; };
static const std::array<Frame,24>& frames() {
    static const auto result=[] {
        std::array<Frame,24> out;
        std::array<int,3> p{0,1,2}; int n=0;
        do {
            int inversions=(p[0]>p[1])+(p[0]>p[2])+(p[1]>p[2]);
            for(int a:{-1,1}) for(int b:{-1,1}) for(int c:{-1,1})
                if(((inversions%2)?-1:1)*a*b*c==1) out[n++]=Frame{p,{a,b,c}};
        } while(std::next_permutation(p.begin(),p.end()));
        return out;
    }();
    return result;
}
static void write_quaternion(float* g,const Mat& frame,const float* parent=nullptr) {
    Eigen::Quaterniond q(frame); q.normalize(); double sign=1;
    if(parent && q.w()*parent[7]+q.x()*parent[8]+q.y()*parent[9]+q.z()*parent[10]<0) sign=-1;
    g[7]=sign*q.w(); g[8]=sign*q.x(); g[9]=sign*q.y(); g[10]=sign*q.z();
}
static Mat align(float* g,const Mat& child,const Mat& parent,const float* parent_g) {
    const Mat dots=child.transpose()*parent;
    double best=-std::numeric_limits<double>::infinity(); int selected=0;
    for(int i=0;i<24;++i) {
        const auto& f=frames()[i]; double score=0;
        for(int j=0;j<3;++j) score+=f.sign[j]*dots(f.perm[j],j);
        if(score>best) { best=score; selected=i; }
    }
    const auto& f=frames()[selected]; Mat out;
    const float scales[3]={g[4],g[5],g[6]};
    for(int j=0;j<3;++j) { out.col(j)=child.col(f.perm[j])*f.sign[j]; g[4+j]=scales[f.perm[j]]; }
    write_quaternion(g,out,parent_g); return out;
}
static bool finalize(const Moments& m,float* g,float* h,Mat& frame) {
    if(!m.count) { frame=Mat::Identity(); return true; }
    Eigen::SelfAdjointEigenSolver<Mat> solver(m.scatter/m.mass);
    if(solver.info()!=Eigen::Success) return false;
    const Vec scale=solver.eigenvalues().cwiseMax(1e-12).cwiseSqrt();
    frame=solver.eigenvectors(); if(frame.determinant()<0) frame.col(0)*=-1;
    for(int k=0;k<3;++k) { g[k]=m.mean[k]; g[4+k]=std::log(scale[k]); }
    g[3]=std::log(std::max(m.mass/area(scale),1e-12)); write_quaternion(g,frame);
    for(int k=0;k<48;++k) h[k]=m.sh[k]/m.mass;
    return true;
}
static void partition(const float* g,int64_t* ids,int begin,int end,int slot,int capacity,int span,int64_t* order) {
    if(begin==end) return;
    if(capacity==span) {
        // Canonical family order, independent of nth_element or OpenMP order.
        std::sort(ids+begin,ids+end); std::copy(ids+begin,ids+end,order+slot); return;
    }
    Vec lo=Vec::Constant(std::numeric_limits<double>::infinity()),hi=-lo;
    for(int i=begin;i<end;++i) for(int k=0;k<3;++k) {
        lo[k]=std::min(lo[k],double(g[ids[i]*11+k])); hi[k]=std::max(hi[k],double(g[ids[i]*11+k]));
    }
    Eigen::Index axis; (hi-lo).maxCoeff(&axis);
    const int mid=begin+(end-begin+1)/2;
    std::nth_element(ids+begin,ids+mid,ids+end,[&](int64_t a,int64_t b) {
        float x=g[a*11+axis],y=g[b*11+axis]; return x<y || (x==y && a<b);
    });
    partition(g,ids,begin,mid,slot,capacity/2,span,order);
    partition(g,ids,mid,end,slot+capacity/2,capacity/2,span,order);
}
template<class T> static void check(const Array<T>& a,std::initializer_list<py::ssize_t> shape,bool writable=false) {
    if(a.ndim()!=int(shape.size())) throw std::invalid_argument("array rank mismatch");
    int dim=0;
    for(auto size:shape) if(a.shape(dim++)!=size) throw std::invalid_argument("array shape mismatch");
    if(writable && !a.writeable()) throw std::invalid_argument("output must be writable");
}
// pread does not change a shared file position. Retry short reads and signals;
// report failures after leaving OpenMP rather than throwing inside a worker.
static int read_rows(int fd,int64_t offset,float* buffer,size_t bytes) {
    char* dst=reinterpret_cast<char*>(buffer);
    while(bytes) {
        const ssize_t n=::pread(fd,dst,bytes,static_cast<off_t>(offset));
        if(n<0) { if(errno==EINTR) continue; return errno; }
        if(n==0) return EIO;
        dst+=n; offset+=n; bytes-=n;
    }
    return 0;
}
static void check_file(int fd,int64_t offset,int64_t rows,int columns) {
    struct stat status;
    if(fd<0 || offset<0 || rows<0 || ::fstat(fd,&status)!=0)
        throw std::invalid_argument("invalid tensor file descriptor or offset");
    if(offset>status.st_size || rows>(status.st_size-offset)/(columns*sizeof(float)))
        throw std::invalid_argument("tensor data exceeds file size (truncated source)");
}
template<class Reader> static void initialize_impl(int64_t source_rows,const Reader& read,int raw_columns,
    Array<int64_t> offsets,Array<int64_t> source_starts,Array<int64_t> order_starts,
    int S,int span,int coarse_span,int threads,int64_t progress_every,
    Array<float> leaves,Array<float> leaf_sh,Array<float> proxies,Array<float> proxy_sh,
    Array<int64_t> order,Array<int32_t> counts,Array<float> minimum,Array<float> maximum) {
    if(S<2 || (S&(S-1)) || span<2 || (span&(span-1)) || coarse_span<=span ||
       (coarse_span&(coarse_span-1)) || S%coarse_span || threads<1 || progress_every<0)
        throw std::invalid_argument("invalid block size, merge spans, threads or source offset");
    if(offsets.ndim()!=1 || offsets.size()<1) throw std::invalid_argument("invalid source layout");
    const int64_t B=offsets.size()-1;
    const int M=S/span,C=S/coarse_span,P=M+C,family=coarse_span/span;
    check(leaves,{B*S,11},true); check(leaf_sh,{B*S,48},true);
    check(proxies,{B*P,11},true); check(proxy_sh,{B*P,48},true); check(order,{B,S},true);
    check(counts,{B,S+P},true); check(minimum,{B,3},true); check(maximum,{B,3},true);
    check(source_starts,{B}); check(order_starts,{B});
    const auto* off=offsets.data(); const auto* src=source_starts.data();
    const auto* original=order_starts.data();
    if(off[0]!=0) throw std::invalid_argument("invalid block offsets");
    for(int64_t b=0;b<B;++b) {
        const int64_t n=off[b+1]-off[b];
        if(n<0 || n>S || src[b]<0 || src[b]>source_rows-n || original[b]<0)
            throw std::invalid_argument("invalid block count or source start");
    }
    float* lg=leaves.mutable_data(); float* lh=leaf_sh.mutable_data();
    float* pg=proxies.mutable_data(); float* ph=proxy_sh.mutable_data();
    int64_t* ord=order.mutable_data(); int32_t* cnt=counts.mutable_data();
    float* low=minimum.mutable_data(); float* high=maximum.mutable_data();
    std::atomic<int64_t> error{-1},io_error{-1}; int io_errno=0; frames();
    std::atomic<int64_t> completed{0}; int64_t reported=0;
    const auto started=std::chrono::steady_clock::now();
    const size_t buffer_floats=size_t(S)*(11+48+raw_columns);
    py::gil_scoped_release release;
    const int workers=static_cast<int>(std::min<int64_t>(threads,std::max<int64_t>(B,1)));
    #pragma omp parallel num_threads(workers)
    {
        // One allocation per worker for the entire scene: decoded geometry/SH
        // and raw on-disk rows occupy nonoverlapping regions of this buffer.
        // Only the current block's n valid rows are read and inspected.
        std::vector<float> parameter_buffer(buffer_floats);
        float* bg=parameter_buffer.data(); float* bh=bg+S*11;
        float* raw=bh+S*48;
        std::vector<Leaf> cached(S); std::vector<int64_t> ids(S);
        std::vector<Moments> mid(M); std::vector<Mat> mid_frame(M),coarse_frame(C);
        #pragma omp single
        {
            if(progress_every) {
                std::fprintf(stdout,"LOD_INIT_IO workers=%d parameter_buffer_bytes_per_worker=%zu\n",
                             omp_get_num_threads(),buffer_floats*sizeof(float));
                std::fflush(stdout);
            }
        }
        // Assign nearby blocks as workers become available: this keeps disk
        // reads localized and balances partial/empty blocks without reallocating
        // any worker buffer. Per-block arithmetic is independent of scheduling.
        #pragma omp for schedule(dynamic,1)
        for(int64_t b=0;b<B;++b) {
            if(io_error.load()>=0) continue;
            const int n=int(off[b+1]-off[b]);
            const int failure=n ? read(b,src[b],n,bg,bh,raw) : 0;
            if(failure) {
                #pragma omp critical(lod_initialization_io_error)
                {
                    if(io_error.load()<0) { io_errno=failure; io_error.store(b); }
                }
                continue;
            }
            float* out=lg+b*S*11; float* out_h=lh+b*S*48; float* proxy=pg+b*P*11; float* proxy_h=ph+b*P*48;
            int64_t* dst_order=ord+b*S; int32_t* dst_counts=cnt+b*(S+P);
            std::fill(out,out+S*11,0.f); std::fill(out_h,out_h+S*48,0.f);
            std::fill(proxy,proxy+P*11,0.f); std::fill(proxy_h,proxy_h+P*48,0.f);
            std::fill(dst_order,dst_order+S,int64_t(-1)); std::fill(dst_counts,dst_counts+S+P,0);
            for(int i=0;i<S;++i) { out[i*11+3]=-30; out[i*11+7]=1; for(int k=4;k<7;++k) out[i*11+k]=-15; }
            for(int i=0;i<P;++i) { proxy[i*11+3]=-30; proxy[i*11+7]=1; for(int k=4;k<7;++k) proxy[i*11+k]=-15; }
            bool valid=true;
            for(int i=0;i<n;++i) {
                for(int k=0;k<11;++k) valid &= std::isfinite(bg[i*11+k]);
                for(int k=0;k<48;++k) valid &= std::isfinite(bh[i*48+k]);
            }
            if(!valid) { error.store(b); continue; }
            std::iota(ids.begin(),ids.begin()+n,0); partition(bg,ids.data(),0,n,0,S,span,dst_order);
            // Activate once per leaf, shared by middle/coarse moments and alignment.
            for(int i=0;i<n;++i) {
                const float* x=bg+i*11; auto& leaf=cached[i];
                leaf.mean=Vec(x[0],x[1],x[2]); leaf.frame=rotation(x+7);
                Vec scales; for(int k=0;k<3;++k) scales[k]=std::exp(clipped(x[4+k],-30,30));
                leaf.covariance=leaf.frame*scales.cwiseProduct(scales).asDiagonal()*leaf.frame.transpose();
                leaf.mass=area(scales)/(1+std::exp(-clipped(x[3],-80,80)));
            }
            for(int m=0;m<M;++m) {
                mid[m]=Moments();
                for(int slot=m*span;slot<(m+1)*span;++slot) {
                    const int64_t id=dst_order[slot]; if(id<0) continue;
                    std::copy(bg+id*11,bg+(id+1)*11,out+slot*11); std::copy(bh+id*48,bh+(id+1)*48,out_h+slot*48);
                    dst_counts[slot]=1; mid[m].add(cached[id],bh+id*48);
                }
                dst_counts[S+m]=mid[m].count; valid &= finalize(mid[m],proxy+m*11,proxy_h+m*48,mid_frame[m]);
            }
            for(int c=0;c<C;++c) {
                Moments merged; for(int m=c*family;m<(c+1)*family;++m) merged.combine(mid[m]);
                dst_counts[S+M+c]=merged.count;
                valid &= finalize(merged,proxy+(M+c)*11,proxy_h+(M+c)*48,coarse_frame[c]);
            }
            if(!valid) { error.store(b); continue; }
            for(int m=0;m<M;++m) if(mid[m].count) {
                int c=m/family;
                mid_frame[m]=align(proxy+m*11,mid_frame[m],coarse_frame[c],proxy+(M+c)*11);
                for(int slot=m*span;slot<(m+1)*span;++slot) if(dst_order[slot]>=0) {
                    const int64_t id=dst_order[slot]; align(out+slot*11,cached[id].frame,mid_frame[m],proxy+m*11);
                    dst_order[slot]=original[b]+id;
                }
            }
            Vec lo=Vec::Constant(std::numeric_limits<double>::infinity()),hi=-lo;
            for(int i=0;i<S+P;++i) if(dst_counts[i]) {
                const float* x=i<S ? out+i*11 : proxy+(i-S)*11;
                double radius=3*std::exp(clipped(std::max({x[4],x[5],x[6]}),-30,30));
                for(int k=0;k<3;++k) { lo[k]=std::min(lo[k],x[k]-radius); hi[k]=std::max(hi[k],x[k]+radius); }
            }
            for(int k=0;k<3;++k) {
                low[b*3+k]=n ? std::nextafter(float(lo[k]),-std::numeric_limits<float>::infinity()) : 0;
                high[b*3+k]=n ? std::nextafter(float(hi[k]),std::numeric_limits<float>::infinity()) : 0;
            }
            if(progress_every) {
                const int64_t done=completed.fetch_add(1,std::memory_order_relaxed)+1;
                if(done==1 || done==B || done%progress_every==0) {
                    // Only progress output is serialized. Each worker writes
                    // parameters directly to the disjoint global range b*S.
                    #pragma omp critical(lod_initialization_progress)
                    {
                        if(done>reported) {
                            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                            std::fprintf(stdout,"LOD_INIT blocks=%lld/%lld seconds=%.1f\n",
                                         static_cast<long long>(done),static_cast<long long>(B),seconds);
                            std::fflush(stdout); reported=done;
                        }
                    }
                }
            }
        }
    }
    if(io_error.load()>=0) throw std::runtime_error("failed reading leaf block "+
        std::to_string(io_error.load())+": "+std::strerror(io_errno));
    if(error.load()>=0) throw std::runtime_error("nonfinite input or failed covariance decomposition in block "+std::to_string(error.load()));
}
// Array fixtures and binary PLY use the identical arithmetic/order above.
static void initialize_arrays(Array<float> geometry,Array<float> sh,Array<int64_t> offsets,
    Array<int64_t> source_starts,Array<int64_t> order_starts,
    int S,int span,int coarse_span,int threads,int64_t progress_every,
    Array<float> leaves,Array<float> leaf_sh,Array<float> proxies,Array<float> proxy_sh,
    Array<int64_t> order,Array<int32_t> counts,Array<float> minimum,Array<float> maximum) {
    if(geometry.ndim()!=2) throw std::invalid_argument("geometry must have rank two");
    const int64_t rows=geometry.shape(0);
    check(geometry,{rows,11}); check(sh,{rows,48});
    const float* g=geometry.data(); const float* h=sh.data();
    auto read=[&](int64_t,int64_t first,int n,float* bg,float* bh,float*) {
        std::copy(g+first*11,g+(first+n)*11,bg);
        std::copy(h+first*48,h+(first+n)*48,bh);
        return 0;
    };
    initialize_impl(rows,read,0,offsets,source_starts,order_starts,
                    S,span,coarse_span,threads,progress_every,
                    leaves,leaf_sh,proxies,proxy_sh,order,counts,minimum,maximum);
}
static void initialize_ply(int fd,int64_t data_offset,int columns,Array<int64_t> geometry_columns,
    Array<int64_t> sh_columns,int64_t source_rows,Array<int64_t> byte_offsets,Array<int64_t> offsets,
    Array<int64_t> source_starts,Array<int64_t> order_starts,
    int S,int span,int coarse_span,int threads,int64_t progress_every,
    Array<float> leaves,Array<float> leaf_sh,Array<float> proxies,Array<float> proxy_sh,
    Array<int64_t> order,Array<int32_t> counts,Array<float> minimum,Array<float> maximum) {
    if(columns<59 || columns>4096)
        throw std::invalid_argument("invalid PLY columns");
    check_file(fd,data_offset,source_rows,columns);
    if(offsets.ndim()!=1 || offsets.size()<1) throw std::invalid_argument("invalid source layout");
    const int64_t B=offsets.size()-1;
    check(byte_offsets,{B});
    check(source_starts,{B});
    const int64_t* addresses=byte_offsets.data(); const int64_t* src=source_starts.data();
    // Validate the shared table once before entering the parallel region.
    for(int64_t b=0;b<B;++b) {
        if(src[b]<0 || src[b]>source_rows ||
           addresses[b]!=data_offset+src[b]*columns*sizeof(float))
            throw std::invalid_argument("invalid precomputed PLY block address");
    }
    check(geometry_columns,{11}); check(sh_columns,{48});
    const auto* gc=geometry_columns.data(); const auto* hc=sh_columns.data();
    for(int k=0;k<11;++k) if(gc[k]<0 || gc[k]>=columns) throw std::invalid_argument("invalid geometry field");
    for(int k=0;k<48;++k) if(hc[k]<0 || hc[k]>=columns) throw std::invalid_argument("invalid SH field");
    auto read=[&](int64_t block,int64_t,int n,float* bg,float* bh,float* raw) {
        // pread leaves the shared file position untouched and overwrites the
        // same worker-local storage on every block, with no resize/allocation.
        int failure=read_rows(fd,addresses[block],raw,size_t(n)*columns*sizeof(float));
        if(failure) return failure;
        for(int i=0;i<n;++i) {
            const float* row=raw+size_t(i)*columns;
            for(int k=0;k<11;++k) bg[i*11+k]=row[gc[k]];
            for(int k=0;k<48;++k) bh[i*48+k]=row[hc[k]];
        }
        return 0;
    };
    initialize_impl(source_rows,read,columns,offsets,source_starts,order_starts,
                    S,span,coarse_span,threads,progress_every,
                    leaves,leaf_sh,proxies,proxy_sh,order,counts,minimum,maximum);
}
void bind_lod_initialize(py::module_& m) {
    m.def("initialize_lod_arrays",&initialize_arrays);
    m.def("initialize_lod_ply",&initialize_ply);
}
