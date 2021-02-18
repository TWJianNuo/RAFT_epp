#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA forward declarations
void epp_inflation_cuda(
    torch::Tensor instance,
    torch::Tensor infdst,
    torch::Tensor infsrc,
    int height,
    int width,
    int bs,
    int infheight,
    int infwidth
    );
void epp_compression_cuda(
    torch::Tensor instance,
    torch::Tensor compdst,
    torch::Tensor compsrc,
    int height,
    int width,
    int bs,
    int compheight,
    int compwidth
    );
void epp_batchselection_cuda(
    torch::Tensor batchidx,
    torch::Tensor seldst,
    torch::Tensor selsrc,
    int insnum,
    int bs,
    int srch,
    int srcw
    );
void epp_batchselection_backward_cuda(
    torch::Tensor batchidx,
    torch::Tensor selbckdst,
    torch::Tensor selbcksrc,
    int insnum,
    int bs,
    int srch,
    int srcw
    );
// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//torch::Tensor bnmorph_find_coorespond_pts(
void epp_inflation(
    torch::Tensor instance,
    torch::Tensor infdst,
    torch::Tensor infsrc,
    int height,
    int width,
    int bs,
    int infheight,
    int infwidth
    ) {
    CHECK_INPUT(instance);
    CHECK_INPUT(infdst);
    CHECK_INPUT(infsrc);
    epp_inflation_cuda(instance, infdst, infsrc, height, width, bs, infheight, infwidth);
    return;
}

void epp_compression(
    torch::Tensor instance,
    torch::Tensor compdst,
    torch::Tensor compsrc,
    int height,
    int width,
    int bs,
    int compheight,
    int compwidth
    ) {
    CHECK_INPUT(instance);
    CHECK_INPUT(compdst);
    CHECK_INPUT(compsrc);
    epp_compression_cuda(instance, compdst, compsrc, height, width, bs, compheight, compwidth);
    return;
}

void epp_batchselection(
    torch::Tensor batchidx,
    torch::Tensor seldst,
    torch::Tensor selsrc,
    int insnum,
    int bs,
    int srch,
    int srcw
    ) {
    CHECK_INPUT(batchidx);
    CHECK_INPUT(seldst);
    CHECK_INPUT(selsrc);
    epp_batchselection_cuda(batchidx, seldst, selsrc, insnum, bs, srch, srcw);
    return;
}

void epp_batchselection_backward(
    torch::Tensor batchidx,
    torch::Tensor selbckdst,
    torch::Tensor selbcksrc,
    int insnum,
    int bs,
    int srch,
    int srcw
    ) {
    CHECK_INPUT(batchidx);
    CHECK_INPUT(selbckdst);
    CHECK_INPUT(selbcksrc);
    epp_batchselection_backward_cuda(batchidx, selbckdst, selbcksrc, insnum, bs, srch, srcw);
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("epp_inflation", &epp_inflation, "inflation operation");
  m.def("epp_compression", &epp_compression, "compression operation");
  m.def("epp_batchselection", &epp_batchselection, "batch selection operation");
  m.def("epp_batchselection_backward", &epp_batchselection_backward, "batch selection operation backward");
}
