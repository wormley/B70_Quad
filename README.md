# B70_Quad
Recipes For LLMs on 4x Intel Arc B70 Cards

Assume all content is 100% LLM generated unless otherwise noted.


qfn-38-fp8-4xb70-mtp Mostly working recipe for 4 x B70 and 128GB RAM to run Qwen Flash Next 3.8 at FP8 with MTP. 
 - This version does not presently support both MTP and Prefix caching at the same time, check READMEs for details

llama.cpp <https://github.com/wormley/llama.cpp> Fork with optimizations for Qwen 3.8 Flash Next Q8 and UD-Q6_K_XL as well as GLM 5.3 Flash UD-Q4_K_XL, check <https://github.com/wormley/llama.cpp/blob/glm-rebase/README_FORK.md> fr the current change list.
