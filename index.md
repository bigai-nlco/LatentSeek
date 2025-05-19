---
layout: default

title: > 
  Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space
authors:
    - name: Hengli Li
      tag: <i class="fas fa-star" style='font-size:11px'></i> 1, 2
      url: https://github.com/Henry839
    - name: Chenxi Li
      tag: <i class="fas fa-star" style='font-size:11px'></i> 3
      url: https://openreview.net/profile?id=~Chenxi_Li7
    - name: Tong Wu
      tag: 2
      url: https://wutong4012.github.io/
    - name: Xuekai Zhu
      tag: 2, 4
      url: https://xuekai-zhu.github.io/Xuekai-Zhu/
    - name: Yuxuan Wang
      tag: 2
      url: https://github.com/patrick-tssn
    - name: Zhaoxin Yu
      tag: 5
      url: https://openreview.net/profile?id=~Zhaoxin_Yu1
    - name: Eric Hanchen Jiang
      tag: 6
      url: https://www.ericjiang.info/
    - name: Zixia Jia
      tag: 2
      url: https://scholar.google.com/citations?user=FdwGDyoAAAAJ&hl=zh-CN
    - name: Song-Chun Zhu
      tag: 1,2,3
      url: https://zhusongchun.net/
    - name: Ying Nian Wu
      tag: 6
      url: http://www.stat.ucla.edu/~ywu/
    - name: Zilong Zheng
      url: https://zilongzheng.github.io
      tag: 2, <i class="fa fa-envelope"></i>
affiliations:
    - name: Institute for Artificial Intelligence, Peking University
      tag: 1
    - name: NLCo Lab, Beijing Institute for General Artificial Intelligence
      tag: 2
    - name: Department of Automation, Tsinghua University
      tag: 3
    - name: Shanghai Jiao Tong University 
      tag: 4
    - name: Institute of Automation, Chinese Academy of Sciences 
      tag: 5
    - name: University of California, Los Angeles
      tag: 6
misc: > 
  <sup><i class="fas fa-star" style='font-size:11px'></i></sup> Equal Contribution.
  <sup><i class="fa fa-envelope"></i></sup> Corresponding authors.

arxiv: https://arxiv.org/abs/2502.18890
code: https://github.com/bigai-nlco/LatentSeek
---

<div class="container is-max-desktop">
<div class="hero-body">
<figure class="image" style="display: flex; justify-content: center; align-items: center; flex-direction: column;" id="table1">
  <img src="{{ 'LatentSeek/assets/img/LatentSeek.jpg' | relative_url }}" style="width: 100%; max-width: 1000px; height: auto"/>
      <figcaption><span class="dnerf">Figure 1.</span> Comparison of LatentSeek with RL-based fine-tuning and Prompt Engineering. RL-based fine-tuning methods generally require iterative updates to model parameters guided by reward signals. Prompt engineering approaches depend heavily on manually designed prompts. In contrast, LatentSeek performs optimization within the latent space.</figcaption>
</figure>
</div>
</div>

<section class="section">
    <div class="container is-max-desktop" markdown="1"> 
<h2 style="font-size: 2em; font-weight: bold;">Introduction</h2>
Reasoning ability, a core component of human intelligence, continues to pose a significant challenge for Large Language Models (LLMs) in the pursuit of AGI. Although model performance has improved under the training scaling law, significant challenges remain, particularly with respect to training algorithms—such as catastrophic forgetting—and the limited availability of novel training data. As an alternative, test-time scaling enhances reasoning performance by increasing test-time computation without parameter updating. Unlike prior methods in this paradigm focused on token space, we propose leveraging latent space for more effective reasoning and better adherence to the test-time scaling law.
<br/>
The introduced framework, named LatentSeek, enhances LLM reasoning through <b>Test Time Instance-level Adaptation (TTIA)</b> within the model's <b>latent space</b>. The latent representations are optimized during the test time using the policy gradient method, to maximize the expected reward. These optimized representations are subsequently decoded into token sequences, which are utilized to compute a new reward, which are then used to guide the next iteration.
      
</div>
</section>

<section class="section">
    <div class="container is-max-desktop" markdown="1"> 
<h2 style="font-size: 2em; font-weight: bold;">TTIA in Latent Space</h2>
<br/>
Given a reasoning problem instance $$\mathbf{c}$$ as a context prompt, a pre-trained auto-regressive language model $$\pi$$, a reasoning token sequence $$\mathbf{x} = (x\_1, x\_2, \ldots, x\_T)$$, and denote the corresponding sequence of latent representations of $$\mathbf{x}$$ as $$\mathbf{z} = (z\_1, z\_2, z\_3, \ldots, z\_T)$$, the objective is:

$$ \mathbf{z}^* = \arg\max_{\mathbf{z}} \mathbb{E}_{\mathbf{x} \sim \pi(\mathbf{x}|\mathbf{z})}[R(\mathbf{x}, \mathbf{c})]， $$

where $$R(\mathbf{x}, \mathbf{c})$$ is the reward function.
<h2 style="font-size: 1em; font-weight: bold;">Test-Time Optimization of Latent Representations</h2>

Assuming the *independence of the latent representations*, the test-time optimization is:

$$ \mathbf{z} \leftarrow \mathbf{z} + \eta  \nabla_{\mathbf{z}} \mathcal{J}(\mathbf{z}), $$

and the gradient is calculated as follows:

$$ [\nabla_{\mathbf{z}}\mathcal{J}(\mathbf{z})]_t =\mathbb{E}_{\mathbf{x}\sim\pi(\mathbf{x}|\mathbf{z})}\left[R(\mathbf{x},\mathbf{c})\nabla_{z_t} \log\pi(x_t|z_t)\right], $$

where $$t$$ denotes the position of the latent representation.

</div>
</section>

<section class="section">
    <div class="container is-max-desktop" markdown="1"> 
<h4 style="font-size: 2em; font-weight: bold;">LatentSeek Algorithm</h4>
<figure class="image" style="display: flex; justify-content: center; align-items: center; flex-direction: column;" id="table1">
<img src="{{ 'LatentSeek/assets/img/image-20250519142719249.png' | relative_url }}" style="width: 100%; max-width: 1000px; height: auto"/>
<figcaption><span class="dnerf">Algorithm 1.</span> The LatentSeek Algorithm.</figcaption>
</figure>
<br/>

The LatentSeek algorithm is described in Algorithm 1. This algorithm iteratively refines the latent representations based on the rewards of generated reasoning paths, effectively performing a guided search through the reasoning space specific to the given problem instance.  After each refinement step, the latent representations are decoded into tokens to calculate a reward signal. This signal is then employed to direct the search process in the subsequent iteration. Along with the reward signal, the final output $$\tilde{\mathbf{x}}$$ is also explicitly provided. The process runs for a small number of iterations (typically 2-10), stopping early if the reward exceeds a threshold.

</div>
</section>

<section class="section">
    <div class="container is-max-desktop" markdown="1"> 
<h2 style="font-size: 2em; font-weight: bold;">Experiments</h2>
<br/>

<h2 style="font-size: 1.5em; font-weight: bold;">Results</h2>

<figure class="image" style="display: flex; justify-content: center; align-items: center; flex-direction: column;" id="table1">
  <img src="{{ 'LatentSeek/assets/img/table1.jpg' | relative_url }}" style="width: 100%; max-width: 1000px; height: auto"/>
  <figcaption><span class="dnerf">Table 1.</span> Accuracy Score (%) on GSM8K, MATH-500 and AIME2024. Self: self-reward. Perfect Sparse Reward Model (PSRM): A reward value of 0 is assigned exclusively when the generated final answer exactly matches the ground truth. In all other cases, a reward of -1 is given.</figcaption>
</figure>
<br/>

<figure class="image" style="display: flex; justify-content: center; align-items: center; flex-direction: column;" id="table1">
  <img src="{{ 'LatentSeek/assets/img/table2.jpg' | relative_url }}" style="width: 100%; max-width: 1000px; height: auto"/>
  <figcaption><span class="dnerf">Table 2.</span> Accuracy score (%) compared with more baseline methods on GSM8K and MATH-500 datasets with Llama3.1-8B as backbone. Self: self-reward. Perfect Sparse Reward Model (PSRM): A reward value of 0 is assigned exclusively when the generated final answer exactly matches the ground truth. In all other cases, a reward of -1 is given.</figcaption>
</figure>

<br/>

**Work or Not: Is Latent Space Powerful Enough?**
<br/>
<ol>
  <li><b>Best Performance on GSM8K, MATH-500:</b> As demonstrated in Table 2, our method outperforms all baseline approaches across all GSM8K, and MATH-500 datasets.</li>
  <li><b>Superior Performance on Complex Problems:</b> As shown in Table 1, our approach consistently outperforms all baselines, achieving an average improvement of 4.73% points over CoT across all model families and prompt configurations.</li>
  <li><b>Generalizable across backbones:</b> LatentSeek demonstrates superior performance across multiple model families. Also, in terms of model scales, our method consistently outperforms all baseline models across diverse datasets and prompt types</li>
  <li><b>Generalizable across prompts:</b> The Qwen2.5 series was explicitly trained using Prompt 1; nevertheless, our methods still achieve notable performance gains.</li>
  <li><b>The large potential of LatentSeek, even when guided by sparse reward:</b> when using PSRM, LatentSeek achieves an average improvement of 19.12% score over the CoT method and surpasses the self-reward version by an average of 12.57% score.</li>
</ol>

<br/>
<br/>



<figure class="image" style="display: flex; justify-content: center; align-items: center; flex-direction: column;" id="table1">
  <img src="{{ 'LatentSeek/assets/img/scaling.jpg' | relative_url }}" style="width: 100%; max-width: 1000px; height: auto"/>
  <figcaption><span class="dnerf">Figure 2.</span> Test-Time Scaling. Performance with respect to the number of iterations. Blue: self-reward. Orange: PSRM.</figcaption>
</figure>
<br/>

<figure class="image" style="display: flex; justify-content: center; align-items: center; flex-direction: column;" id="table1">
  <img src="{{ 'LatentSeek/assets/img/extreme_scaling.png' | relative_url }}" style="width: 100%; max-width: 1000px; height: auto"/>
  <figcaption><span class="dnerf">Figure 3.</span> Performance of Extreme Scaling on MATH-500 \cite{hendrycksmath2021} and AIME2024. Setting the maximum update iteration to 256. K: average number of outputs or iterations.</figcaption>
</figure>
<br/>

**A dream of AGI: Can it be a method for Test-Time Scaling?**
<br/>
<ol>
  <li>Test-time scaling can be achieved <b>without necessitating a dense reward function</b> in our setting. (Figure 2)</li>
  <li>Searching through the latent space offers a <b>promising new direction</b> for test-time scaling. (Figure 2)</li>
  <li>The latent space represents a more <b>efficient option</b> for test-time scaling compared to the explicit space. (Figure 3)</li>
</ol>
<br/>


</div>
</section>

<section class="section">
    <div class="container is-max-desktop" markdown="1"> 
<h2 style="font-size: 2em; font-weight: bold;">BibTex</h2>

```bibtex
TBA

```

</div>
</section>
